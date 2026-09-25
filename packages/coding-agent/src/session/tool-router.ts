/**
 * Native Jev tool router (POC).
 *
 * Operates on the generic `Model + Context + SimpleStreamOptions` triple inside
 * {@link createSettingsAwareStreamFn}, BEFORE `mapOptionsForApi()` and any
 * provider-specific serialization. Output is always generic {@link ToolChoice}
 * (`{ type: "function", name }` or `"none"`); the existing provider mappers
 * (`mapOpenAiToolChoice`, `mapGoogleToolChoice`, `mapAnthropicToolChoice`,
 * `normalizeCodexToolChoice`) do the rest. There is intentionally NO
 * provider-conditional code here.
 *
 * Scope: named/forced tool, none, passthrough/fail-open. No direct mode, no
 * Jev-generated arguments, no provider wire manipulation.
 *
 * No-recursion argument (structural, not a flag): Jev backends never pass
 * through this wrapper. {@link TypeSafeJudge} talks HTTP directly, and the
 * chat fallback (`chatTextBackend` in `packages/ai/src/judgment/chat.ts`)
 * calls core `completeSimple` -> `streamSimple` directly. The wrapper is
 * composed per consumer in `sdk.ts` (`main`, `advisor`, `capture`,
 * `side-channel`) but only the `main` scope is eligible for routing, so a
 * judgment side request cannot re-enter the router.
 *
 * Fail-open contract: any guard trip, validation failure, error, or timeout
 * returns the caller's options object untouched.
 */
import type { ChoiceQuestion, Context, Judge, Model, SimpleStreamOptions, ToolChoice } from "@oh-my-pi/pi-ai";
import { logger } from "@oh-my-pi/pi-utils";
import type { Settings } from "../config/settings";
import { cfgToolRouterEnabled, cfgToolRouterMinConfidence, cfgToolRouterTimeoutMs } from "../tools/settings";

/** Choice label meaning "the request needs no tool; answer in prose". */
export const TOOL_ROUTER_NO_TOOL = "no_tool_needed";

/** Conservative default: mirrors the registered `toolRouter.minConfidence` default. */
export const TOOL_ROUTER_DEFAULT_MIN_CONFIDENCE: number = cfgToolRouterMinConfidence.default;
/** Conservative default: mirrors the registered `toolRouter.timeoutMs` default. */
export const TOOL_ROUTER_DEFAULT_TIMEOUT_MS: number = cfgToolRouterTimeoutMs.default;
/** Upper bound even a user-configured timeout cannot exceed. */
export const TOOL_ROUTER_MAX_TIMEOUT_MS = 10_000;

/**
 * Which consumer owns the inference. Only `"main"` is eligible for Jev
 * routing; every other scope takes the base stream path untouched with zero
 * judge calls. Scope describes WHO infers, never provider/model — a main
 * retry/fallback that switches provider mid-turn stays `"main"`.
 *
 * Maintenance, handoff, branch summaries, and ephemeral side channels all
 * flow through the shared side-stream function, so they share `"side-channel"`.
 * Subagent sessions build their own main-scoped wrapper per
 * `createAgentSession` call; the seam carries no reliable primary-vs-subagent
 * marker, so subagent primary turns inherit `"main"` with identical
 * guards and fail-open behavior (no heuristic invented).
 */
export type ToolRouterScope = "main" | "advisor" | "capture" | "side-channel";

/** Judge provider injected by the host (sdk wiring uses the `judge` role chain). */
export interface ToolRouterSource {
	getJudge: () => Judge | undefined;
	/** Owning consumer; only `"main"` routes. Required so wiring stays explicit. */
	scope: ToolRouterScope;
}

/** What the router decided, with the options the caller must forward. */
export interface ToolRouterOutcome {
	/** Options to forward to `base`: identical reference on passthrough, copy on route. */
	options: SimpleStreamOptions;
	routed: boolean;
	/** Machine-readable reason (`disabled`, `named-tool`, `none`, `timeout`, ...). */
	reason: string;
	/**
	 * Jev-selected tool name when known: the forced tool on `named-tool`,
	 * and the raw (below-threshold or unknown) selection on fail-open
	 * outcomes. Observability only — never changes the fallback.
	 */
	choice?: string;
	/** Jev-reported confidence, when a decision was reached. */
	confidence?: number;
	/** Wall-clock ms spent reaching the outcome (0 for sync guard trips). */
	latencyMs: number;
}

const TOOL_ROUTE_QUESTION_ID = "route";
/** Recent intent chars forwarded to Jev; hard bound, never the full transcript. */
const MAX_INTENT_CHARS = 2000;
/** Per-tool description chars forwarded to Jev. */
const MAX_DESCRIPTION_CHARS = 500;
/** Max tools described to Jev; bounds the judgment state size. */
const MAX_TOOLS_IN_STATE = 64;

const TOOL_ROUTE_INSTRUCTIONS =
	"Decide which tool the assistant should call next, if any. Choose the single tool " +
	"whose purpose matches the user's intent, or no_tool_needed when the intent is fully " +
	"answerable in prose or no listed tool fits.";

/** Rejection arrival past this deadline; the main request always proceeds. */
class ToolRouterTimeoutError extends Error {
	override readonly name = "ToolRouterTimeoutError";
	constructor(readonly timeoutMs: number) {
		super(`tool router decision exceeded ${timeoutMs}ms`);
	}
}

function truncate(text: string, max: number): string {
	return text.length > max ? text.slice(0, max) : text;
}

function contentText(content: string | Array<{ type: string; text?: string }>): string {
	if (typeof content === "string") return content;
	let out = "";
	for (const part of content) {
		if (part.type === "text" && typeof part.text === "string") out += (out ? "\n" : "") + part.text;
	}
	return out;
}

/** Most recent user/developer message text; empty when the context has none. */
function extractIntent(context: Context): string {
	for (let i = context.messages.length - 1; i >= 0; i--) {
		const message = context.messages[i];
		if (message === undefined || (message.role !== "user" && message.role !== "developer")) continue;
		const text = contentText(message.content).trim();
		if (text) return truncate(text, MAX_INTENT_CHARS);
	}
	return "";
}

function normalizeTimeoutMs(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return TOOL_ROUTER_DEFAULT_TIMEOUT_MS;
	return Math.min(Math.max(1, Math.floor(value)), TOOL_ROUTER_MAX_TIMEOUT_MS);
}

function normalizeMinConfidence(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) return TOOL_ROUTER_DEFAULT_MIN_CONFIDENCE;
	return Math.min(Math.max(0, value), 1);
}

function isChoiceAnswer(answer: unknown): answer is { choice: unknown; confidence: unknown } {
	return typeof answer === "object" && answer !== null && (answer as { type?: unknown }).type === "choice";
}

function withDeadline<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timeout = new Promise<never>((_, reject) => {
		timer = setTimeout(() => reject(new ToolRouterTimeoutError(timeoutMs)), timeoutMs);
	});
	const raced = Promise.race([promise, timeout]);
	const clear = (): void => {
		if (timer !== undefined) clearTimeout(timer);
	};
	void raced.then(clear, clear);
	return raced;
}

function isTimeoutError(error: unknown): boolean {
	return (
		error instanceof ToolRouterTimeoutError ||
		(error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError"))
	);
}

interface RoutePlan {
	toolNames: string[];
	intent: string;
	timeoutMs: number;
	minConfidence: number;
	judge: Judge;
}

function logDecision(
	model: Model,
	plan: Pick<RoutePlan, "toolNames"> & { timeoutMs?: number },
	outcome: Omit<ToolRouterOutcome, "options">,
	extra: { scope: ToolRouterScope; sessionId?: string },
): void {
	logger.debug("tool-router decision", {
		source: extra.scope,
		...(extra.sessionId !== undefined ? { session: extra.sessionId } : {}),
		provider: model.provider,
		model: model.id,
		tools: plan.toolNames.length,
		result: outcome.routed ? "routed" : "passthrough",
		reason: outcome.reason,
		...(outcome.choice !== undefined ? { choice: outcome.choice } : {}),
		...(outcome.confidence !== undefined ? { confidence: outcome.confidence } : {}),
		latencyMs: outcome.latencyMs,
	});
}

/**
 * Per-wrapper turn gate: remembers the last user turn this instance decided,
 * so one turn pays at most one Jev call. Single entry, replaced on every new
 * turn — bounded by construction, never shared across sessions (each
 * `createSettingsAwareStreamFn` site owns one).
 */
export interface ToolRouterTurnState {
	last?: {
		/** Identity of the decided user turn (last user/developer message + tool names). */
		fingerprint: string;
		/** `toolResult` count when decided; growth means post-tool follow-up. */
		toolResults: number;
		/** Resolved options to reuse for same-inference retries/fallbacks. */
		options: SimpleStreamOptions;
		routed: boolean;
		reason: string;
		choice?: string;
		confidence?: number;
	};
}

/** Fresh turn gate for one wrapper instance. */
export function createToolRouterTurnState(): ToolRouterTurnState {
	return {};
}

/** Chars of the last user text folded into the turn fingerprint (bounded key). */
const FINGERPRINT_TEXT_CHARS = 500;

/** Count `toolResult` messages: grows exactly when tools execute mid-turn. */
function countToolResults(context: Context): number {
	let count = 0;
	for (const message of context.messages) {
		if (message?.role === "toolResult") count++;
	}
	return count;
}

/**
 * Identify the user turn driving this request from data already at the seam:
 * the most recent user/developer message (role + timestamp + text) plus the
 * offered tool names. A new user message changes the key; retries, fallbacks,
 * and post-tool follow-ups of the same turn keep it. No turn plumbing needed.
 */
function turnFingerprint(context: Context): { fingerprint: string; toolResults: number } {
	let lastUser = "none";
	for (let i = context.messages.length - 1; i >= 0; i--) {
		const message = context.messages[i];
		if (message === undefined || (message.role !== "user" && message.role !== "developer")) continue;
		lastUser = `${message.role}:${message.timestamp}:${truncate(contentText(message.content).trim(), FINGERPRINT_TEXT_CHARS)}`;
		break;
	}
	const tools = (context.tools ?? []).map(tool => tool.name).join(",");
	return { fingerprint: `${lastUser}|${tools}`, toolResults: countToolResults(context) };
}

/**
 * Decide the generic {@link ToolChoice} for this request. Returns synchronously
 * (no Jev call) whenever a guard trips; otherwise returns a promise bounded by
 * `toolRouter.timeoutMs` that fail-opens to the input options on any failure.
 *
 * When `turn` is provided, one user turn pays at most one Jev call: same-turn
 * retries/fallbacks reuse the stored resolution, post-tool follow-ups pass
 * through undecided, and a new user message re-arms the router.
 *
 * Never mutates `options` or `context`. Never sends credentials: Jev sees only
 * the recent intent text plus tool names/descriptions.
 */
export function decideToolRoute(
	model: Model,
	context: Context,
	options: SimpleStreamOptions,
	source: ToolRouterSource | undefined,
	settings: Settings,
	turn?: ToolRouterTurnState,
): ToolRouterOutcome | Promise<ToolRouterOutcome> {
	const passthrough = (reason: string): ToolRouterOutcome => ({ options, routed: false, reason, latencyMs: 0 });

	if (cfgToolRouterEnabled.get(settings) !== true) return passthrough("disabled");
	// First POC routes the primary main loop only. Advisor, capture, and
	// side-channel consumers take the base path with zero judge calls. An
	// absent source predates scoped wiring and never routes either.
	if (source === undefined) return passthrough("no-source");
	if (source.scope !== "main") return passthrough("non-main-source");
	const tools = context.tools ?? [];
	if (tools.length === 0) return passthrough("no-tools");
	// Explicit caller intent — including explicit "auto" — always wins.
	if (options.toolChoice !== undefined) return passthrough("explicit-tool-choice");
	const { fingerprint, toolResults } = turnFingerprint(context);
	const decided = turn?.last;
	if (decided && decided.fingerprint === fingerprint) {
		if (decided.toolResults === toolResults) {
			// Same inference retried (provider retry/fallback): reuse the
			// stored resolution without paying another judge call. The copy
			// keeps the cached entry immutable for further attempts.
			const outcome: ToolRouterOutcome = {
				options: { ...decided.options },
				routed: decided.routed,
				reason: "reused-turn-decision",
				latencyMs: 0,
			};
			if (decided.choice !== undefined) outcome.choice = decided.choice;
			if (decided.confidence !== undefined) outcome.confidence = decided.confidence;
			logDecision(model, { toolNames: tools.slice(0, MAX_TOOLS_IN_STATE).map(tool => tool.name) }, outcome, {
				scope: source.scope,
				sessionId: options.sessionId,
			});
			return outcome;
		}
		// Same user turn, but tool results grew (or history shrank): the
		// follow-up answers from tool output, so it must neither re-judge
		// nor re-force the decided choice.
		return passthrough("post-tool-followup");
	}
	const intent = extractIntent(context);
	if (!intent) return passthrough("no-user-intent");
	let judge: Judge | undefined;
	try {
		judge = source?.getJudge();
	} catch {
		return passthrough("judge-unavailable");
	}
	if (!judge) return passthrough("no-judge");

	const plan: RoutePlan = {
		toolNames: tools.slice(0, MAX_TOOLS_IN_STATE).map(tool => tool.name),
		intent,
		timeoutMs: normalizeTimeoutMs(cfgToolRouterTimeoutMs.get(settings)),
		minConfidence: normalizeMinConfidence(cfgToolRouterMinConfidence.get(settings)),
		judge,
	};
	return resolveRoute(model, context, options, plan, source.scope, turn, { fingerprint, toolResults });
}

async function resolveRoute(
	model: Model,
	context: Context,
	options: SimpleStreamOptions,
	plan: RoutePlan,
	scope: ToolRouterScope,
	turn: ToolRouterTurnState | undefined,
	turnKey: { fingerprint: string; toolResults: number },
): Promise<ToolRouterOutcome> {
	const started = Date.now();
	const logContext = { scope, sessionId: options.sessionId };
	const remember = (outcome: ToolRouterOutcome): void => {
		if (turn === undefined) return;
		turn.last = {
			fingerprint: turnKey.fingerprint,
			toolResults: turnKey.toolResults,
			options: outcome.options,
			routed: outcome.routed,
			reason: outcome.reason,
			...(outcome.choice !== undefined ? { choice: outcome.choice } : {}),
			...(outcome.confidence !== undefined ? { confidence: outcome.confidence } : {}),
		};
	};
	const fail = (reason: string, confidence?: number, choice?: string, transient = false): ToolRouterOutcome => {
		const outcome: ToolRouterOutcome = { options, routed: false, reason, latencyMs: Date.now() - started };
		if (confidence !== undefined) outcome.confidence = confidence;
		if (choice !== undefined) outcome.choice = choice;
		logDecision(model, plan, outcome, logContext);
		// Deterministic answers arm the turn gate; transient failures
		// (judge error, timeout, caller abort) leave it so a retry re-judges.
		if (!transient) remember(outcome);
		return outcome;
	};

	const criteria: Record<string, string | null> = {};
	for (const tool of (context.tools ?? []).slice(0, MAX_TOOLS_IN_STATE)) {
		const description = tool.description?.trim();
		criteria[tool.name] = description ? truncate(description, MAX_DESCRIPTION_CHARS) : null;
	}
	criteria[TOOL_ROUTER_NO_TOOL] = "No tool fits, or the intent is fully answerable in prose without tools.";
	const question: ChoiceQuestion<string> = {
		type: "choice",
		instructions: TOOL_ROUTE_INSTRUCTIONS,
		criteria,
	};
	const deadline = AbortSignal.timeout(plan.timeoutMs);
	const signal = options.signal ? AbortSignal.any([options.signal, deadline]) : deadline;

	let choice: unknown;
	let confidence: unknown;
	try {
		const result = await withDeadline(
			plan.judge.judge(
				{ state: { intent: plan.intent }, questions: { [TOOL_ROUTE_QUESTION_ID]: question } },
				{ signal },
			),
			plan.timeoutMs,
		);
		const answer = (result.answers as Record<string, unknown> | undefined)?.[TOOL_ROUTE_QUESTION_ID];
		if (!isChoiceAnswer(answer)) return fail("malformed-answer");
		choice = answer.choice;
		confidence = answer.confidence;
	} catch (error) {
		if (options.signal?.aborted) return fail("caller-aborted", undefined, undefined, true);
		return fail(isTimeoutError(error) ? "timeout" : "judge-error", undefined, undefined, true);
	}
	if (typeof choice !== "string" || typeof confidence !== "number" || Number.isNaN(confidence)) {
		return fail("malformed-answer");
	}
	if (confidence < plan.minConfidence) return fail("low-confidence", confidence, choice);
	let toolChoice: ToolChoice;
	let routeChoice: string | undefined;
	if (choice === TOOL_ROUTER_NO_TOOL) {
		toolChoice = "none";
	} else if (plan.toolNames.includes(choice)) {
		toolChoice = { type: "function", name: choice };
		routeChoice = choice;
	} else {
		return fail("unknown-tool", confidence, choice);
	}
	const outcome: ToolRouterOutcome = {
		options: { ...options, toolChoice },
		routed: true,
		reason: routeChoice !== undefined ? "named-tool" : "none",
		confidence,
		latencyMs: Date.now() - started,
	};
	if (routeChoice !== undefined) outcome.choice = routeChoice;
	logDecision(model, plan, outcome, logContext);
	remember(outcome);
	return outcome;
}
