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
 * calls core `completeSimple` -> `streamSimple` directly. This wrapper is only
 * composed in `sdk.ts` for the main/advisor/side-channel stream functions, so
 * a judgment side request cannot re-enter the router.
 *
 * Fail-open contract: any guard trip, validation failure, error, or timeout
 * returns the caller's options object untouched.
 */
import type { ChoiceQuestion, Context, Judge, Model, SimpleStreamOptions, ToolChoice } from "@oh-my-pi/pi-ai";
import { logger } from "@oh-my-pi/pi-utils";
import type { Settings } from "../config/settings";

/** Choice label meaning "the request needs no tool; answer in prose". */
export const TOOL_ROUTER_NO_TOOL = "no_tool_needed";

/** Conservative default: minimum Jev confidence required to override the caller. */
export const TOOL_ROUTER_DEFAULT_MIN_CONFIDENCE = 0.7;
/** Conservative default: hard deadline for the Jev side request. */
export const TOOL_ROUTER_DEFAULT_TIMEOUT_MS = 1500;
/** Upper bound even a user-configured timeout cannot exceed. */
export const TOOL_ROUTER_MAX_TIMEOUT_MS = 10_000;
/** Model selector value the POC resolves through the normal `judge` role chain. */
export const TOOL_ROUTER_JUDGE_MODEL = "@judge";

/** Single routing behavior shipped by the POC; reserved for future extension. */
export type ToolRouterMode = "conservative";

/** Judge provider injected by the host (sdk wiring uses the `judge` role chain). */
export interface ToolRouterSource {
	getJudge: () => Judge | undefined;
}

/** What the router decided, with the options the caller must forward. */
export interface ToolRouterOutcome {
	/** Options to forward to `base`: identical reference on passthrough, copy on route. */
	options: SimpleStreamOptions;
	routed: boolean;
	/** Machine-readable reason (`disabled`, `named-tool`, `none`, `timeout`, ...). */
	reason: string;
	/** Selected tool name, when a named tool was forced. */
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
	outcome: Omit<ToolRouterOutcome, "options"> & { mode: ToolRouterMode },
): void {
	logger.debug("tool-router decision", {
		provider: model.provider,
		model: model.id,
		tools: plan.toolNames.length,
		mode: outcome.mode,
		result: outcome.routed ? "routed" : "passthrough",
		reason: outcome.reason,
		...(outcome.choice !== undefined ? { choice: outcome.choice } : {}),
		...(outcome.confidence !== undefined ? { confidence: outcome.confidence } : {}),
		latencyMs: outcome.latencyMs,
	});
}

/**
 * Decide the generic {@link ToolChoice} for this request. Returns synchronously
 * (no Jev call) whenever a guard trips; otherwise returns a promise bounded by
 * `toolRouter.timeoutMs` that fail-opens to the input options on any failure.
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
): ToolRouterOutcome | Promise<ToolRouterOutcome> {
	const passthrough = (reason: string): ToolRouterOutcome => ({ options, routed: false, reason, latencyMs: 0 });

	if (settings.get("toolRouter.enabled") !== true) return passthrough("disabled");
	const tools = context.tools ?? [];
	if (tools.length === 0) return passthrough("no-tools");
	// Explicit caller intent — including explicit "auto" — always wins.
	if (options.toolChoice !== undefined) return passthrough("explicit-tool-choice");
	if (settings.get("toolRouter.model") !== TOOL_ROUTER_JUDGE_MODEL) return passthrough("unsupported-model");
	if (settings.get("toolRouter.mode") !== "conservative") return passthrough("unsupported-mode");
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
		timeoutMs: normalizeTimeoutMs(settings.get("toolRouter.timeoutMs")),
		minConfidence: normalizeMinConfidence(settings.get("toolRouter.minConfidence")),
		judge,
	};
	return resolveRoute(model, context, options, plan);
}

async function resolveRoute(
	model: Model,
	context: Context,
	options: SimpleStreamOptions,
	plan: RoutePlan,
): Promise<ToolRouterOutcome> {
	const started = Date.now();
	const fail = (reason: string, confidence?: number): ToolRouterOutcome => {
		const outcome: ToolRouterOutcome = { options, routed: false, reason, latencyMs: Date.now() - started };
		if (confidence !== undefined) outcome.confidence = confidence;
		logDecision(model, plan, { ...outcome, mode: "conservative" });
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
		if (options.signal?.aborted) return fail("caller-aborted");
		return fail(isTimeoutError(error) ? "timeout" : "judge-error");
	}
	if (typeof choice !== "string" || typeof confidence !== "number" || Number.isNaN(confidence)) {
		return fail("malformed-answer");
	}
	if (confidence < plan.minConfidence) return fail("low-confidence", confidence);
	let toolChoice: ToolChoice;
	let routeChoice: string | undefined;
	if (choice === TOOL_ROUTER_NO_TOOL) {
		toolChoice = "none";
	} else if (plan.toolNames.includes(choice)) {
		toolChoice = { type: "function", name: choice };
		routeChoice = choice;
	} else {
		return fail("unknown-tool", confidence);
	}
	const outcome: ToolRouterOutcome = {
		options: { ...options, toolChoice },
		routed: true,
		reason: routeChoice !== undefined ? "named-tool" : "none",
		confidence,
		latencyMs: Date.now() - started,
	};
	if (routeChoice !== undefined) outcome.choice = routeChoice;
	logDecision(model, plan, { ...outcome, mode: "conservative" });
	return outcome;
}
