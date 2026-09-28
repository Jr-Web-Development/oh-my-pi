/**
 * Display-only publishing contract for the native Jev tool router.
 *
 * The router itself (`session/tool-router.ts`) stays a pure decision function:
 * it returns the options to forward and why. This module is the seam that turns
 * that outcome into a short transcript indicator, without ever touching agent
 * state, the transcript model, or the LLM context — the event only travels over
 * the session event bus to the TUI.
 *
 * An outcome is either APPLIED (the router forced a tool, so the marker belongs
 * to that tool's row) or a ROUTER STATUS (the router consulted Jev and did not
 * force anything: advisory passthrough, the prose route, or an operational
 * failure). The two are carried by the same typed event so the consumer can
 * tell them apart; only the APPLIED event is allowed to annotate a tool row.
 */
import { isRecord } from "@oh-my-pi/pi-utils";

/** Session event channel for the Jev route of the current inference. */
export const TOOL_ROUTER_DECISION_EVENT_CHANNEL = "session:tool-router-decision";

/**
 * An applied route: the router forced `tool` for this inference. Provenance of
 * that tool's row, and of nothing else.
 */
export interface ToolRouterAppliedRoute {
	readonly kind: "applied";
	/** Tool name the router forced. */
	readonly tool: string;
	/** Router-reported confidence, as reported by the judge. */
	readonly confidence: number;
	/** Machine-readable applied reason (`named-tool`, `reused-turn-decision`). */
	readonly reason: string;
}

/**
 * A Jev selection that was NOT applied because it fell below the confidence
 * threshold (or named a tool outside the roster). The model still chose for
 * itself: this line reports the router's status, never a forced tool.
 */
export interface ToolRouterAdvisoryRoute {
	readonly kind: "advisory";
	/** Tool Jev selected but that was not forced. */
	readonly choice: string;
	/** Router-reported confidence of that selection. */
	readonly confidence: number;
	/** `low-confidence` or `unknown-tool`. */
	readonly reason: string;
}

/** The prose route: the router decided no tool should be forced. */
export interface ToolRouterNoneRoute {
	readonly kind: "none";
	readonly confidence: number;
	readonly reason: string;
}

/**
 * The router consulted Jev and reached no decision at all (timeout, judge
 * error, unusable answer). No tool was forced and no choice is known.
 */
export interface ToolRouterFailureRoute {
	readonly kind: "failure";
	/** `timeout`, `judge-error`, or `malformed-answer`. */
	readonly reason: string;
}

/** Every display event published for a router outcome. */
export type ToolRouterRouteEvent =
	| ToolRouterAppliedRoute
	| ToolRouterAdvisoryRoute
	| ToolRouterNoneRoute
	| ToolRouterFailureRoute;

/**
 * The router statuses rendered as a standalone line of the turn, never attached
 * to a tool row (there is no forced tool to attach them to).
 */
export type ToolRouterStatusRoute = Exclude<ToolRouterRouteEvent, { kind: "applied" }>;

/** Operational failures of a judge call that was actually attempted. */
const TOOL_ROUTER_FAILURE_REASONS: Record<string, true> = {
	timeout: true,
	"judge-error": true,
	"malformed-answer": true,
};

function finiteConfidence(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * The one user-facing line for a route event.
 *
 * APPLIED renders as tool provenance (`◆`); every other kind is a router status
 * line (`◇`) and must never be confused with a tool the router forced.
 */
export function toolRouterRouteLine(event: ToolRouterRouteEvent): string {
	switch (event.kind) {
		case "applied":
			return `◆ JEV → ${event.tool} · ${Math.round(event.confidence * 100)}%`;
		case "advisory":
			return `◇ JEV · ${event.choice} · ${Math.round(event.confidence * 100)}% · passthrough`;
		case "none":
			return `◇ JEV · none · ${Math.round(event.confidence * 100)}%`;
		case "failure":
			return `◇ JEV · ${event.reason}`;
	}
}

/** Validate an untyped event-bus payload before rendering it. */
export function isToolRouterRouteEvent(value: unknown): value is ToolRouterRouteEvent {
	if (!isRecord(value)) return false;
	switch (value.kind) {
		case "applied":
			return (
				typeof value.tool === "string" &&
				value.tool.length > 0 &&
				finiteConfidence(value.confidence) !== undefined &&
				typeof value.reason === "string" &&
				value.reason.length > 0
			);
		case "advisory":
			return (
				typeof value.choice === "string" &&
				value.choice.length > 0 &&
				finiteConfidence(value.confidence) !== undefined &&
				typeof value.reason === "string" &&
				value.reason.length > 0
			);
		case "none":
			return finiteConfidence(value.confidence) !== undefined && typeof value.reason === "string";
		case "failure":
			return typeof value.reason === "string" && TOOL_ROUTER_FAILURE_REASONS[value.reason] === true;
		default:
			return false;
	}
}

/**
 * The emit gate: the single place that decides whether a router outcome earns a
 * transcript line, and which kind of line it is.
 *
 * An outcome that reached Jev and forced a tool becomes `applied`; one that
 * reached Jev and did not is `advisory` (a below-threshold or off-roster
 * selection the model overrode), `none` (the prose route), or `failure` (the
 * judge call timed out, errored, or answered unusably).
 *
 * Guards that never consulted Jev (`disabled`, `non-main-source`, `no-tools`,
 * `explicit-tool-choice`, `post-tool-followup`, `no-user-intent`, `no-judge`,
 * …) publish nothing: they are configuration or flow facts, not router
 * activity, and rendering them would put a line in front of every turn. The
 * same goes for `caller-aborted`, which is the user cancelling rather than a
 * router failure.
 *
 * `reused-turn-decision` re-publishes the SAME decision as the original
 * outcome — a provider retry or fallback re-runs this wrapper, and its forced
 * tool must keep correct provenance rather than looking unrouted.
 */
export function toolRouterRouteEvent(outcome: {
	routed: boolean;
	reason: string;
	choice?: string;
	confidence?: number;
}): ToolRouterRouteEvent | undefined {
	const reason = outcome.reason;
	if (outcome.routed) {
		if (reason === "none") {
			const confidence = finiteConfidence(outcome.confidence);
			return confidence === undefined ? undefined : { kind: "none", confidence, reason };
		}
		// `named-tool` always carries the forced tool; a re-published decision
		// carries one only when the reused decision was a named tool.
		if (reason === "named-tool" || reason === "reused-turn-decision") {
			const confidence = finiteConfidence(outcome.confidence);
			if (confidence === undefined) return undefined;
			const tool = outcome.choice;
			if (tool !== undefined) return { kind: "applied", tool, confidence, reason };
			return reason === "reused-turn-decision" ? { kind: "none", confidence, reason } : undefined;
		}
		return undefined;
	}
	if (reason === "low-confidence" || reason === "unknown-tool") {
		const tool = outcome.choice;
		const confidence = finiteConfidence(outcome.confidence);
		if (tool === undefined || confidence === undefined) return undefined;
		return { kind: "advisory", choice: tool, confidence, reason };
	}
	if (TOOL_ROUTER_FAILURE_REASONS[reason] === true) return { kind: "failure", reason };
	return undefined;
}

/**
 * Display-only marker for the current turn's applied route.
 *
 * `consume` exists because the annotation must land on the row of the tool the
 * router forced and nowhere else. Only a matching call consumes the marker; any
 * other tool leaves it in place, because the forced tool's row can be added
 * after an unrelated row the model produced first — a mismatched call must
 * neither be annotated itself nor destroy the provenance of the forced call that
 * has not rendered yet. The per-turn reset (`clear`) drops a decision whose tool
 * never ran, so a marker cannot leak into a later turn.
 */
export class PendingToolRoute {
	#decision: ToolRouterAppliedRoute | undefined;

	/** Record the applied decision for this turn; the last write wins. */
	note(decision: ToolRouterAppliedRoute): void {
		this.#decision = decision;
	}

	/** Drop any recorded decision (per-turn reset). */
	clear(): void {
		this.#decision = undefined;
	}

	/**
	 * Take the recorded decision for `toolName`. A matching call consumes it;
	 * any other tool returns `undefined` and leaves the marker pending.
	 */
	consume(toolName: string): ToolRouterAppliedRoute | undefined {
		const decision = this.#decision;
		if (decision === undefined || decision.tool !== toolName) return undefined;
		this.#decision = undefined;
		return decision;
	}
}

/**
 * Display-only latch for the current turn's standalone router status lines.
 *
 * A retry or fallback re-publishes the same decision, so the same line would
 * otherwise print twice for one turn. Every line already shown this turn is
 * remembered (not just the last one), so a status repeated after an intervening
 * different status is still dropped — the contract is "one representation per
 * status per turn". The per-turn reset (`clear`) lets the next turn report its
 * own status, including one identical to the previous turn's.
 */
export class TurnRouteStatusLine {
	#shown = new Set<string>();

	/** Record `line`; `true` when this turn has not shown it yet. */
	accept(line: string): boolean {
		if (this.#shown.has(line)) return false;
		this.#shown.add(line);
		return true;
	}

	/** Drop the recorded lines (per-turn reset). */
	clear(): void {
		this.#shown.clear();
	}
}
