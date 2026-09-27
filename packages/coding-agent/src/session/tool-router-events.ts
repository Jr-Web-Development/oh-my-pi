/**
 * Display-only publishing contract for the native Jev tool router.
 *
 * The router itself (`session/tool-router.ts`) stays a pure decision function:
 * it returns the options to forward and why. This module is the seam that turns
 * the APPLIED part of that outcome into a short transcript indicator, without
 * ever touching agent state, the transcript model, or the LLM context — the
 * event only travels over the session event bus to the TUI.
 */
import { isRecord } from "@oh-my-pi/pi-utils";

/** Session event channel for the applied Jev route of the current inference. */
export const TOOL_ROUTER_DECISION_EVENT_CHANNEL = "session:tool-router-decision";

/**
 * One applied router decision, shaped for display only.
 *
 * Only APPLIED routes are ever published: passthrough outcomes (disabled,
 * timeout, low-confidence, judge error, post-tool follow-up) carry no forced
 * tool, so rendering them as a decision would claim "JEV chose" where the model
 * chose. Never enters agent state or the LLM context.
 */
export interface ToolRouterDecisionEvent {
	/** Tool name the router forced for this inference. */
	readonly tool: string;
	/** Router-reported confidence, as reported by the judge. */
	readonly confidence: number;
	/** Machine-readable applied reason (`named-tool`, `reused-turn-decision`). */
	readonly reason: string;
}

/** Validate an untyped event-bus payload before rendering it. */
export function isToolRouterDecisionEvent(value: unknown): value is ToolRouterDecisionEvent {
	if (!isRecord(value)) return false;
	const decision = value;
	return (
		typeof decision.tool === "string" &&
		decision.tool.length > 0 &&
		typeof decision.confidence === "number" &&
		Number.isFinite(decision.confidence) &&
		typeof decision.reason === "string" &&
		decision.reason.length > 0
	);
}

/**
 * The emit gate: the single place that decides whether an outcome is worth a
 * transcript line.
 *
 * Publishes nothing for passthrough outcomes — `routed: false` (disabled,
 * timeout, low-confidence, judge error) must never render as "JEV chose" — nor
 * for a routable reason with no forced tool (`none`/`no_tool_needed`), nor for
 * any reason outside the applied set.
 *
 * `reused-turn-decision` re-publishes the SAME applied decision as the original
 * `named-tool` outcome: a provider retry or fallback re-runs this wrapper, and
 * its forced tool must keep correct provenance rather than looking unrouted.
 */
export function toolRouterDecisionEvent(outcome: {
	routed: boolean;
	reason: string;
	choice?: string;
	confidence?: number;
}): ToolRouterDecisionEvent | undefined {
	if (!outcome.routed) return undefined;
	const tool = outcome.choice;
	if (tool === undefined) return undefined;
	if (outcome.reason !== "named-tool" && outcome.reason !== "reused-turn-decision") return undefined;
	const confidence = outcome.confidence;
	if (typeof confidence !== "number" || !Number.isFinite(confidence)) return undefined;
	return { tool, confidence, reason: outcome.reason };
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
	#decision: ToolRouterDecisionEvent | undefined;

	/** Record the applied decision for this turn; the last write wins. */
	note(decision: ToolRouterDecisionEvent): void {
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
	consume(toolName: string): ToolRouterDecisionEvent | undefined {
		const decision = this.#decision;
		if (decision === undefined || decision.tool !== toolName) return undefined;
		this.#decision = undefined;
		return decision;
	}
}
