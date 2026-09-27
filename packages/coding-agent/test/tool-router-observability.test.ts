/**
 * Contract: display-only observability for the native Jev tool router.
 *
 * `session/tool-router-events.ts` decides WHICH router outcomes deserve a
 * transcript line and carries the applied decision from the stream wrapper to
 * the tool row it caused. The consumer-visible failures guarded here:
 *
 * - a passthrough outcome (disabled, timeout, low confidence, judge error)
 *   rendering as "JEV chose" — the gate must publish nothing for it;
 * - the `none` route (answer in prose) rendering a tool indicator for a tool
 *   that was never chosen;
 * - a provider retry/fallback losing provenance: `reused-turn-decision`
 *   re-publishes the same applied decision;
 * - a tool the model chose itself being annotated because a stale decision
 *   leaked past the row it belonged to (the marker must be consumed once, and
 *   only for the matching tool);
 * - the wrapper not observing the resolved outcome at all (wiring).
 */
import { describe, expect, it } from "bun:test";
import type { StreamFn } from "@oh-my-pi/pi-agent-core";
import type { Context, Judge, JudgmentResult, Model, Questions } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { createSettingsAwareStreamFn } from "@oh-my-pi/pi-coding-agent/session/settings-stream-fn";
import { TOOL_ROUTER_NO_TOOL, type ToolRouterOutcome } from "@oh-my-pi/pi-coding-agent/session/tool-router";
import {
	isToolRouterDecisionEvent,
	PendingToolRoute,
	toolRouterDecisionEvent,
} from "@oh-my-pi/pi-coding-agent/session/tool-router-events";

const stubModel = { api: "openai-completions", provider: "test", id: "test-model" } as unknown as Model;

function makeContext(tools: Array<{ name: string; description?: string }>): Context {
	return {
		messages: [{ role: "user", content: "read hello.txt and return only its contents", timestamp: Date.now() }],
		tools: tools.map(tool => ({
			name: tool.name,
			description: tool.description ?? `${tool.name} tool`,
			parameters: { type: "object", properties: {} },
		})),
	} as unknown as Context;
}

const readWriteContext = () =>
	makeContext([
		{ name: "read", description: "Read a file from disk" },
		{ name: "write", description: "Write a file to disk" },
	]);

function choiceResult(choice: unknown, confidence: unknown): JudgmentResult<Questions> {
	return {
		api: "typesafe",
		provider: "typesafe",
		model: "jev-latest",
		answers: {
			route: { type: "choice", choice, probabilities: {}, confidence },
		},
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
	} as unknown as JudgmentResult<Questions>;
}

function stubJudge(behavior: () => Promise<JudgmentResult<Questions>>): Judge {
	return {
		label: "stub/jev",
		judge: (async () => behavior()) as unknown as Judge["judge"],
	} as Judge;
}

function captureBase(): StreamFn {
	return (_model, _context, _options) => new AssistantMessageEventStream();
}

function enabledSettings(): Settings {
	return Settings.isolated({ "toolRouter.enabled": true });
}

async function observedOutcomes(
	streamSettings: Settings,
	route: unknown,
	confidence: unknown,
	context: Context,
): Promise<ToolRouterOutcome[]> {
	const observed: ToolRouterOutcome[] = [];
	const wrapped = createSettingsAwareStreamFn(
		streamSettings,
		captureBase(),
		undefined,
		{ getJudge: () => stubJudge(async () => choiceResult(route, confidence)), scope: "main" },
		outcome => observed.push(outcome),
	);
	await wrapped(stubModel, context, undefined);
	return observed;
}

describe("toolRouterDecisionEvent gate", () => {
	it("publishes a fresh named-tool route and preserves tool/confidence/reason", () => {
		expect(toolRouterDecisionEvent({ routed: true, reason: "named-tool", choice: "read", confidence: 0.9 })).toEqual({
			tool: "read",
			confidence: 0.9,
			reason: "named-tool",
		});
	});

	it("publishes nothing for passthrough, so it can never render as 'JEV chose'", () => {
		// Low confidence: the router forwards the caller options untouched.
		expect(
			toolRouterDecisionEvent({ routed: false, reason: "low-confidence", choice: "read", confidence: 0.2 }),
		).toBeUndefined();
		// Timeout: bounded judge wait expired, no decision was applied.
		expect(toolRouterDecisionEvent({ routed: false, reason: "timeout" })).toBeUndefined();
		// Disabled is the default state.
		expect(toolRouterDecisionEvent({ routed: false, reason: "disabled" })).toBeUndefined();
	});

	it("publishes nothing for the routed `none` answer (no forced tool)", () => {
		expect(toolRouterDecisionEvent({ routed: true, reason: "none", confidence: 0.95 })).toBeUndefined();
		expect(
			toolRouterDecisionEvent({ routed: true, reason: "none", choice: undefined, confidence: 0.95 }),
		).toBeUndefined();
	});

	it("republishes on reused-turn-decision, so a retry keeps the same provenance", () => {
		expect(
			toolRouterDecisionEvent({ routed: true, reason: "reused-turn-decision", choice: "write", confidence: 0.81 }),
		).toEqual({ tool: "write", confidence: 0.81, reason: "reused-turn-decision" });
	});

	it("refuses a routable reason with no finite confidence", () => {
		expect(toolRouterDecisionEvent({ routed: true, reason: "named-tool", choice: "read" })).toBeUndefined();
		expect(
			toolRouterDecisionEvent({ routed: true, reason: "named-tool", choice: "read", confidence: Number.NaN }),
		).toBeUndefined();
	});

	it("ignores reasons outside the applied set", () => {
		expect(
			toolRouterDecisionEvent({ routed: true, reason: "post-tool-followup", choice: "read", confidence: 0.9 }),
		).toBeUndefined();
	});
});

describe("isToolRouterDecisionEvent", () => {
	it("accepts a well-formed decision", () => {
		expect(isToolRouterDecisionEvent({ tool: "read", confidence: 0.5, reason: "named-tool" })).toBe(true);
	});

	it("rejects malformed payloads", () => {
		expect(isToolRouterDecisionEvent(undefined)).toBe(false);
		expect(isToolRouterDecisionEvent("read")).toBe(false);
		expect(isToolRouterDecisionEvent({ confidence: 0.5, reason: "named-tool" })).toBe(false);
		expect(isToolRouterDecisionEvent({ tool: "", confidence: 0.5, reason: "named-tool" })).toBe(false);
		expect(isToolRouterDecisionEvent({ tool: "read", confidence: Number.NaN, reason: "named-tool" })).toBe(false);
		expect(isToolRouterDecisionEvent({ tool: "read", confidence: 0.5, reason: "" })).toBe(false);
		expect(isToolRouterDecisionEvent({ tool: "read", confidence: 0.5 })).toBe(false);
	});
});

describe("PendingToolRoute", () => {
	const decision = { tool: "read", confidence: 0.9, reason: "named-tool" };

	it("consumes once for the matching tool, then yields nothing", () => {
		const marker = new PendingToolRoute();
		marker.note(decision);
		expect(marker.consume("read")).toEqual(decision);
		expect(marker.consume("read")).toBeUndefined();
	});

	it("yields nothing for a non-matching tool and keeps the marker for the forced one", () => {
		const marker = new PendingToolRoute();
		marker.note(decision);
		// The model produced `write` first inside the forced schema: it must not be
		// annotated, and it must not destroy the provenance of the forced `read`
		// row that can still be created later in the same turn.
		expect(marker.consume("write")).toBeUndefined();
		expect(marker.consume("read")).toEqual(decision);
	});

	it("clear() drops the marker", () => {
		const marker = new PendingToolRoute();
		marker.note(decision);
		marker.clear();
		expect(marker.consume("read")).toBeUndefined();
	});

	it("a second note replaces the first", () => {
		const marker = new PendingToolRoute();
		marker.note(decision);
		marker.note({ tool: "write", confidence: 0.4, reason: "reused-turn-decision" });
		// The replacement is the only decision left: `write` resolves it, and the
		// superseded `read` note is unreachable (consume already cleared it).
		expect(marker.consume("write")).toEqual({ tool: "write", confidence: 0.4, reason: "reused-turn-decision" });
		expect(marker.consume("read")).toBeUndefined();
	});
});

describe("createSettingsAwareStreamFn observer wiring", () => {
	it("observes the resolved applied route exactly once", async () => {
		const observed = await observedOutcomes(enabledSettings(), "read", 0.9, readWriteContext());
		expect(observed.length).toBe(1);
		const outcome = observed[0];
		if (outcome === undefined) throw new Error("expected one observed outcome");
		expect(outcome.routed).toBe(true);
		expect(outcome.reason).toBe("named-tool");
		expect(toolRouterDecisionEvent(outcome)).toEqual({ tool: "read", confidence: 0.9, reason: "named-tool" });
	});

	it("observes passthrough too, so the gate (not silence) decides", async () => {
		const observed = await observedOutcomes(Settings.isolated(), "read", 0.9, readWriteContext());
		expect(observed.length).toBe(1);
		const outcome = observed[0];
		if (outcome === undefined) throw new Error("expected one observed outcome");
		expect(outcome.routed).toBe(false);
		expect(toolRouterDecisionEvent(outcome)).toBeUndefined();
	});

	it("observes the prose route as an unroutable-to-tool outcome", async () => {
		const observed = await observedOutcomes(enabledSettings(), TOOL_ROUTER_NO_TOOL, 0.95, readWriteContext());
		expect(observed.length).toBe(1);
		const outcome = observed[0];
		if (outcome === undefined) throw new Error("expected one observed outcome");
		expect(outcome.reason).toBe("none");
		expect(toolRouterDecisionEvent(outcome)).toBeUndefined();
	});
});
