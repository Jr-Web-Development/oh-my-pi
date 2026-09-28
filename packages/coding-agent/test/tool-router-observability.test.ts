/**
 * Contract: display-only observability for the native Jev tool router.
 *
 * `session/tool-router-events.ts` decides WHICH router outcomes deserve a
 * transcript line, which KIND of line they get, and carries the applied
 * decision from the stream wrapper to the tool row it caused. The
 * consumer-visible failures guarded here:
 *
 * - a passthrough outcome (timeout, low confidence, judge error) rendering as
 *   "JEV chose" — the router status is a `◇` line, never a `◆` tool marker;
 * - the `none` route (answer in prose) rendering a tool indicator for a tool
 *   that was never chosen;
 * - a guard that never consulted Jev (disabled, non-main, post-tool follow-up)
 *   spamming a line on every turn;
 * - a provider retry/fallback losing provenance: `reused-turn-decision`
 *   re-publishes the same applied decision, and the same standalone status is
 *   only ever shown once per turn;
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
	isToolRouterRouteEvent,
	PendingToolRoute,
	toolRouterRouteEvent,
	toolRouterRouteLine,
	TurnRouteStatusLine,
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

describe("toolRouterRouteEvent gate", () => {
	it("publishes an applied route, preserving tool/confidence/reason", () => {
		expect(toolRouterRouteEvent({ routed: true, reason: "named-tool", choice: "read", confidence: 0.9 })).toEqual({
			kind: "applied",
			tool: "read",
			confidence: 0.9,
			reason: "named-tool",
		});
	});

	it("publishes a low-confidence selection as an advisory router status, never as applied", () => {
		expect(
			toolRouterRouteEvent({ routed: false, reason: "low-confidence", choice: "eval", confidence: 0.62 }),
		).toEqual({ kind: "advisory", choice: "eval", confidence: 0.62, reason: "low-confidence" });
	});

	it("publishes an off-roster selection as an advisory router status", () => {
		expect(
			toolRouterRouteEvent({ routed: false, reason: "unknown-tool", choice: "teleport", confidence: 0.91 }),
		).toEqual({ kind: "advisory", choice: "teleport", confidence: 0.91, reason: "unknown-tool" });
	});

	it("publishes the prose route as none (no forced tool)", () => {
		expect(toolRouterRouteEvent({ routed: true, reason: "none", confidence: 0.82 })).toEqual({
			kind: "none",
			confidence: 0.82,
			reason: "none",
		});
	});

	it("publishes the operational failures of a judge call that was attempted", () => {
		expect(toolRouterRouteEvent({ routed: false, reason: "timeout" })).toEqual({
			kind: "failure",
			reason: "timeout",
		});
		expect(toolRouterRouteEvent({ routed: false, reason: "judge-error" })).toEqual({
			kind: "failure",
			reason: "judge-error",
		});
		expect(toolRouterRouteEvent({ routed: false, reason: "malformed-answer" })).toEqual({
			kind: "failure",
			reason: "malformed-answer",
		});
	});

	it("publishes nothing for guards that never consulted Jev", () => {
		for (const reason of [
			"disabled",
			"no-source",
			"non-main-source",
			"no-tools",
			"explicit-tool-choice",
			"post-tool-followup",
			"no-user-intent",
			"no-judge",
			"judge-unavailable",
			// Caller cancellation is the user aborting, not a router failure.
			"caller-aborted",
		]) {
			expect(toolRouterRouteEvent({ routed: false, reason })).toBeUndefined();
		}
	});

	it("publishes nothing for a routed outcome whose line cannot be rendered", () => {
		// No confidence at all: there is no honest percentage to show.
		expect(toolRouterRouteEvent({ routed: true, reason: "none" })).toBeUndefined();
		expect(toolRouterRouteEvent({ routed: true, reason: "none", confidence: Number.NaN })).toBeUndefined();
		// `named-tool` without the forced tool is not an applied route.
		expect(toolRouterRouteEvent({ routed: true, reason: "named-tool", confidence: 0.9 })).toBeUndefined();
		expect(toolRouterRouteEvent({ routed: true, reason: "something-new", choice: "read", confidence: 0.9 })).toBe(
			undefined,
		);
		// An advisory without its choice carries no information.
		expect(toolRouterRouteEvent({ routed: false, reason: "low-confidence", confidence: 0.2 })).toBeUndefined();
	});

	it("republishes reused-turn-decision with the same kind as the original decision", () => {
		// A retry of a forced tool keeps its provenance...
		expect(
			toolRouterRouteEvent({ routed: true, reason: "reused-turn-decision", choice: "write", confidence: 0.81 }),
		).toEqual({ kind: "applied", tool: "write", confidence: 0.81, reason: "reused-turn-decision" });
		// ...and a retry of the prose route stays `none`, not a bogus tool.
		expect(toolRouterRouteEvent({ routed: true, reason: "reused-turn-decision", confidence: 0.81 })).toEqual({
			kind: "none",
			confidence: 0.81,
			reason: "reused-turn-decision",
		});
	});
});

describe("toolRouterRouteLine", () => {
	it("renders each kind as its own user-facing line", () => {
		expect(toolRouterRouteLine({ kind: "applied", tool: "read", confidence: 0.91, reason: "named-tool" })).toBe(
			"◆ JEV → read · 91%",
		);
		expect(
			toolRouterRouteLine({ kind: "advisory", choice: "eval", confidence: 0.62, reason: "low-confidence" }),
		).toBe("◇ JEV · eval · 62% · passthrough");
		expect(toolRouterRouteLine({ kind: "none", confidence: 0.82, reason: "none" })).toBe("◇ JEV · none · 82%");
		expect(toolRouterRouteLine({ kind: "failure", reason: "timeout" })).toBe("◇ JEV · timeout");
		expect(toolRouterRouteLine({ kind: "failure", reason: "judge-error" })).toBe("◇ JEV · judge-error");
	});

	it("rounds the percentage and never prints a confidence for a failure", () => {
		expect(
			toolRouterRouteLine({ kind: "advisory", choice: "eval", confidence: 0.615, reason: "low-confidence" }),
		).toBe("◇ JEV · eval · 62% · passthrough");
		expect(toolRouterRouteLine({ kind: "none", confidence: 0.995, reason: "none" })).toBe("◇ JEV · none · 100%");
	});
});

describe("isToolRouterRouteEvent", () => {
	it("accepts well-formed events of every kind", () => {
		expect(isToolRouterRouteEvent({ kind: "applied", tool: "read", confidence: 0.5, reason: "named-tool" })).toBe(
			true,
		);
		expect(
			isToolRouterRouteEvent({ kind: "advisory", choice: "eval", confidence: 0.5, reason: "low-confidence" }),
		).toBe(true);
		expect(isToolRouterRouteEvent({ kind: "none", confidence: 0.5, reason: "none" })).toBe(true);
		expect(isToolRouterRouteEvent({ kind: "failure", reason: "timeout" })).toBe(true);
	});

	it("rejects malformed payloads", () => {
		expect(isToolRouterRouteEvent(undefined)).toBe(false);
		expect(isToolRouterRouteEvent("read")).toBe(false);
		// The pre-status payload shape: no discriminant, so nothing renders.
		expect(isToolRouterRouteEvent({ tool: "read", confidence: 0.5, reason: "named-tool" })).toBe(false);
		expect(isToolRouterRouteEvent({ kind: "applied", tool: "", confidence: 0.5, reason: "named-tool" })).toBe(false);
		expect(
			isToolRouterRouteEvent({ kind: "applied", tool: "read", confidence: Number.NaN, reason: "named-tool" }),
		).toBe(false);
		expect(isToolRouterRouteEvent({ kind: "applied", tool: "read", confidence: 0.5, reason: "" })).toBe(false);
		expect(isToolRouterRouteEvent({ kind: "advisory", confidence: 0.5, reason: "low-confidence" })).toBe(false);
		expect(isToolRouterRouteEvent({ kind: "none", reason: "none" })).toBe(false);
		// A failure must be one the router can actually report.
		expect(isToolRouterRouteEvent({ kind: "failure", reason: "disabled" })).toBe(false);
		expect(isToolRouterRouteEvent({ kind: "failure" })).toBe(false);
	});
});

describe("PendingToolRoute", () => {
	const decision = { kind: "applied", tool: "read", confidence: 0.9, reason: "named-tool" } as const;

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
		const reused = { kind: "applied", tool: "write", confidence: 0.4, reason: "reused-turn-decision" } as const;
		marker.note(decision);
		marker.note(reused);
		// The replacement is the only decision left: `write` resolves it, and the
		// superseded `read` note is unreachable (consume already cleared it).
		expect(marker.consume("write")).toEqual(reused);
		expect(marker.consume("read")).toBeUndefined();
	});
});

describe("TurnRouteStatusLine", () => {
	it("shows a status once and drops an identical repeat from a retry", () => {
		const latch = new TurnRouteStatusLine();
		const line = toolRouterRouteLine({
			kind: "advisory",
			choice: "eval",
			confidence: 0.62,
			reason: "low-confidence",
		});
		expect(latch.accept(line)).toBe(true);
		expect(latch.accept(line)).toBe(false);
	});

	it("still shows a different status reached later in the same turn", () => {
		const latch = new TurnRouteStatusLine();
		expect(latch.accept("◇ JEV · timeout")).toBe(true);
		expect(latch.accept("◇ JEV · none · 82%")).toBe(true);
	});

	it("drops a repeat even after an intervening different status", () => {
		const latch = new TurnRouteStatusLine();
		const timeout = toolRouterRouteLine({ kind: "failure", reason: "timeout" });
		// Transient failures are re-judged on retry, so a turn can reach `timeout`
		// twice around a routed attempt; the second one must not reprint.
		expect(latch.accept(timeout)).toBe(true);
		expect(latch.accept("◇ JEV · none · 82%")).toBe(true);
		expect(latch.accept(timeout)).toBe(false);
	});

	it("clear() lets the next turn report the same status again", () => {
		const latch = new TurnRouteStatusLine();
		const line = toolRouterRouteLine({ kind: "none", confidence: 0.82, reason: "none" });
		expect(latch.accept(line)).toBe(true);
		latch.clear();
		expect(latch.accept(line)).toBe(true);
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
		expect(toolRouterRouteEvent(outcome)).toEqual({
			kind: "applied",
			tool: "read",
			confidence: 0.9,
			reason: "named-tool",
		});
	});

	it("observes the disabled guard too, so the gate (not silence) decides", async () => {
		const observed = await observedOutcomes(Settings.isolated(), "read", 0.9, readWriteContext());
		expect(observed.length).toBe(1);
		const outcome = observed[0];
		if (outcome === undefined) throw new Error("expected one observed outcome");
		expect(outcome.routed).toBe(false);
		expect(toolRouterRouteEvent(outcome)).toBeUndefined();
	});

	it("observes the below-threshold selection as an advisory line", async () => {
		const settings = Settings.isolated({ "toolRouter.enabled": true, "toolRouter.minConfidence": 0.99 });
		const observed = await observedOutcomes(settings, "eval", 0.62, readWriteContext());
		expect(observed.length).toBe(1);
		const outcome = observed[0];
		if (outcome === undefined) throw new Error("expected one observed outcome");
		expect(outcome.reason).toBe("low-confidence");
		expect(toolRouterRouteEvent(outcome)).toEqual({
			kind: "advisory",
			choice: "eval",
			confidence: 0.62,
			reason: "low-confidence",
		});
	});

	it("observes the prose route as `none`, and a judge error as a failure", async () => {
		const prose = await observedOutcomes(enabledSettings(), TOOL_ROUTER_NO_TOOL, 0.95, readWriteContext());
		expect(toolRouterRouteEvent(prose[0]!)).toEqual({ kind: "none", confidence: 0.95, reason: "none" });

		const failing: ToolRouterOutcome[] = [];
		const wrapped = createSettingsAwareStreamFn(
			enabledSettings(),
			captureBase(),
			undefined,
			{
				getJudge: () =>
					({
						label: "stub/jev",
						judge: (async () => {
							throw new Error("judge exploded");
						}) as unknown as Judge["judge"],
					}) as Judge,
				scope: "main",
			},
			outcome => failing.push(outcome),
		);
		await wrapped(stubModel, readWriteContext(), undefined);
		expect(toolRouterRouteEvent(failing[0]!)).toEqual({ kind: "failure", reason: "judge-error" });
	});
});
