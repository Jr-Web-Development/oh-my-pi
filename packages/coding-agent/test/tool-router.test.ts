/**
 * Contract: native Jev tool router (`session/tool-router.ts`) placed in the
 * `createSettingsAwareStreamFn` seam.
 *
 * Guards (zero Jev calls): disabled default, no tools, explicit caller
 * `toolChoice` (including explicit `"auto"`). Routing: valid named choice ->
 * generic `{ type: "function", name }`, `no_tool_needed` -> generic `"none"`.
 * Fail-open (caller options untouched): low confidence, unknown tool,
 * malformed answer, judge error, timeout. The router never mutates inputs,
 * never sees credentials, and judgment side requests bypass it structurally
 * (backends call core `streamSimple`, never the wrapper).
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import type { StreamFn } from "@oh-my-pi/pi-agent-core";
import type { Context, Judge, JudgmentResult, Model, Questions, SimpleStreamOptions } from "@oh-my-pi/pi-ai";
import { normalizeCodexToolChoice } from "@oh-my-pi/pi-ai/providers/openai-codex-responses";
import { mapAnthropicToolChoice, mapGoogleToolChoice } from "@oh-my-pi/pi-ai/stream";
import type { ToolChoice } from "@oh-my-pi/pi-ai/types";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { logger } from "@oh-my-pi/pi-utils";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	cfgToolRouterEnabled,
	cfgToolRouterMinConfidence,
	cfgToolRouterTimeoutMs,
} from "@oh-my-pi/pi-coding-agent/tools/settings";
import { createSettingsAwareStreamFn } from "@oh-my-pi/pi-coding-agent/session/settings-stream-fn";
import { TOOL_ROUTER_NO_TOOL } from "@oh-my-pi/pi-coding-agent/session/tool-router";

const stubModel = { api: "openai-completions", provider: "test", id: "test-model" } as unknown as Model;

function makeContext(
	tools: Array<{ name: string; description?: string }>,
	intent = "read hello.txt and return only its contents",
): Context {
	return {
		messages: [{ role: "user", content: intent, timestamp: Date.now() }],
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

interface JudgeProbe {
	judge: Judge;
	calls: Array<{ state: unknown; questions: unknown }>;
}

function stubJudge(behavior: (callIndex: number) => Promise<JudgmentResult<Questions>>): JudgeProbe {
	const calls: JudgeProbe["calls"] = [];
	const judge: Judge = {
		label: "stub/jev",
		judge: (async (request: { state: unknown; questions: unknown }) => {
			calls.push({ state: request.state, questions: request.questions });
			return behavior(calls.length);
		}) as unknown as Judge["judge"],
	};
	return { judge, calls };
}

function captureBase(): { fn: StreamFn; calls: Array<{ options?: SimpleStreamOptions }> } {
	const calls: Array<{ options?: SimpleStreamOptions }> = [];
	const fn: StreamFn = (_model, _context, options) => {
		calls.push({ options });
		return new AssistantMessageEventStream();
	};
	return { fn, calls };
}

function enabledSettings(): Settings {
	return Settings.isolated({ "toolRouter.enabled": true });
}

async function routedOptions(
	settings: Settings,
	probe: JudgeProbe,
	context: Context,
	callerOptions?: SimpleStreamOptions,
): Promise<{ calls: Array<{ options?: SimpleStreamOptions }>; judgeCalls: number }> {
	const { fn: base, calls } = captureBase();
	const wrapped = createSettingsAwareStreamFn(settings, base, undefined, {
		getJudge: () => probe.judge,
		scope: "main",
	});
	await wrapped(stubModel, context, callerOptions);
	return { calls, judgeCalls: probe.calls.length };
}

describe("toolRouter settings defaults", () => {
	it("stays disabled with conservative bounded defaults", () => {
		expect(cfgToolRouterEnabled.default).toBe(false);
		expect(cfgToolRouterMinConfidence.default).toBe(0.7);
		expect(cfgToolRouterTimeoutMs.default).toBe(1500);
	});

	it("slowModeContext and toolRouter coexist on separate arguments", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(
			enabledSettings(),
			base,
			{},
			{
				getJudge: () => probe.judge,
				scope: "main",
			},
		);

		await wrapped(stubModel, readWriteContext(), undefined);

		expect(probe.calls.length).toBe(1);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
	});
});

describe("tool router guards", () => {
	it("disabled: zero judge calls, original options, stays synchronous", () => {
		const probe = stubJudge(async () => choiceResult("read", 0.99));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(Settings.isolated({}), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});

		const returned = wrapped(stubModel, readWriteContext(), undefined);

		expect(probe.calls.length).toBe(0);
		expect(returned).not.toBeInstanceOf(Promise);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it("no tools: zero judge calls", async () => {
		const emptyTools = stubJudge(async () => choiceResult("read", 0.99));
		const noToolsField = stubJudge(async () => choiceResult("read", 0.99));

		const emptyResult = await routedOptions(enabledSettings(), emptyTools, makeContext([]));
		const bareContext = { messages: [{ role: "user", content: "hi", timestamp: Date.now() }] } as unknown as Context;
		const missingResult = await routedOptions(enabledSettings(), noToolsField, bareContext);

		expect(emptyResult.judgeCalls).toBe(0);
		expect(missingResult.judgeCalls).toBe(0);
		expect(emptyResult.calls[0]?.options?.toolChoice).toBeUndefined();
		expect(missingResult.calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it('explicit toolChoice "auto": zero judge calls, preserved', async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.99));

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext(), {
			toolChoice: "auto",
		});

		expect(judgeCalls).toBe(0);
		expect(calls[0]?.options?.toolChoice).toBe("auto");
	});

	it("explicit named toolChoice: zero judge calls, preserved", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.99));

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext(), {
			toolChoice: { type: "function", name: "write" },
		});

		expect(judgeCalls).toBe(0);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "write" });
	});
});

describe("tool router decisions", () => {
	it("valid tool with sufficient confidence: generic named ToolChoice", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const callerOptions: SimpleStreamOptions = { temperature: 0 };

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext(), callerOptions);

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		expect(calls[0]?.options).not.toBe(callerOptions);
		expect(calls[0]?.options?.temperature).toBe(0);
	});

	it("no_tool_needed: generic none", async () => {
		const probe = stubJudge(async () => choiceResult(TOOL_ROUTER_NO_TOOL, 0.95));

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBe("none");
	});

	it("low confidence: original path", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.5));

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it("unknown tool: original path", async () => {
		const probe = stubJudge(async () => choiceResult("shell_exec", 0.99));

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it("malformed answer: original path", async () => {
		const missing = stubJudge(
			async () => ({ ...choiceResult("read", 0.99), answers: {} }) as unknown as JudgmentResult<Questions>,
		);
		const wrongType = stubJudge(
			async () =>
				({
					...choiceResult("read", 0.99),
					answers: { route: { type: "noul", noul: 0.9 } },
				}) as unknown as JudgmentResult<Questions>,
		);

		const missingResult = await routedOptions(enabledSettings(), missing, readWriteContext());
		const wrongTypeResult = await routedOptions(enabledSettings(), wrongType, readWriteContext());

		expect(missingResult.judgeCalls).toBe(1);
		expect(wrongTypeResult.judgeCalls).toBe(1);
		expect(missingResult.calls[0]?.options?.toolChoice).toBeUndefined();
		expect(wrongTypeResult.calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it("judge error: original path without throwing", async () => {
		const probe = stubJudge(async () => {
			throw new Error("jev unavailable");
		});

		const { calls, judgeCalls } = await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
	});

	it("timeout: original path within the configured deadline", async () => {
		const probe = stubJudge(() => new Promise<never>(() => {}));
		const settings = Settings.isolated({ "toolRouter.enabled": true, "toolRouter.timeoutMs": 50 });
		const started = Date.now();

		const { calls, judgeCalls } = await routedOptions(settings, probe, readWriteContext());
		const elapsed = Date.now() - started;

		expect(judgeCalls).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
		expect(elapsed).toBeLessThan(2000);
	});

	it("does not mutate input options, model, or context", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const callerOptions: SimpleStreamOptions = { temperature: 0 };
		const context = readWriteContext();
		const beforeOptions = JSON.stringify(callerOptions);
		const beforeContext = JSON.stringify(context);

		const { calls } = await routedOptions(enabledSettings(), probe, context, callerOptions);

		expect(JSON.stringify(callerOptions)).toBe(beforeOptions);
		expect(JSON.stringify(context)).toBe(beforeContext);
		expect(callerOptions.toolChoice).toBeUndefined();
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
	});

	it("judge side requests do not re-enter the router", async () => {
		const { fn: base, calls } = captureBase();
		let innerCalls = 0;
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			scope: "main",
			getJudge: () =>
				({
					label: "stub/online-backend",
					judge: (async () => {
						// Mirrors the production online backend path
						// (chatTextBackend -> completeSimple -> core streamSimple):
						// judgment work goes straight to the base stream fn.
						await base(stubModel, makeContext([], "keyword reply"), undefined);
						innerCalls += 1;
						return choiceResult("read", 0.9);
					}) as unknown as Judge["judge"],
				}) as Judge,
		});

		await wrapped(stubModel, readWriteContext(), undefined);

		expect(innerCalls).toBe(1);
		expect(calls.length).toBe(2);
		// Inner judgment side request went straight to base, untouched.
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
		// Outer request carries the routed generic named choice.
		expect(calls[1]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
	});

	it("sends only semantic input to jev, never credentials", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));

		await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(probe.calls.length).toBe(1);
		const seen = probe.calls[0];
		const serialized = JSON.stringify(seen);
		expect(seen?.state).toEqual({ intent: "read hello.txt and return only its contents" });
		const questions = (seen?.questions ?? {}) as Record<
			string,
			{ type: string; criteria: Record<string, string | null> }
		>;
		expect(questions["route"]?.type).toBe("choice");
		expect(Object.keys(questions["route"]?.criteria ?? {}).sort()).toEqual(["no_tool_needed", "read", "write"]);
		expect(serialized).not.toContain("Bearer");
		expect(serialized).not.toContain("apiKey");
		expect(serialized).not.toContain("Authorization");
	});
});

describe("tool router observability", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	function lastDecision(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown> {
		const entry = spy.mock.calls.find((args: [unknown, ...unknown[]]) => args[0] === "tool-router decision");
		if (!entry) throw new Error("expected a tool-router decision log");
		return entry[1] as Record<string, unknown>;
	}

	it("logs choice on low-confidence fail-open", async () => {
		const spy = vi.spyOn(logger, "debug").mockImplementation(() => {});
		const probe = stubJudge(async () => choiceResult("read", 0.5));

		await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(lastDecision(spy)).toMatchObject({
			source: "main",
			provider: "test",
			model: "test-model",
			tools: 2,
			result: "passthrough",
			reason: "low-confidence",
			choice: "read",
			confidence: 0.5,
		});
	});

	it("logs the invented name on unknown-tool fail-open", async () => {
		const spy = vi.spyOn(logger, "debug").mockImplementation(() => {});
		const probe = stubJudge(async () => choiceResult("shell_exec", 0.99));

		await routedOptions(enabledSettings(), probe, readWriteContext());

		expect(lastDecision(spy)).toMatchObject({
			result: "passthrough",
			reason: "unknown-tool",
			choice: "shell_exec",
			confidence: 0.99,
		});
	});

	it("records scope and session id, never prompt or secrets", async () => {
		const spy = vi.spyOn(logger, "debug").mockImplementation(() => {});
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});

		await wrapped(stubModel, readWriteContext(), { sessionId: "sess-1" });

		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		const fields = lastDecision(spy);
		expect(fields["source"]).toBe("main");
		expect(fields["session"]).toBe("sess-1");
		expect(Object.keys(fields).sort()).toEqual(
			[
				"choice",
				"confidence",
				"latencyMs",
				"model",
				"provider",
				"reason",
				"result",
				"session",
				"source",
				"tools",
			].sort(),
		);
	});

	it.each(["advisor", "capture", "side-channel"] as const)(
		"source %s: zero judge calls, base options unchanged",
		async scope => {
			const probe = stubJudge(async () => choiceResult("read", 0.99));
			const { fn: base, calls } = captureBase();
			const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
				getJudge: () => probe.judge,
				scope,
			});

			const returned = wrapped(stubModel, readWriteContext(), undefined);

			expect(probe.calls.length).toBe(0);
			expect(returned).not.toBeInstanceOf(Promise);
			expect(calls[0]?.options?.toolChoice).toBeUndefined();
		},
	);

	it("main fallback across providers reuses the turn decision without a second judge call", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});
		const fallbackModel = { api: "openai-completions", provider: "other", id: "other-model" } as unknown as Model;
		// Same inference retried on another provider: the agent loop reuses
		// the identical context object, so the turn fingerprint matches.
		const context = readWriteContext();

		await wrapped(stubModel, context, undefined);
		await wrapped(fallbackModel, context, undefined);

		expect(probe.calls.length).toBe(1);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		expect(calls[1]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		expect(calls[0]?.options).not.toBe(calls[1]?.options);
	});
});

describe("tool router turn gate: one jev decision per user turn", () => {
	function appendToolResult(context: Context, text = "file contents"): Context {
		const assistantCall = {
			role: "assistant",
			content: [],
			timestamp: Date.now(),
			api: "test",
			provider: "test",
			model: "test-model",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
		};
		const toolResult = {
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "read",
			content: [{ type: "text", text }],
			isError: false,
			timestamp: Date.now(),
		};
		return { ...context, messages: [...context.messages, assistantCall, toolResult] } as unknown as Context;
	}

	function appendUserTurn(context: Context, intent: string): Context {
		const turn = { role: "user", content: intent, timestamp: Date.now() + 1 };
		return { ...context, messages: [...context.messages, turn] } as unknown as Context;
	}

	it("post-tool follow-up does not re-judge and does not re-force the tool", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});
		const initial = readWriteContext();

		await wrapped(stubModel, initial, undefined);
		await wrapped(stubModel, appendToolResult(initial), undefined);

		expect(probe.calls.length).toBe(1);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		expect(calls[1]?.options?.toolChoice).toBeUndefined();
	});

	it("a new user turn re-arms the router", async () => {
		const probe = stubJudge(async callIndex => choiceResult(callIndex === 1 ? "read" : "write", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});
		const turn1 = readWriteContext();

		await wrapped(stubModel, turn1, undefined);
		await wrapped(stubModel, appendToolResult(turn1), undefined);
		await wrapped(stubModel, appendUserTurn(turn1, "now write the file instead"), undefined);

		expect(probe.calls.length).toBe(2);
		expect(calls[0]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
		expect(calls[1]?.options?.toolChoice).toBeUndefined();
		expect(calls[2]?.options?.toolChoice).toEqual({ type: "function", name: "write" });
	});

	it("none applies only to its own turn", async () => {
		const probe = stubJudge(async callIndex => choiceResult(callIndex === 1 ? TOOL_ROUTER_NO_TOOL : "read", 0.9));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});
		const turn1 = readWriteContext();

		await wrapped(stubModel, turn1, undefined);
		await wrapped(stubModel, appendUserTurn(turn1, "read hello.txt now"), undefined);

		expect(probe.calls.length).toBe(2);
		expect(calls[0]?.options?.toolChoice).toBe("none");
		expect(calls[1]?.options?.toolChoice).toEqual({ type: "function", name: "read" });
	});

	it("low-confidence fail-open does not trigger a post-tool re-judge", async () => {
		const probe = stubJudge(async () => choiceResult("read", 0.5));
		const { fn: base, calls } = captureBase();
		const wrapped = createSettingsAwareStreamFn(enabledSettings(), base, undefined, {
			getJudge: () => probe.judge,
			scope: "main",
		});
		const initial = readWriteContext();

		await wrapped(stubModel, initial, undefined);
		await wrapped(stubModel, appendToolResult(initial), undefined);

		expect(probe.calls.length).toBe(1);
		expect(calls[0]?.options?.toolChoice).toBeUndefined();
		expect(calls[1]?.options?.toolChoice).toBeUndefined();
	});
});

describe("generic choice reaches existing provider mappers", () => {
	const named: ToolChoice = { type: "function", name: "read" };
	const tools = makeContext([{ name: "read" }]).tools ?? [];

	it("openai/codex contract: generic shape is the mapper input", () => {
		expect(named).toEqual({ type: "function", name: "read" });
		expect(normalizeCodexToolChoice(named, tools)).toEqual({ type: "function", name: "read" });
		expect(normalizeCodexToolChoice("none", tools)).toBe("none");
	});

	it("google mapper: generic named choice becomes a one-entry ANY allow-list", () => {
		expect(mapGoogleToolChoice(named)).toEqual({ mode: "ANY", allowedFunctionNames: ["read"] });
		expect(mapGoogleToolChoice("none")).toBe("none");
	});

	it("anthropic mapper: generic named choice becomes a tool pin", () => {
		expect(mapAnthropicToolChoice(named)).toEqual({ type: "tool", name: "read" });
		expect(mapAnthropicToolChoice("none")).toBe("none");
	});
});
