/**
 * Contract: a Jev tool-router judgment is accounted for exactly once, by the
 * upstream judgment infrastructure — never by router-local bookkeeping.
 *
 * The router is a *consumer* of `resolveJudge`: it passes `purpose`,
 * `onUsage` and `telemetry` and lets `ChainJudge` attribute each attempt. These
 * tests drive the real router entry (`createSettingsAwareStreamFn` →
 * `decideToolRoute`) against a real `ChainJudge` over a native TypeSafe
 * candidate, so what is asserted is the consumer-visible result:
 *
 * - a cache MISS runs the provider and journals exactly ONE `model_usage` entry
 *   for this session branch, labelled `purpose: "tool-router"`;
 * - an identical state/question/model is answered from the JudgmentCache with
 *   NO new ledger entry and no second provider request;
 * - a failed billed attempt keeps the upstream ledger semantics (the router
 *   adds no accounting of its own and fails open);
 * - guards that never call the judge (disabled, non-main, no tools) journal
 *   nothing at all;
 * - the configured telemetry receives one `judgment` span stamped
 *   `omp.gen_ai.judgment.purpose = "tool-router"`.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import { Database } from "bun:sqlite";
import * as path from "node:path";
import { OmpGenAIAttr, type AgentTelemetryConfig } from "@oh-my-pi/pi-agent-core/telemetry";
import type { Api, Context, Model } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { type ChainJudge, JudgmentCache, journalJudgmentUsage, resolveJudge } from "@oh-my-pi/pi-coding-agent/judgment";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { createSettingsAwareStreamFn } from "@oh-my-pi/pi-coding-agent/session/settings-stream-fn";
import type { ToolRouterOutcome } from "@oh-my-pi/pi-coding-agent/session/tool-router";
import { BasicTracerProvider, InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createInMemoryAuthStorage } from "./helpers/agent-session-setup";
import { asGlobalFetch } from "./helpers/fetch-mock";

/** Native System One candidate: the API kind the router's judge role resolves to. */
const JEV = {
	id: "jev-preview",
	name: "JEV Preview",
	api: "typesafe",
	provider: "typesafe",
	baseUrl: "https://judge.example.test/",
	kind: "judge",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
} as Model<Api>;

const ROUTED = "read";

function registry(models: Model<Api>[], keys: Record<string, string> = {}): ModelRegistry {
	const authStorage = createInMemoryAuthStorage();
	for (const provider in keys) authStorage.keys.setRuntime(provider, keys[provider]!);
	const model = new ModelRegistry(authStorage, "/nonexistent/router-accounting-models.yml");
	vi.spyOn(model, "getAvailable").mockReturnValue(models);
	return model;
}

function routerContext(): Context {
	return {
		messages: [{ role: "user", content: "read hello.txt and return only its contents", timestamp: Date.now() }],
		tools: [
			{ name: "read", description: "Read a file from disk", parameters: { type: "object", properties: {} } },
			{ name: "write", description: "Write a file to disk", parameters: { type: "object", properties: {} } },
		],
	} as unknown as Context;
}

function enabledSettings(): Settings {
	return Settings.isolated({ "toolRouter.enabled": true, modelRoles: { judge: "typesafe/jev-preview" } });
}

/** One router inference: a fresh wrapper, which is how the router sees a new turn. */
async function routeOnce(
	streamSettings: Settings,
	judge: ChainJudge,
	context: Context,
	scope: "main" | "advisor" = "main",
): Promise<ToolRouterOutcome> {
	const outcomes: ToolRouterOutcome[] = [];
	const wrapped = createSettingsAwareStreamFn(
		streamSettings,
		() => new AssistantMessageEventStream(),
		undefined,
		{ getJudge: () => judge, scope },
		outcome => outcomes.push(outcome),
	);
	await wrapped(JEV, context, undefined);
	const outcome = outcomes[0];
	if (outcome === undefined) throw new Error("expected one observed router outcome");
	return outcome;
}

function choiceBody(): Response {
	return Response.json({
		model: "jev-1.13.0",
		answers: {
			route: {
				type: "choice",
				choice: ROUTED,
				probabilities: { [ROUTED]: 0.9, write: 0.1 },
				confidence: 0.9,
			},
		},
		usage: { input_tokens: 8, output_tokens: 2 },
	});
}

/** The `model_usage` entries this session branch recorded. */
function ledger(manager: SessionManager) {
	return manager.getBranch().filter(entry => entry.type === "model_usage");
}

function cacheCounts(dbPath: string): { states: number; oracle: number; usage: number } {
	const db = new Database(dbPath, { readonly: true });
	const count = (table: string) => db.query<{ n: number }, []>(`SELECT count(*) AS n FROM ${table}`).get()?.n ?? -1;
	const counts = { states: count("states"), oracle: count("oracle"), usage: count("usage") };
	db.close();
	return counts;
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe("tool-router judgment accounting", () => {
	it("reports one billed tool-router attempt on a miss and populates the judgment cache", async () => {
		using tempDir = TempDir.createSync("@omp-router-accounting-");
		const dbPath = path.join(tempDir.path(), "judgment-cache.db");
		const cache = JudgmentCache.open(dbPath);
		const manager = SessionManager.inMemory();
		manager.appendMessage({ role: "user", content: "read hello.txt", timestamp: 1 });
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(asGlobalFetch(async () => choiceBody()));
		const onUsage = vi.fn();

		const judge = resolveJudge({
			settings: enabledSettings(),
			registry: registry([JEV], { typesafe: "ts-key" }),
			sessionId: "session-router-accounting",
			purpose: "tool-router",
			onUsage,
			cache,
		});
		const outcome = await routeOnce(enabledSettings(), judge, routerContext());

		expect(outcome.routed).toBe(true);
		expect(outcome.reason).toBe("named-tool");
		expect(outcome.choice).toBe(ROUTED);
		expect(fetchSpy).toHaveBeenCalledTimes(1);

		// The attempt is attributed exactly once, labelled by purpose.
		expect(ledger(manager)).toHaveLength(0);
		expect(onUsage).toHaveBeenCalledTimes(1);
		expect(onUsage.mock.calls[0]?.[0]).toMatchObject({ purpose: "tool-router", provider: "typesafe" });

		cache.close();
		expect(cacheCounts(dbPath)).toEqual({ states: 1, oracle: 1, usage: 1 });
	});

	it("serves an identical state/question/model from the cache without a second ledger entry", async () => {
		using tempDir = TempDir.createSync("@omp-router-accounting-");
		const dbPath = path.join(tempDir.path(), "judgment-cache.db");
		const cache = JudgmentCache.open(dbPath);
		const manager = SessionManager.inMemory();
		manager.appendMessage({ role: "user", content: "read hello.txt", timestamp: 1 });
		const leafBefore = manager.getLeafId();
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(asGlobalFetch(async () => choiceBody()));
		const deps = {
			settings: enabledSettings(),
			registry: registry([JEV], { typesafe: "ts-key" }),
			sessionId: "session-router-accounting",
			purpose: "tool-router",
			onUsage: journalJudgmentUsage(manager),
			cache,
		};

		// Two independent inferences over byte-identical state and question, which
		// is what a repeat of the same intent under the same roster produces.
		const first = await routeOnce(enabledSettings(), resolveJudge(deps), routerContext());
		const billedRows = cacheCounts(dbPath);
		const second = await routeOnce(enabledSettings(), resolveJudge(deps), routerContext());

		expect(first.choice).toBe(ROUTED);
		expect(second.choice).toBe(ROUTED);
		expect(second.confidence).toBe(first.confidence);
		// The cached answer never reaches the provider and is never billed again:
		// the branch still holds the single entry the miss produced.
		expect(fetchSpy).toHaveBeenCalledTimes(1);
		const entries = ledger(manager);
		expect(entries).toHaveLength(1);
		expect(entries[0]).toMatchObject({
			parentId: leafBefore,
			purpose: "tool-router",
			role: "typesafe",
			provider: "typesafe",
			model: "jev-preview",
			usage: { input: 8, output: 2 },
		});

		cache.close();
		expect(cacheCounts(dbPath)).toEqual(billedRows);
	});

	it("keeps upstream ledger semantics for a failed attempt and adds no router-local accounting", async () => {
		using tempDir = TempDir.createSync("@omp-router-accounting-");
		const cache = JudgmentCache.open(path.join(tempDir.path(), "judgment-cache.db"));
		const manager = SessionManager.inMemory();
		manager.appendMessage({ role: "user", content: "read hello.txt", timestamp: 1 });
		const leafBefore = manager.getLeafId();
		// A rejected account: billed as an attempt (upstream reports it once with
		// `stopReason: "error"`), never retried, and it leaves the turn unrouted.
		const fetchSpy = vi
			.spyOn(globalThis, "fetch")
			.mockImplementation(
				asGlobalFetch(async () => Response.json({ detail: { error_type: "billing_error" } }, { status: 402 })),
			);

		const judge = resolveJudge({
			settings: enabledSettings(),
			registry: registry([JEV], { typesafe: "ts-key" }),
			sessionId: "session-router-accounting",
			purpose: "tool-router",
			onUsage: journalJudgmentUsage(manager),
			cache,
		});
		const outcome = await routeOnce(enabledSettings(), judge, routerContext());

		expect(fetchSpy).toHaveBeenCalledTimes(1);
		expect(outcome.routed).toBe(false);
		expect(outcome.reason).toBe("judge-error");
		// One attributed attempt, still labelled by purpose: the router invents no
		// accounting of its own for a failure.
		const entries = ledger(manager);
		expect(entries).toHaveLength(1);
		expect(entries[0]).toMatchObject({
			parentId: leafBefore,
			purpose: "tool-router",
			model: "jev-preview",
			stopReason: "error",
		});
		cache.close();
	});

	it("journals nothing for guards that never consult the judge", async () => {
		using tempDir = TempDir.createSync("@omp-router-accounting-");
		const cache = JudgmentCache.open(path.join(tempDir.path(), "judgment-cache.db"));
		const manager = SessionManager.inMemory();
		manager.appendMessage({ role: "user", content: "read hello.txt", timestamp: 1 });
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(asGlobalFetch(async () => choiceBody()));
		const onUsage = vi.fn();
		const judgeFor = (settings: Settings) =>
			resolveJudge({
				settings,
				registry: registry([JEV], { typesafe: "ts-key" }),
				sessionId: "session-router-accounting",
				purpose: "tool-router",
				onUsage,
				cache,
			});

		const disabled = await routeOnce(Settings.isolated(), judgeFor(Settings.isolated()), routerContext());
		const nonMain = await routeOnce(enabledSettings(), judgeFor(enabledSettings()), routerContext(), "advisor");
		const noTools = await routeOnce(enabledSettings(), judgeFor(enabledSettings()), {
			messages: [{ role: "user", content: "read hello.txt", timestamp: Date.now() }],
			tools: [],
		} as unknown as Context);

		expect([disabled.reason, nonMain.reason, noTools.reason]).toEqual(["disabled", "non-main-source", "no-tools"]);
		expect(fetchSpy).not.toHaveBeenCalled();
		expect(onUsage).not.toHaveBeenCalled();
		expect(ledger(manager)).toHaveLength(0);
		cache.close();
	});
});

describe("tool-router judgment telemetry", () => {
	function telemetryFixture(): {
		exporter: InMemorySpanExporter;
		telemetry: AgentTelemetryConfig;
		provider: BasicTracerProvider;
	} {
		const exporter = new InMemorySpanExporter();
		const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
		return {
			exporter,
			provider,
			telemetry: { conversationId: "conv-router", tracer: provider.getTracer("router-accounting-test") },
		};
	}

	const judgmentSpans = (exporter: InMemorySpanExporter) =>
		exporter.getFinishedSpans().filter(span => span.attributes[OmpGenAIAttr.JudgmentPurpose] !== undefined);

	it("stamps purpose on the judgment span and reports a cached question on a hit", async () => {
		using tempDir = TempDir.createSync("@omp-router-accounting-");
		const cache = JudgmentCache.open(path.join(tempDir.path(), "judgment-cache.db"));
		const { exporter, telemetry, provider } = telemetryFixture();
		const manager = SessionManager.inMemory();
		manager.appendMessage({ role: "user", content: "read hello.txt", timestamp: 1 });
		vi.spyOn(globalThis, "fetch").mockImplementation(asGlobalFetch(async () => choiceBody()));
		const judge = resolveJudge({
			settings: enabledSettings(),
			registry: registry([JEV], { typesafe: "ts-key" }),
			sessionId: "session-router-accounting",
			purpose: "tool-router",
			onUsage: journalJudgmentUsage(manager),
			telemetry,
			cache,
		});

		await routeOnce(enabledSettings(), judge, routerContext());
		await routeOnce(enabledSettings(), judge, routerContext());
		// The span is emitted from a fire-and-forget report; let it finish.
		for (let i = 0; i < 100 && judgmentSpans(exporter).length < 2; i++) await Bun.sleep(10);

		const spans = judgmentSpans(exporter);
		expect(spans.map(span => span.attributes[OmpGenAIAttr.JudgmentPurpose])).toEqual(["tool-router", "tool-router"]);
		expect(spans[0]?.name.startsWith("judgment ")).toBe(true);
		// Miss: one question asked of the provider. Hit: it came from the cache,
		// which is the upstream contract for a cached answer.
		expect(spans[0]?.attributes[OmpGenAIAttr.JudgmentQuestions]).toBe(1);
		expect(spans[0]?.attributes[OmpGenAIAttr.JudgmentCachedQuestions]).toBe(0);
		expect(spans[1]?.attributes[OmpGenAIAttr.JudgmentCachedQuestions]).toBe(1);

		cache.close();
		await provider.shutdown();
	});
});
