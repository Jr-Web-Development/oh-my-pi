/**
 * Regression contract for the Jev route indicator row.
 *
 * The indicator (`◆ JEV → <tool> · <pct>%`) is printed immediately BEFORE the
 * transcript row of the tool the router forced. The TUI creates that row while
 * the assistant message is still streaming (`message_update`), not at
 * `tool_execution_start` — annotating only in `tool_execution_start` (the
 * original implementation) never rendered the marker in a live session.
 *
 * `showStatus` is the seam the controller uses to emit the marker. The real
 * `UiHelpers.showStatus` mounts a row into `chatContainer`, so the fixture
 * override below appends a real `Text` row there too; that makes the marker's
 * position relative to the tool row structurally assertable instead of relying
 * on a bare spy.
 */
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "bun:test";
import type { AssistantMessage, ToolCall, Usage } from "@oh-my-pi/pi-ai";
import { resetSettingsForTest, Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { EventController } from "@oh-my-pi/pi-coding-agent/modes/controllers/event-controller";
import type { InteractiveModeContext } from "@oh-my-pi/pi-coding-agent/modes/types";
import type { AgentSessionEvent } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type {
	ToolRouterAppliedRoute,
	ToolRouterStatusRoute,
} from "@oh-my-pi/pi-coding-agent/session/tool-router-events";
import { AssistantMessageComponent } from "@oh-my-pi/pi-tui/chat/assistant-message";
import { ReadToolGroupComponent } from "@oh-my-pi/pi-tui/chat/read-tool-group";
import { type Component, Text } from "@oh-my-pi/pi-tui";
import { initTheme } from "@oh-my-pi/pi-tui/theme";
import { createInteractiveModeContext } from "./helpers/interactive-mode-context";

function zeroUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function toolCall(id: string, name: string, args: Record<string, unknown>): ToolCall {
	return { type: "toolCall", id, name, arguments: args };
}

function assistantMessage(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "test",
		provider: "test",
		model: "test-model",
		stopReason: "stop",
		usage: zeroUsage(),
		timestamp: 1,
	};
}

/** Cumulative `message_update` snapshot listing every tool call so far. */
function messageUpdate(
	calls: ToolCall[],
	contentIndex: number,
): Extract<AgentSessionEvent, { type: "message_update" }> {
	const message = assistantMessage(calls);
	return {
		type: "message_update",
		message,
		assistantMessageEvent: {
			type: "toolcall_end",
			contentIndex,
			toolCall: calls[contentIndex]!,
			partial: message,
		},
	} as Extract<AgentSessionEvent, { type: "message_update" }>;
}

function toolStart(call: ToolCall): Extract<AgentSessionEvent, { type: "tool_execution_start" }> {
	return {
		type: "tool_execution_start",
		toolCallId: call.id,
		toolName: call.name,
		args: call.arguments,
	} as Extract<AgentSessionEvent, { type: "tool_execution_start" }>;
}

/** A read whose `path` collapses into the compact read group (filesystem target). */
function readCall(id: string): ToolCall {
	return toolCall(id, "read", { path: "src/modes/controllers/event-controller.ts" });
}

const READ_DECISION: ToolRouterAppliedRoute = { kind: "applied", tool: "read", confidence: 0.98, reason: "named-tool" };
const BASH_DECISION: ToolRouterAppliedRoute = { kind: "applied", tool: "bash", confidence: 0.98, reason: "named-tool" };
const LOW_CONFIDENCE: ToolRouterStatusRoute = {
	kind: "advisory",
	choice: "eval",
	confidence: 0.62,
	reason: "low-confidence",
};
const PROSE_ROUTE: ToolRouterStatusRoute = { kind: "none", confidence: 0.82, reason: "none" };

interface Fixture {
	readonly controller: EventController;
	readonly ctx: InteractiveModeContext;
	/** Marker rows appended to `chatContainer` by the `showStatus` override, in order. */
	readonly markerRows: Text[];
	/** The exact message passed to `showStatus` for each marker row. */
	readonly markerTexts: string[];
}

/**
 * The real `UiHelpers.showStatus` mounts a row into the transcript; mirror that
 * so the marker participates in `chatContainer.children` ordering.
 */
function createFixture(): Fixture {
	const markerRows: Text[] = [];
	const markerTexts: string[] = [];
	const ctx = createInteractiveModeContext({
		session: { isStreaming: true, getToolByName: () => undefined },
		streamingComponent: new AssistantMessageComponent(),
		showStatus: (message: string) => {
			const row = new Text(message, 1, 0);
			ctx.chatContainer.addChild(row);
			markerRows.push(row);
			markerTexts.push(message);
		},
	});
	ctx.chatContainer.setToolActivityVisible(true);
	return { controller: new EventController(ctx), ctx, markerRows, markerTexts };
}

/** The transcript row the controller registered for a tool call. */
function toolRow(ctx: InteractiveModeContext, toolCallId: string): Component {
	const row = ctx.pendingTools.get(toolCallId);
	if (!row) throw new Error(`no transcript row registered for tool call ${toolCallId}`);
	return row as unknown as Component;
}

beforeAll(async () => {
	await initTheme(false);
});

// Shared by both suites: the message-update path reads live settings.
beforeEach(async () => {
	resetSettingsForTest();
	await Settings.init({ inMemory: true, overrides: { "display.smoothStreaming": false } });
});

afterEach(() => {
	resetSettingsForTest();
});

describe("EventController Jev route indicator", () => {
	it("prints the indicator above the streamed tool row (the regression: annotation only at tool_execution_start left the marker unrendered)", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();
		// A non-read tool deliberately: its streamed row is created by the ordinary
		// `ToolExecutionComponent` branch, the other row-creation site that must annotate.
		const call = toolCall("toolu_jev_real_order", "bash", { command: "ls -la" });

		controller.noteJevRouteEvent(BASH_DECISION);

		// The row is created here, during streaming — before any tool_execution_start.
		await controller.handleEvent(messageUpdate([call], 0));
		const rowAfterStream = toolRow(ctx, call.id);
		expect(rowAfterStream).not.toBeInstanceOf(ReadToolGroupComponent);

		await controller.handleEvent(toolStart(call));

		// Exactly one marker, carrying the exact indicator text.
		expect(markerRows).toHaveLength(1);
		expect(markerTexts[0]).toBe("◆ JEV → bash · 98%");

		// And it sits strictly ABOVE the tool row it annotates.
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(
			ctx.chatContainer.children.indexOf(rowAfterStream),
		);

		// The fallback `tool_execution_start` path must not re-annotate an existing row.
		expect(markerRows).toHaveLength(1);
	});

	it("still renders the indicator when only tool_execution_start arrives (no streamed row)", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();
		const call = readCall("toolu_jev_fallback");

		controller.noteJevRouteEvent(READ_DECISION);
		await controller.handleEvent(toolStart(call));

		const row = toolRow(ctx, call.id);
		expect(markerRows).toHaveLength(1);
		expect(markerTexts[0]).toBe("◆ JEV → read · 98%");
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(ctx.chatContainer.children.indexOf(row));
	});

	it("prints the indicator above a read that collapses into the shared read group", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();
		const call = readCall("toolu_jev_read_group");

		controller.noteJevRouteEvent(READ_DECISION);
		await controller.handleEvent(messageUpdate([call], 0));

		const group = toolRow(ctx, call.id);
		expect(group).toBeInstanceOf(ReadToolGroupComponent);
		expect(markerRows).toHaveLength(1);
		expect(markerTexts[0]).toBe("◆ JEV → read · 98%");
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(ctx.chatContainer.children.indexOf(group));
	});

	it("does not mark a follow-up tool row created after the forced tool in the same turn", async () => {
		const { controller, ctx, markerRows } = createFixture();
		const forced = readCall("toolu_jev_followup_read");
		const followUp = toolCall("toolu_jev_followup_bash", "bash", { command: "ls" });

		controller.noteJevRouteEvent(READ_DECISION);
		await controller.handleEvent(messageUpdate([forced], 0));
		await controller.handleEvent(toolStart(forced));
		await controller.handleEvent(toolStart(followUp));

		const followUpRow = toolRow(ctx, followUp.id);
		expect(markerRows).toHaveLength(1);
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(
			ctx.chatContainer.children.indexOf(toolRow(ctx, forced.id)),
		);
		expect(ctx.chatContainer.children.indexOf(followUpRow)).toBeGreaterThan(
			ctx.chatContainer.children.indexOf(markerRows[0]),
		);
		expect(markerRows).toHaveLength(1);
	});

	it("leaves an unrelated tool row unmarked but keeps the marker for the forced tool's later row", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();
		const unrelated = toolCall("toolu_jev_wrong_name_bash", "bash", { command: "ls" });
		const forced = readCall("toolu_jev_wrong_name_read");

		controller.noteJevRouteEvent(READ_DECISION);

		// The model picked its own tool first: it must NOT be annotated as a router decision...
		await controller.handleEvent(messageUpdate([unrelated], 0));
		expect(markerRows).toHaveLength(0);

		// ...and that mismatch must NOT destroy the provenance: the forced tool's row,
		// created later in the same turn, still gets its indicator.
		await controller.handleEvent(messageUpdate([unrelated, forced], 1));
		const forcedRow = toolRow(ctx, forced.id);
		expect(markerRows).toHaveLength(1);
		expect(markerTexts[0]).toBe("◆ JEV → read · 98%");
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(
			ctx.chatContainer.children.indexOf(forcedRow),
		);

		// After the match the marker is spent: a further row gets nothing.
		const after = toolCall("toolu_jev_wrong_name_grep", "grep", { pattern: "x" });
		await controller.handleEvent(toolStart(after));
		expect(markerRows).toHaveLength(1);
	});

	it("renders no indicator when no router decision was applied", async () => {
		const { controller, markerRows } = createFixture();
		const call = readCall("toolu_jev_no_decision");

		// Nothing noted: no APPLIED route. A passthrough/low-confidence/none outcome
		// still renders, but only as the standalone ◇ status line (covered by the
		// status-line suite below) — never as a marker on a tool row.
		await controller.handleEvent(messageUpdate([call], 0));
		expect(markerRows).toHaveLength(0);

		await controller.handleEvent(toolStart(call));
		expect(markerRows).toHaveLength(0);

		// The tool_execution_start creation branch (non-read) is unmarked too.
		const other = toolCall("toolu_jev_no_decision_bash", "bash", { command: "ls" });
		await controller.handleEvent(toolStart(other));
		expect(markerRows).toHaveLength(0);
	});

	it("marks the forced row only once when a republished decision and a repeated row-creation attempt arrive", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();
		const call = readCall("toolu_jev_reused_turn");

		// A retry/fallback republishes the same applied decision for the turn.
		controller.noteJevRouteEvent(READ_DECISION);
		controller.noteJevRouteEvent(READ_DECISION);

		await controller.handleEvent(messageUpdate([call], 0));
		const row = toolRow(ctx, call.id);
		expect(markerRows).toHaveLength(1);
		expect(markerTexts[0]).toBe("◆ JEV → read · 98%");
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(ctx.chatContainer.children.indexOf(row));

		// The same call id streaming again reuses the existing row — no second marker.
		await controller.handleEvent(messageUpdate([call], 0));
		expect(markerRows).toHaveLength(1);

		// And neither does the tool_execution_start fallback for the existing row.
		await controller.handleEvent(toolStart(call));
		expect(markerRows).toHaveLength(1);
	});
});

/** `agent_start` — the only turn boundary the controller resets route state on. */
function agentStart(): Extract<AgentSessionEvent, { type: "agent_start" }> {
	return { type: "agent_start" } as Extract<AgentSessionEvent, { type: "agent_start" }>;
}

/** A model-chosen tool that the router did NOT force. */
function editCall(id: string): ToolCall {
	return toolCall(id, "edit", { file_path: "src/app.ts", old_string: "a", new_string: "b" });
}

describe("EventController Jev router status lines", () => {
	it("prints a low-confidence passthrough as a standalone line and never marks the model's own tool", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();

		controller.noteJevRouteEvent(LOW_CONFIDENCE);
		expect(markerTexts).toEqual(["◇ JEV · eval · 62% · passthrough"]);

		// The model picks `edit` itself: the router status must not attach to its
		// row, and the row must not be labelled as a forced route.
		const call = editCall("toolu_jev_advisory_edit");
		await controller.handleEvent(messageUpdate([call], 0));
		await controller.handleEvent(toolStart(call));

		expect(markerTexts).toEqual(["◇ JEV · eval · 62% · passthrough"]);
		expect(ctx.chatContainer.children.indexOf(markerRows[0])).toBeLessThan(
			ctx.chatContainer.children.indexOf(toolRow(ctx, call.id)),
		);
	});

	it("prints the prose route as a standalone `none` line", async () => {
		const { controller, markerTexts } = createFixture();

		controller.noteJevRouteEvent(PROSE_ROUTE);
		expect(markerTexts).toEqual(["◇ JEV · none · 82%"]);

		// The model still answers with a tool: no `◆` marker is created for it.
		const call = editCall("toolu_jev_none_edit");
		await controller.handleEvent(messageUpdate([call], 0));
		expect(markerTexts).toEqual(["◇ JEV · none · 82%"]);
	});

	it("prints operational failures with no confidence", async () => {
		const { controller, markerTexts } = createFixture();

		controller.noteJevRouteEvent({ kind: "failure", reason: "timeout" });
		expect(markerTexts).toEqual(["◇ JEV · timeout"]);

		controller.noteJevRouteEvent({ kind: "failure", reason: "judge-error" });
		expect(markerTexts).toEqual(["◇ JEV · timeout", "◇ JEV · judge-error"]);
	});

	it("shows a re-published status once per turn and a different status afterwards", async () => {
		const { controller, markerTexts } = createFixture();

		// A retry/fallback re-publishes the same status for the same turn.
		controller.noteJevRouteEvent(LOW_CONFIDENCE);
		controller.noteJevRouteEvent(LOW_CONFIDENCE);
		expect(markerTexts).toEqual(["◇ JEV · eval · 62% · passthrough"]);

		// A different status reached later is still reported.
		controller.noteJevRouteEvent(PROSE_ROUTE);
		expect(markerTexts).toEqual(["◇ JEV · eval · 62% · passthrough", "◇ JEV · none · 82%"]);
	});

	it("keeps an applied route's provenance after a status line shown in the same turn", async () => {
		const { controller, ctx, markerRows, markerTexts } = createFixture();

		// A transient failure line, then the re-judged applied route: the status
		// latch is separate state and must not swallow the pending provenance.
		controller.noteJevRouteEvent({ kind: "failure", reason: "timeout" });
		controller.noteJevRouteEvent(READ_DECISION);
		const call = readCall("toolu_jev_status_then_applied");
		await controller.handleEvent(messageUpdate([call], 0));
		const row = toolRow(ctx, call.id);

		expect(markerTexts).toEqual(["◇ JEV · timeout", "◆ JEV → read · 98%"]);
		expect(ctx.chatContainer.children.indexOf(markerRows[1])).toBeLessThan(ctx.chatContainer.children.indexOf(row));
	});

	it("leaks no route state into the next turn", async () => {
		const { controller, markerTexts } = createFixture();
		// A decision published last turn whose tool never ran...
		controller.noteJevRouteEvent(BASH_DECISION);
		controller.noteJevRouteEvent(PROSE_ROUTE);
		await controller.handleEvent(agentStart());

		// ...must not annotate this turn's first row, and a status already shown
		// last turn must be shown again for this turn.
		const call = toolCall("toolu_jev_turn_leak_bash", "bash", { command: "ls" });
		await controller.handleEvent(messageUpdate([call], 0));
		expect(markerTexts).toEqual(["◇ JEV · none · 82%"]);

		controller.noteJevRouteEvent(PROSE_ROUTE);
		expect(markerTexts).toEqual(["◇ JEV · none · 82%", "◇ JEV · none · 82%"]);
	});
});
