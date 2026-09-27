<system-notice>
User message contains **jevify** → bulk classification through the `eval` kernel's `judge_batch()`. You decide once, up front; the host-owned batch processes the bulk — it outlives the cell, so you drain it across as many cells as you need; you read only what it flags. This overrides the tendency to split the data up and scan it yourself.

<critical>
- NEVER read bulk items before the rubric is frozen. Rubric first, data second.
- NEVER hand-scan the bulk or delegate scanning to subagents. `judge_batch` classifies; you read only flagged items.
- Rubric changes mid-run invalidate every prior verdict: re-judge everything.
</critical>

<when>
Any list of ≥ ~20 homogeneous items with a bucket/yes-no/score question: commit or PR file diffs ("what here is unrelated?"), log lines, test names, search hits, issues, review findings, catalog rows. Under ~20 items or a question that needs cross-item reasoning: read directly.
</when>

<workflow>
1. **Decide** — one `eval` cell, before any data is loaded, define as constants:
   - **Unit**: what one `state` is (file diff, hunk, log line, row). Prefer the smallest unit that still carries enough context to answer.
   - **Questions**: independent questions with fixed ids. One `choice` for the primary bucket; optional `bool`/`score` for secondary facts. Every criterion label is one sentence of observable evidence; labels exhaustive + mutually exclusive; include an explicit catch-all ("unrelated"/"other") and a "mixed" label when a unit can straddle.
   - **Pre-filter**: deterministic exclusions (path prefix, extension, size, pure deletions) that skip judging. Log how many units it removed.
   - **Escalation rule**: which verdicts and which uncertainty (e.g. top probability `< 0.7`, `bool` in `0.3..0.7`, error) you will read yourself.
   - **Cap**: max state size; truncate with a visible marker and count truncations.
2. **Partition** — load every unit in the kernel (`git show`, `glob`, `read`, parsers). Apply the pre-filter. Store units in a dict keyed by a stable id. Exactly one unit survives → `judge()`; two or more homogeneous states → one `judge_batch()`.
3. **Judge** — `judge_batch(states, questions, …)` creates the host-owned run and returns a batch; every question is asked in the same call, and the run survives the cell. Pull settled items with `await b.drain(timeout=…)` — or `async for key, item in b.drain_iter(timeout)` — re-checking `.status().running` after each pull, since a pull ends on its timeout as well as on completion; keep `{id: choice, top_p, extra facts}` and give `item.error` its own row. A later cell (or a reset kernel) rejoins the same still-open run with `judge_batch.attach(b.id)`; `b.close()` releases the finished run.
4. **Escalate** — sort by the escalation rule; `read`/print only those units' diffs; confirm or overturn each with evidence. Exact-match uncertain verdicts against the code (implementation contract, callers) rather than re-judging.
5. **Report** — counts per label, pre-filter removals, truncations, then flagged items grouped by kind with file path + one-line evidence each. Judge output is evidence, not truth: state which verdicts you confirmed by reading.
</workflow>

<judge>
**Fallback for exactly one unit:** `await judge(state, questions)` → `{id: answer}` (Python and JavaScript both await it).

**Everything else:** `judge_batch(states, questions, …)` on the host, one batch per bulk.
- Python: `judge_batch(states, questions, *, concurrency=None, retries=None, min_ok=None, intent=None)` — synchronous call returning the batch (no await).
- JavaScript: `await judgeBatch(states, questions, { intent, concurrency, retries, minOk })`.
- `states` is `{key: state}` or a list (keys are indices). Defaults: `concurrency` 32 (clamped), `retries` 1, `minOk` 1; `intent` is a nonempty run/progress label.
- `state`: `str` | JSON object | JSON array. Every question sees the same state.
- `{type: "choice", instructions, criteria: {label: rubric, …}}` → `{choice, probabilities, confidence}`.
- `{type: "bool", instructions, criteria?: {true, false}}` → `{bool: P(yes)}`.
- `{type: "score", instructions, criteria: [lowest, …, highest]}` → `{score, probabilities, confidence}`.
- Batch: `.id`, `.total`, `.intent`; `.status()` (`running`/`done`/`total`), `.cancel()`, `.close()`, `.results()` → `{key: answers}`, `.failed()` → `{key: error}`; pulls are Python `await b.drain(timeout=…)` → `[(key, item)]` and `async for key, item in b.drain_iter(timeout)`, JavaScript `await b.drain({ timeout })` → `[[key, item]]` and `for await (const [key, item] of b.drainIter({ timeout }))` (`[]` when a pull times out empty); both forms stop at the timeout or on completion, so re-check `.status().running` before `close()`.
- `JudgmentItem`: `.key`, `.answers`, `.error`, `.model`, `.ok`.
- Item failures never raise — they land in `item.error` and stay in the tabulation. Only a run that dies wholesale rejects `drain()`.
- `judgeBatch.attach(id)` / `judge_batch.attach(id)` re-creates the batch ref after a kernel reset or from another cell; the run is host-owned and outlives the cell, and `close()` releases it.
Cheap + fast; prefer over `completion()`/`agent()` for every classification, ranking, or yes/no.
</judge>

<example>
**Python:**

```python
SHA = "abc123"
SUBJECT = "refactor: new tui framework"
QUESTIONS = {
    "verdict": {"type": "choice",
        "instructions": f"One file diff from commit '{SUBJECT}'. Does this change belong to that refactor?",
        "criteria": {
            "belongs": "Every hunk is required by or mechanically follows from the stated refactor.",
            "mixed": "Mostly the refactor, plus at least one hunk changing unrelated behavior.",
            "unrelated": "No hunk relates to the stated refactor.",
        }},
    "logic": {"type": "bool", "instructions": "Does any hunk change runtime behavior outside the refactor's subsystem (not renames/imports/types)?"},
}
CAP = 24_000
def prefilter(path): return path.startswith("packages/tui/") or path.endswith(".tsx")
```

```python
import subprocess
def git(*args): return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
all_files = git("show", "--name-only", "--format=", SHA).split()
files = [f for f in all_files if not prefilter(f)]
skipped = len(all_files) - len(files)
states, truncated = {}, []
for f in files:
    d = git("show", "--format=", SHA, "--", f)
    if len(d) > CAP:
        truncated.append(f); d = d[:CAP] + "\n…[truncated]"
    states[f] = {"file": f, "subject": SUBJECT, "diff": d}
b = judge_batch(states, QUESTIONS, concurrency=16, retries=1, min_ok=1, intent=f"Classifying {len(states)} diffs")
rows = {}
while True:
    for key, item in await b.drain(timeout=30):        # a pull ends on timeout too
        rows[key] = item.answers if item.ok else {"error": item.error}
    if not b.status()["running"]:                      # not running => every item has settled
        break
# before close(): rows == (b.results() | b.failed())
b.close()                                              # releases the finished host-side run; attach(id) cannot revive it
flag = [k for k, r in rows.items()
        if "error" in r or r["verdict"]["choice"] != "belongs"
        or r["verdict"]["probabilities"]["belongs"] < 0.7 or r["logic"]["bool"] >= 0.5]
print(f"pre-filter removed {skipped}, truncated {len(truncated)}, flagged {len(flag)} of {len(states)}")
```

```python
# if a cell ends before the run finishes, keep b.id and re-attach from a later cell —
# only while that run is still open; attach() never revives a closed run
rows = {}
b = judge_batch.attach("batch_abc123")
while True:
    for key, item in await b.drain(timeout=30):
        rows[key] = item.answers if item.ok else {"error": item.error}
    if not b.status()["running"]:
        break
b.close()
```

**JavaScript** (frozen constants `SHA`, `SUBJECT`, `QUESTIONS`, `CAP`, and `prefilter` from the block above):

```js
const { execFileSync } = await import("node:child_process");
const git = (...args) => execFileSync("git", args, { encoding: "utf8" });
const allFiles = git("show", "--name-only", "--format=", SHA).split("\n").filter(Boolean);
const files = allFiles.filter(f => !prefilter(f));
const skipped = allFiles.length - files.length;
const states = {}, truncated = [];
for (const f of files) {
    let d = git("show", "--format=", SHA, "--", f);
    if (d.length > CAP) { truncated.push(f); d = d.slice(0, CAP) + "\n…[truncated]"; }
    states[f] = { file: f, subject: SUBJECT, diff: d };
}
const b = await judgeBatch(states, QUESTIONS, { intent: `Classifying ${files.length} diffs`, concurrency: 16, retries: 1, minOk: 1 });
const rows = {};
while (true) {
    for (const [key, item] of await b.drain({ timeout: 30 })) rows[key] = item.ok ? item.answers : { error: item.error };
    if (!(await b.status()).running) break; // not running => every item has settled
}
await b.close(); // releases the finished host-side run
const flag = files.filter(f => {
    const r = rows[f];
    return "error" in r || r.verdict.choice !== "belongs" || r.verdict.probabilities.belongs < 0.7 || r.logic.bool >= 0.5;
});
console.log(`pre-filter removed ${skipped}, truncated ${truncated.length}, flagged ${flag.length} of ${files.length}`);
```

```js
// if a cell ends before the run finishes, keep b.id and re-attach from a later cell —
// only while that run is still open; attach() never revives a closed run
const b2 = await judgeBatch.attach("batch_abc123");
while (true) {
    for (const [key, item] of await b2.drain({ timeout: 30 })) rows[key] = item.ok ? item.answers : { error: item.error };
    if (!(await b2.status()).running) break;
}
await b2.close();
```

Then print only `diffs[f]` for `f in flag`, confirm each against the code, and report.
</example>

<anti-patterns>
- Reading the first N items "to get a feel" before writing the rubric.
- One `judge()` call per item instead of one `judge_batch()` over the whole bulk.
- Fanning items to `task` subagents to "review" them: they read; the batch classifies.
- Treating a judge verdict as final without reading the flagged unit.
- Dropping errored or truncated items silently instead of counting and escalating them.
- Vague labels ("bad", "suspicious") instead of one-sentence observable evidence.
</anti-patterns>

<critical>
Rubric frozen before data. `judge_batch` classifies the bulk. You read only what it flags. Report counts, then evidence.
</critical>
</system-notice>
