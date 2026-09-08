# 0. Data

**Why:** the fine-tune is only as trustworthy as its data. ToolACE's public schema needs
to be mapped onto this company's *actual* internal API contracts, and the split needs to
carry a `risk_tier` field so phase 2's evaluation can be broken out by risk tier from day
one — a miss on a high-value transfer function has to be tracked separately from a miss
on a low-risk one, so risk tier can't be bolted on after the fact.

## Source

[`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) — confirmed via
the Hugging Face Hub API: a single `default/train` config, 11,300 rows, two columns:

- `system` (string) — the system prompt, which embeds the available tool/function
  definitions as inline JSON plus instructions.
- `conversations` (list of `{from, value}`) — the dialogue turns. `from` role names are
  normalized defensively (see `ROLE_MAP` in `prepare_dataset.py`) rather than assumed,
  since the dataset card doesn't pin an exact role vocabulary.

## Code structure

`prepare_dataset.py` implements a small `DatasetProcessor` interface —
`extract() -> prepare(raw) -> save(prepared, output_dir)` — with `ACEToolDatasetProcessor`
as the Team-ACE/ToolACE implementation:

- **`extract()`** — loads `Team-ACE/ToolACE` (`train` split) via `datasets.load_dataset`.
- **`prepare(raw)`** — every preprocessing fix lives here: tool-schema extraction (and
  recovery — see below), role normalization, `call_type` classification, `risk_tier`
  tagging, and drop-reason tracking. Delegates the per-row work to the
  `normalize_example` staticmethod, which returns `(example, None)` on success or
  `(None, drop_reason)` otherwise, so drops are logged with *why*, not just a count.
- **`save(prepared, output_dir)`** — the train/val/test split (90/5/5, stratified on
  `call_type`) and JSONL writing.

`main()` is now a thin CLI wrapper: `extract() -> prepare() -> save()`.

The lower-level parsing functions (`extract_tool_schema`, `try_parse_calls`,
`classify_example`, etc.) stay as module-level functions rather than methods — they're
generic parsing utilities reused directly by `1_training/1_sft/train_sft.py` and
`2_evaluations/run_internal_eval.py`, independent of the processor class.

`call_type` is one of: `single`, `parallel` (>1 call in one assistant turn), `multi_turn`
(more than one assistant call across the conversation), or `no_call` (a
clarification/refusal turn). `risk_tier` is tagged `"unclassified"` for every row —
ToolACE has no concept of fintech risk tiers. **This is a placeholder**:
`add_internal_examples()` is a stubbed hook, called from `prepare()`, for merging in this
company's own internal-API examples (which should carry a real `low`/`medium`/`high`
risk tier, plus adversarial/refusal cases) once available. Do not treat `unclassified` as
a real tier in downstream reporting.

## Known limitation: ToolACE isn't one template

ToolACE rows aren't all generated from the same system-prompt template. Two drop reasons
show up in `prepare()`'s log line, and mean different things:

- **`unrecoverable_tool_schema`** — the row uses a tool-list template this pipeline
  doesn't parse. One variant *is* recovered: `{"tool_name": ..., "arguments": [...]}`
  objects (`try_recover_alternate_tool_schema` remaps them to the standard
  `{"name", "description", "parameters"}` shape used everywhere else, including by the
  BFCL eval harness). What's still dropped, deliberately, is anything using a markdown
  bullet-list, LaTeX `tabular`, or HTML `<table>` sub-template
  (`_BRITTLE_TEMPLATE_MARKERS`) — because a single system prompt can mix several of
  these for *different* tools in the same tool list, recovering only some of them would
  leave an incomplete `tools` list, and `run_internal_eval.py`'s hallucination check
  would then misfire on the tools that got missed. **TODO: revisit** if this dataset's
  coverage is ever worth building parsers for those free-form formats too.
- **`foreign_call_syntax`** — the tool schema recovered fine, but an assistant turn looks
  like it's attempting a call (starts with `[`/`(`/`{`) and doesn't parse under the
  standard `Name(arg="val")` syntax. ToolACE's alternate templates pair with several
  other call notations (`Name-(param--'val')`, `(Name|{'k'/v})`,
  `{Name:{"k"--v}}`, `[Name]=>(k=v)`, ...) that `try_parse_calls` doesn't understand —
  rather than silently mislabel a real call as `no_call`, the row is dropped.
  **TODO: revisit** if any of these syntaxes turn out to be common enough to warrant a
  dedicated parser.

## Verification performed

Ran end-to-end against the real `Team-ACE/ToolACE` dataset: 11,300 raw rows → 727
dropped (`unrecoverable_tool_schema=691`, `foreign_call_syntax=36`) → 10,573 normalized →
split 9,515 / 529 / 529 (train/val/test), stratified by `call_type`.

Cross-checked the call-syntax parser (`try_parse_calls`, reused by
`2_evaluations/run_internal_eval.py`) for self-consistency: for every instance across all
three splits, fed the *ground-truth* call back through the eval's own scoring function as
if it were the model's prediction. A correct parser must score 100% against itself.
**Result: 13,092 / 13,092 instances (100%) scored correctly.** Three real parser bugs
were caught and fixed this way before landing on that number:
- Function names that themselves contain parentheses (e.g. `Get Rounds (Esports) by
  Event ID`) were splitting name-from-arguments at the wrong `(`.
- Possessive apostrophes in function names (e.g. `Get User's Likes`) were being misread
  as opening a string literal, scrambling everything parsed after them.
- A `[Name]=>(k=v)`-style foreign call (recovered via the alternate-template path) was
  slipping past the foreign-syntax guard because `_parse_single_call` didn't reject
  names containing notation characters (`[`, `]`, `{`, `}`, `<`, `>`) that never appear
  in a real ToolACE function name — it now does.

**That 100% figure is self-consistency, not BFCL compatibility — the two are not the
same claim.** It proves `try_parse_calls` agrees with itself, which is true by
construction; it originally said nothing about whether BFCL's own scorer would accept
the same string, and — before the fix below — it didn't: BFCL's `ast_parse` runs a call
turn through real Python syntax parsing (`ast.parse(..., mode="eval")`), and ToolACE's
own function/parameter names routinely aren't valid Python identifiers — spaces
(`Short Code Check`), apostrophes (`Get User's Likes`), embedded parentheses (`Get
Rounds (Esports) by Event ID`), hyphens (`vin-identifier`), even bare reserved words used
as parameter names (`from='JPY'` is a `SyntaxError` — `from` is a keyword). Measured
against BFCL's own real `ast_parse` (cross-checked directly against the installed
`bfcl-eval==2025.8.6.2` package, not guessed): **only 52.0% of call turns parsed**
(5,176/9,954 across all three splits) before this was fixed.

**Fix: `rename_tools_for_bfcl`.** Every tool name and every one of its parameter names
gets deterministically sanitized into a valid Python identifier (`sanitize_identifier`
— strips invalid characters, guards against leading digits and reserved keywords), and
the rename is applied consistently in both places a name has to agree with itself: the
tool-list JSON spliced back into the system prompt (`_find_tool_list_span` locates that
exact span so only the JSON, not the surrounding instruction text, is touched — see
below for why that distinction mattered), and every call turn that invokes the tool,
re-rendered (`render_calls`) with the new name and `repr()`-formatted argument values
(which, as a side effect of just being normal Python literals, also fixes a separate,
smaller gap: JSON-style lowercase `true`/`false` in argument values, valid Python syntax
but not a value BFCL's own resolver handles, ~1pp of turns pre-fix). Only applied to the
standard tool-list-JSON-array template — `try_recover_alternate_tool_schema`'s rows (a
different, scattered text layout, and a small minority of the corpus) are left
unrenamed, matching that function's own established "don't guess on a recovery path"
convention.

Verified against every record in the real prepared data already on disk (raw ToolACE
itself isn't reachable in an offline check — this simulates `normalize_example`'s new
step against each record's own already-parsed system/tools/turns, which is what the
real pipeline would have fed it): **9,954/9,954 (100%) of call turns now parse under
BFCL's real `ast_parse`, zero rows dropped, zero same-example name collisions, zero
tool-list/call-turn name disagreements, and zero argument values changed** (checked by
value, accounting for the key rename — a raw dict-equality check would show every
renamed argument as "different" since its key changed, which isn't the same as its
*value* changing). `bfcl_ast_parseable` — the same real-`ast_parse` reproduction used for
this measurement — now runs as a standing check in `prepare()`'s own summary logging, so
a future regression shows up as a logged rate dropping, not silence.

Consequence for reading eval numbers: `2_evaluations/run_internal_eval.py`'s
`parse_prediction` reads this same native syntax (not JSON — see
`1_training/1_sft/train_sft.py`'s `to_messages` for why training was changed to match),
so internal-eval and BFCL scores are now measuring the same output format as each other,
and both are now free of the identifier-validity gap this section used to document as an
open problem.

## Run

```bash
python prepare_dataset.py --output-dir data --seed 42
```

## Walkthrough notebook

[`dataset_walkthrough.ipynb`](dataset_walkthrough.ipynb) runs **end-to-end from the live
Hugging Face source** — it doesn't depend on `data/*.jsonl` already existing. It pulls
`Team-ACE/ToolACE` directly from the Hub via `ACEToolDatasetProcessor.extract()`, shows
one raw row exactly as the Hub returns it (before any processing), then runs the same
`.prepare()` this script uses (imported, not reimplemented) live over the full
11,300-row dataset, and shows the same row before/after normalization side by side.

From there it shows real examples of: the exact system/user messages sent to the model
for a single call, a parallel call, a multi-turn tool-use conversation, and a
no-call/refusal case; the target the model is trained to produce in both ToolACE's native
notation and the JSON schema actually used for training; and the fully rendered chat-
template string plus its real token count, using the `Qwen/Qwen3-8B` tokenizer (with
thinking mode explicitly disabled, matching the non-thinking behavior
`2_evaluations/` expects at inference — tokenizer files only, no model weights
downloaded). It does not perform the
train/val/test split — that stays this script's job.
