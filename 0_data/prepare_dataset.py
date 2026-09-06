"""Download Team-ACE/ToolACE, normalize it into the internal function-calling schema,
and write a stratified train/val/test split.

Usage:
    python prepare_dataset.py --output-dir data --seed 42
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DATASET_ID = "Team-ACE/ToolACE"

# ToolACE's `conversations[i].from` role vocabulary isn't pinned by the dataset card, so
# we normalize defensively instead of assuming exact strings. Unknown roles fall back to
# "user" with a one-time warning rather than crashing the whole prep run.
ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "function_call": "assistant",
    "function": "tool",
    "tool": "tool",
    "observation": "tool",
    "tool_response": "tool",
}
_warned_roles: set[str] = set()


def normalize_role(raw_role: str) -> str:
    role = ROLE_MAP.get(raw_role.strip().lower())
    if role is None:
        if raw_role not in _warned_roles:
            log.warning("Unrecognized conversation role %r — defaulting to 'user'", raw_role)
            _warned_roles.add(raw_role)
        role = "user"
    return role


def extract_tool_schema(system_text: str) -> list[dict[str, Any]] | None:
    """Pull the embedded tool/function-definition JSON array out of a `system` string.

    ToolACE's system prompts wrap the tool list in instruction text rather than storing
    it as a separate field, so we locate the outermost JSON array by bracket-matching
    instead of assuming a fixed prefix/suffix.
    """
    start = system_text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(system_text)):
        if system_text[i] == "[":
            depth += 1
        elif system_text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = system_text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, list):
                    return parsed
                return None
    return None


# A quote only opens a string literal when it appears where a value can start (right
# after "=", "(", "[", "{", ",", or ":", ignoring whitespace) — never bare. Without this,
# a possessive apostrophe inside an unquoted function name (e.g. "Get User's Likes")
# gets misread as opening a string literal, scrambling everything parsed after it. ":" is
# included for JSON-object scanning (`"key": "value"`), used by _find_all_brace_objects.
_VALUE_START_CHARS = "=([{,:"


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` only outside of quotes/brackets/parens (depth-aware)."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    prev_significant = ""
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"" and prev_significant in _VALUE_START_CHARS:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        if not ch.isspace():
            prev_significant = ch
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_call_paren_span(call_str: str) -> tuple[int, int] | None:
    """Find the (start, end) indices of the '(...)' pair that closes the string.

    ToolACE function names sometimes contain their own parentheses (e.g.
    "Get Rounds (Esports) by Event ID"), so the *first* "(" in the string is not
    reliably where the argument list starts — the real argument list is whichever
    top-level paren group's close paren is the call's very last character.
    """
    depth = 0
    quote: str | None = None
    prev_significant = ""
    open_idx = None
    last_span = None
    i = 0
    while i < len(call_str):
        ch = call_str[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"" and prev_significant in _VALUE_START_CHARS:
            quote = ch
        elif ch == "(":
            if depth == 0:
                open_idx = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and open_idx is not None:
                last_span = (open_idx, i)
        if not ch.isspace():
            prev_significant = ch
        i += 1
    if last_span and last_span[1] == len(call_str) - 1:
        return last_span
    return None


def _parse_single_call(call_str: str) -> dict[str, Any] | None:
    """Parse `Func Name(arg="val", arg2=1)` — ToolACE call syntax, not JSON.

    Function names may contain spaces (and even their own parentheses), so this can't be
    parsed as plain Python; instead we locate the argument-list paren span (see
    _find_call_paren_span), then split arguments on top-level commas and evaluate each
    value with ast.literal_eval.
    """
    call_str = call_str.strip()
    span = _find_call_paren_span(call_str)
    if span is None:
        return None
    paren_idx, end_idx = span
    name = call_str[:paren_idx].strip()
    # A real ToolACE function name (standard template) never contains these — they're
    # only seen in other call notations this parser doesn't support (e.g. the
    # "[Name]=>(...)" variant that otherwise slips through as a garbage "name").
    # Rejecting them here, rather than accepting a garbled parse, is what lets
    # _looks_like_foreign_call_syntax correctly flag and drop those rows instead.
    if not name or any(c in name for c in "[]{}<>"):
        return None
    args_str = call_str[paren_idx + 1 : end_idx]
    arguments: dict[str, Any] = {}
    for arg in _split_top_level(args_str, ","):
        if "=" not in arg:
            return None
        key, _, value_str = arg.partition("=")
        try:
            value = ast.literal_eval(value_str.strip())
        except (ValueError, SyntaxError):
            value = value_str.strip().strip("'\"")
        arguments[key.strip()] = value
    return {"name": name, "arguments": arguments}


def try_parse_calls(value: str) -> list[dict[str, Any]] | None:
    """If an assistant turn's value is a ToolACE-style call list, parse it.

    Returns a list of {name, arguments} dicts, or None if the turn is plain natural
    language (a clarification question or final answer) rather than a function call.
    """
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return None
    calls = []
    for part in _split_top_level(inner, ","):
        parsed = _parse_single_call(part)
        if parsed is None:
            return None  # not a clean call list — treat the whole turn as natural language
        calls.append(parsed)
    return calls or None


def classify_example(turns: list[dict[str, str]]) -> str:
    assistant_calls = [
        try_parse_calls(t["content"]) for t in turns if t["role"] == "assistant"
    ]
    call_counts = [len(c) for c in assistant_calls if c]
    if not call_counts:
        return "no_call"
    if len(call_counts) > 1:
        return "multi_turn"
    return "parallel" if call_counts[0] > 1 else "single"


def add_internal_examples() -> list[dict[str, Any]]:
    """Hook for merging this company's own internal-API examples into the training set.

    Stubbed until real internal-API data (with genuine low/medium/high risk tiers, plus
    adversarial/refusal cases) is available. Expected shape per example matches the
    normalized records this script produces: {system, tools, turns, call_type, risk_tier}.
    """
    return []


def is_standard_tool_schema(tools: list[Any]) -> bool:
    """True if `tools` is already in the standard {"name", "parameters"} shape.

    ToolACE isn't generated from a single template — a minority of rows describe tools
    some other way (see try_recover_alternate_tool_schema for the one variant we
    recover, and _BRITTLE_TEMPLATE_MARKERS for the ones we still drop).
    """
    return bool(tools) and all(isinstance(t, dict) and "name" in t for t in tools)


def _parse_json_or_literal(text: str) -> Any | None:
    """Try JSON first, then Python-literal syntax as a fallback.

    Some ToolACE rows serialize the same structures with single quotes (valid Python,
    invalid JSON) instead of double-quoted JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _find_all_brace_objects(text: str) -> list[str]:
    """Find every top-level {...} span in text (depth-aware, quote-aware).

    Used to recover ToolACE's alternate tool-definition template, where each tool is a
    standalone {"tool_name": ..., "arguments": [...]} object rather than an entry in one
    JSON list — see try_recover_alternate_tool_schema.
    """
    spans = []
    depth = 0
    quote: str | None = None
    prev_significant = ""
    start = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"" and prev_significant in _VALUE_START_CHARS:
            quote = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
        if not ch.isspace():
            prev_significant = ch
        i += 1
    return spans


# ToolACE has several other tool-list templates beyond the standard JSON list and the
# {"tool_name", "arguments"} variant handled below: a markdown bullet-list style
# ("- **tool_name**: ..."), a LaTeX `tabular` style, and an HTML `<table>` style — and a
# single system prompt can mix several of these for *different* tools in the same list.
# Recovering those safely would mean parsing free-form markdown/LaTeX/HTML *and*
# guaranteeing the complete tool list comes back (a partial list would make the
# hallucination check in run_internal_eval.py misfire on tools we failed to recover).
# That's too much brittle parsing surface for a slice of a public synthetic dataset.
# TODO: revisit if dataset coverage becomes worth the added complexity.
_BRITTLE_TEMPLATE_MARKERS = ("<table", "\\begin{tabular}", "- **tool_name**", "\ntool_name:")


def _remap_alt_tool_def(defn: Any) -> dict[str, Any] | None:
    """Remap one alternate-template tool definition to the standard shape.

    ToolACE's alternate template describes a tool as
    `{"tool_name": ..., "definition": ..., "arguments": [{"parameter_name": ...}, ...],
    "required": [...]}`. This remaps it to `{"name", "description", "parameters"}` —
    the shape the rest of this pipeline (and the BFCL eval harness) expects.
    """
    if not isinstance(defn, dict) or "tool_name" not in defn or "arguments" not in defn:
        return None
    if not isinstance(defn["arguments"], list):
        return None
    properties: dict[str, Any] = {}
    for arg in defn["arguments"]:
        if not isinstance(arg, dict) or "parameter_name" not in arg:
            return None
        pname = arg["parameter_name"]
        properties[pname] = {k: v for k, v in arg.items() if k != "parameter_name"}
    return {
        "name": defn["tool_name"],
        "description": defn.get("definition", ""),
        "parameters": {
            "type": "dict",
            "properties": properties,
            "required": defn.get("required") or [],
        },
    }


def try_recover_alternate_tool_schema(system_text: str) -> list[dict[str, Any]] | None:
    """Recover tools from ToolACE's `{"tool_name": ..., "arguments": [...]}` template —
    one or more standalone JSON/Python-literal objects, instead of one JSON list.

    Returns None (caller drops the row) if:
    - the system prompt also contains a brittle sub-template we don't parse (see
      _BRITTLE_TEMPLATE_MARKERS), or
    - any "tool_name"-bearing object in the text fails to parse or remap — a partially
      recovered tool list is worse than none, since run_internal_eval.py's hallucination
      check would then misfire on the tools we failed to recover.
    """
    if any(marker in system_text for marker in _BRITTLE_TEMPLATE_MARKERS):
        return None

    expected_count = system_text.count('"tool_name"') + system_text.count("'tool_name'")
    if expected_count == 0:
        return None

    tools = []
    for span in _find_all_brace_objects(system_text):
        parsed = _parse_json_or_literal(span)
        if not isinstance(parsed, dict) or "tool_name" not in parsed:
            continue
        remapped = _remap_alt_tool_def(parsed)
        if remapped is None:
            return None  # a real tool_name object we couldn't cleanly remap — don't guess
        tools.append(remapped)

    if len(tools) != expected_count:
        return None  # didn't recover every tool_name object seen — don't return a partial list

    return tools or None


def _looks_like_foreign_call_syntax(content: str) -> bool:
    """True if `content` looks like it's trying to be a function call (starts with a
    bracket/paren/brace) but doesn't parse under the standard `Name(arg="val")` syntax
    try_parse_calls understands.

    ToolACE's alternate templates pair with several other call notations we don't
    parse — e.g. `(Name|{'k'/v})`, `{Name:{"k"--v}}` — so a row whose tool schema we
    recovered (see try_recover_alternate_tool_schema) can still be using one of those.
    Silently calling that a "no_call" example would mislabel a real call, so
    ACEToolDatasetProcessor.normalize_example drops the row instead when this fires.
    """
    stripped = content.strip()
    return bool(stripped) and stripped[0] in "([{" and try_parse_calls(content) is None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class DatasetProcessor(ABC):
    """extract -> prepare -> save pipeline for turning a source dataset into the
    internal function-calling training format."""

    @abstractmethod
    def extract(self) -> Any:
        """Pull the raw dataset from its source."""

    @abstractmethod
    def prepare(self, raw: Any) -> list[dict[str, Any]]:
        """Normalize raw records into the internal schema.

        Every preprocessing fix — tool-schema extraction/recovery, role normalization,
        call-type classification, risk-tier tagging — happens here.
        """

    @abstractmethod
    def save(self, prepared: list[dict[str, Any]], output_dir: Path) -> None:
        """Split and persist the prepared examples."""


class ACEToolDatasetProcessor(DatasetProcessor):
    """DatasetProcessor for Team-ACE/ToolACE — see module docstring."""

    def __init__(self, seed: int = 42, train_frac: float = 0.90, val_frac: float = 0.05):
        self.seed = seed
        self.train_frac = train_frac
        self.val_frac = val_frac

    def extract(self) -> Any:
        log.info("Loading %s ...", DATASET_ID)
        raw = load_dataset(DATASET_ID, split="train")
        log.info("Loaded %d raw rows", len(raw))
        return raw

    def prepare(self, raw: Any) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        drop_reasons: Counter[str] = Counter()
        for row in raw:
            example, reason = self.normalize_example(row)
            if example is None:
                drop_reasons[reason] += 1
                continue
            prepared.append(example)

        if drop_reasons:
            total_dropped = sum(drop_reasons.values())
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(drop_reasons.items()))
            log.warning("Dropped %d/%d rows (%s)", total_dropped, len(raw), breakdown)

        prepared.extend(add_internal_examples())
        return prepared

    def save(self, prepared: list[dict[str, Any]], output_dir: Path) -> None:
        labels = [ex["call_type"] for ex in prepared]
        train, rest = train_test_split(
            prepared, train_size=self.train_frac, random_state=self.seed, stratify=labels
        )
        rest_labels = [ex["call_type"] for ex in rest]
        val_frac_of_rest = self.val_frac / (1 - self.train_frac)
        val, test = train_test_split(
            rest, train_size=val_frac_of_rest, random_state=self.seed, stratify=rest_labels
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "train.jsonl", train)
        write_jsonl(output_dir / "val.jsonl", val)
        write_jsonl(output_dir / "test.jsonl", test)

        def summarize(name: str, split: list[dict[str, Any]]) -> None:
            counts = Counter(ex["call_type"] for ex in split)
            breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            log.info("%-6s n=%-6d %s", name, len(split), breakdown)

        log.info("Split summary:")
        summarize("train", train)
        summarize("val", val)
        summarize("test", test)
        log.info("Wrote splits to %s", output_dir)

    @staticmethod
    def normalize_example(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Normalize one raw HF row into the internal schema.

        Returns (example, None) on success, or (None, drop_reason) if the row can't be
        safely recovered — callers should track drop_reason rather than discard it, so
        a regression in the parsing pipeline shows up as a reason-breakdown change
        instead of just a silently bigger drop count.
        """
        system_text = row["system"]
        tools = extract_tool_schema(system_text)
        recovered = False
        if tools is None or not is_standard_tool_schema(tools):
            tools = try_recover_alternate_tool_schema(system_text)
            if tools is None:
                return None, "unrecoverable_tool_schema"
            recovered = True

        turns = [
            {"role": normalize_role(t["from"]), "content": t["value"]}
            for t in row["conversations"]
        ]
        if not turns:
            return None, "empty_conversation"

        if recovered:
            # TODO: revisit — see try_recover_alternate_tool_schema's docstring. We can
            # recover the tool *schema* for these rows, but several ToolACE templates use
            # call syntaxes try_parse_calls doesn't understand (e.g. "(Name|{'k'/v})",
            # '{Name:{"k"--v}}'). Rather than silently mislabel a real call as "no_call",
            # drop the row if any assistant turn looks call-shaped but won't parse.
            if any(
                _looks_like_foreign_call_syntax(t["content"])
                for t in turns
                if t["role"] == "assistant"
            ):
                return None, "foreign_call_syntax"

        return {
            "system": system_text,
            "tools": tools,
            "turns": turns,
            "call_type": classify_example(turns),
            # Placeholder — see add_internal_examples() and the phase-0 README.
            "risk_tier": "unclassified",
        }, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.90)
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()

    processor = ACEToolDatasetProcessor(
        seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac
    )
    raw = processor.extract()
    prepared = processor.prepare(raw)
    processor.save(prepared, args.output_dir)


if __name__ == "__main__":
    main()
