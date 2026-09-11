import json
import copy
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
from trl import GRPOConfig

from test_sft_dataset import example, tokenizer, tiny_model, write_examples

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_training" / "2_grpo"))
import train_grpo


def reward(text, mode="dense", expected=None):
    return train_grpo.compute_reward(text, ["transfer"], True,
        expected or [{"name": "transfer", "arguments": {"to": "XYZ", "amount": 5}}], reward_mode=mode)


def test_more_correct_arguments_receive_more_credit():
    wrong = reward("[transfer(to='BAD', amount=9)]")
    partial = reward("[transfer(to='XYZ', amount=9)]")
    exact = reward("[transfer(to='XYZ', amount=5)]")
    assert 0 < wrong < partial < 0.8 < exact == 1
    assert reward("[transfer(to='BAD', amount=9)]", "legacy") == 0.3
    assert reward("[transfer(to='XYZ', amount=9)]", "legacy") == 0.3


@pytest.mark.parametrize("actual,expected", [(True, 1), (1, 1.0), ([True], [1]), ({"a": 1}, {"a": 1.0})])
def test_type_equality_is_recursive(actual, expected):
    assert not train_grpo.typed_equal(actual, expected)
    assert train_grpo.typed_equal(expected, expected)


def test_argument_credit_penalizes_missing_extra_and_wrong_types():
    expected = {"a": 1, "b": 2}
    assert train_grpo.argument_credit(expected, expected) == 1
    assert train_grpo.argument_credit({"a": 1}, expected) == 0.5
    assert train_grpo.argument_credit({"a": 1, "b": 2, "extra": 3}, expected) == pytest.approx(2 / 3)
    assert train_grpo.argument_credit({"a": "1", "b": 2}, expected) == 0.5
    assert train_grpo.argument_credit({}, {}) == 1


def test_repeated_calls_are_matched_one_to_one_and_order_independent():
    expected = [{"name": "transfer", "arguments": {"amount": x}} for x in [1, 2]]
    assert reward("[transfer(amount=2), transfer(amount=1)]", expected=expected) == 1
    duplicated = reward("[transfer(amount=1), transfer(amount=1)]", expected=expected)
    assert duplicated == pytest.approx(0.1 + 0.7 * (1 + 0.25) / 2)
    assert reward("[transfer(amount=1)]", expected=expected) == -0.5
    assert reward("[transfer(amount=1), transfer(amount=2), transfer(amount=2)]", expected=expected) == -0.5


def test_blank_refusal_is_not_rewarded_and_penalties_remain():
    assert train_grpo.compute_reward("  ", ["transfer"], False, []) == -1
    assert train_grpo.compute_reward("Please provide the recipient.", ["transfer"], False, []) == 1
    assert train_grpo.compute_reward("[transfer(amount=5)]", ["transfer"], False, []) == -1
    assert reward("Please clarify.") == -1
    assert reward("[invented(amount=5)]") == -2


def test_reward_contract_preserves_alignment_and_rejects_length_mismatch():
    targets = json.dumps([{"name": "transfer", "arguments": {"to": "XYZ", "amount": 5}}])
    kwargs = dict(completions=[[{"content": "[transfer(to='XYZ', amount=5)]"}], [{"content": ""}]],
                  tool_names=[["transfer"], ["transfer"]], is_call_case=[True, False],
                  expected_calls=[targets, "[]"])
    assert train_grpo.reward_func(**kwargs) == [1, -1]
    kwargs["tool_names"] = []
    with pytest.raises(ValueError, match="matching lengths"):
        train_grpo.reward_func(**kwargs)


def test_grpo_dataset_contains_history_and_separate_reward_targets(tmp_path, example):
    data = train_grpo.build_grpo_dataset(write_examples(tmp_path, [example]))
    assert len(data) == 2
    assert data[0]["prompt"][-1]["content"] == "Look up account ABC."
    assert not any(t["content"].startswith("[transfer") for t in data[1]["prompt"])
    assert any(t["role"] == "tool" for t in data[1]["prompt"])
    assert json.loads(data[1]["expected_calls"]) == [{"name": "transfer", "arguments": {"to": "XYZ", "amount": 5}}]


def pilot_examples(example):
    rows = []
    for i in range(8):
        row = copy.deepcopy(example)
        row["call_type"] = "no_call" if i < 4 else "single"
        row["turns"] = [
            {"role": "user", "content": f"Request {i}: " + ("Transfer money." if i < 4 else "Transfer 5 to XYZ.")},
            {"role": "assistant", "content": "Please provide the recipient and amount." if i < 4 else "[transfer(to='XYZ', amount=5)]"},
        ]
        rows.append(row)
    return rows


def test_pilot_selection_balanced_deterministic_and_excludes_validation(tmp_path, example):
    rows = pilot_examples(example)
    train = write_examples(tmp_path, rows)
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps(rows[0]) + "\n")
    data, manifest = train_grpo.build_pilot_dataset(train, val, size=4)
    assert sum(data["is_call_case"]) == 2
    ids = [r["source_conversation_id"] for r in manifest["examples"]]
    assert len(set(ids)) == 4 and 0 not in ids
    assert manifest == train_grpo.build_pilot_dataset(train, val, size=4)[1]
    assert set(data.column_names) == {"prompt", "tool_names", "is_call_case", "expected_calls"}
    with pytest.raises(ValueError, match="Not enough"):
        train_grpo.build_pilot_dataset(train, val, size=8)
    with pytest.raises(ValueError, match="even"):
        train_grpo.build_pilot_dataset(train, val, size=3)


def test_grounding_excludes_schema_and_assistant_inventions():
    context = [{"role": "system", "content": "XYZ"}, {"role": "assistant", "content": "XYZ"},
               {"role": "user", "content": "Send 5 to ABC."}]
    assert not train_grpo.grounded_arguments([{"arguments": {"to": "XYZ"}}], context)
    assert train_grpo.grounded_arguments([{"arguments": {"to": "ABC", "amount": 5}}], context)
    assert not train_grpo.grounded_arguments([{"arguments": {"amount": 50}}], context)


def test_probe_and_preset(tmp_path, example):
    path = write_examples(tmp_path, pilot_examples(example))
    ids = train_grpo.select_probe_ids(path, size=4)
    assert len(ids) == 4 and sum(i < 4 for i in ids) == 2
    assert ids == train_grpo.select_probe_ids(path, size=4)
    args = SimpleNamespace(pilot=True, max_steps=7)
    train_grpo.apply_pilot_defaults(args, ["--max-steps=7"])
    assert args.max_steps == 7 and args.num_generations == 4
    train_grpo.apply_pilot_defaults(args, [])
    assert args.max_steps == 100 and args.eval_steps == args.save_steps == 25


def test_error_gaps_expose_repeated_counts_and_truncation():
    from metrics import score_detailed, summarize_error_gaps
    instance = {"context": [{"role": "user", "content": "First transfer 1, then transfer 2."}],
                "tool_names": {"transfer"},
                "expected_calls": [{"name": "transfer", "arguments": {"amount": n}} for n in [1, 2]]}
    prediction = "[transfer(amount=1)]"
    score = score_detailed(instance, prediction, train_grpo.parse_prediction, train_grpo.normalize_calls)
    conversations = [{"conversation_id": 7, "call_type": "parallel", "turns": [
        {**instance, "prediction": prediction, "score": score,
         "turn_index": 0, "reached_token_limit": True}]}]
    gaps = summarize_error_gaps(conversations, train_grpo.parse_prediction)
    for key in ("call", "parallel", "repeated_tool_calls", "prerequisite_proxy"):
        assert gaps[key]["n"] == gaps[key]["wrong_call_count"] == gaps[key]["truncated"] == 1
        assert gaps[key]["accuracy"] == 0
        assert gaps[key]["failures"][0]["conversation_id"] == 7


@pytest.mark.parametrize("pilot", [False, True])
def test_grpo_main_cpu_smoke(tmp_path, tokenizer, example, monkeypatch, pilot):
    base = tmp_path / "base"
    tiny_model(tokenizer).save_pretrained(base)
    tokenizer.save_pretrained(base)
    path = write_examples(tmp_path, pilot_examples(example) if pilot else [example])
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps(example) + "\n")

    def cpu_config(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False, disable_tqdm=True)
        return GRPOConfig(**kwargs)

    monkeypatch.setattr(train_grpo, "GRPOConfig", cpu_config)
    monkeypatch.setattr(sys, "argv", [
        "train_grpo.py", "--base-model", str(base), "--train-file", str(path), "--val-file", str(val),
        "--output-dir", str(tmp_path / "output"), "--max-steps", "1", "--eval-steps", "1",
        "--save-steps", "1", "--num-generations", "2", "--num-generations-eval", "2",
        "--per-device-batch-size", "2", "--grad-accum", "1", "--eval-batch-size", "2",
        "--max-completion-length", "2", "--metrics-max-new-tokens", "2",
    ] + (["--pilot", "--pilot-train-samples", "4", "--pilot-eval-samples", "1"] if pilot else []))
    train_grpo.main()
    assert (tmp_path / "output" / "config.json").exists()
    assert (tmp_path / "output" / "validation_predictions" / "step-000001.json").exists()
    if pilot:
        assert (tmp_path / "output" / "pilot_selection.json").exists()
    report = json.loads((tmp_path / "output" / "validation_predictions" / "step-000001.json").read_text())
    assert report["error_gaps"]["call"]["n"] == 2
