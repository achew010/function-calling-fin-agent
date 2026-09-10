import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_training" / "1_sft"))
from metrics import compare_validation_reports


def report(groups):
    """Groups of binary call outcomes, with deliberately unequal conversation sizes."""
    return {
        "schema_version": 1, "dataset_sha256": "same-data", "generation": {"do_sample": False},
        "scorer_sha256": "same-scorer",
        "step": 0,
        "conversations": [
            {"conversation_id": i, "call_type": "multi_turn", "turns": [
                {"turn_index": j, "context": [{"role": "user", "content": f"{i}/{j}"}],
                 "expected_calls": [{"name": "lookup", "arguments": {}}],
                 "score": {"is_call_case": True, "param_correct": bool(correct)}}
                for j, correct in enumerate(group)
            ]} for i, group in enumerate(groups)
        ],
    }


def test_identical_predictions_have_zero_delta_and_interval():
    a = report([[1], [0, 1, 0]])
    comparison = compare_validation_reports(a, copy.deepcopy(a), n_bootstrap=100)
    for metric in comparison["metrics"].values():
        assert metric["delta"] == metric["ci_low"] == metric["ci_high"] == 0


def test_pairs_by_conversation_id_not_report_order():
    a = report([[1], [0, 1, 0]])
    b = copy.deepcopy(a)
    b["conversations"].reverse()
    assert compare_validation_reports(a, b, n_bootstrap=50)["metrics"]["full_call_accuracy"]["delta"] == 0


def test_bootstrap_retains_entire_clusters_and_turn_weighting():
    a = report([[0], [0, 0, 0]])
    b = report([[1], [0, 0, 0]])
    result = compare_validation_reports(a, b, n_bootstrap=1000, seed=42)
    call = result["metrics"]["full_call_accuracy"]
    assert call["delta"] == 0.25  # turn-weighted, not mean of conversation accuracies (0.5)
    # Drawing both clusters from conversation 0 gives 1.0; both from 1 gives 0.0.
    # Both occur with probability 1/4. A turn-level bootstrap cannot reproduce this CI.
    assert call["ci_low"] == 0
    assert call["ci_high"] == 1
    assert result["metrics"]["trajectory_accuracy"]["delta"] == 0.5
    assert result == compare_validation_reports(a, b, n_bootstrap=1000, seed=42)


def test_refusal_only_conversations_have_no_call_metric():
    a = report([[0], [0]])
    for conv in a["conversations"]:
        conv["call_type"] = "no_call"
        conv["turns"][0]["expected_calls"] = None
        conv["turns"][0]["score"] = {"is_call_case": False, "refusal_correct": False}
    b = copy.deepcopy(a)
    for conv in b["conversations"]:
        conv["turns"][0]["score"]["refusal_correct"] = True
    result = compare_validation_reports(a, b, n_bootstrap=100)
    assert set(result["metrics"]) == {"refusal_accuracy"}
    assert result["metrics"]["refusal_accuracy"]["delta"] == 1


@pytest.mark.parametrize("change", ["dataset", "scorer", "generation", "missing", "duplicate", "turn", "label"])
def test_rejects_unpaired_or_incompatible_reports(change):
    a = report([[1], [0, 1]])
    b = copy.deepcopy(a)
    if change == "dataset":
        b["dataset_sha256"] = "different"
    elif change == "scorer":
        b["scorer_sha256"] = "different"
    elif change == "generation":
        b["generation"]["do_sample"] = True
    elif change == "missing":
        b["conversations"].pop()
    elif change == "duplicate":
        b["conversations"].append(copy.deepcopy(b["conversations"][0]))
    elif change == "turn":
        b["conversations"][0]["turns"][0]["turn_index"] = 99
    else:
        b["conversations"][0]["turns"][0]["score"]["is_call_case"] = False
    with pytest.raises(ValueError):
        compare_validation_reports(a, b, n_bootstrap=10)


def test_rejects_nonpositive_bootstrap_count():
    with pytest.raises(ValueError, match="positive"):
        compare_validation_reports(report([[1]]), report([[1]]), n_bootstrap=0)
