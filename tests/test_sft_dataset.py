"""CPU-only tests; no model downloads or GPU required.

Run with: tox -e sft-tests
"""

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_training" / "1_sft"))
import train_sft


@pytest.fixture
def tokenizer():
    # An offline byte tokenizer with a non-thinking assistant prefix, deliberately
    # without Jinja generation tags. Completion masking must not require those tags.
    special = ["<|pad|>", "<|im_start|>", "<|im_end|>"]
    vocab = {s: i for i, s in enumerate(special + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=special[0],
        eos_token=special[2],
        additional_special_tokens=[special[1]],
    )
    tok.chat_template = (
        "{% for m in messages %}"
        "{{ '<|im_start|>' + m.role + '\n' }}"
        "{% if m.role == 'assistant' and loop.last %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
        "{{ m.content + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
        "{% if enable_thinking is defined and not enable_thinking %}"
        "{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}"
    )
    return tok


@pytest.fixture
def example():
    return {
        "system": "Use lookup and transfer. Preserve argument types.",
        "tools": [{"name": "lookup"}, {"name": "transfer"}],
        "call_type": "multi_turn",
        "risk_tier": "unclassified",
        "turns": [
            {"role": "user", "content": "Look up account ABC."},
            {"role": "assistant", "content": "[lookup(account='ABC')]"},
            {"role": "tool", "content": '{"balance": 25}'},
            {"role": "user", "content": "Transfer 5 to XYZ."},
            {"role": "assistant", "content": "[transfer(to='XYZ', amount=5)]"},
        ],
    }


def write_examples(tmp_path, examples):
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(json.dumps(ex) for ex in examples) + "\n")
    return path


def tiny_model(tokenizer):
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=2048,
        bos_token_id=None, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))


def test_load_examples_preserves_json_types_and_order(tmp_path, example):
    second = copy.deepcopy(example)
    second["call_type"] = "parallel"
    path = write_examples(tmp_path, [example, second])
    assert train_sft.load_examples(path) == [example, second]


def test_to_messages_preserves_roles_content_and_input(example):
    original = copy.deepcopy(example)
    assert train_sft.to_messages(example) == [
        {"role": "system", "content": example["system"]}, *example["turns"],
    ]
    assert example == original


def test_expands_one_row_per_assistant_and_keeps_prior_history(tmp_path, tokenizer, example):
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    assert data.column_names == ["prompt", "completion"]
    assert len(data) == 2
    assert example["system"] in data[0]["prompt"]
    assert "Look up account ABC." in data[0]["prompt"]
    assert "[lookup" not in data[0]["prompt"]
    assert "Transfer 5" not in data[0]["prompt"]
    assert "[lookup(account='ABC')]" in data[1]["prompt"]
    assert '{"balance": 25}' in data[1]["prompt"]
    assert "Transfer 5 to XYZ." in data[1]["prompt"]
    assert "[transfer" not in data[1]["prompt"]


def test_completion_preserves_native_arguments_and_end_of_turn(tmp_path, tokenizer, example):
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    assert data[0]["completion"] == "[lookup(account='ABC')]<|im_end|>"
    assert data[1]["completion"] == "[transfer(to='XYZ', amount=5)]<|im_end|>"


def test_parallel_calls_and_nested_values_stay_in_one_target(tmp_path, tokenizer, example):
    target = "[lookup(account='ABC'), lookup(account='XYZ', filters={'tags': ['日本', 'a,b'], 'active': True})]"
    example["call_type"] = "parallel"
    example["turns"] = [example["turns"][0], {"role": "assistant", "content": target}]
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    assert len(data) == 1
    assert data[0]["completion"] == target + tokenizer.eos_token


def test_eos_normalization_preserves_whitespace_inside_response(tmp_path, tokenizer, example):
    example["turns"] = [
        example["turns"][0], {"role": "assistant", "content": "Please clarify.  \n"},
    ]
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    assert data[0]["completion"] == "Please clarify.  \n<|im_end|>"


def test_prompt_matches_eval_and_reconstructs_target_chat(tmp_path, tokenizer, example):
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    instances = train_sft.build_eval_instances(example)
    for row, instance, target in zip(data, instances, [example["turns"][1], example["turns"][4]]):
        assert row["prompt"] == tokenizer.apply_chat_template(
            instance["context"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        assert row["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert row["prompt"] + row["completion"] == tokenizer.apply_chat_template(
            instance["context"] + [target], tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        ).rstrip()


@pytest.mark.parametrize("call_type", train_sft.CALL_TYPES)
def test_filter_applies_before_turn_expansion(tmp_path, tokenizer, example, call_type):
    rows = []
    for kind in train_sft.CALL_TYPES:
        row = copy.deepcopy(example)
        row["call_type"] = kind
        row["system"] = f"Schema for {kind}"
        rows.append(row)
    data = train_sft.build_dataset(write_examples(tmp_path, rows), tokenizer, call_type)
    assert len(data) == 2
    assert all(f"Schema for {call_type}" in row["prompt"] for row in data)


def test_no_call_answers_remain_supervised(tmp_path, tokenizer, example):
    example["call_type"] = "no_call"
    example["turns"] = [
        {"role": "user", "content": "Transfer some money."},
        {"role": "assistant", "content": "What amount and recipient?"},
    ]
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    assert data[0]["completion"] == "What amount and recipient?<|im_end|>"


@pytest.mark.parametrize("missing_filter", [False, True])
def test_rejects_dataset_without_assistant_targets(tmp_path, tokenizer, example, missing_filter):
    if not missing_filter:
        example["turns"] = [{"role": "user", "content": "Hello"}]
    with pytest.raises(ValueError, match="no assistant targets"):
        train_sft.build_dataset(
            write_examples(tmp_path, [example]), tokenizer, "no_call" if missing_filter else None,
        )


def test_rejects_template_with_incompatible_prompt_boundary(tmp_path, tokenizer, example):
    tokenizer.chat_template = "{{ 'prompt' if add_generation_prompt else 'different history' }}"
    with pytest.raises(ValueError, match="example 0, turn 2.*nonempty completion"):
        train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)


def test_trl_masks_prompt_history_and_padding_but_supervises_completion(tmp_path, tokenizer, example):
    data = train_sft.build_dataset(write_examples(tmp_path, [example]), tokenizer, None)
    trainer = SFTTrainer(
        model=tiny_model(tokenizer), processing_class=tokenizer,
        train_dataset=data, eval_dataset=data,
        args=SFTConfig(
            output_dir=str(tmp_path / "trainer"), use_cpu=True, bf16=False,
            completion_only_loss=True, max_length=2048, report_to=[],
        ),
    )
    # Check both paths after TRL preprocessing AND its actual training collator.
    for prepared in [trainer.train_dataset, trainer.eval_dataset]:
        batch = trainer.data_collator([prepared[0], prepared[1]])
        assert (batch["attention_mask"] == 0).any(), "Fixture must exercise padding"
        for i, raw in enumerate(data):
            labels = batch["labels"][i]
            ids = batch["input_ids"][i]
            prompt_length = len(tokenizer(raw["prompt"], add_special_tokens=False)["input_ids"])
            assert torch.all(labels[:prompt_length] == -100)
            assert torch.all(labels[batch["attention_mask"][i] == 0] == -100)
            supervised = labels != -100
            assert torch.equal(labels[supervised], ids[supervised])
            assert tokenizer.decode(labels[supervised].tolist()) == raw["completion"]
            assert (labels[supervised] == tokenizer.eos_token_id).sum() == 1


def test_main_evaluates_step_zero_before_first_update(tmp_path, tokenizer, example, monkeypatch):
    path = write_examples(tmp_path, [example, example])
    model = tiny_model(tokenizer)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    events = []

    class Observe(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
            events.append(("eval", state.global_step, dict(metrics)))
            if state.global_step == 0:
                # The real LoRA wrapper has not changed the backbone yet.
                for name, p in model.get_base_model().named_parameters():
                    original_name = name.replace(".base_layer.", ".")
                    if original_name in before:
                        assert torch.equal(p.detach(), before[original_name])

        def on_step_begin(self, args, state, control, **kwargs):
            events.append(("step", state.global_step, {}))

    def cpu_config(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, gradient_checkpointing=False, disable_tqdm=True)
        return SFTConfig(**kwargs)

    def observed_trainer(**kwargs):
        assert kwargs["args"].completion_only_loss is True
        assert kwargs["args"].eval_on_start is True
        kwargs["callbacks"].append(Observe())
        return SFTTrainer(**kwargs)

    monkeypatch.setattr(train_sft.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(train_sft.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(train_sft, "SFTConfig", cpu_config)
    monkeypatch.setattr(train_sft, "SFTTrainer", observed_trainer)
    monkeypatch.setattr(sys, "argv", [
        "train_sft.py", "--train-file", str(path), "--val-file", str(path),
        "--output-dir", str(tmp_path / "output"), "--max-steps", "1",
        "--eval-steps", "1", "--save-steps", "1", "--logging-steps", "1",
        "--per-device-batch-size", "1", "--eval-batch-size", "1",
        "--metrics-max-new-tokens", "2",
        "--max-seq-length", "2048",
    ])
    train_sft.main()
    assert [(kind, step) for kind, step, _ in events] == [("eval", 0), ("step", 0), ("eval", 1)]
    assert "eval_loss" in events[0][2]
    assert "eval_fc_call_correctness" in events[0][2]
    reports = tmp_path / "output" / "validation_predictions"
    baseline = json.loads((reports / "step-000000.json").read_text())
    candidate = json.loads((reports / "step-000001.json").read_text())
    assert baseline["metrics"]["n_conversations"] == 2
    assert baseline["metrics"]["n_instances"] == 4
    assert baseline["metrics"]["n_trajectories"] == 2
    assert [c["conversation_id"] for c in baseline["conversations"]] == [0, 1]
    assert "prediction" in baseline["conversations"][0]["turns"][0]
    assert "comparison_to_baseline" not in baseline
    assert candidate["comparison_to_baseline"]["baseline_step"] == 0
    assert candidate["comparison_to_baseline"]["n_bootstrap"] == 10000


def test_callback_full_validation_default_and_fixed_smoke_subset(tmp_path, example):
    path = write_examples(tmp_path, [example] * 60)
    full = train_sft.FunctionCallEvalCallback(path)
    assert len(full.examples) == 60
    small = train_sft.FunctionCallEvalCallback(path, eval_samples=3)
    repeated = train_sft.FunctionCallEvalCallback(path, eval_samples=3)
    assert len(small.examples) == 3
    assert small.examples == repeated.examples
    with pytest.raises(ValueError, match="nonnegative"):
        train_sft.FunctionCallEvalCallback(path, eval_samples=-1)
