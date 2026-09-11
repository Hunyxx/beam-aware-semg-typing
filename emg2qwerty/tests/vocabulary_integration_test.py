"""Offline round trips through the actual tiny Qwen/PEFT implementation."""

from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import models, pre_tokenizers, Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from emg2qwerty.transcription.beam_corrector import (
    BeamAwareQwenModule,
    load_beam_checkpoint,
)
from emg2qwerty.transcription.vocabulary import (
    special_tokens_for_policy,
    TEXT_ONLY_VOCABULARY,
)


@pytest.mark.parametrize("base_rows", [8, 32])
def test_text_tokenizer_checkpoint_and_training_round_trip(
    tmp_path: Path, base_rows: int
) -> None:
    base = tmp_path / "base"
    tokens = [
        "<unk>",
        "<pad>",
        "<eos>",
        "hello",
        "world",
        "user",
        "assistant",
        "system",
    ]
    backend = Tokenizer(
        models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    config = Qwen3_5TextConfig(
        vocab_size=base_rows,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        eos_token_id=2,
        pad_token_id=1,
        max_position_embeddings=512,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    Qwen3_5ForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    options: dict[str, Any] = dict(
        base_model_revision="a" * 40,
        dtype="fp32",
        gradient_checkpointing=False,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        max_seq_length=512,
        max_new_tokens=2,
    )
    module = BeamAwareQwenModule(str(base), **options)
    expected_tokens = special_tokens_for_policy(TEXT_ONLY_VOCABULARY)
    expected_size = 8 + len(expected_tokens)
    assert len(module.tokenizer) == expected_size
    assert module.model.get_input_embeddings().num_embeddings == expected_size
    assert module.tokenizer.padding_side == "left"
    assert [
        module.tokenizer.convert_tokens_to_ids(token) for token in expected_tokens
    ] == list(range(8, expected_size))
    assert set(module.tokenizer.get_vocab()) == set(tokens) | set(expected_tokens)
    assert module.tokenizer.encode("<NO_CONTEXT>", add_special_tokens=False) == [11]
    batch = module.prepare_response_only_batch(
        ["hello", "hello world"], ["hello", "world"]
    )
    assert batch["attention_mask"][0, 0] == 0
    assert batch["labels"][0, 0] == -100
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad], lr=1e-3
    )
    loss = module.model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    module.eval()
    with torch.no_grad():
        expected = module.model(**batch).logits
    checkpoint: dict[str, Any] = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "pytorch-lightning_version": "2.6.1",
    }
    module.on_save_checkpoint(checkpoint)
    assert all(key.startswith("model.") for key in checkpoint["state_dict"])
    assert checkpoint["transcription_metadata"]["embedding_weights_saved"] is (
        base_rows == 8
    )
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    restored = load_beam_checkpoint(path)
    with torch.no_grad():
        torch.testing.assert_close(restored.model(**batch).logits, expected)
    assert len(restored.correct(["hello"], ["hello"], [[["hello", "world"]]])) == 1
    resumed = BeamAwareQwenModule.load_from_checkpoint(path, weights_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(resumed.model(**batch).logits, expected)

    bad_checkpoint = dict(checkpoint)
    bad_checkpoint["transcription_metadata"] = dict(
        checkpoint["transcription_metadata"], tokenizer_vocab_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="tokenizer_vocab_sha256"):
        resumed.on_load_checkpoint(bad_checkpoint)
    for mode in ("missing", "unexpected", "shape"):
        bad_state = dict(checkpoint["state_dict"])
        key = next(key for key in bad_state if "lora_" in key)
        if mode == "missing":
            del bad_state[key]
        elif mode == "unexpected":
            bad_state["unknown.weight"] = torch.ones(1)
        else:
            bad_state[key] = torch.ones(1)
        with pytest.raises(RuntimeError, match="checkpoint tensors"):
            resumed.load_state_dict(bad_state)

    # Development checkpoints contain model.* LoRA keys and an old template path.
    legacy = dict(checkpoint)
    legacy.pop("transcription_metadata")
    legacy["hyper_parameters"] = dict(checkpoint["hyper_parameters"])
    legacy["hyper_parameters"].pop("user_template")
    legacy["hyper_parameters"][
        "user_template_path"
    ] = "/old/training/config/template.txt"
    legacy["hyper_parameters"]["lora_target_modules"] = None
    legacy["hyper_parameters"]["projector_lr"] = 0.001
    legacy_path = tmp_path / "development.ckpt"
    torch.save(legacy, legacy_path)
    legacy_restored = load_beam_checkpoint(legacy_path)
    with torch.no_grad():
        torch.testing.assert_close(legacy_restored.model(**batch).logits, expected)

    # A real Trainer must restore the optimizer as well as the LoRA tensors.
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    loader = DataLoader([{"payload": "hello", "target": "world"}], batch_size=1)
    train_options = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    first = pl.Trainer(
        **train_options,
        default_root_dir=tmp_path / "train",
        callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)],
    )
    first.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    saved = Path(first.checkpoint_callback.last_model_path)
    saved_payload = torch.load(saved, weights_only=True)
    assert first.global_step == 1
    assert saved_payload["optimizer_states"][0]["state"]
    trained = load_beam_checkpoint(saved)
    with torch.no_grad():
        trained.model.eval()
        module.model.eval()
        torch.testing.assert_close(
            trained.model(**batch).logits, module.model(**batch).logits
        )
    continued = BeamAwareQwenModule(str(base), **options)
    second = pl.Trainer(
        **{**train_options, "max_epochs": 2}, default_root_dir=tmp_path / "resume"
    )
    second.fit(
        continued, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=saved
    )
    assert second.global_step == 2
    optimizer_steps = [
        int(state["step"]) for state in second.optimizers[0].state.values()
    ]
    assert optimizer_steps and set(optimizer_steps) == {2}
