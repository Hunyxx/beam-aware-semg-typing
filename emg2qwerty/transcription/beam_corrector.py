"""Isolated Qwen LoRA module for beam-aware transcription correction.

Transformers and PEFT are imported only when the module is instantiated, so
CPU-only dataset, protocol, and metric tooling remains lightweight.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR

from emg2qwerty.transcription.beam_protocol import (
    build_structural_bad_words_ids,
    parse_model_output,
    shared_output_filter,
)
from emg2qwerty.transcription.checkpoint import read_lightning_checkpoint
from emg2qwerty.transcription.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    load_user_template,
    prompt_sha256,
    render_user_prompt,
)
from emg2qwerty.transcription.text import normalize_text
from emg2qwerty.transcription.vocabulary import (
    configure_text_tokenizer,
    TEXT_ONLY_VOCABULARY,
    tokenizer_vocab_sha256,
)


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _resolve_dtype(name: str) -> torch.dtype | str:
    normalized = str(name).lower().replace("torch.", "")
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {name!r}")
    return mapping[normalized]


def _cosine_lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between zero and total_steps")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_tokenizer_embeddings(tokenizer: Any, model: Any) -> None:
    """Allow padded embedding tables, but reject tokens outside their bounds."""

    token_ids = list(tokenizer.get_vocab().values())
    token_ids.extend(
        int(value)
        for value in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if value is not None
    )
    if not token_ids or any(not isinstance(value, int) for value in token_ids):
        raise ValueError("tokenizer vocabulary must contain integer token IDs")
    embeddings = model.get_input_embeddings().num_embeddings
    if min(token_ids) < 0 or max(token_ids) >= embeddings:
        raise ValueError("tokenizer token IDs exceed the model embedding vocabulary")


class BeamAwareQwenModule(pl.LightningModule):
    """Response-only causal-LM fine-tuning over top-k beam prompts."""

    def __init__(
        self,
        model_id: str,
        *,
        base_model_revision: str | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 0,
        max_seq_length: int = 4096,
        max_new_tokens: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_loading_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template_path: str | Path | None = None,
        user_template: str | None = None,
        dtype: str = "bf16",
        gradient_checkpointing: bool = True,
        vocabulary_policy: str | None = None,
    ) -> None:
        super().__init__()
        if max_seq_length <= 0 or max_new_tokens <= 0:
            raise ValueError("sequence limits must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if int(lora_r) <= 0 or int(lora_alpha) <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= float(lora_dropout) < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not lora_target_modules:
            raise ValueError("lora_target_modules must be non-empty")

        self.model_id = str(model_id)
        self.vocabulary_policy = vocabulary_policy or TEXT_ONLY_VOCABULARY
        if self.vocabulary_policy != TEXT_ONLY_VOCABULARY:
            raise ValueError("unsupported vocabulary policy")
        self.base_model_revision = (
            None if base_model_revision is None else str(base_model_revision)
        )
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.max_seq_length = int(max_seq_length)
        self.max_new_tokens = int(max_new_tokens)
        self.generation_kwargs = dict(generation_kwargs or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if bool(self.generation_kwargs.get("do_sample", False)):
            raise ValueError(
                "the deterministic beam correction protocol requires "
                "deterministic generation"
            )
        forbidden_generation_keys = {
            "bad_words_ids",
            "eos_token_id",
            "max_new_tokens",
            "pad_token_id",
        }
        conflicts = forbidden_generation_keys & set(self.generation_kwargs)
        if conflicts:
            raise ValueError(
                f"generation_kwargs must not override: {sorted(conflicts)}"
            )
        self.dtype_name = str(dtype).lower().replace("torch.", "")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.system_prompt = str(system_prompt)
        self.user_template = load_user_template(
            user_template_path,
            template=user_template,
        )
        self.prompt_digest = prompt_sha256(
            self.system_prompt,
            self.user_template,
        )
        tokenizer_options = dict(tokenizer_kwargs or {})
        model_options = dict(model_loading_kwargs or {})

        try:
            from peft import get_peft_model, LoraConfig, TaskType
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        except ImportError as exc:
            raise ImportError(
                "BeamAwareQwenModule requires the optional LLM dependencies; "
                "install the pinned dependencies from requirements.txt"
            ) from exc

        if self.base_model_revision is not None:
            for name, options in (
                ("tokenizer_kwargs", tokenizer_options),
                ("model_loading_kwargs", model_options),
            ):
                configured_revision = options.get("revision")
                if configured_revision not in (None, self.base_model_revision):
                    raise ValueError(
                        f"{name}.revision conflicts with base_model_revision"
                    )
                options["revision"] = self.base_model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **tokenizer_options,
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("the base tokenizer must define an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        configure_text_tokenizer(self.tokenizer, self.vocabulary_policy)

        model_options.setdefault("torch_dtype", _resolve_dtype(dtype))
        base_model = Qwen3_5ForCausalLM.from_pretrained(
            self.model_id,
            **model_options,
        )
        self.base_model_class = type(base_model).__name__
        self.resolved_model_revision = (
            getattr(base_model.config, "_commit_hash", None) or self.base_model_revision
        )
        self.base_embedding_rows = base_model.get_input_embeddings().num_embeddings
        self.embedding_weights_saved = len(self.tokenizer) > self.base_embedding_rows
        base_model.resize_token_embeddings(len(self.tokenizer))
        if gradient_checkpointing:
            base_model.config.use_cache = False
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules),
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = tuple(str(value) for value in lora_target_modules)

        validate_tokenizer_embeddings(self.tokenizer, self.model)

        self.save_hyperparameters(
            {
                "model_id": self.model_id,
                "base_model_revision": self.base_model_revision,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "max_seq_length": self.max_seq_length,
                "max_new_tokens": self.max_new_tokens,
                "lora_r": int(lora_r),
                "lora_alpha": int(lora_alpha),
                "lora_dropout": float(lora_dropout),
                "lora_target_modules": list(lora_target_modules),
                "tokenizer_kwargs": tokenizer_options,
                "model_loading_kwargs": model_options,
                "generation_kwargs": self.generation_kwargs,
                "chat_template_kwargs": self.chat_template_kwargs,
                "system_prompt": self.system_prompt,
                "user_template_path": None,
                "user_template": self.user_template,
                "dtype": self.dtype_name,
                "gradient_checkpointing": self.gradient_checkpointing,
                "vocabulary_policy": self.vocabulary_policy,
            }
        )

    def format_prompt(self, payload: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": render_user_prompt(
                    payload,
                    self.user_template,
                    development_format=True,
                ),
            },
        ]
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        )

    def prepare_response_only_batch(
        self,
        payloads: Sequence[object],
        targets: Sequence[object],
    ) -> dict[str, torch.Tensor]:
        if len(payloads) != len(targets):
            raise ValueError("payload and target batch sizes differ")
        encoded_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        eos_id = int(self.tokenizer.eos_token_id)

        for payload, target in zip(payloads, targets):
            prompt_ids = self.tokenizer.encode(
                self.format_prompt(normalize_text(payload)),
                add_special_tokens=False,
            )
            response_ids = self.tokenizer.encode(
                normalize_text(target),
                add_special_tokens=False,
            ) + [eos_id]
            token_ids = [int(value) for value in prompt_ids + response_ids]
            if len(token_ids) > self.max_seq_length:
                raise ValueError(
                    "tokenized training example exceeds max_seq_length: "
                    f"{len(token_ids)} > {self.max_seq_length}"
                )
            encoded_rows.append(token_ids)
            label_rows.append([-100] * len(prompt_ids) + [int(v) for v in response_ids])

        if not encoded_rows:
            raise ValueError("empty training batch")
        width = max(len(row) for row in encoded_rows)
        batch_size = len(encoded_rows)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full(
            (batch_size, width),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, width),
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.full(
            (batch_size, width),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        for row_index, (row, row_labels) in enumerate(zip(encoded_rows, label_rows)):
            length = len(row)
            start = width - length if self.tokenizer.padding_side == "left" else 0
            end = start + length
            input_ids[row_index, start:end] = torch.tensor(
                row,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[row_index, start:end] = 1
            labels[row_index, start:end] = torch.tensor(
                row_labels,
                dtype=torch.long,
                device=self.device,
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def forward(self, **batch: torch.Tensor) -> Any:
        return self.model(**batch)

    def _loss_step(
        self,
        phase: str,
        batch: Mapping[str, Sequence[object]],
    ) -> torch.Tensor:
        encoded = self.prepare_response_only_batch(
            batch["payload"],
            batch["target"],
        )
        loss = self.model(**encoded).loss
        self.log(
            f"{phase}/loss",
            loss,
            batch_size=len(batch["payload"]),
            prog_bar=phase != "train",
            sync_dist=True,
        )
        return loss

    def training_step(
        self,
        batch: Mapping[str, Sequence[object]],
        batch_index: int,
    ) -> torch.Tensor:
        del batch_index
        return self._loss_step("train", batch)

    def validation_step(
        self,
        batch: Mapping[str, Sequence[object]],
        batch_index: int,
    ) -> torch.Tensor:
        del batch_index
        return self._loss_step("val", batch)

    def test_step(
        self,
        batch: Mapping[str, Sequence[object]],
        batch_index: int,
    ) -> torch.Tensor:
        del batch_index
        return self._loss_step("test", batch)

    def configure_optimizers(self) -> dict[str, Any]:
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 0:
            return {"optimizer": optimizer}

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_lr_scale(
                step, total_steps, self.warmup_steps
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.inference_mode()
    def generate_text(self, payloads: Sequence[object]) -> list[str]:
        if not payloads:
            return []
        prompts = [self.format_prompt(normalize_text(payload)) for payload in payloads]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max().item()) > self.max_seq_length:
            raise ValueError("generation prompt exceeds max_seq_length")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generation_options = dict(self.generation_kwargs)
        generation_options["do_sample"] = False
        generation_options["max_new_tokens"] = self.max_new_tokens
        generation_options["bad_words_ids"] = build_structural_bad_words_ids(
            self.tokenizer
        )
        generation_options.setdefault(
            "pad_token_id",
            int(self.tokenizer.pad_token_id),
        )
        generation_options.setdefault(
            "eos_token_id",
            int(self.tokenizer.eos_token_id),
        )

        was_training = self.model.training
        self.model.eval()
        try:
            generated = self.model.generate(
                **encoded,
                **generation_options,
            )
        finally:
            if was_training:
                self.model.train()
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        return [
            self.tokenizer.decode(row, skip_special_tokens=False)
            for row in continuation
        ]

    @torch.inference_mode()
    def correct(
        self,
        payloads: Sequence[object],
        primaries: Sequence[object],
        candidates_per_sample: Sequence[Sequence[Sequence[str]]],
    ) -> list[dict[str, Any]]:
        if not (len(payloads) == len(primaries) == len(candidates_per_sample)):
            raise ValueError("correction inputs have different batch sizes")
        raw_outputs = self.generate_text(payloads)
        results: list[dict[str, Any]] = []
        for raw, primary, candidates in zip(
            raw_outputs,
            primaries,
            candidates_per_sample,
        ):
            parsed = parse_model_output(raw, tokenizer=self.tokenizer)
            corrected, diagnostics = shared_output_filter(
                str(primary),
                candidates,
                parsed,
            )
            results.append(
                {
                    "raw": raw,
                    "parsed": parsed,
                    "corrected": corrected,
                    "diagnostics": diagnostics,
                }
            )
        return results

    def _checkpoint_keys(self) -> set[str]:
        keys = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if self.embedding_weights_saved:
            embeddings = (
                self.model.get_input_embeddings(),
                self.model.get_output_embeddings(),
            )
            for name, module in self.named_modules():
                if any(module is embedding for embedding in embeddings):
                    keys.update(f"{name}.{key}" for key in module.state_dict())
        return keys

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model_revision": self.resolved_model_revision,
            "prompt_sha256": self.prompt_digest,
            "tokenizer_vocab_sha256": tokenizer_vocab_sha256(self.tokenizer),
            "tokenizer_vocab_size": len(self.tokenizer),
            "embedding_weights_saved": self.embedding_weights_saved,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Save LoRA tensors with their model.* parameter names.
        state = checkpoint.get("state_dict", self.state_dict())
        keys = self._checkpoint_keys()
        missing = keys - set(state)
        if missing:
            raise RuntimeError(
                f"checkpoint is missing trainable tensors: {sorted(missing)}"
            )
        checkpoint["state_dict"] = {
            key: state[key].detach().cpu() for key in state if key in keys
        }
        checkpoint["trainable_state_keys"] = sorted(keys)
        checkpoint["transcription_metadata"] = self.checkpoint_metadata()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        metadata = checkpoint.get("transcription_metadata")
        if metadata is not None:
            expected = self.checkpoint_metadata()
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"checkpoint {key} does not match the configured module"
                    )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        expected = self.state_dict()
        missing = sorted(self._checkpoint_keys() - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        mismatched = [
            key
            for key in set(state_dict) & set(expected)
            if not torch.is_tensor(state_dict[key])
            or state_dict[key].shape != expected[key].shape
        ]
        if mismatched or (strict and (missing or unexpected)):
            raise RuntimeError(
                f"checkpoint tensors do not match the model: missing={missing}, "
                f"unexpected={unexpected}, shape_mismatches={sorted(mismatched)}"
            )
        super().load_state_dict(state_dict, strict=False, assign=assign)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)


def load_beam_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | None = None,
    model_id: str | None = None,
    base_model_revision: str | None = None,
    max_new_tokens: int | None = None,
    max_seq_length: int | None = None,
) -> BeamAwareQwenModule:
    """Load a trained beam corrector directly from a Lightning .ckpt file."""
    checkpoint = read_lightning_checkpoint(path)
    recorded = checkpoint.get("hyper_parameters")
    if not isinstance(recorded, Mapping) or not recorded.get("model_id"):
        raise ValueError(
            "corrector checkpoint must record its model_id in hyper_parameters"
        )
    parameters = inspect.signature(BeamAwareQwenModule.__init__).parameters
    options = {
        key: value
        for key, value in recorded.items()
        if key in parameters and key != "self"
    }
    if model_id is not None:
        options["model_id"] = model_id
    metadata = checkpoint.get("transcription_metadata", {})
    revision = (
        base_model_revision
        or metadata.get("base_model_revision")
        or options.get("base_model_revision")
    )
    options["base_model_revision"] = revision
    # Checkpoints embed the template or record its path on the training machine.
    if options.get("user_template") is not None:
        options["user_template_path"] = None
    elif (
        options.get("user_template_path")
        and not Path(options["user_template_path"]).is_file()
    ):
        options["user_template_path"] = None
    if options.get("lora_target_modules") is None:
        options["lora_target_modules"] = DEFAULT_LORA_TARGET_MODULES
    generation = dict(options.get("generation_kwargs") or {})
    generation.pop("bad_word_strings", None)
    options["generation_kwargs"] = generation
    options["gradient_checkpointing"] = False
    loading = dict(options.get("model_loading_kwargs") or {})
    loading.pop("device_map", None)
    loading.pop("revision", None)
    # Runtime precision can be chosen independently of checkpoint storage precision.
    loading.pop("torch_dtype", None)
    loading.pop("dtype", None)
    device = torch.device(device)
    if device.type == "cuda":
        loading["device_map"] = {"": str(device)}
    options["model_loading_kwargs"] = loading
    tokenizer_options = dict(options.get("tokenizer_kwargs") or {})
    tokenizer_options.pop("revision", None)
    options["tokenizer_kwargs"] = tokenizer_options
    if dtype is not None:
        options["dtype"] = dtype
    if max_new_tokens is not None:
        options["max_new_tokens"] = max_new_tokens
    if max_seq_length is not None:
        options["max_seq_length"] = max_seq_length
    module = BeamAwareQwenModule(**options)
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.to(device).eval()
    return module


__all__ = ["BeamAwareQwenModule", "load_beam_checkpoint", "DEFAULT_LORA_TARGET_MODULES"]
