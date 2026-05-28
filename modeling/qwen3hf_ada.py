"""HF Qwen3-VL-MoE + Adapter."""

from __future__ import annotations

from collections.abc import Container
from typing import Any, Literal

import torch
import torch.nn as nn
from unittest.mock import patch
import os
from transformers import AutoConfig, AutoModelForImageTextToText

import modeling.qwen2hf_ada as qwen2hf_ada

import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as qwen3vlmoe_hf
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)


# =========================
# 1) Adapter + Config
# =========================
class Qwen3Adapter(qwen2hf_ada.Qwen2Adapter):
    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(hidden_size=hidden_size, dtype=dtype, device=device)


class Qwen3VLMoeTextConfigWithAdapter(Qwen3VLMoeTextConfig):
    model_type = "qwen3_vl_moe_text"

    def __init__(
        self,
        adapter_layers: Literal["all", "none"] | Container[int] = "all",
        adapter_dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapter_layers = adapter_layers
        self.adapter_dtype = adapter_dtype

    def has_adapter(self, layer_idx: int) -> bool:
        if self.adapter_layers == "all":
            return True
        if self.adapter_layers in ("none", None):
            return False
        return layer_idx in self.adapter_layers


class Qwen3VLMoeConfigWithAdapter(Qwen3VLMoeConfig):
    model_type = "qwen3_vl_moe"

    sub_configs = {
        "vision_config": Qwen3VLMoeVisionConfig,
        "text_config": Qwen3VLMoeTextConfigWithAdapter,
    }

    def __init__(
        self,
        adapter_layers: Literal["all", "none"] | Container[int] = "all",
        adapter_dtype: torch.dtype | None = None,
        text_config=None,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        self.text_config.adapter_layers = adapter_layers
        self.text_config.adapter_dtype = adapter_dtype


# =========================
# 2) Patch Text Decoder Layer
# =========================
class Qwen3VLMoeTextDecoderLayerWithAdapter(qwen3vlmoe_hf.Qwen3VLMoeTextDecoderLayer):
    def __init__(self, config: Qwen3VLMoeTextConfigWithAdapter, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        ref_p = self.self_attn.v_proj.weight  # Access a known initialized parameter
        ref_dtype = ref_p.dtype if ref_p is not None else torch.bfloat16
        ref_device = ref_p.device if ref_p is not None else torch.device("cuda")

        self.adapter = (
            Qwen3Adapter(
                config.hidden_size,
                dtype=(config.adapter_dtype or ref_dtype),
                device=ref_device,
            )
            if config.has_adapter(layer_idx)
            else None
        )

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, Any]:
        outputs = super().forward(*args, **kwargs)

        if self.adapter is None:
            return outputs
        if self.adapter is not None:
            if not torch.isfinite(self.adapter.steer).all():
                raise RuntimeError(f"adapter steer non-finite layer={self.layer_idx}")
        return self.adapter(outputs)


# =========================
# 3) Patch text branch only
# =========================
class Qwen3VLMoeTextModelWithAdapter(qwen3vlmoe_hf.Qwen3VLMoeTextModel):
    def __init__(self, config):
        with patch.object(
            qwen3vlmoe_hf,
            "Qwen3VLMoeTextDecoderLayer",
            Qwen3VLMoeTextDecoderLayerWithAdapter,
        ):
            super().__init__(config=config)


class Qwen3VLMoeModelWithAdapter(qwen3vlmoe_hf.Qwen3VLMoeModel):
    def __init__(self, config):
        with patch.object(
            qwen3vlmoe_hf,
            "Qwen3VLMoeTextModel",
            Qwen3VLMoeTextModelWithAdapter,
        ):
            super().__init__(config=config)


class Qwen3VLMoeForConditionalGenerationWithAdapter(
    qwen3vlmoe_hf.Qwen3VLMoeForConditionalGeneration
):
    config_class = Qwen3VLMoeConfigWithAdapter
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayerWithAdapter"]

    def __init__(self, config: Qwen3VLMoeConfigWithAdapter):
        with patch.object(qwen3vlmoe_hf, "Qwen3VLMoeModel", Qwen3VLMoeModelWithAdapter):
            super().__init__(config=config)
        self.freeze_base()

        print("[INFO] HF Qwen3-VL-MoE with Adapter loaded.")

    def freeze_base(self):
        total = 0
        trainable = 0
        base_train = True if "qwen3vllm_base" in os.environ.get("VLLM_MODELS", "") else False
        ada_train = True if "qwen3vllm_ada" in os.environ.get("VLLM_MODELS", "") else False
        
        s = os.environ.get("VLLM_ADA_TRAIN_LAYERS", "").strip()
        train_layers = None if s == "" else {int(x) for x in s.split(",")}      

        for name, param in self.named_parameters():
            total += param.numel()
            param.requires_grad = False

            if ada_train and "adapter" in name:
                if train_layers is None:
                    param.requires_grad = True
                else:
                    for layer_idx in train_layers:
                        if f".layers.{layer_idx}." in name:
                            param.requires_grad = True
                            break
            if base_train and "adapter" not in name:
                param.requires_grad = True

            if param.requires_grad:
                trainable += param.numel()

        print(f"[INFO] Total params: {total:,}")
        print(f"[INFO] Trainable params: {trainable:,}")
        print(f"[INFO] Frozen params: {total - trainable:,}")
        print(f"[INFO] VLLM_ADA_TRAIN_LAYERS={s if s else 'ALL'}")


def register():
    AutoConfig.register("qwen3_vl_moe_text", Qwen3VLMoeTextConfigWithAdapter, exist_ok=True)
    AutoConfig.register("qwen3_vl_moe", Qwen3VLMoeConfigWithAdapter, exist_ok=True)

    AutoModelForImageTextToText.register(
        Qwen3VLMoeConfigWithAdapter,
        Qwen3VLMoeForConditionalGenerationWithAdapter,
        exist_ok=True,
    )

    print("[INFO] HF register: qwen3_vl_moe -> Qwen3VLMoeForConditionalGenerationWithAdapter")