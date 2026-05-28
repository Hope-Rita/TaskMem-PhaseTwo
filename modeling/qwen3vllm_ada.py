import os
import torch
import torch.distributed as dist
from unittest.mock import patch

from vllm import LLM, ModelRegistry
from vllm.config import VllmConfig

import vllm.model_executor.models.qwen3_moe as qwen3moe
import vllm.model_executor.models.qwen3_vl_moe as qwen3vlmoe

import modeling.qwen3hf_ada as qwen3hf_ada


# =========================
# 1) DecoderLayer
# =========================
class Qwen3MoeDecoderLayerWithAdapter(qwen3moe.Qwen3MoeDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        prefix = kwargs.get("prefix", "")
        try:
            self.layer_idx = int(prefix.split(".")[-1])
        except Exception:
            self.layer_idx = -1

        vllm_config = None
        if len(args) > 0 and hasattr(args[0], "model_config"):
            vllm_config = args[0]
        else:
            vllm_config = kwargs.get("vllm_config", None)

        text_cfg = None
        if vllm_config is not None:
            mc = vllm_config.model_config
            text_cfg = getattr(mc, "hf_text_config", None)
            if text_cfg is None:
                hf_cfg = getattr(mc, "hf_config", None)
                text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        else:
            text_cfg = kwargs.get("config", None) or (args[0] if len(args) > 0 else None)

        has_adapter = getattr(text_cfg, "has_adapter", None)
        enable_adapter = bool(has_adapter(self.layer_idx)) if callable(has_adapter) else False

        ref_p = next(self.parameters(), None)
        ref_dtype = ref_p.dtype if ref_p is not None else torch.bfloat16
        ref_device = ref_p.device if ref_p is not None else torch.device("cuda")

        self.adapter = (
            qwen3hf_ada.Qwen3Adapter(
                text_cfg.hidden_size,
                dtype=ref_dtype,
                device=ref_device,
            )
            if enable_adapter
            else None
        )

    def forward(self, positions, hidden_states, residual):
        hidden_states, residual = super().forward(positions, hidden_states, residual)
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        return hidden_states, residual


# =========================
# 2) MoE Text Model
# =========================
class Qwen3MoeModelWithAdapter(qwen3moe.Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        with patch.object(qwen3moe, "Qwen3MoeDecoderLayer", Qwen3MoeDecoderLayerWithAdapter):
            super().__init__(vllm_config=vllm_config, prefix=prefix)


# =========================
# 3) Patch internal Qwen3MoeModel reference
# =========================
from contextlib import ExitStack

class Qwen3MoeLLMModelWithAdapter(qwen3vlmoe.Qwen3MoeLLMModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        g = qwen3moe.Qwen3MoeModel.__init__.__globals__

        old_global = g.get("Qwen3MoeDecoderLayer", None)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(qwen3moe, "Qwen3MoeDecoderLayer", Qwen3MoeDecoderLayerWithAdapter)
            )
            if hasattr(qwen3vlmoe, "Qwen3MoeDecoderLayer"):
                stack.enter_context(
                    patch.object(qwen3vlmoe, "Qwen3MoeDecoderLayer", Qwen3MoeDecoderLayerWithAdapter)
                )

            g["Qwen3MoeDecoderLayer"] = Qwen3MoeDecoderLayerWithAdapter

            try:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            finally:
                if old_global is None:
                    g.pop("Qwen3MoeDecoderLayer", None)
                else:
                    g["Qwen3MoeDecoderLayer"] = old_global


# =========================
# 4) Qwen3VLMoeForConditionalGeneration
# =========================
class Qwen3VLMoeForConditionalGenerationWithAdapter(qwen3vlmoe.Qwen3VLMoeForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        hf_cfg = vllm_config.model_config.hf_config
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)

        if not hasattr(text_cfg, "has_adapter"):
            adapter_layers = getattr(text_cfg, "adapter_layers", "all")
            setattr(text_cfg, "adapter_layers", adapter_layers)

            def _has_adapter(idx: int) -> bool:
                if adapter_layers == "all":
                    return True
                if adapter_layers in ("none", None, []):
                    return False
                return idx in adapter_layers

            text_cfg.has_adapter = _has_adapter

        if hasattr(vllm_config.model_config, "hf_text_config") and vllm_config.model_config.hf_text_config is not None:
            vllm_config.model_config.hf_text_config.has_adapter = text_cfg.has_adapter
            vllm_config.model_config.hf_text_config.adapter_layers = getattr(text_cfg, "adapter_layers", "all")

        with patch.object(qwen3vlmoe, "Qwen3MoeLLMModel", Qwen3MoeLLMModelWithAdapter):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

        print(f"[INFO] vLLM Qwen3-VL-MoE with Adapter loaded: {self}")


def register():
    qwen3hf_ada.register()

    ModelRegistry.register_model(
        "Qwen3VLMoeForConditionalGeneration",
        "modeling.qwen3vllm_ada:Qwen3VLMoeForConditionalGenerationWithAdapter",
    )
    print("[INFO] vLLM register: Qwen3VLMoeForConditionalGeneration -> WithAdapter")
