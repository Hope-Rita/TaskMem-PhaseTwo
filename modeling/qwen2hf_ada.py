from collections.abc import Container
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from unittest.mock import patch

import transformers.models.qwen2.modeling_qwen2 as qwen2hf


class Qwen2ConfigWithAdapter(qwen2hf.Qwen2Config):
    def __init__(
        self,
        adapter_layers: Literal["all", "none"] | Container[int] = "all",
        adapter_dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapter_layers = adapter_layers
        self.adapter_dtype = adapter_dtype

    def has_adapter(self, layer_idx: int):
        return self.adapter_layers == "all" or \
              (self.adapter_layers != "none" and layer_idx in self.adapter_layers)


class Qwen2Adapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.steer = nn.Parameter(torch.randn(self.hidden_size, dtype=dtype, device=device))
    
    def forward(self, hidden_states: torch.Tensor):
        steer = self.steer.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + steer
    
    def __repr__(self):
        return f"{self.__class__.__name__}(({self.hidden_size},))"
    
    def _init_weights(self, module):
        module.steer.zero_()

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True


class Qwen2DecoderLayerWithAdapter(qwen2hf.Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen2ConfigWithAdapter,
        layer_idx: int,
    ):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.adapter = Qwen2Adapter(
            config.hidden_size,
            dtype=self.mlp.down_proj.weight.dtype,
            device=self.mlp.down_proj.weight.device,
        ) if config.has_adapter(layer_idx) else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.Tensor, Any]:
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.adapter is not None:
            hidden_states = outputs[0]
            hidden_states = self.adapter(hidden_states)
            outputs = (hidden_states,) + outputs[1:]

        
        return outputs


class Qwen2ModelWithAdapter(qwen2hf.Qwen2Model):
    config_class = Qwen2ConfigWithAdapter
    _no_split_modules = ["Qwen2DecoderLayerWithAdapter"]

    def __init__(self, config: Qwen2ConfigWithAdapter):
        with patch.object(qwen2hf, "Qwen2DecoderLayer", Qwen2DecoderLayerWithAdapter):
            super().__init__(config=config)


class Qwen2ForCausalLMWithAdapter(qwen2hf.Qwen2ForCausalLM):
    config_class = Qwen2ConfigWithAdapter
    _no_split_modules = ["Qwen2DecoderLayerWithAdapter"]
    
    def __init__(self, config: Qwen2ConfigWithAdapter):
        with patch.object(qwen2hf, "Qwen2Model", Qwen2ModelWithAdapter):
            super().__init__(config=config)

        self.freeze_base()
        print(f"[INFO] HF model: {self}")
    
    def freeze_base(self):
        for name, param in self.named_parameters():
            param.requires_grad = "adapter" in name
        
        print(f"[INFO] Frozen base model parameters: {sum(p.numel() for p in self.parameters() if not p.requires_grad):,}")
        print(f"[INFO] Trainable adapter parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")


def register():
    AutoConfig.register("qwen2", Qwen2ConfigWithAdapter, exist_ok=True)
    AutoModelForCausalLM.register(Qwen2ConfigWithAdapter, Qwen2ForCausalLMWithAdapter, exist_ok=True)

    print(f"[INFO] HF register: model_type 'qwen2' -> Qwen2ForCausalLMWithAdapter")
