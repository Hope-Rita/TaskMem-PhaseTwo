import torch
import torch.nn as nn
from vllm import ModelRegistry
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from unittest.mock import patch

import vllm.model_executor.models.qwen2 as qwen2vllm
import modeling.qwen2hf_ada as qwen2hf_ada


class Qwen2DecoderLayerWithAdapter(qwen2vllm.Qwen2DecoderLayer):
    def __init__(
        self,
        config: qwen2hf_ada.Qwen2ConfigWithAdapter,
        cache_config: CacheConfig = None,
        quant_config: QuantizationConfig = None,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.config = config
        self.layer_idx = int(prefix.split(".")[-1])

        self.adapter = qwen2hf_ada.Qwen2Adapter(
            config.hidden_size,
            dtype=self.mlp.down_proj.weight.dtype,
            device=self.mlp.down_proj.weight.device,
        ) if config.has_adapter(self.layer_idx) else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        return hidden_states, residual


class Qwen2ModelWithAdapter(qwen2vllm.Qwen2Model):
    def __init__(
        self, *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayerWithAdapter,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )

        # [BC] for version <0.8.4: no `decoder_layer_type` argument
        # with patch.object(qwen2vllm, "Qwen2DecoderLayer", Qwen2DecoderLayerWithAdapter):
        #     super().__init__(vllm_config=vllm_config, prefix=prefix)


class Qwen2ForCausalLMWithAdapter(qwen2vllm.Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        assert isinstance(vllm_config.model_config.hf_config, qwen2hf_ada.Qwen2ConfigWithAdapter), \
              "HF config not registered!"
        
        with patch.object(qwen2vllm, "Qwen2Model", Qwen2ModelWithAdapter):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        print(f"[INFO] vLLM model: {self}")


def register():
    qwen2hf_ada.register()
    ModelRegistry.register_model("Qwen2ForCausalLM", "modeling.qwen2vllm_ada:Qwen2ForCausalLMWithAdapter")

    print(f"[INFO] vLLM register: architecture 'Qwen2ForCausalLM' -> Qwen2ForCausalLMWithAdapter")