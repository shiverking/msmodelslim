# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import glob
import os
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
)
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    AscendV1SaveInterface,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import (
    MAX_READ_FILE_SIZE_32G,
    get_valid_read_path,
    json_safe_dump,
    json_safe_load,
)


CALIBRATION_DTYPE = torch.float16
DEFAULT_ASR_PROMPT = "Transcribe this audio accurately."


class Qwen3ASRCalibrationModel(nn.Module):
    """Expose the thinker forward while preserving its checkpoint prefix."""

    def __init__(self, thinker: nn.Module):
        super().__init__()
        self.thinker = thinker
        self.config = thinker.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, **inputs):
        return self.thinker(**inputs)

    def generate(self, *args, **kwargs):
        return self.thinker.generate(*args, **kwargs)


@logger_setter()
class Qwen3ASRModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    AscendV1SaveInterface,
):
    """Layer-wise Qwen3-ASR adapter that quantizes only the text decoder."""

    USE_VLM_DATASET_LOADER = True

    def __init__(
        self, model_type: str, model_path: str, trust_remote_code: bool = False
    ):
        self._processor = None
        super().__init__(model_type, model_path, trust_remote_code)

    def _load_config(self, trust_remote_code: bool = False):
        try:
            from qwen_asr.core.transformers_backend import Qwen3ASRConfig
        except ImportError as error:
            raise ImportError(
                "Qwen3-ASR support requires qwen-asr. Install qwen-asr==0.0.6."
            ) from error
        return Qwen3ASRConfig.from_pretrained(
            str(self.model_path),
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def get_model_pedigree(self) -> str:
        return "qwen3_asr"

    def get_model_type(self) -> str:
        return self.model_type

    def get_global_model_torch_dtype(self) -> torch.dtype:
        return CALIBRATION_DTYPE

    def handle_dataset(
        self, dataset: Any, device: DeviceType = DeviceType.NPU
    ) -> List[Any]:
        """Load audio and build the same prompt inputs used by Qwen3-ASR."""
        try:
            import librosa
            from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        except ImportError as error:
            raise ImportError(
                "Qwen3-ASR calibration requires qwen-asr==0.0.6 and librosa."
            ) from error

        self._processor = Qwen3ASRProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        sample_rate = self._processor.feature_extractor.sampling_rate
        processed_data = []
        for line_id, item in enumerate(
            tqdm(dataset, desc="Processing Qwen3-ASR calibration dataset"), 1
        ):
            text = item.text if hasattr(item, "text") else item.get("text")
            audio_path = item.audio if hasattr(item, "audio") else item.get("audio")
            if not audio_path:
                get_logger().warning(
                    "Line %d: missing audio, skip this sample.", line_id
                )
                continue
            valid_audio_path = get_valid_read_path(
                str(audio_path), extensions=[".wav", ".mp3"]
            )
            waveform, _ = librosa.load(valid_audio_path, sr=sample_rate, mono=True)
            messages = [
                {"role": "system", "content": text or DEFAULT_ASR_PROMPT},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ]
            prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=prompt, audio=waveform, return_tensors="pt", padding=True
            )
            processed_inputs = self._collect_inputs_to_device(
                inputs,
                device,
                keys=[
                    "input_ids",
                    "input_features",
                    "attention_mask",
                    "feature_attention_mask",
                    "audio_feature_lengths",
                    "position_ids",
                ],
            )
            if processed_inputs["input_features"] is not None:
                processed_inputs["input_features"] = processed_inputs[
                    "input_features"
                ].to(dtype=CALIBRATION_DTYPE)
            processed_data.append(processed_inputs)
        if not processed_data:
            raise InvalidDatasetError(
                "No valid audio samples found for Qwen3-ASR calibration.",
                action="Provide at least one readable WAV or MP3 sample.",
            )
        get_logger().info("Processed %d Qwen3-ASR audio samples.", len(processed_data))
        return processed_data

    def handle_dataset_by_batch(
        self,
        dataset: Any,
        batch_size: int,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        """Keep audio calibration at batch size one for the legacy pipeline."""
        if batch_size != 1:
            get_logger().warning(
                "Qwen3-ASR calibration only supports batch size 1; "
                "processing samples sequentially."
            )
        return self.handle_dataset(dataset, device)

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Load the complete outer model for legacy W8A8S calibration."""
        try:
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )
        except ImportError as error:
            raise ImportError("Please install qwen-asr==0.0.6.") from error

        outer_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=CALIBRATION_DTYPE,
            local_files_only=True,
            device_map="auto" if device == DeviceType.NPU else "cpu",
            attn_implementation="eager",
            use_safetensors=True,
        ).eval()
        outer_model.config.use_cache = False
        outer_model.thinker.config.use_cache = False
        model = Qwen3ASRCalibrationModel(outer_model.thinker).eval()
        get_logger().info(
            "Initialized full Qwen3-ASR in FP16 for legacy W8A8S calibration."
        )
        return model

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Load the audio tower and one decoder layer; load the rest on demand."""
        try:
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRThinkerForConditionalGeneration,
            )
        except ImportError as error:
            raise ImportError("Please install qwen-asr==0.0.6.") from error
        self.model_path = get_valid_read_path(
            str(self.model_path), is_dir=True, check_user_stat=True
        )
        thinker_config = self.config.thinker_config
        original_num_layers = thinker_config.text_config.num_hidden_layers
        thinker_config.text_config.num_hidden_layers = 1
        thinker_config.text_config.use_cache = False
        try:
            model = Qwen3ASRThinkerForConditionalGeneration.from_pretrained(
                self.model_path,
                config=thinker_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=CALIBRATION_DTYPE,
                local_files_only=True,
                device_map="cpu",
                attn_implementation="eager",
                use_safetensors=True,
            ).eval()
        finally:
            thinker_config.text_config.num_hidden_layers = original_num_layers
        model.config.use_cache = False
        get_logger().info(
            "Initialized Qwen3-ASR in FP16 with one resident decoder layer."
        )
        return model

    def generate_model_visit(
        self, model: nn.Module
    ) -> Generator[ProcessRequest, Any, None]:
        # Audio features are required by decoder calibration. The quant config
        # excludes *audio_tower*, keeping the encoder floating point.
        yield ProcessRequest("audio_tower", model.audio_tower, (), {})
        yield from generated_decoder_layer_visit_func(
            model, transformer_blocks=self.generate_decoder_layer(model)
        )

    def generate_model_forward(
        self, model: nn.Module, inputs: Any
    ) -> Generator[ProcessRequest, Any, None]:
        """Reproduce the Qwen3-ASR prefill path and expose each decoder layer."""
        from transformers.masking_utils import create_causal_mask

        sample = inputs[0] if isinstance(inputs, list) else inputs
        input_ids = sample.get("input_ids")
        input_features = sample.get("input_features")
        attention_mask = sample.get("attention_mask")
        feature_attention_mask = sample.get("feature_attention_mask")
        audio_feature_lengths = sample.get("audio_feature_lengths")
        position_ids = sample.get("position_ids")
        if input_ids is None or input_features is None:
            raise InvalidDatasetError(
                "Qwen3-ASR input must contain input_ids and input_features."
            )
        if input_ids.shape[0] != 1:
            raise InvalidDatasetError(
                "Qwen3-ASR layer-wise calibration currently requires batch size 1."
            )

        inputs_embeds = model.get_input_embeddings()(input_ids)
        if feature_attention_mask is not None:
            audio_feature_lengths = feature_attention_mask.sum(dim=1)
        if audio_feature_lengths is None:
            raise InvalidDatasetError(
                "Qwen3-ASR input is missing audio feature lengths."
            )
        feature_len = audio_feature_lengths[0]
        audio_dtype = next(model.audio_tower.parameters()).dtype
        audio_output = yield ProcessRequest(
            "audio_tower",
            model.audio_tower,
            (input_features[0, :, :feature_len].to(audio_dtype),),
            {"feature_lens": feature_len.unsqueeze(0)},
        )
        if hasattr(audio_output, "last_hidden_state"):
            audio_features = audio_output.last_hidden_state
        elif isinstance(audio_output, tuple):
            audio_features = audio_output[0]
        else:
            audio_features = audio_output
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        audio_mask = model.get_placeholder_mask(input_ids, inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = model.get_rope_index(attention_mask)
            model.rope_deltas = rope_deltas - delta0
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        text_position_ids = position_ids[0]
        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        causal_mask = create_causal_mask(
            config=model.model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        for name, layer in self.generate_decoder_layer(model):
            hidden_states = yield ProcessRequest(
                name,
                layer,
                (hidden_states,),
                {
                    "position_embeddings": position_embeddings,
                    "attention_mask": causal_mask,
                    "position_ids": text_position_ids,
                    "past_key_values": None,
                    "use_cache": False,
                    "cache_position": cache_position,
                },
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

    def generate_decoder_layer(
        self, model: nn.Module
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = self.config.thinker_config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"model.layers.{layer_idx}"
            yield name, self._load_decoder_if_not_exist(model, name, layer_idx)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.config.use_cache = need_kv_cache
        model.model.config.use_cache = need_kv_cache

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        num_layers = self.config.thinker_config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"
            adapter_config.extend(
                [
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=MappingConfig(
                            source=f"{prefix}.input_layernorm",
                            targets=[
                                f"{prefix}.self_attn.q_proj",
                                f"{prefix}.self_attn.k_proj",
                                f"{prefix}.self_attn.v_proj",
                            ],
                        ),
                    ),
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=MappingConfig(
                            source=f"{prefix}.post_attention_layernorm",
                            targets=[
                                f"{prefix}.mlp.gate_proj",
                                f"{prefix}.mlp.up_proj",
                            ],
                        ),
                    ),
                    AdapterConfig(
                        subgraph_type="ov",
                        mapping=MappingConfig(
                            source=f"{prefix}.self_attn.v_proj",
                            targets=[f"{prefix}.self_attn.o_proj"],
                        ),
                        extra_config={"group_method": "max"},
                    ),
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=MappingConfig(
                            source=f"{prefix}.mlp.up_proj",
                            targets=[f"{prefix}.mlp.down_proj"],
                        ),
                    ),
                ]
            )
        return adapter_config

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """Restore the outer-model ``thinker.`` prefix expected by vLLM."""
        prefix = "thinker."
        config_path = os.path.join(save_directory, "config.json")
        config = json_safe_load(config_path)
        thinker_config = config.get("thinker_config", {})
        thinker_config["dtype"] = "float16"
        for sub_config_name in ("audio_config", "text_config"):
            if sub_config_name in thinker_config:
                thinker_config[sub_config_name]["dtype"] = "float16"
        json_safe_dump(config, config_path, indent=2)

        for name in (
            "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
        ):
            path = os.path.join(save_directory, name)
            data = json_safe_load(path)
            if "weight_map" in data:
                data["weight_map"] = {
                    self._add_prefix(key, prefix): value
                    for key, value in data["weight_map"].items()
                }
            else:
                data = {
                    self._add_prefix(key, prefix): value for key, value in data.items()
                }
            json_safe_dump(data, path, indent=2)
        for path in glob.glob(os.path.join(save_directory, "*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as file:
                tensors = {
                    self._add_prefix(key, prefix): file.get_tensor(key)
                    for key in file.keys()
                }
            save_file(tensors, path)

    @staticmethod
    def _add_prefix(name: str, prefix: str) -> str:
        return name if name.startswith(prefix) else f"{prefix}{name}"

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            return json_safe_load(index_path)["weight_map"]

        file_name = "model.safetensors"
        file_path = get_valid_read_path(
            os.path.join(self.model_path, file_name),
            extensions="safetensors",
            size_max=MAX_READ_FILE_SIZE_32G,
        )
        with safe_open(file_path, framework="pt", device="cpu") as file:
            return {key: file_name for key in file.keys()}

    def _get_state_dict(
        self, module: nn.Module, prefix: str = ""
    ) -> Dict[str, torch.Tensor]:
        weight_map = self._get_weight_map()
        file_groups = defaultdict(list)
        for param_name, _ in module.named_parameters():
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_groups[weight_map[full_name]].append(param_name)
        state_dict = {}
        for file_name, names in tqdm(
            file_groups.items(), desc=f"Loading {prefix}", leave=False
        ):
            file_path = get_valid_read_path(
                os.path.join(self.model_path, file_name),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as file:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = file.get_tensor(full_name)
        return state_dict

    def _load_decoder_if_not_exist(
        self, model: nn.Module, name: str, idx: int
    ) -> nn.Module:
        try:
            decoder = model.get_submodule(name)
            _ = decoder.input_layernorm.weight.device
            return decoder
        except (AttributeError, RuntimeError):
            pass
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRThinkerTextDecoderLayer,
            )

            decoder = Qwen3ASRThinkerTextDecoderLayer(
                config=model.config.text_config, layer_idx=idx
            )
            state_dict = self._get_state_dict(
                decoder, prefix=f"thinker.model.layers.{idx}"
            )
            decoder.load_state_dict(state_dict, strict=True)
            decoder.eval()
            layers: nn.ModuleList = model.model.layers
            if len(layers) <= idx:
                layers.append(decoder)
            else:
                layers[idx] = decoder
        return decoder
