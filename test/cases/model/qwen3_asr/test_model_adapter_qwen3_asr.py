# -*- coding: UTF-8 -*-

"""Unit tests for the Qwen3-ASR ModelSlim adapter."""

import json
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.qwen3_asr.model_adapter import Qwen3ASRModelAdapter
from msmodelslim.utils.exception import InvalidDatasetError


def _make_adapter(tmp_path: Path, num_layers: int = 2) -> Qwen3ASRModelAdapter:
    with patch(
        "msmodelslim.model.qwen3_asr.model_adapter.VLMBaseModelAdapter.__init__",
        return_value=None,
    ):
        adapter = Qwen3ASRModelAdapter("Qwen3-ASR-1.7B", str(tmp_path))
    text_config = SimpleNamespace(num_hidden_layers=num_layers)
    adapter.config = SimpleNamespace(
        thinker_config=SimpleNamespace(text_config=text_config)
    )
    adapter.model_type = "Qwen3-ASR-1.7B"
    adapter.model_path = str(tmp_path)
    adapter.trust_remote_code = False
    return adapter


def _fake_qwen_modules(processor_class):
    qwen_asr = ModuleType("qwen_asr")
    core = ModuleType("qwen_asr.core")
    backend = ModuleType("qwen_asr.core.transformers_backend")
    backend.Qwen3ASRProcessor = processor_class
    qwen_asr.core = core
    core.transformers_backend = backend
    return {
        "qwen_asr": qwen_asr,
        "qwen_asr.core": core,
        "qwen_asr.core.transformers_backend": backend,
    }


def test_model_identity_and_fp16_dtype(tmp_path):
    adapter = _make_adapter(tmp_path)
    assert adapter.get_model_pedigree() == "qwen3_asr"
    assert adapter.get_model_type() == "Qwen3-ASR-1.7B"
    assert adapter.get_global_model_torch_dtype() == torch.float16


def test_subgraph_configuration_targets_only_text_decoder(tmp_path):
    adapter = _make_adapter(tmp_path, num_layers=2)
    configs = adapter.get_adapter_config_for_subgraph()
    assert len(configs) == 8
    paths = []
    for config in configs:
        paths.extend(config.mapping.targets)
        if config.mapping.source:
            paths.append(config.mapping.source)
    assert all(path.startswith("model.layers.") for path in paths)
    assert all("audio_tower" not in path for path in paths)


def test_dataset_handler_accepts_audio_only_samples(tmp_path):
    adapter = _make_adapter(tmp_path)
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    processor = MagicMock()
    processor.feature_extractor.sampling_rate = 16000
    processor.apply_chat_template.return_value = "<|audio_pad|>"
    processor.return_value = SimpleNamespace(
        input_ids=torch.ones((1, 2), dtype=torch.long),
        input_features=torch.ones((1, 2, 4)),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        feature_attention_mask=torch.ones((1, 4), dtype=torch.long),
    )
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = processor
    adapter._collect_inputs_to_device = MagicMock(return_value={"ok": True})

    fake_modules = _fake_qwen_modules(fake_processor_class)
    fake_modules["librosa"] = SimpleNamespace(
        load=MagicMock(return_value=(torch.zeros(160).numpy(), 16000))
    )
    with (
        patch.dict("sys.modules", fake_modules),
        patch(
            "msmodelslim.model.qwen3_asr.model_adapter.get_valid_read_path",
            return_value=str(audio_path),
        ),
    ):
        result = adapter.handle_dataset(
            [VlmCalibSample(text="Transcribe.", audio=str(audio_path))],
            DeviceType.CPU,
        )

    assert result == [{"ok": True}]
    call_kwargs = processor.call_args.kwargs
    assert call_kwargs["return_tensors"] == "pt"
    assert call_kwargs["padding"] is True
    messages = processor.apply_chat_template.call_args.args[0]
    assert messages[0] == {"role": "system", "content": "Transcribe."}
    assert messages[1]["content"][0]["type"] == "audio"


def test_dataset_handler_rejects_missing_audio(tmp_path):
    adapter = _make_adapter(tmp_path)
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value.feature_extractor.sampling_rate = 16000
    fake_modules = _fake_qwen_modules(fake_processor_class)
    fake_modules["librosa"] = MagicMock()
    with patch.dict("sys.modules", fake_modules):
        try:
            adapter.handle_dataset([VlmCalibSample(text="Transcribe.")], DeviceType.CPU)
        except InvalidDatasetError:
            pass
        else:
            raise AssertionError("Missing audio must raise InvalidDatasetError")


def test_save_postprocess_restores_thinker_prefix(tmp_path):
    adapter = _make_adapter(tmp_path)
    config = {
        "thinker_config": {
            "dtype": "bfloat16",
            "audio_config": {"dtype": None},
            "text_config": {"dtype": None},
        }
    }
    description = {"model.layers.0.self_attn.q_proj.weight": {"dtype": "int8"}}
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.layers.0.self_attn.q_proj.weight": "part.safetensors"},
    }
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.ones(1)},
        str(tmp_path / "part.safetensors"),
    )

    adapter.ascendv1_save_postprocess(MagicMock(), str(tmp_path))

    saved_index = json.loads(
        (tmp_path / "quant_model_weights.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_index["metadata"] == {"total_size": 4}
    assert list(saved_index["weight_map"]) == [
        "thinker.model.layers.0.self_attn.q_proj.weight"
    ]
    with safe_open(tmp_path / "part.safetensors", framework="pt") as file:
        assert list(file.keys()) == ["thinker.model.layers.0.self_attn.q_proj.weight"]
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved_config["thinker_config"]["dtype"] == "float16"
    assert saved_config["thinker_config"]["audio_config"]["dtype"] == "float16"
    assert saved_config["thinker_config"]["text_config"]["dtype"] == "float16"


def test_weight_map_supports_single_safetensors_file(tmp_path):
    adapter = _make_adapter(tmp_path)
    save_file(
        {"thinker.model.layers.0.weight": torch.ones(1)},
        str(tmp_path / "model.safetensors"),
    )

    assert adapter._get_weight_map() == {
        "thinker.model.layers.0.weight": "model.safetensors"
    }
