import torch

import comfy.model_detection

from comfyui_fl_seedvr2.modules import loader
from comfyui_fl_seedvr2.modules.loader import _validate_model_keys, require_supported_comfyui
from comfyui_fl_seedvr2.modules.model_info import SEEDVR2_1_4B_CONFIG


def test_explicit_six_block_config_matches_seedvr2():
    state_dict = {
        "positive_conditioning": torch.empty(58, 5120),
        "negative_conditioning": torch.empty(64, 5120),
    }
    config = comfy.model_detection.model_config_from_unet_config(
        SEEDVR2_1_4B_CONFIG,
        state_dict,
    )

    assert type(config).__name__ == "SeedVR2"
    assert config.unet_config["num_layers"] == 6
    assert config.unet_config["mm_layers"] == 6


def test_plain_checkpoint_without_conditioning_does_not_match(monkeypatch):
    monkeypatch.setattr(comfy.model_detection.logging, "error", lambda message: None)
    assert comfy.model_detection.model_config_from_unet_config(
        SEEDVR2_1_4B_CONFIG,
        {},
    ) is None


def test_model_key_validation_is_strict():
    wrapper = torch.nn.Module()
    wrapper.diffusion_model = torch.nn.Linear(2, 2)

    state_dict = {
        "weight": torch.empty(2, 2),
        "bias": torch.empty(2),
    }
    _validate_model_keys(wrapper, state_dict)

    state_dict["extra"] = torch.empty(1)
    try:
        _validate_model_keys(wrapper, state_dict)
    except ValueError as error:
        assert "unexpected keys" in str(error)
    else:
        raise AssertionError("unexpected checkpoint keys must fail")


def test_old_comfyui_version_fails_before_model_loading(monkeypatch):
    monkeypatch.setattr(loader, "COMFYUI_VERSION", "0.27.1")

    try:
        require_supported_comfyui()
    except RuntimeError as error:
        assert "requires ComfyUI 0.28.0 or newer" in str(error)
        assert "do not need to be downloaded again" in str(error)
    else:
        raise AssertionError("unsupported ComfyUI versions must fail")


def test_minimum_comfyui_version_is_supported(monkeypatch):
    monkeypatch.setattr(loader, "COMFYUI_VERSION", "0.28.0")
    require_supported_comfyui()


def test_text_branch_mismatch_reports_comfyui_update():
    wrapper = torch.nn.Module()
    wrapper.diffusion_model = torch.nn.Linear(2, 2)
    state_dict = {
        "weight": torch.empty(2, 2),
        "bias": torch.empty(2),
        "blocks.0.attn.proj_out.txt.bias": torch.empty(2),
    }

    try:
        _validate_model_keys(wrapper, state_dict)
    except RuntimeError as error:
        assert "incompatible SeedVR2 architecture" in str(error)
        assert "checkpoint is valid" in str(error)
        assert "does not need to be downloaded again" in str(error)
    else:
        raise AssertionError("legacy SeedVR2 text-key mismatches must fail with update guidance")
