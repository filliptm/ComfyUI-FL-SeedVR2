import torch

import comfy.model_detection

from comfyui_fl_seedvr2.modules.loader import _validate_model_keys
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
