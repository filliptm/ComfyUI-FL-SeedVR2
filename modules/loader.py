import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils

from .model_info import SEEDVR2_1_4B_CONFIG


def _validate_model_keys(model, state_dict):
    expected = set(model.diffusion_model.state_dict())
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"{len(missing)} missing keys, first: {missing[0]}")
        if unexpected:
            details.append(f"{len(unexpected)} unexpected keys, first: {unexpected[0]}")
        raise ValueError(f"SeedVR2 1.4B checkpoint does not match the six-block architecture: {'; '.join(details)}.")


def load_seedvr2_model(model_path):
    model_path = str(model_path)
    state_dict = comfy.utils.load_torch_file(model_path)
    model_config = comfy.model_detection.model_config_from_unet_config(
        SEEDVR2_1_4B_CONFIG,
        state_dict,
    )
    if model_config is None:
        raise RuntimeError(
            "This checkpoint is not the ComfyUI SeedVR2 1.4B variant. "
            "Use seedvr2_distill_6L_1.4B_sharp_fp16_comfyui.safetensors."
        )

    parameters = comfy.utils.calculate_parameters(state_dict)
    weight_dtype = comfy.utils.weight_dtype(state_dict)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    supported_dtypes = list(model_config.supported_inference_dtypes)
    unet_dtype = comfy.model_management.unet_dtype(
        model_params=parameters,
        supported_dtypes=supported_dtypes,
        weight_dtype=weight_dtype,
    )
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype,
        load_device,
        supported_dtypes,
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    model = model_config.get_model(state_dict, "")
    _validate_model_keys(model, state_dict)

    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
    )
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(state_dict, "", assign=patcher.is_dynamic())
    patcher.cached_patcher_init = (load_seedvr2_model, (model_path,))
    return patcher


def load_seedvr2_vae(vae_path):
    vae_path = str(vae_path)
    state_dict, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
    vae.throw_exception_if_invalid()
    vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (vae_path, metadata, None))
    return vae
