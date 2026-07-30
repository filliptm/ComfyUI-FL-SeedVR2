import torch

import comfy.utils
import nodes
from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess


def _scaled_dimension(size, multiplier):
    value = max(2, round(size * multiplier))
    return value - value % 2


def _resize(image, width, height):
    image = image.movedim(-1, 1)
    image = comfy.utils.common_upscale(image, width, height, "lanczos", "disabled")
    return image.movedim(1, -1)


def _upscale_one(model, vae, image, width, height, seed, color_correction, tile_size):
    resized = _resize(image, width, height)
    preprocessed = SeedVR2Preprocess.execute(resized).result[0]
    overlap = tile_size // 4
    latent = nodes.VAEEncodeTiled().encode(
        vae,
        preprocessed,
        tile_size=tile_size,
        overlap=overlap,
        temporal_size=64,
        temporal_overlap=8,
    )[0]
    positive, negative = SeedVR2Conditioning.execute(model, latent).result
    sampled = nodes.common_ksampler(
        model,
        seed,
        1,
        1.0,
        "euler",
        "simple",
        positive,
        negative,
        latent,
        denoise=1.0,
    )[0]
    decoded = nodes.VAEDecodeTiled().decode(
        vae,
        sampled,
        tile_size=tile_size,
        overlap=overlap,
        temporal_size=64,
        temporal_overlap=8,
    )[0]
    return SeedVR2PostProcessing.execute(decoded, resized, color_correction).result[0]


def upscale_images(model, vae, images, scale_multiplier, seed, color_correction, tile_size):
    if images.ndim != 4:
        raise ValueError(f"FL SeedVR2 expected IMAGE input with shape (batch, height, width, channels), got {tuple(images.shape)}.")
    if images.shape[0] < 1:
        raise ValueError("FL SeedVR2 expected at least one image.")

    height = _scaled_dimension(images.shape[1], scale_multiplier)
    width = _scaled_dimension(images.shape[2], scale_multiplier)
    outputs = []
    for index in range(images.shape[0]):
        output = _upscale_one(
            model=model,
            vae=vae,
            image=images[index:index + 1],
            width=width,
            height=height,
            seed=(seed + index) & 0xffffffffffffffff,
            color_correction=color_correction,
            tile_size=tile_size,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=0)
