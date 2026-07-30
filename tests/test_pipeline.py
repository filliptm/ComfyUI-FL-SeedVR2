import torch

from comfyui_fl_seedvr2.modules import pipeline


def test_scaled_dimensions_are_even():
    assert pipeline._scaled_dimension(101, 1.0) == 100
    assert pipeline._scaled_dimension(128, 4.0) == 512


def test_batches_are_processed_independently(monkeypatch):
    calls = []

    def upscale_one(model, vae, image, width, height, seed, color_correction, tile_size):
        calls.append((tuple(image.shape), width, height, seed, color_correction, tile_size))
        return torch.full((1, height, width, 3), float(seed))

    monkeypatch.setattr(pipeline, "_upscale_one", upscale_one)
    images = torch.zeros(2, 8, 10, 3)
    output = pipeline.upscale_images(
        model=object(),
        vae=object(),
        images=images,
        scale_multiplier=2.0,
        seed=5,
        color_correction="none",
        tile_size=256,
    )

    assert output.shape == (2, 16, 20, 3)
    assert calls == [
        ((1, 8, 10, 3), 20, 16, 5, "none", 256),
        ((1, 8, 10, 3), 20, 16, 6, "none", 256),
    ]


def test_seedvr2_postprocess_preserves_rgba_alpha():
    decoded = torch.zeros(1, 8, 10, 3)
    resized = torch.zeros(1, 8, 10, 4)
    resized[..., 3] = 0.25

    output = pipeline.SeedVR2PostProcessing.execute(decoded, resized, "none").result[0]

    assert output.shape == (1, 8, 10, 4)
    assert torch.all(output[..., 3] == 0.25)


def test_invalid_image_rank_fails():
    try:
        pipeline.upscale_images(object(), object(), torch.zeros(8, 8, 3), 2.0, 0, "none", 512)
    except ValueError as error:
        assert "shape (batch, height, width, channels)" in str(error)
    else:
        raise AssertionError("invalid IMAGE tensors must fail")
