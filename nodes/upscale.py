from comfy_api.latest import io

from ..modules.pipeline import upscale_images


class FLSeedVR2Upscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FLSeedVR2Upscale",
            display_name="FL SeedVR2 1.4B Upscale",
            category="FL/SeedVR2",
            description="Restore and upscale images with the one-step SeedVR2 1.4B model. Images in a batch are processed independently.",
            search_aliases=["seedvr2", "seed vr2", "1.4b", "upscale", "restore", "super resolution"],
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Image.Input("images"),
                io.Float.Input(
                    "scale_multiplier",
                    default=4.0,
                    min=1.0,
                    max=8.0,
                    step=0.25,
                    tooltip="Output scale. The model is intended for 2x to 4x; 8x is supported but weaker.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                ),
                io.Combo.Input(
                    "color_correction",
                    options=["none", "lab", "wavelet", "adain"],
                    default="none",
                ),
                io.Combo.Input(
                    "vae_tile_size",
                    options=["512", "256"],
                    default="512",
                    advanced=True,
                    tooltip="512 is recommended. Use 256 when VAE encoding or decoding runs out of memory.",
                ),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, images, scale_multiplier, seed, color_correction, vae_tile_size):
        output = upscale_images(
            model=model,
            vae=vae,
            images=images,
            scale_multiplier=scale_multiplier,
            seed=seed,
            color_correction=color_correction,
            tile_size=int(vae_tile_size),
        )
        return io.NodeOutput(output)
