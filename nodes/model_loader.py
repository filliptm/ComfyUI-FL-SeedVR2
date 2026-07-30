from comfy_api.latest import io

from ..modules.downloader import ensure_model_files
from ..modules.loader import load_seedvr2_model, load_seedvr2_vae


class FLSeedVR2ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FLSeedVR2ModelLoader",
            display_name="FL SeedVR2 1.4B Loader",
            category="FL/SeedVR2",
            description="Load the SeedVR2 1.4B transformer and VAE. Missing files are downloaded only when this node executes.",
            search_aliases=["seedvr2", "seed vr2", "1.4b", "upscale", "restoration"],
            inputs=[
                io.Boolean.Input(
                    "download_if_missing",
                    default=True,
                    tooltip="Download the pinned SeedVR2 1.4B transformer and VAE into the standard ComfyUI model folders when missing.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Vae.Output("vae"),
            ],
        )

    @classmethod
    def execute(cls, download_if_missing):
        model_path, vae_path = ensure_model_files(download_if_missing)
        model = load_seedvr2_model(model_path)
        vae = load_seedvr2_vae(vae_path)
        return io.NodeOutput(model, vae)
