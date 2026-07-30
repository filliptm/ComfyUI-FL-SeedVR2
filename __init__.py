from comfy_api.latest import ComfyExtension

from .nodes import FLSeedVR2ModelLoader, FLSeedVR2Upscale


class FLSeedVR2Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            FLSeedVR2ModelLoader,
            FLSeedVR2Upscale,
        ]


async def comfy_entrypoint():
    return FLSeedVR2Extension()


WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "FLSeedVR2Extension", "WEB_DIRECTORY"]
