from dataclasses import dataclass


REPO_ID = "lvladikov/SeedVR2-1.4B"
REVISION = "7694e0f361dde8521668e9f8e1d242a1ee90035a"


@dataclass(frozen=True)
class ModelArtifact:
    folder: str
    remote_name: str
    local_name: str
    size: int
    sha256: str


TRANSFORMER = ModelArtifact(
    folder="diffusion_models",
    remote_name="comfyui/seedvr2_distill_6L_1.4B_sharp_fp16_comfyui.safetensors",
    local_name="seedvr2_distill_6L_1.4B_sharp_fp16_comfyui.safetensors",
    size=2_886_486_040,
    sha256="bdce240cdbfcfa90ec8ad8b535667b93507cbaa0fb96b2970cc10b1389cc1080",
)

VAE = ModelArtifact(
    folder="vae",
    remote_name="ema_vae_fp16.safetensors",
    local_name="seedvr2_ema_vae_fp16.safetensors",
    size=501_324_814,
    sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
)

SEEDVR2_1_4B_CONFIG = {
    "image_model": "seedvr2",
    "vid_dim": 3072,
    "heads": 24,
    "num_layers": 6,
    "mm_layers": 6,
    "norm_eps": 1e-5,
    "rope_type": "rope3d",
    "rope_dim": 64,
    "mlp_type": "normal",
}
