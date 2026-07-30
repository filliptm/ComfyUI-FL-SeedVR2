import hashlib
import io

import torch
from safetensors.torch import save

from comfyui_fl_seedvr2.modules import downloader
from comfyui_fl_seedvr2.modules.model_info import ModelArtifact


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_is_verified_and_moved_atomically(tmp_path, monkeypatch):
    payload = save({
        "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight": torch.empty(1024, 1, 1, 1, 1),
    })
    artifact = ModelArtifact(
        folder="vae",
        remote_name="vae.safetensors",
        local_name="vae.safetensors",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, timeout: Response(payload))

    target = tmp_path / artifact.local_name
    downloader._download(artifact, target)

    assert target.read_bytes() == payload
    assert not target.with_name(f"{target.name}.download").exists()


def test_invalid_existing_file_is_not_overwritten(tmp_path, monkeypatch):
    artifact = ModelArtifact(
        folder="vae",
        remote_name="vae.safetensors",
        local_name="vae.safetensors",
        size=10,
        sha256="0" * 64,
    )
    path = tmp_path / artifact.local_name
    path.write_bytes(b"invalid")
    monkeypatch.setattr(downloader, "_registered_path", lambda value: path)

    try:
        downloader._ensure_artifact(artifact, True)
    except RuntimeError as error:
        assert "Invalid SeedVR2 model file" in str(error)
    else:
        raise AssertionError("invalid existing files must not be overwritten")

    assert path.read_bytes() == b"invalid"
