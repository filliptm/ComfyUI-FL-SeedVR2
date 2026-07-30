import hashlib
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from safetensors import safe_open

import comfy.model_management
import comfy.utils
import folder_paths

from .model_info import REPO_ID, REVISION, TRANSFORMER, VAE


CHUNK_SIZE = 4 * 1024 * 1024


def _registered_path(artifact):
    path = folder_paths.get_full_path(artifact.folder, artifact.local_name)
    return Path(path) if path is not None else None


def _download_path(artifact):
    path = Path(folder_paths.models_dir) / artifact.folder / artifact.local_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tensor_shape(handle, key):
    if key not in handle.keys():
        return None
    return tuple(handle.get_slice(key).get_shape())


def _validate_header(path, artifact):
    if path.stat().st_size != artifact.size:
        raise ValueError(
            f"{artifact.local_name} has size {path.stat().st_size:,} bytes; expected {artifact.size:,}."
        )

    with safe_open(path, framework="pt", device="cpu") as handle:
        if artifact is TRANSFORMER:
            block5 = "blocks.5.mlp.vid.proj_out.weight"
            block6 = "blocks.6.mlp.vid.proj_out.weight"
            block5_shape = _tensor_shape(handle, block5)
            if block5_shape is None or block5_shape[0] != 3072 or block6 in handle.keys():
                raise ValueError(f"{artifact.local_name} is not the supported six-block SeedVR2 1.4B model.")
            if _tensor_shape(handle, "positive_conditioning") != (58, 5120):
                raise ValueError(
                    f"{artifact.local_name} is missing the ComfyUI positive conditioning tensor."
                )
            if _tensor_shape(handle, "negative_conditioning") != (64, 5120):
                raise ValueError(
                    f"{artifact.local_name} is missing the ComfyUI negative conditioning tensor."
                )
        else:
            key = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"
            shape = _tensor_shape(handle, key)
            if shape is None or shape[0] != 1024:
                raise ValueError(f"{artifact.local_name} is not the supported SeedVR2 VAE.")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_url(artifact):
    remote_name = urllib.parse.quote(artifact.remote_name, safe="/")
    return f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{remote_name}?download=true"


def _download(artifact, target):
    temp = target.with_name(f"{target.name}.download")
    offset = temp.stat().st_size if temp.exists() else 0
    if offset > artifact.size:
        temp.unlink()
        offset = 0

    request = urllib.request.Request(
        _remote_url(artifact),
        headers={
            "Range": f"bytes={offset}-" if offset else "bytes=0-",
            "User-Agent": "ComfyUI-FL-SeedVR2/1.0",
        },
    )
    progress = comfy.utils.ProgressBar(artifact.size)
    progress.update_absolute(offset)

    logging.info("FL SeedVR2: downloading %s", artifact.local_name)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset == artifact.size:
            response = None
        else:
            raise RuntimeError(f"Could not download {artifact.local_name}: HTTP {error.code}.") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not download {artifact.local_name}: {error.reason}.") from error

    if response is not None:
        append = offset > 0 and response.status == 206
        if offset > 0 and not append:
            offset = 0
            progress.update_absolute(0)
        with response, temp.open("ab" if append else "wb") as output:
            downloaded = offset
            while chunk := response.read(CHUNK_SIZE):
                comfy.model_management.throw_exception_if_processing_interrupted()
                output.write(chunk)
                downloaded += len(chunk)
                progress.update_absolute(downloaded)
            output.flush()
            os.fsync(output.fileno())

    if not temp.exists() or temp.stat().st_size != artifact.size:
        actual = temp.stat().st_size if temp.exists() else 0
        raise RuntimeError(
            f"Download of {artifact.local_name} is incomplete: {actual:,} of {artifact.size:,} bytes."
        )
    if _sha256(temp) != artifact.sha256:
        temp.unlink()
        raise RuntimeError(f"Checksum failed for {artifact.local_name}; the incomplete download was removed.")

    _validate_header(temp, artifact)
    os.replace(temp, target)
    logging.info("FL SeedVR2: saved %s", target)


def _ensure_artifact(artifact, download_if_missing):
    path = _registered_path(artifact)
    if path is not None:
        try:
            _validate_header(path, artifact)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid SeedVR2 model file at {path}: {error}") from error
        return path

    target = _download_path(artifact)
    if target.exists():
        try:
            _validate_header(target, artifact)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid SeedVR2 model file at {target}: {error}") from error
        return target

    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing {artifact.local_name}. Put it in ComfyUI/models/{artifact.folder}/ "
            "or enable download_if_missing."
        )

    free = shutil.disk_usage(target.parent).free
    if free < artifact.size + 512 * 1024 * 1024:
        raise RuntimeError(
            f"Not enough free space to download {artifact.local_name}; "
            f"{(artifact.size + 512 * 1024 * 1024) / (1024 ** 3):.1f} GiB is required."
        )
    _download(artifact, target)
    return target


def ensure_model_files(download_if_missing=True):
    model_path = _ensure_artifact(TRANSFORMER, download_if_missing)
    vae_path = _ensure_artifact(VAE, download_if_missing)
    return model_path, vae_path
