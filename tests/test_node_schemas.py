import comfyui_fl_seedvr2

from comfyui_fl_seedvr2.nodes import FLSeedVR2ModelLoader, FLSeedVR2Upscale
from comfyui_fl_seedvr2.nodes import model_loader


def test_node_schemas_validate():
    for node in (FLSeedVR2ModelLoader, FLSeedVR2Upscale):
        schema = node.define_schema()
        schema.validate()


def test_upscale_schema_keeps_one_step_controls_internal():
    inputs = {value.id for value in FLSeedVR2Upscale.define_schema().inputs}
    assert "steps" not in inputs
    assert "cfg" not in inputs
    assert "sampler" not in inputs
    assert "scheduler" not in inputs
    assert "denoise" not in inputs


def test_frontend_extension_is_registered():
    assert comfyui_fl_seedvr2.WEB_DIRECTORY == "./web"


def test_loader_checks_comfyui_version_before_download(monkeypatch):
    def reject_comfyui():
        raise RuntimeError("unsupported ComfyUI")

    def reject_download(download_if_missing):
        raise AssertionError("download started")

    monkeypatch.setattr(model_loader, "require_supported_comfyui", reject_comfyui)
    monkeypatch.setattr(model_loader, "ensure_model_files", reject_download)

    try:
        FLSeedVR2ModelLoader.execute(True)
    except RuntimeError as error:
        assert str(error) == "unsupported ComfyUI"
    else:
        raise AssertionError("ComfyUI compatibility must be checked before model download")
