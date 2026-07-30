import comfyui_fl_seedvr2

from comfyui_fl_seedvr2.nodes import FLSeedVR2ModelLoader, FLSeedVR2Upscale


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
