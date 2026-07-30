import json
from pathlib import Path


def test_example_workflow_is_connected():
    path = Path(__file__).resolve().parents[1] / "examples" / "seedvr2_1_4b_upscale.json"
    workflow = json.loads(path.read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert {node["type"] for node in nodes.values()} == {
        "LoadImage",
        "FLSeedVR2ModelLoader",
        "FLSeedVR2Upscale",
        "SaveImage",
    }
    assert len(links) == 4

    for link_id, source_id, source_slot, target_id, target_slot, link_type in links.values():
        assert nodes[source_id]["outputs"][source_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["type"] == link_type
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
