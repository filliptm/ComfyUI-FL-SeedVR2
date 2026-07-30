import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FL-SeedVR2.appearance",
    nodeCreated(node) {
        if (node.comfyClass.startsWith("FLSeedVR2")) {
            node.color = "#16727c";
            node.bgcolor = "#4F0074";
        }
    },
});
