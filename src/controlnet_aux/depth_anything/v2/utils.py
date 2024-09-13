DEPTH_ANYTHING_V2_MODEL_NAME_DICT = {
    "depth_anything_v2_vits.pth": "depth-anything/Depth-Anything-V2-Small",
    "depth_anything_v2_vitb.pth": "depth-anything/Depth-Anything-V2-Base",
    "depth_anything_v2_vitl.pth": "depth-anything/Depth-Anything-V2-Large",
    "depth_anything_v2_vitg.pth": "depth-anything/Depth-Anything-V2-Giant",
    "depth_anything_v2_metric_vkitti_vitl.pth": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
    "depth_anything_v2_metric_hypersim_vitl.pth": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
}

MODEL_CONFIGS = {
    "depth_anything_v2_vits.pth": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "depth_anything_v2_vitb.pth": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "depth_anything_v2_vitl.pth": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "depth_anything_v2_vitg.pth": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
    "depth_anything_v2_metric_vkitti_vitl.pth": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "depth_anything_v2_metric_hypersim_vitl.pth": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}
