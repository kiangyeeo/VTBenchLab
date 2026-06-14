import os
from .clip_encoder import CLIPVisionTower
import sys
import torch
# Add the parent directory of Light_Align to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from light_align.model.vision_model import ImageEmbedding
from light_align.model.sail_model import AlignmentLayer

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs), None
    # Load custom SAIL - Dinov2 with alignment layer
    elif vision_tower.startswith("facebook"):
        vision_tower = ImageEmbedding(vision_tower)
        vlhead = AlignmentLayer(
            vision_dimesion=vision_tower.model.config.hidden_size * 2, 
            target_dimension=vision_tower_cfg.target_dimension, 
            linear_type=vision_tower_cfg.linear_type,
            only_vision=True
            )
        assert vision_tower_cfg.vlhead_weights_path is not None
        print(f"Loading vlhead weights from {vision_tower_cfg.vlhead_weights_path}")
        weights = torch.load(vision_tower_cfg.vlhead_weights_path)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        new_weights = {}
        for k, v in weights.items():
            if 'model.vlhead' in k:
                new_weights[k.replace('model.vlhead.', '')] = v
            else:   
                new_weights[k] = v
        # load only vision weights
        weights = {k: v for k, v in new_weights.items() if ('text' not in k)}
        msg = vlhead.load_state_dict(weights, strict=False)
        print(f"vlhead weights loaded status: {msg}")
        return vision_tower, vlhead
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
