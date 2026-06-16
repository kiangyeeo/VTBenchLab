import open_clip


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
    force_quick_gelu = bool(pretrained_cfg and pretrained_cfg.get("quick_gelu"))
    model, _, transform = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        force_quick_gelu=force_quick_gelu,
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
