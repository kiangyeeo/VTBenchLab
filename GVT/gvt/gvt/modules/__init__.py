def __getattr__(name):
    if name == "GVT":
        from .modeling_gvt import GVT
        return GVT
    if name == "MLLM_LLAVA":
        from .modeling_llava import MLLM_LLAVA
        return MLLM_LLAVA
    if name == "MLLM_MINIGPT4":
        from .modeling_minigpt4 import MLLM_MINIGPT4
        return MLLM_MINIGPT4
    if name == "MLLM_BLIP2":
        from .modeling_blip2 import MLLM_BLIP2
        return MLLM_BLIP2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GVT", "MLLM_LLAVA", "MLLM_MINIGPT4", "MLLM_BLIP2"]
