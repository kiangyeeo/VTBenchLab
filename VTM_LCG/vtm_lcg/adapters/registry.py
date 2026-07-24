from __future__ import annotations

from typing import Any, Mapping

from .base import SequenceTokenizerAdapter
from .metaclip import MetaClipL14Adapter
from .openai_clip import OpenAIClipL14Adapter


ADAPTERS = {
    "openai_clip": OpenAIClipL14Adapter,
    "metaclip": MetaClipL14Adapter,
}


def create_adapter(
    tokenizer_config: Mapping[str, Any],
    preprocess_config: Mapping[str, Any],
) -> SequenceTokenizerAdapter:
    adapter_name = str(tokenizer_config.get("adapter", ""))
    try:
        adapter_class = ADAPTERS[adapter_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported adapter {adapter_name!r}; choices={sorted(ADAPTERS)}"
        ) from error
    return adapter_class(tokenizer_config, preprocess_config)

