"""Text embedding utilities for trainers."""

from typing import Any, List
import torch
from genrl.diffusers_patch.wan_prompt_embedding import encode_prompt


def wan_compute_text_embeddings(
    prompt: str | List[str],
    text_encoders: List[Any],
    tokenizers: List[Any],
    max_sequence_length: int,
    device: torch.device,
) -> torch.FloatTensor:
    """Encode prompts into embeddings on target device.

    Args:
        prompt: String or list of strings to encode.
        text_encoders: Sequence of text encoder modules.
        tokenizers: Sequence of tokenizers aligned with encoders.
        max_sequence_length: Max token length for encoding.
        device: Target device for embeddings.

    Returns:
        Tensor of encoded prompt embeddings on `device`.
    """
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds
