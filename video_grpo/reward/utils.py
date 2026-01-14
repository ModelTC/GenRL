import contextlib
from typing import Tuple, Union

import numpy as np
import torch


def prepare_images(images: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Prepare images for reward evaluation.

    Converts torch.Tensor images to numpy arrays in HWC format.
    Handles both 4D (NCHW) and 5D (NFCHW) tensors for images and videos.

    Args:
        images: Input images as torch.Tensor (NCHW or NFCHW) or np.ndarray.

    Returns:
        Tuple of (numpy array in NHWC or NFHWC format, is_video flag).
        - is_video: True if input is 5D (video), False if 4D (image).
    """
    if isinstance(images, torch.Tensor):
        # Handle 4D (NCHW) and 5D (NFCHW) tensors
        if images.dim() == 4 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
            is_video = False
        elif images.dim() == 5 and images.shape[2] == 3:
            images = images.permute(0, 1, 3, 4, 2)  # NFCHW -> NFHWC
            is_video = True
        else:
            raise ValueError(f"Unsupported tensor shape: {images.shape}")
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        # Assume numpy array - determine if video based on dimensions
        if images.ndim == 4:
            is_video = False
        elif images.ndim == 5:
            is_video = True
        else:
            raise ValueError(f"Unsupported array shape: {images.shape}")
        # Ensure uint8 dtype
        if images.dtype != np.uint8:
            images = (images * 255).round().clip(0, 255).astype(np.uint8)

    return images, is_video


def preserve_accelerate_state():
    """Context manager to preserve Accelerate global state during reward init."""
    try:
        from accelerate.state import AcceleratorState, PartialState
    except Exception:
        # If accelerate isn't available, no-op.
        return contextlib.nullcontext()

    class _StatePreserver:
        def __enter__(self):
            self._acc_state = dict(AcceleratorState._shared_state)
            self._partial_state = dict(PartialState._shared_state)

        def __exit__(self, exc_type, exc, tb):
            AcceleratorState._shared_state.clear()
            AcceleratorState._shared_state.update(self._acc_state)
            PartialState._shared_state.clear()
            PartialState._shared_state.update(self._partial_state)
            return False

    return _StatePreserver()
