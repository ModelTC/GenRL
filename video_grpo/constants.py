"""Training constants for VideoGRPO."""

# Numerical stability constants
EPSILON = 1e-4  # For std normalization in advantage computation
ADVANTAGE_EPSILON = 1e-6  # For zero advantage handling

# Default values
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_VIDEO_FPS = 16
