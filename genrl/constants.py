"""Training constants for VideoGRPO."""

# Numerical stability constants
EPSILON = 1e-4  # For std normalization in advantage computation
ADVANTAGE_EPSILON = 1e-6  # For zero advantage handling

# Default values
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_VIDEO_FPS = 16

# Seeding strategy
# Multiplier used when deriving per-epoch / per-batch seeds to avoid collisions.
SEED_EPOCH_STRIDE = 10_000
