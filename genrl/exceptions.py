"""Custom exceptions for VideoGRPO."""


class ConfigurationError(ValueError):
    """Raised when configuration is invalid."""

    pass


class TrainingError(RuntimeError):
    """Raised when training encounters an error."""

    pass
