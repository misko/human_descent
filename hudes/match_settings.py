"""Small, inference-free policy for user-configurable match settings."""

from dataclasses import replace

DIRECTIONS_PER_SET = 6
MAX_DIMENSIONS = 6006


def valid_dimension_limit(value):
    return value is None or (
        type(value) is int
        and DIRECTIONS_PER_SET <= value <= MAX_DIMENSIONS
        and value % DIRECTIONS_PER_SET == 0
    )


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) - {"batchSize", "maxDimensions"}:
        raise ValueError("Invalid match settings")
    if "batchSize" in settings and (
        type(settings["batchSize"]) is not int or not 1 <= settings["batchSize"] <= 512
    ):
        raise ValueError("Batch size must be an integer from 1 to 512")
    if not valid_dimension_limit(settings.get("maxDimensions")):
        raise ValueError(
            "Dimension limit must be a multiple of 6 from 6 to 6006, or unlimited"
        )
    return settings


def configure_preset(preset, settings):
    validate_settings(settings)
    return replace(
        preset,
        batch_size=settings.get("batchSize", preset.batch_size),
        max_dimensions=settings.get("maxDimensions", preset.max_dimensions),
    )


def public_settings(preset):
    return {"batchSize": preset.batch_size, "maxDimensions": preset.max_dimensions}


def switches_remaining(preset, generation):
    if preset.max_dimensions is None:
        return None
    return max(0, preset.max_dimensions // DIRECTIONS_PER_SET - 1 - generation)
