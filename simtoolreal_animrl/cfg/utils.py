"""Small configuration helpers retained from AnimRL's Python cfg style."""


def config_to_dict(obj):
    """Recursively convert an instantiated config class to plain values."""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        if isinstance(value, (list, tuple)):
            result[key] = [config_to_dict(item) for item in value]
        else:
            result[key] = config_to_dict(value)
    return result


def update_config_from_dict(obj, values, strict=True):
    """Recursively apply a saved configuration to an instantiated config."""
    if not isinstance(values, dict):
        raise TypeError("Configuration values must be a dictionary")
    for key, value in values.items():
        if not hasattr(obj, key):
            if strict:
                raise KeyError("Unknown configuration key: {}".format(key))
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dict__") and isinstance(value, dict):
            update_config_from_dict(current, value, strict=strict)
        else:
            setattr(obj, key, value)
    return obj
