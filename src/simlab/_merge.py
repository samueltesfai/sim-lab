import copy


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Deep-copies ``base`` first so that multiple merges sharing the same
    ``base`` (e.g. every profile merging against the same resolved
    ``agent.defaults``) never share a nested dict object -- mutating one
    profile's merged settings must never affect another's.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
