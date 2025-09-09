def to_int(x, d=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def to_float(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def to_bool(x, d=False):
    if x is None:
        return d
    s = str(x).strip().lower()
    return s in ("1", "true", "on", "yes")
