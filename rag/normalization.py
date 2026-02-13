import re

_ws = re.compile(r"\s+")

def _norm_key(k: str) -> str:
    return _ws.sub("", k.replace("\n", "").replace('"', "").replace("'", "")).lower()

def normalize_keys(obj):
    if isinstance(obj, dict):
        return {_norm_key(k): normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    return obj
