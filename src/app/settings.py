import os


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def get_api_host() -> str:
    return os.getenv("API_HOST", "0.0.0.0")


def get_api_port() -> int:
    return int(os.getenv("API_PORT", "8000"))


def get_api_reload() -> bool:
    return _as_bool(os.getenv("API_RELOAD"), True)


def get_predict_use_two_stage() -> bool:
    return _as_bool(os.getenv("PREDICT_USE_TWO_STAGE"), True)


def get_predict_normal_threshold() -> float:
    raw = os.getenv("PREDICT_NORMAL_THRESHOLD", "0.50")
    try:
        value = float(raw)
    except ValueError:
        value = 0.50
    return max(0.0, min(1.0, value))
