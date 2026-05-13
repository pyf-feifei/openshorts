import os
from typing import Mapping, Optional


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 1:
        return None
    return parsed


def resolve_thread_count(
    env_name: str,
    *,
    default_cap: int = 4,
    reserve_cores: int = 2,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    env = environ if environ is not None else os.environ
    override = _parse_positive_int(env.get(env_name))
    if override is not None:
        return override

    cpu_count = os.cpu_count() or 2
    half_cpu = max(1, cpu_count // 2)
    after_reserve = max(1, cpu_count - reserve_cores)
    return max(1, min(default_cap, half_cpu, after_reserve))


def resolve_worker_count(env_name: str, *, default: int = 1, environ: Optional[Mapping[str, str]] = None) -> int:
    env = environ if environ is not None else os.environ
    override = _parse_positive_int(env.get(env_name))
    return override if override is not None else max(1, default)
