import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from gemini_client import create_gemini_client, normalize_gemini_base_url


GEMINI_POOL_HEADER = "X-Gemini-Pool"


def fingerprint_gemini_key(api_key: str) -> str:
    value = (api_key or "").strip()
    if len(value) <= 8:
        return value
    return f"{value[:4]}...{value[-4:]}"


@dataclass
class GeminiErrorClassification:
    state: str
    summary: str
    cooldown_seconds: int = 0


def _extract_retry_delay_seconds(error_text: str) -> int:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", error_text)
    if match:
        return int(match.group(1))
    match = re.search(r"retry in\s+([0-9.]+)s", error_text, re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))))
    return 0


def classify_gemini_error(error_text: str) -> GeminiErrorClassification:
    text = str(error_text or "")
    lower = text.lower()
    if "quota_limit_value" in text and re.search(r"quota_limit_value['\"]?\s*:\s*['\"]0['\"]", text):
        return GeminiErrorClassification("exhausted", "Gemini quota limit is 0 for this key/project")
    if "free_tier" in lower and "limit: 0" in lower:
        return GeminiErrorClassification("exhausted", "Gemini free-tier quota is 0 for this model/key/project")
    if "permission to access the file" in lower or ("file" in lower and "may not exist" in lower):
        return GeminiErrorClassification("file_permission", "Gemini file belongs to a different key/project or no longer exists")
    if "403" in text or "permission_denied" in lower or "invalid api key" in lower:
        return GeminiErrorClassification("disabled", "Gemini key permission denied or invalid")
    if "429" in text or "resource_exhausted" in lower:
        delay = _extract_retry_delay_seconds(text) or 60
        return GeminiErrorClassification("cooldown", "Gemini rate limit exceeded", delay)
    if any(marker in lower for marker in ["500", "503", "unavailable", "timeout", "timed out"]):
        return GeminiErrorClassification("cooldown", "Gemini transient server or network error", 30)
    return GeminiErrorClassification("terminal", "Gemini request failed")


@dataclass
class GeminiEvent:
    fingerprint: str
    operation: str
    status: str
    timestamp: float
    summary: str = ""
    cooldown_seconds: int = 0

    def to_dict(self) -> Dict:
        data = {
            "fingerprint": self.fingerprint,
            "operation": self.operation,
            "status": self.status,
            "timestamp": self.timestamp,
        }
        if self.summary:
            data["summary"] = self.summary
        if self.cooldown_seconds:
            data["cooldownSeconds"] = self.cooldown_seconds
        return data


@dataclass
class GeminiPoolSession:
    pool: "GeminiKeyPool"
    api_key: str
    base_url: str
    fingerprint: str

    @property
    def client(self):
        return create_gemini_client(self.api_key, self.base_url)

    def record_success(self, operation: str = "models.generate_content", response=None) -> None:
        self.pool.record_success(self.fingerprint, operation)

    def record_error(self, exc: Exception, operation: str = "models.generate_content") -> GeminiErrorClassification:
        return self.pool.record_error(self.fingerprint, operation, exc)


@dataclass
class GeminiKeyPool:
    mode: str
    keys: List[str]
    base_url: str = ""
    stats: Dict[str, Dict] = field(default_factory=dict)
    events: List[GeminiEvent] = field(default_factory=list)
    now: Callable[[], float] = time.time
    _cursor: int = 0

    def __post_init__(self) -> None:
        self.mode = (self.mode or "custom_proxy").strip() or "custom_proxy"
        self.keys = [str(key).strip() for key in self.keys if str(key or "").strip()]
        self.base_url = "" if self.mode == "official_pool" else normalize_gemini_base_url(self.base_url)

    def checkout(self, capability: str = "generate") -> GeminiPoolSession:
        if not self.keys:
            raise ValueError("Missing Gemini API Key")
        if self.mode != "official_pool":
            api_key = self.keys[0]
            return GeminiPoolSession(self, api_key, self.base_url, fingerprint_gemini_key(api_key))

        now = self.now()
        candidates = []
        cooling = []
        for offset in range(len(self.keys)):
            index = (self._cursor + offset) % len(self.keys)
            key = self.keys[index]
            fingerprint = fingerprint_gemini_key(key)
            stat = self.stats.get(fingerprint, {})
            state = stat.get("state", "healthy")
            cooldown_until = float(stat.get("cooldownUntil") or 0)
            if state in {"disabled", "exhausted"}:
                continue
            if cooldown_until > now:
                cooling.append(cooldown_until - now)
                continue
            candidates.append((index, key, fingerprint))

        if not candidates:
            if cooling:
                raise RuntimeError(f"All Gemini keys are cooling. Retry in {int(min(cooling))}s.")
            raise RuntimeError("All Gemini keys are disabled or exhausted.")

        index, key, fingerprint = candidates[0]
        self._cursor = (index + 1) % len(self.keys)
        return GeminiPoolSession(self, key, "", fingerprint)

    def record_success(self, fingerprint: str, operation: str) -> None:
        stat = self.stats.setdefault(fingerprint, {})
        stat["state"] = "healthy"
        stat["successes"] = int(stat.get("successes") or 0) + 1
        stat["lastSuccess"] = self.now()
        stat.pop("cooldownUntil", None)
        self.events.append(GeminiEvent(fingerprint, operation, "success", self.now()))

    def record_error(self, fingerprint: str, operation: str, exc: Exception) -> GeminiErrorClassification:
        classification = classify_gemini_error(str(exc))
        stat = self.stats.setdefault(fingerprint, {})
        if classification.state == "cooldown":
            stat["cooldownUntil"] = self.now() + classification.cooldown_seconds
            stat["state"] = "cooling"
            stat["errors429"] = int(stat.get("errors429") or 0) + 1
        elif classification.state in {"disabled", "exhausted"}:
            stat["state"] = classification.state
            if classification.state == "disabled":
                stat["errors403"] = int(stat.get("errors403") or 0) + 1
        elif classification.state == "terminal":
            stat["errorsOther"] = int(stat.get("errorsOther") or 0) + 1
        stat["lastError"] = classification.summary
        self.events.append(
            GeminiEvent(
                fingerprint=fingerprint,
                operation=operation,
                status=classification.state,
                timestamp=self.now(),
                summary=classification.summary,
                cooldown_seconds=classification.cooldown_seconds,
            )
        )
        return classification

    def event_dicts(self) -> List[Dict]:
        return [event.to_dict() for event in self.events]


def _load_pool_config(value: Optional[str]) -> Dict:
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_gemini_pool_config(
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict] = None,
    form: Optional[Dict] = None,
) -> GeminiKeyPool:
    headers = headers or {}
    body = body or {}
    form = form or {}
    pool_value = (
        headers.get(GEMINI_POOL_HEADER)
        or headers.get(GEMINI_POOL_HEADER.lower())
        or body.get("gemini_pool")
        or form.get("gemini_pool")
    )
    config = _load_pool_config(pool_value)
    if config.get("mode") == "official_pool":
        return GeminiKeyPool(
            mode="official_pool",
            keys=config.get("keys") or [],
            stats=config.get("stats") or {},
        )

    api_key = (
        headers.get("X-Gemini-Key")
        or headers.get("x-gemini-key")
        or body.get("api_key")
        or body.get("gemini_key")
        or form.get("api_key")
        or form.get("gemini_key")
        or ""
    )
    base_url = (
        headers.get("X-Gemini-Base-URL")
        or headers.get("x-gemini-base-url")
        or body.get("gemini_base_url")
        or form.get("gemini_base_url")
        or ""
    )
    return GeminiKeyPool(mode="custom_proxy", keys=[api_key], base_url=base_url)
