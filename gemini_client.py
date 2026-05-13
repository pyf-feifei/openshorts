import os
from typing import Optional

from google import genai
from google.genai import types

DEFAULT_GEMINI_HTTP_TIMEOUT_MS = 20 * 60 * 1000


def normalize_gemini_base_url(base_url: Optional[str] = None) -> str:
    value = (base_url or os.environ.get("GEMINI_BASE_URL", "")).strip()
    return value.rstrip("/")


def resolve_gemini_http_timeout_ms() -> int:
    try:
        value = int(os.environ.get("OPENSHORTS_GEMINI_HTTP_TIMEOUT_MS", str(DEFAULT_GEMINI_HTTP_TIMEOUT_MS)).strip())
    except (TypeError, ValueError):
        value = DEFAULT_GEMINI_HTTP_TIMEOUT_MS
    return max(60_000, value)


def create_gemini_client(api_key: str, base_url: Optional[str] = None) -> genai.Client:
    normalized_base_url = normalize_gemini_base_url(base_url)
    http_options = types.HttpOptions(timeout=resolve_gemini_http_timeout_ms())
    if not normalized_base_url:
        return genai.Client(api_key=api_key, http_options=http_options)

    http_options.base_url = normalized_base_url
    return genai.Client(
        api_key=api_key,
        http_options=http_options,
    )
