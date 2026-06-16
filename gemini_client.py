import os
from typing import Dict, List, Optional

from google import genai
from google.genai import types

DEFAULT_GEMINI_HTTP_TIMEOUT_MS = 20 * 60 * 1000


def normalize_gemini_base_url(base_url: Optional[str] = None) -> str:
    value = (base_url or os.environ.get("GEMINI_BASE_URL", "")).strip()
    value = value.rstrip("/")
    for suffix in ("/v1beta", "/v1"):
        if value.lower().endswith(suffix):
            return value[: -len(suffix)].rstrip("/")
    return value


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


def _model_field(model, field: str) -> str:
    if isinstance(model, dict):
        return str(model.get(field) or "").strip()
    return str(getattr(model, field, "") or "").strip()


def list_gemini_models(api_key: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    client = create_gemini_client(api_key, base_url)
    models = []
    for model in client.models.list():
        name = _model_field(model, "name")
        if not name:
            continue
        model_id = name.removeprefix("models/")
        models.append({
            "id": model_id,
            "name": name,
            "display_name": _model_field(model, "display_name") or model_id,
        })
    return models
