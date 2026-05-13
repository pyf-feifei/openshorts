# Gemini Official Key Pool Design

## Context

OpenShorts currently sends a single `X-Gemini-Key` and optional `X-Gemini-Base-URL` from the browser to the FastAPI backend. The backend creates a direct `google.genai.Client` for each feature call. This works for one official key or one proxy, but it does not handle multiple official Gemini keys, per-key health, quota cooling, or usage/error statistics.

The new design adds an official Gemini multi-key mode that covers every OpenShorts Gemini feature while preserving the existing custom proxy mode.

## Constraints From Gemini

Gemini API rate limits are applied per project, not per API key. Multiple keys from the same Google Cloud or AI Studio project may share the same RPM, TPM, and RPD limits, so key rotation cannot be treated as guaranteed quota multiplication.

Gemini Files API has its own lifecycle and access boundary. Large videos are uploaded as files, then polled until `ACTIVE`, then referenced by URI in `generate_content`. A file URI uploaded with one key/project may not be accessible from another key/project. For video workflows, the same selected key must be used for upload, file polling, and generation. If the flow retries with another key, it must upload the file again.

Files API limits matter for OpenShorts commentary analysis: single files can be large, project file storage is finite, and files expire automatically. OpenShorts should avoid unnecessary reuploads unless a selected key fails in a way that merits failover.

## User Experience

Settings will expose a Gemini access mode selector:

- `Custom proxy`: current behavior. One API key plus optional Gemini Base URL. Requests use the configured Base URL.
- `Official key pool`: no Base URL. The app uses official Gemini API only and accepts multiple Gemini API keys.

In official key pool mode, the settings panel shows a key list. Each key is displayed by fingerprint only, for example `AIza...TfQY`, never in full after entry. Users can add keys in bulk by pasting one key per line, remove keys, disable or re-enable a key, and clear statistics.

Each key row displays:

- State: healthy, cooling, disabled, or exhausted
- Success count
- 429 count
- 403 or auth error count
- timeout or 5xx count
- Last success time
- Last error summary
- Cooldown remaining time, when applicable

The existing privacy promise remains: keys are stored in the browser and are sent to the backend only for the current request. Backend job status may return key fingerprints and usage events, but not raw keys.

## Request Contract

The frontend sends a new Gemini configuration payload for every request that uses Gemini.

Headers remain compatible for existing single-key callers:

- `X-Gemini-Key`: single key fallback
- `X-Gemini-Base-URL`: custom proxy fallback

New official key pool mode sends one JSON header or form field containing:

```json
{
  "mode": "official_pool",
  "keys": ["..."],
  "stats": {
    "fingerprint": {
      "state": "healthy",
      "cooldownUntil": 0,
      "successes": 0,
      "errors429": 0,
      "errors403": 0,
      "errors5xx": 0,
      "lastError": ""
    }
  }
}
```

For JSON endpoints, the pool can be sent in the request body or a compact header. For multipart upload endpoints, it should be sent as a form field. The backend normalizes both into the same internal `GeminiKeyPool`.

## Backend Architecture

Add a `gemini_pool.py` module with these responsibilities:

- Parse the incoming Gemini configuration.
- Normalize single-key and pool modes into one internal interface.
- Fingerprint keys without exposing raw values.
- Select a key using health-first round robin.
- Create official Gemini clients with no Base URL in pool mode.
- Create proxy clients only in custom proxy mode.
- Classify Gemini errors into retryable, cooldown, exhausted, disabled, or terminal.
- Record usage events that can be returned in job status.

The core interface should look like this conceptually:

```python
pool = GeminiKeyPool.from_request(...)
with pool.checkout(capability="files+generate") as session:
    client = session.client
    # use client for this Gemini operation
    session.record_success(response)
```

For text-only requests, a failed key can be retried on the next healthy key without extra cleanup.

For Files API workflows, one checkout owns the complete upload/poll/generate sequence. If the selected key fails before upload completes, the next key can retry from the upload step. If it fails after upload, the next key must also retry from the upload step because the previous file URI is not assumed transferable.

## Error Classification

The pool tracks these classes:

- Success: update success count, token usage if available, last success time.
- `429` with `RetryInfo`: cool the key until the retry delay expires.
- `429` without `RetryInfo`: short cooldown, default 60 seconds.
- `quota_limit_value: 0`, RPD exhausted, or free-tier limit 0: mark exhausted or long-cooldown. This key is skipped until manual reset or the next configured day boundary.
- `403`, invalid key, permission denied, API disabled: disable the key until the user re-enables it.
- `400` unsupported model or unsupported media: terminal for the request, not a key health failure unless the message clearly identifies key/project access.
- `5xx`, network timeout, transient transport error: short cooldown and retry another key.
- JSON parsing or model response shape errors: request failure, not key health.

The backend should preserve detailed error text in logs but send concise UI-safe summaries in job status.

## Load Balancing

Selection order is health-first round robin:

1. Skip disabled keys.
2. Skip keys whose cooldown has not expired.
3. Prefer keys with no recent errors.
4. Round-robin among remaining candidates.
5. If all keys are cooling, return a clear error with the shortest remaining cooldown.
6. If all keys are disabled or exhausted, return a clear error asking the user to add or re-enable a key.

This avoids pure random selection because random can repeatedly hit a bad key and makes statistics harder to reason about.

## Coverage Across Features

All Gemini call sites should move through the same pool interface:

- Core short-video editing in `app.py` and `editor.py`
- Commentary remix in `commentary.py`
- Thumbnail studio in `thumbnail.py`
- SaaS shorts in `saasshorts.py`
- Translation and clip actions that call Gemini through result cards or backend endpoints

The first implementation should keep endpoint behavior compatible with the existing single-key headers while adding pool-aware parsing.

## Frontend Data Model

Local storage keys:

- `gemini_access_mode`: `custom_proxy` or `official_pool`
- `gemini_key`: existing single-key value for compatibility
- `gemini_base_url`: existing proxy URL for custom proxy mode
- `gemini_key_pool_v1`: encrypted or obfuscated JSON array of official keys and user labels
- `gemini_key_pool_stats_v1`: JSON statistics keyed by fingerprint

The current encryption helper is weak browser-local obfuscation, not real security. UI text should continue to state that keys are stored only in the browser and are sent to the backend only for requests.

## Job Status And Statistics

Backend jobs should include Gemini pool events:

```json
{
  "gemini_events": [
    {
      "fingerprint": "AIza...TfQY",
      "operation": "files.upload",
      "status": "success",
      "timestamp": 1777731949
    },
    {
      "fingerprint": "AIza...TfQY",
      "operation": "models.generate_content",
      "status": "429",
      "cooldownSeconds": 42,
      "summary": "Rate limit exceeded"
    }
  ]
}
```

The frontend merges these events into local stats after polling each job status. This keeps backend stateless and consistent with existing privacy behavior.

## Testing

Backend tests should cover:

- Single-key custom proxy mode still creates a client with Base URL.
- Official pool mode never passes a Base URL.
- Round-robin selection skips disabled and cooling keys.
- 429 with retry delay sets cooldown.
- `quota_limit_value: 0` is treated as exhausted/long cooldown.
- 403 permission errors disable the key.
- Text operation failover retries another key.
- Files operation failover reuploads with the new key rather than reusing the old URI.

Frontend tests should cover:

- Settings mode switch preserves existing single-key/proxy behavior.
- Adding multiple keys stores fingerprints and raw keys correctly in browser storage.
- Gemini headers or payloads are built correctly for custom proxy and official pool modes.
- Job events update local statistics.
- Disabled/cooling/exhausted states render accurately.

## Rollout

Implement in three phases:

1. Add data model, parser, backend pool selection, and tests while preserving existing single-key behavior.
2. Migrate all Gemini backend call sites to the pool interface.
3. Add the settings UI, job event display, and local statistics merge.

This order keeps the risky backend behavior testable before exposing the full UI.
