import json
import time
import unittest

from gemini_pool import (
    GeminiKeyPool,
    classify_gemini_error,
    fingerprint_gemini_key,
    parse_gemini_pool_config,
)


class GeminiPoolTests(unittest.TestCase):
    def test_fingerprint_hides_full_key(self):
        self.assertEqual("AIza...TfQY", fingerprint_gemini_key("AIzaSyBOMrVoq6wAwsfDN2nrbvtEMD_ffK0TfQY"))

    def test_parse_legacy_headers_as_custom_proxy(self):
        pool = parse_gemini_pool_config(
            headers={
                "X-Gemini-Key": " key-one ",
                "X-Gemini-Base-URL": " https://proxy.example.com/ ",
            }
        )

        self.assertEqual("custom_proxy", pool.mode)
        self.assertEqual("key-one", pool.keys[0])
        self.assertEqual("https://proxy.example.com", pool.base_url)

    def test_parse_official_pool_from_json_header(self):
        config = {
            "mode": "official_pool",
            "keys": ["key-one", "key-two"],
            "stats": {
                fingerprint_gemini_key("key-one"): {"state": "disabled"},
            },
        }
        pool = parse_gemini_pool_config(headers={"X-Gemini-Pool": json.dumps(config)})

        self.assertEqual("official_pool", pool.mode)
        self.assertEqual(["key-one", "key-two"], pool.keys)
        self.assertEqual("", pool.base_url)
        self.assertEqual("disabled", pool.stats[fingerprint_gemini_key("key-one")]["state"])

    def test_round_robin_skips_disabled_and_cooling_keys(self):
        now = time.time()
        pool = GeminiKeyPool(
            mode="official_pool",
            keys=["key-one", "key-two", "key-three"],
            stats={
                fingerprint_gemini_key("key-one"): {"state": "disabled"},
                fingerprint_gemini_key("key-two"): {"cooldownUntil": now + 60},
            },
            now=lambda: now,
        )

        session = pool.checkout()

        self.assertEqual("key-three", session.api_key)
        self.assertEqual("", session.base_url)

    def test_429_retry_info_sets_cooldown(self):
        classification = classify_gemini_error(
            "429 RESOURCE_EXHAUSTED {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '42s'}"
        )

        self.assertEqual("cooldown", classification.state)
        self.assertEqual(42, classification.cooldown_seconds)

    def test_zero_quota_is_exhausted(self):
        classification = classify_gemini_error(
            "429 RESOURCE_EXHAUSTED {'quota_limit_value': '0', 'quota_location': 'asia-east1'}"
        )

        self.assertEqual("exhausted", classification.state)
        self.assertIn("quota", classification.summary.lower())

    def test_permission_denied_disables_key(self):
        classification = classify_gemini_error("403 PERMISSION_DENIED invalid API key")

        self.assertEqual("disabled", classification.state)


if __name__ == "__main__":
    unittest.main()
