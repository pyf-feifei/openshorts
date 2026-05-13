import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gemini_client


class GeminiClientTests(unittest.TestCase):
    def test_official_client_uses_configured_http_timeout(self):
        calls = []

        def fake_client(**kwargs):
            calls.append(kwargs)
            return object()

        with patch.object(gemini_client.genai, "Client", side_effect=fake_client), \
            patch.dict(os.environ, {"OPENSHORTS_GEMINI_HTTP_TIMEOUT_MS": "123456"}):
            gemini_client.create_gemini_client("key")

        self.assertEqual("key", calls[0]["api_key"])
        self.assertEqual(123456, calls[0]["http_options"].timeout)

    def test_custom_base_url_client_uses_same_http_timeout(self):
        calls = []

        def fake_client(**kwargs):
            calls.append(kwargs)
            return object()

        with patch.object(gemini_client.genai, "Client", side_effect=fake_client), \
            patch.dict(os.environ, {"OPENSHORTS_GEMINI_HTTP_TIMEOUT_MS": "654321"}):
            gemini_client.create_gemini_client("key", "https://proxy.example.com/")

        self.assertEqual("https://proxy.example.com", calls[0]["http_options"].base_url)
        self.assertEqual(654321, calls[0]["http_options"].timeout)


if __name__ == "__main__":
    unittest.main()
