import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app


class YouTubeCookiesVerifyTests(unittest.TestCase):
    def test_normalize_accepts_export_with_fields_split_across_lines(self):
        raw = "\n".join([
            "# Netscape HTTP Cookie File",
            ".youtube.com",
            "TRUE",
            "/",
            "TRUE",
            "1811082292",
            "SID",
            "sid-value",
            ".youtube.com",
            "TRUE",
            "/",
            "TRUE",
            "1798167371",
            "LOGIN_INFO",
            "login-value",
        ])

        normalized = app.normalize_youtube_cookies(raw)

        self.assertIn(".youtube.com\tTRUE\t/\tTRUE\t1811082292\tSID\tsid-value", normalized)
        self.assertIn(".youtube.com\tTRUE\t/\tTRUE\t1798167371\tLOGIN_INFO\tlogin-value", normalized)

    def test_verify_reports_successful_extract_without_downloading(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.extract_info.return_value = {
            "title": "Demo",
            "duration": 12,
            "formats": [{"vcodec": "avc1.64001F"}],
        }

        with tempfile.NamedTemporaryFile() as cookies_file, \
            patch.object(app.yt_dlp, "YoutubeDL", return_value=ydl):
            result = app.verify_youtube_cookies_for_url("https://youtube.test/watch?v=1", cookies_file.name)

        self.assertTrue(result["ok"])
        self.assertEqual("Demo", result["title"])
        ydl.extract_info.assert_called_once_with("https://youtube.test/watch?v=1", download=False)

    def test_verify_reports_cookie_or_bot_failure(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.extract_info.side_effect = Exception("Sign in to confirm you are not a bot")

        with tempfile.NamedTemporaryFile() as cookies_file, \
            patch.object(app.yt_dlp, "YoutubeDL", return_value=ydl):
            result = app.verify_youtube_cookies_for_url("https://youtube.test/watch?v=1", cookies_file.name)

        self.assertFalse(result["ok"])
        self.assertIn("cookies", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
