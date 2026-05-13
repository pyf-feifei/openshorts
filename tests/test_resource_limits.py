import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import main
import resource_limits


class ResourceLimitTests(unittest.TestCase):
    def test_resolve_thread_count_uses_env_override(self):
        self.assertEqual(
            6,
            resource_limits.resolve_thread_count(
                "OPENSHORTS_TEST_THREADS",
                environ={"OPENSHORTS_TEST_THREADS": "6"},
            ),
        )

    def test_resolve_thread_count_uses_cpu_count_when_env_is_missing_or_invalid(self):
        with patch("os.cpu_count", return_value=8):
            self.assertEqual(
                4,
                resource_limits.resolve_thread_count(
                    "OPENSHORTS_TEST_THREADS",
                    environ={"OPENSHORTS_TEST_THREADS": "invalid"},
                ),
            )

        with patch("os.cpu_count", return_value=2):
            self.assertEqual(
                1,
                resource_limits.resolve_thread_count(
                    "OPENSHORTS_TEST_THREADS",
                    environ={},
                ),
            )

    def test_resolve_worker_count_defaults_to_one_unless_overridden(self):
        self.assertEqual(1, resource_limits.resolve_worker_count("OPENSHORTS_TEST_WORKERS", environ={}))
        self.assertEqual(
            3,
            resource_limits.resolve_worker_count(
                "OPENSHORTS_TEST_WORKERS",
                environ={"OPENSHORTS_TEST_WORKERS": "3"},
            ),
        )

    def test_transcribe_video_uses_limited_cpu_threads_and_workers(self):
        created = {}

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)

            def transcribe(self, *args, **kwargs):
                return [], type("Info", (), {"language": "en", "language_probability": 1.0})()

        with patch.dict(os.environ, {
            "OPENSHORTS_WHISPER_CPU_THREADS": "2",
            "OPENSHORTS_WHISPER_WORKERS": "1",
        }), patch.dict(sys.modules, {"faster_whisper": type("Module", (), {"WhisperModel": FakeWhisperModel})}):
            result = main.transcribe_video("video.mp4")

        self.assertEqual("en", result["language"])
        self.assertEqual(2, created["cpu_threads"])
        self.assertEqual(1, created["num_workers"])

    def test_transcribe_video_defaults_threads_from_cpu_count(self):
        created = {}

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)

            def transcribe(self, *args, **kwargs):
                return [], type("Info", (), {"language": "en", "language_probability": 1.0})()

        with patch.dict(os.environ, {}, clear=True), \
            patch("os.cpu_count", return_value=8), \
            patch.dict(sys.modules, {"faster_whisper": type("Module", (), {"WhisperModel": FakeWhisperModel})}):
            result = main.transcribe_video("video.mp4")

        self.assertEqual("en", result["language"])
        self.assertEqual(4, created["cpu_threads"])
        self.assertEqual(1, created["num_workers"])

    def test_transcribe_video_keeps_small_cpu_defaults_conservative(self):
        created = {}

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)

            def transcribe(self, *args, **kwargs):
                return [], type("Info", (), {"language": "en", "language_probability": 1.0})()

        with patch.dict(os.environ, {}, clear=True), \
            patch("os.cpu_count", return_value=2), \
            patch.dict(sys.modules, {"faster_whisper": type("Module", (), {"WhisperModel": FakeWhisperModel})}):
            main.transcribe_video("video.mp4")

        self.assertEqual(1, created["cpu_threads"])
        self.assertEqual(1, created["num_workers"])


if __name__ == "__main__":
    unittest.main()
