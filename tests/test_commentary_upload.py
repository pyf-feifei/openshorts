import os
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

import app


class ImmediateThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class CommentaryUploadTests(unittest.TestCase):
    def test_commentary_form_defaults_to_two_to_four_target_duration(self):
        req = app.commentary_request_from_form({})

        self.assertEqual("two_to_four", req.target_duration)

    def test_openai_edit_first_stage_mapping_tracks_real_chain(self):
        self.assertEqual(
            ("openai_source_frames", "准备全片抽帧分析"),
            app.resolve_commentary_stage(
                "Generating OpenAI-compatible edit-first commentary: full-video analysis, intermediate visual cut, then final narration on the edited video...",
                None,
            ),
        )
        self.assertEqual(
            ("openai_source_analysis", "全片多模态分析"),
            app.resolve_commentary_stage(
                "OpenAI-compatible multimodal visual analysis batch 1/3...",
                "openai_source_frames",
            ),
        )
        self.assertEqual(
            ("openai_edit", "生成中间视觉剪辑"),
            app.resolve_commentary_stage(
                "OpenAI-compatible edit-first flow locked visual cut: 6 source ranges, 142.0s edited video.",
                "openai_source_analysis",
            ),
        )
        self.assertEqual(
            ("openai_edited_frames", "抽取剪辑片视觉帧"),
            app.resolve_commentary_stage(
                "Extracted OpenAI-compatible analysis frames: 8/24",
                "openai_edited_frames",
            ),
        )
        self.assertEqual(
            ("openai_edited_analysis", "剪辑片多模态分析"),
            app.resolve_commentary_stage(
                "OpenAI-compatible multimodal visual analysis batch 2/3 done (2/3)",
                "openai_edited_frames",
            ),
        )
        self.assertEqual(
            ("openai_final_script", "基于剪辑片写最终解说"),
            app.resolve_commentary_stage(
                "OpenAI-compatible model is writing commentary script from transcript and visual timeline...",
                "openai_edited_analysis",
            ),
        )

    def test_gemini_models_endpoint_requires_gemini_access(self):
        client = TestClient(app.app)

        response = client.get("/api/settings/gemini-models")

        self.assertEqual(400, response.status_code, response.text)
        self.assertEqual("Missing X-Gemini-Key header", response.json()["detail"])

    def test_gemini_models_endpoint_lists_models_for_configured_access(self):
        returned_models = [
            {"id": "gemini-2.5-flash", "name": "models/gemini-2.5-flash", "display_name": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "name": "models/gemini-2.5-pro", "display_name": "Gemini 2.5 Pro"},
        ]
        with patch.object(app, "list_gemini_models", return_value=returned_models) as list_models:
            client = TestClient(app.app)
            response = client.get(
                "/api/settings/gemini-models",
                headers={
                    "X-Gemini-Key": "gemini-key",
                    "X-Gemini-Base-URL": "https://proxy.example.com/",
                },
            )

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual({"models": returned_models}, response.json())
        list_models.assert_called_once_with("gemini-key", "https://proxy.example.com")

    def test_commentary_generate_accepts_uploaded_video_file(self):
        with tempfile.TemporaryDirectory() as uploads_dir, \
            tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "UPLOAD_DIR", uploads_dir), \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "final.mp4"),
                "video_url": "/videos/job/final.mp4",
                "title": "Uploaded Commentary",
            }) as generate:

            app.commentary_jobs.clear()
            client = TestClient(app.app)
            response = client.post(
                "/api/commentary/generate",
                headers={"X-Gemini-Key": "gemini-key"},
                data={
                    "language": "zh",
                    "style": "custom",
                    "custom_style_prompt": "用第一人称紧张整活口吻，短句优先。",
                    "target_duration": "medium",
                    "analysis_mode": "video",
                    "gemini_model": "gemini-2.5-pro",
                    "tts_provider": "edge",
                    "original_audio_volume": "0.12",
                    "pause_original_audio_volume": "0.7",
                    "background_music_enabled": "true",
                    "background_music_track": "aodebiao_caravan",
                    "background_music_volume": "0.22",
                    "subtitles": "true",
                    "aspect_mode": "auto",
                },
                files={"file": ("demo.mp4", b"fake-video", "video/mp4")},
            )
            job_id = response.json()["job_id"]
            task_path = os.path.join(output_dir, job_id, "commentary_task.json")
            with open(task_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)

        self.assertEqual(200, response.status_code, response.text)
        kwargs = generate.call_args.kwargs
        self.assertEqual("file", kwargs["source_type"])
        self.assertTrue(kwargs["source"].endswith("demo.mp4"))
        self.assertEqual("video", kwargs["analysis_mode"])
        self.assertEqual("gemini-2.5-pro", kwargs["gemini_model"])
        self.assertEqual("custom", kwargs["style"])
        self.assertEqual("用第一人称紧张整活口吻，短句优先。", kwargs["custom_style_prompt"])
        self.assertEqual(0.12, kwargs["original_audio_volume"])
        self.assertEqual(0.7, kwargs["pause_original_audio_volume"])
        self.assertTrue(kwargs["background_music_enabled"])
        self.assertEqual("aodebiao_caravan", kwargs["background_music_track"])
        self.assertEqual(0.22, kwargs["background_music_volume"])
        self.assertTrue(os.path.basename(kwargs["source"]).startswith(job_id))
        self.assertEqual("completed", task_data["status"])
        self.assertEqual(job_id, task_data["job_id"])
        self.assertTrue(task_data["source_path"].endswith("demo.mp4"))

    def test_commentary_status_loads_persisted_task_after_memory_clear(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir):
            job_id = "persisted-job"
            os.makedirs(os.path.join(output_dir, job_id))
            with open(os.path.join(output_dir, job_id, "commentary_task.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "failed",
                    "stage_label": "生成失败",
                    "logs": ["saved failure"],
                    "error": "saved error",
                }, f)
            app.commentary_jobs.clear()

            client = TestClient(app.app)
            response = client.get(f"/api/commentary/status/{job_id}")

        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("failed", response.json()["status"])
        self.assertEqual(["saved failure"], response.json()["logs"])

    def test_commentary_retry_reuses_saved_source_and_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "retry", "final.mp4"),
                "video_url": "/videos/retry/final.mp4",
                "title": "Retried Commentary",
            }) as generate:
            job_id = "retry"
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir)
            source_path = os.path.join(job_dir, "source.mp4")
            analysis_path = os.path.join(job_dir, "source_gemini_360p.mp4")
            open(source_path, "wb").close()
            open(analysis_path, "wb").close()
            with open(os.path.join(job_dir, "commentary_task.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "failed",
                    "logs": ["Error: old failure"],
                    "error": "Gemini narration_blocks do not cover enough",
                    "request": {
                        "url": "",
                        "language": "zh",
                        "style": "funny",
                        "target_duration": "full",
                        "analysis_mode": "video",
                        "gemini_model": "old-thinking-model",
                        "tts_provider": "edge",
                        "subtitles": True,
                        "aspect_mode": "auto",
                    },
                    "source_type": "file",
                    "source_path": source_path,
                    "source_value": source_path,
                    "analysis_video_path": analysis_path,
                    "gemini_file_uri": "https://files.example/video-1",
                    "gemini_file_name": "files/video-1",
                    "gemini_file_mime_type": "video/mp4",
                }, f)
            app.commentary_jobs.clear()

            client = TestClient(app.app)
            response = client.post(
                f"/api/commentary/jobs/{job_id}/retry",
                headers={"X-Gemini-Key": "gemini-key"},
                json={"analysis_mode": "current", "gemini_model": "Qwen3.7-Plus"},
            )

        self.assertEqual(200, response.status_code, response.text)
        kwargs = generate.call_args.kwargs
        self.assertEqual(source_path, kwargs["source"])
        self.assertEqual("file", kwargs["source_type"])
        self.assertEqual(analysis_path, kwargs["prepared_analysis_video_path"])
        self.assertEqual("current", kwargs["analysis_mode"])
        self.assertEqual("Qwen3.7-Plus", kwargs["gemini_model"])
        self.assertEqual(0.6, kwargs["pause_original_audio_volume"])
        self.assertFalse(kwargs["background_music_enabled"])
        self.assertEqual("aodebiao_caravan", kwargs["background_music_track"])
        self.assertEqual(app.DEFAULT_BACKGROUND_MUSIC_VOLUME, kwargs["background_music_volume"])
        self.assertEqual("https://files.example/video-1", kwargs["gemini_file"]["uri"])
        self.assertIn("narration_blocks", kwargs["previous_error"])

    def test_commentary_retry_preserves_historical_proxy_timeout_for_fallback(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "retry", "final.mp4"),
                "video_url": "/videos/retry/final.mp4",
                "title": "Retried Commentary",
            }) as generate:
            job_id = "historical-proxy-timeout"
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir)
            source_path = os.path.join(job_dir, "source.mp4")
            analysis_path = os.path.join(job_dir, "source_gemini_360p.mp4")
            open(source_path, "wb").close()
            open(analysis_path, "wb").close()
            with open(os.path.join(job_dir, "commentary_task.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "failed",
                    "logs": [
                        "Error: 500 INTERNAL. {'error': {'message': 'Response timeout for 60000ms, see https://github.com/ali-sdk/ali-oss#responsetimeouterror'}}",
                        "Error: Gemini returned invalid JSON for the commentary script",
                    ],
                    "error": "AI narration block claims a completed packing/ending action that is not supported by its selected visual range.",
                    "request": {
                        "url": "",
                        "language": "zh",
                        "style": "hustle",
                        "target_duration": "medium",
                        "analysis_mode": "video",
                        "tts_provider": "edge",
                        "subtitles": True,
                        "aspect_mode": "auto",
                    },
                    "source_type": "file",
                    "source_path": source_path,
                    "source_value": source_path,
                    "analysis_video_path": analysis_path,
                    "gemini_file_uri": "files/video-1",
                    "gemini_file_name": "files/video-1",
                    "gemini_file_mime_type": "video/mp4",
                }, f)
            app.commentary_jobs.clear()

            client = TestClient(app.app)
            response = client.post(
                f"/api/commentary/jobs/{job_id}/retry",
                headers={"X-Gemini-Key": "gemini-key"},
            )

        self.assertEqual(200, response.status_code, response.text)
        kwargs = generate.call_args.kwargs
        self.assertEqual("video", kwargs["analysis_mode"])
        self.assertIn("ali-oss", kwargs["previous_error"])
        self.assertIn("completed packing/ending", kwargs["previous_error"])

    def test_commentary_retry_allows_stale_processing_task_after_restart(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "retry", "final.mp4"),
                "video_url": "/videos/retry/final.mp4",
                "title": "Retried Commentary",
            }) as generate:
            job_id = "stale-processing"
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir)
            source_path = os.path.join(job_dir, "source.mp4")
            open(source_path, "wb").close()
            script_path = os.path.join(job_dir, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump({"script": {"narration_blocks": []}}, f)
            with open(os.path.join(job_dir, "commentary_task.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "processing",
                    "stage": "voice",
                    "logs": ["Generating synced commentary block 4/16..."],
                    "error": None,
                    "request": {
                        "url": "",
                        "language": "zh",
                        "style": "hustle",
                        "target_duration": "full",
                        "analysis_mode": "openai",
                        "tts_provider": "edge",
                        "subtitles": True,
                        "aspect_mode": "auto",
                    },
                    "source_type": "file",
                    "source_path": source_path,
                    "source_value": source_path,
                    "script_path": script_path,
                }, f)
            app.commentary_jobs.clear()
            app.active_commentary_job_ids.clear()

            client = TestClient(app.app)
            response = client.post(f"/api/commentary/jobs/{job_id}/retry")

        self.assertEqual(200, response.status_code, response.text)
        self.assertTrue(generate.called)
        kwargs = generate.call_args.kwargs
        self.assertEqual(source_path, kwargs["source"])

    def test_commentary_retry_with_cached_video_script_does_not_require_gemini_key(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "retry", "final.mp4"),
                "video_url": "/videos/retry/final.mp4",
                "title": "Retried Commentary",
            }) as generate:
            job_id = "cached-video-script"
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir)
            source_path = os.path.join(job_dir, "source.mp4")
            script_path = os.path.join(job_dir, "script.json")
            open(source_path, "wb").close()
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump({"script": {"narration_blocks": []}}, f)
            with open(os.path.join(job_dir, "commentary_task.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "failed",
                    "stage": "failed",
                    "logs": ["TTS failed after script validation"],
                    "error": "Edge TTS interrupted",
                    "request": {
                        "url": "",
                        "language": "zh",
                        "style": "hustle",
                        "target_duration": "full",
                        "analysis_mode": "video",
                        "tts_provider": "edge",
                        "subtitles": True,
                        "aspect_mode": "auto",
                    },
                    "source_type": "file",
                    "source_path": source_path,
                    "source_value": source_path,
                    "script_path": script_path,
                }, f)
            app.commentary_jobs.clear()
            app.active_commentary_job_ids.clear()

            client = TestClient(app.app)
            response = client.post(f"/api/commentary/jobs/{job_id}/retry")

        self.assertEqual(200, response.status_code, response.text)
        kwargs = generate.call_args.kwargs
        self.assertEqual(source_path, kwargs["source"])
        self.assertEqual("", kwargs["gemini_key"])

    def test_commentary_retry_rejects_active_processing_task(self):
        with tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "OUTPUT_DIR", output_dir):
            job_id = "active-processing"
            job_dir = os.path.join(output_dir, job_id)
            os.makedirs(job_dir)
            source_path = os.path.join(job_dir, "source.mp4")
            open(source_path, "wb").close()
            app.commentary_jobs.clear()
            app.active_commentary_job_ids.clear()
            app.commentary_jobs[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "logs": [],
                "request": {
                    "url": "",
                    "language": "zh",
                    "style": "hustle",
                    "target_duration": "full",
                    "analysis_mode": "video",
                    "tts_provider": "edge",
                    "subtitles": True,
                    "aspect_mode": "auto",
                },
                "source_type": "file",
                "source_path": source_path,
                "source_value": source_path,
            }
            app.active_commentary_job_ids.add(job_id)

            client = TestClient(app.app)
            response = client.post(f"/api/commentary/jobs/{job_id}/retry", headers={"X-Gemini-Key": "gemini-key"})

        self.assertEqual(409, response.status_code, response.text)

    def test_commentary_generate_accepts_official_gemini_pool(self):
        pool_config = {
            "mode": "official_pool",
            "keys": ["key-one", "key-two"],
            "stats": {"key-...-one": {"state": "healthy"}},
        }
        with tempfile.TemporaryDirectory() as uploads_dir, \
            tempfile.TemporaryDirectory() as output_dir, \
            patch.object(app, "UPLOAD_DIR", uploads_dir), \
            patch.object(app, "OUTPUT_DIR", output_dir), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "generate_commentary_video", return_value={
                "video_path": os.path.join(output_dir, "final.mp4"),
                "video_url": "/videos/job/final.mp4",
                "title": "Uploaded Commentary",
            }) as generate:

            app.commentary_jobs.clear()
            client = TestClient(app.app)
            response = client.post(
                "/api/commentary/generate",
                data={
                    "gemini_pool": json.dumps(pool_config),
                    "language": "zh",
                    "style": "documentary",
                    "target_duration": "medium",
                    "analysis_mode": "video",
                    "tts_provider": "edge",
                    "subtitles": "true",
                    "aspect_mode": "auto",
                },
                files={"file": ("demo.mp4", b"fake-video", "video/mp4")},
            )

        self.assertEqual(200, response.status_code, response.text)
        pool = generate.call_args.kwargs["gemini_pool"]
        self.assertEqual("official_pool", pool.mode)
        self.assertEqual(["key-one", "key-two"], pool.keys)
        self.assertEqual("", generate.call_args.kwargs["gemini_key"])


if __name__ == "__main__":
    unittest.main()
