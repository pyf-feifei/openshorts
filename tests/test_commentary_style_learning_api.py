import json
import os
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


class CommentaryStyleLearningApiTests(unittest.TestCase):
    def test_douyin_cookie_settings_status_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "DOUYIN_COOKIES_PATH", os.path.join(tmpdir, "douyin_cookies.txt")):
            client = TestClient(app.app)

            initial = client.get("/api/settings/douyin-cookies")
            response = client.post(
                "/api/settings/douyin-cookies",
                json={
                    "cookies": "\n".join([
                        "# Netscape HTTP Cookie File",
                        ".douyin.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tabc",
                        "notdouyin.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tbad",
                    ])
                },
            )
            status = client.get("/api/settings/douyin-cookies")

        self.assertEqual(200, initial.status_code, initial.text)
        self.assertFalse(initial.json()["configured"])
        self.assertEqual(200, response.status_code, response.text)
        self.assertTrue(response.json()["configured"])
        self.assertEqual(1, response.json()["rows"])
        self.assertEqual([".douyin.com"], response.json()["domains"])
        self.assertEqual(200, status.status_code, status.text)
        self.assertEqual(1, status.json()["rows"])

    def test_commentary_style_list_and_delete_uses_backend_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "COMMENTARY_STYLES_PATH", os.path.join(tmpdir, "styles.json")):
            with open(app.COMMENTARY_STYLES_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "styles": [{
                        "id": "custom:learned-api",
                        "label": "Learned API Style",
                        "prompt": "Use concise visual-first narration without copying source wording.",
                        "custom": True,
                    }]
                }, f)
            client = TestClient(app.app)

            listed = client.get("/api/commentary/styles")
            deleted = client.delete("/api/commentary/styles/custom%3Alearned-api")
            listed_after = client.get("/api/commentary/styles")

        self.assertEqual(200, listed.status_code, listed.text)
        self.assertEqual("custom:learned-api", listed.json()["styles"][0]["id"])
        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(200, listed_after.status_code, listed_after.text)
        self.assertEqual([], listed_after.json()["styles"])

    def test_style_learning_create_does_not_require_cookies_and_validates_openai_headers_before_job_start(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "DOUYIN_COOKIES_PATH", os.path.join(tmpdir, "missing_cookies.txt")), \
            patch.object(app, "OUTPUT_DIR", os.path.join(tmpdir, "output")):
            app.style_learning_jobs.clear()
            client = TestClient(app.app)
            missing_openai = client.post(
                "/api/commentary/style-learning/jobs",
                json={"profile_url": "https://www.douyin.com/user/MS4wLjABAAAAabc"},
            )

            missing_profile = client.post(
                "/api/commentary/style-learning/jobs",
                json={"profile_url": ""},
            )

        self.assertEqual(400, missing_openai.status_code, missing_openai.text)
        self.assertIn("OpenAI-compatible", missing_openai.json()["detail"])
        self.assertEqual(400, missing_profile.status_code, missing_profile.text)
        self.assertIn("profile URL", missing_profile.json()["detail"])

    def test_style_learning_create_runs_job_and_persists_completed_status(self):
        style = {
            "id": "custom:learned-route",
            "label": "Route Learned Style",
            "prompt": "Use reusable visual-first commentary rhythm.",
            "custom": True,
        }

        def fake_run_commentary_style_learning(**kwargs):
            kwargs["checkpoint"]({
                "selected_count": 1,
                "downloaded_count": 1,
                "transcript_count": 1,
                "selected_videos": [{
                    "aweme_id": "a1",
                    "video_url": "https://www.douyin.com/video/a1",
                    "like_count": 10,
                    "save_count": 5,
                    "rank_score": 15,
                    "rank_index": 1,
                }],
                "failed_videos": [],
            })
            return {
                "style": style,
                "selected_count": 1,
                "downloaded_count": 1,
                "transcript_count": 1,
                "selected_videos": [{
                    "aweme_id": "a1",
                    "video_url": "https://www.douyin.com/video/a1",
                    "like_count": 10,
                    "save_count": 5,
                    "rank_score": 15,
                    "rank_index": 1,
                }],
                "failed_videos": [],
                "result_path": os.path.join(kwargs["output_dir"], "style_learning_result.json"),
            }

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "DOUYIN_COOKIES_PATH", os.path.join(tmpdir, "douyin_cookies.txt")), \
            patch.object(app, "OUTPUT_DIR", os.path.join(tmpdir, "output")), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "run_commentary_style_learning", side_effect=fake_run_commentary_style_learning) as run_job:
            os.makedirs(tmpdir, exist_ok=True)
            app.style_learning_jobs.clear()
            app.style_learning_job_cancel_events.clear()
            app.style_learning_job_threads.clear()
            client = TestClient(app.app)

            created = client.post(
                "/api/commentary/style-learning/jobs",
                headers={
                    "X-OpenAI-Compatible-Key": "openai-key",
                    "X-OpenAI-Compatible-Base-URL": "https://api.example.com/v1",
                    "X-OpenAI-Compatible-Model": "model",
                },
                json={
                    "profile_url": "https://www.douyin.com/user/MS4wLjABAAAAabc",
                    "style_name": "Route Learned Style",
                    "max_videos": 200,
                    "language": "zh",
                },
            )
            job_id = created.json()["job_id"]
            status = client.get(f"/api/commentary/style-learning/jobs/{job_id}")
            list_response = client.get("/api/commentary/style-learning/jobs")

        self.assertEqual(200, created.status_code, created.text)
        self.assertEqual("processing", created.json()["status"])
        self.assertEqual(200, status.status_code, status.text)
        self.assertEqual("completed", status.json()["status"])
        self.assertEqual("Route Learned Style", status.json()["style"]["label"])
        self.assertEqual(1, status.json()["selected_count"])
        self.assertEqual(200, list_response.status_code, list_response.text)
        self.assertEqual(job_id, list_response.json()["jobs"][0]["job_id"])
        self.assertEqual(100, run_job.call_args.kwargs["max_videos"])
        self.assertEqual("zh", run_job.call_args.kwargs["language"])

    def test_style_learning_media_log_does_not_reset_stage_to_fetch(self):
        style = {
            "id": "custom:learned-media-route",
            "label": "Media Route Style",
            "prompt": "Use reusable visual-first commentary rhythm.",
            "custom": True,
        }
        observed_stages = []

        def fake_run_commentary_style_learning(**kwargs):
            kwargs["checkpoint"]({
                "selected_count": 18,
                "total_videos": 18,
                "selected_videos": [
                    {
                        "aweme_id": str(index),
                        "video_url": f"https://www.douyin.com/video/{index}",
                        "rank_index": index,
                    }
                    for index in range(1, 19)
                ],
            })
            kwargs["progress"]("Fetching public Douyin video detail for media URL: 7641867461817617698")
            observed_stages.append(next(iter(app.style_learning_jobs.values()))["stage"])
            return {
                "style": style,
                "selected_count": 18,
                "downloaded_count": 0,
                "transcript_count": 0,
                "selected_videos": [],
                "failed_videos": [],
                "result_path": os.path.join(kwargs["output_dir"], "style_learning_result.json"),
            }

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "DOUYIN_COOKIES_PATH", os.path.join(tmpdir, "douyin_cookies.txt")), \
            patch.object(app, "OUTPUT_DIR", os.path.join(tmpdir, "output")), \
            patch.object(app.threading, "Thread", ImmediateThread), \
            patch.object(app, "run_commentary_style_learning", side_effect=fake_run_commentary_style_learning):
            app.style_learning_jobs.clear()
            app.style_learning_job_cancel_events.clear()
            app.style_learning_job_threads.clear()
            client = TestClient(app.app)

            created = client.post(
                "/api/commentary/style-learning/jobs",
                headers={
                    "X-OpenAI-Compatible-Key": "openai-key",
                    "X-OpenAI-Compatible-Base-URL": "https://api.example.com/v1",
                    "X-OpenAI-Compatible-Model": "model",
                },
                json={
                    "profile_url": "https://www.douyin.com/user/MS4wLjABAAAAabc",
                    "language": "zh",
                },
            )
            job_id = created.json()["job_id"]
            status = client.get(f"/api/commentary/style-learning/jobs/{job_id}")

        self.assertEqual(200, created.status_code, created.text)
        self.assertEqual("completed", status.json()["status"])
        self.assertEqual("Media Route Style", status.json()["style"]["label"])
        self.assertEqual(["media"], observed_stages)
        self.assertIn("media URL", "\n".join(status.json()["logs"]))

    def test_style_learning_delete_removes_persisted_job_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "OUTPUT_DIR", os.path.join(tmpdir, "output")):
            app.style_learning_jobs.clear()
            app.style_learning_job_cancel_events.clear()
            job_id = "style-delete-one"
            app.style_learning_jobs[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "stage": "queued",
                "stage_label": "Queued",
                "logs": [],
                "created_at": app.now_iso(),
                "updated_at": app.now_iso(),
            }
            app.save_style_learning_task(job_id)
            job_dir = app.style_learning_job_dir(job_id)
            self.assertTrue(os.path.exists(app.style_learning_task_path(job_id)))

            client = TestClient(app.app)
            deleted = client.delete(f"/api/commentary/style-learning/jobs/{job_id}")
            listed = client.get("/api/commentary/style-learning/jobs")

        self.assertEqual(200, deleted.status_code, deleted.text)
        self.assertTrue(deleted.json()["deleted"])
        self.assertNotIn(job_id, app.style_learning_jobs)
        self.assertFalse(os.path.exists(job_dir))
        self.assertEqual([], listed.json()["jobs"])

    def test_style_learning_bulk_delete_deduplicates_and_reports_invalid_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(app, "OUTPUT_DIR", os.path.join(tmpdir, "output")):
            app.style_learning_jobs.clear()
            app.style_learning_job_cancel_events.clear()
            for job_id in ("style-delete-a", "style-delete-b"):
                app.style_learning_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "completed",
                    "stage": "done",
                    "stage_label": "Done",
                    "logs": [],
                    "created_at": app.now_iso(),
                    "updated_at": app.now_iso(),
                }
                app.save_style_learning_task(job_id)
            first_dir = app.style_learning_job_dir("style-delete-a")
            second_dir = app.style_learning_job_dir("style-delete-b")

            client = TestClient(app.app)
            deleted = client.post(
                "/api/commentary/style-learning/jobs/delete",
                json={"job_ids": ["style-delete-a", "style-delete-a", "../bad", "style-delete-b"]},
            )

        self.assertEqual(200, deleted.status_code, deleted.text)
        body = deleted.json()
        self.assertEqual(["style-delete-a", "style-delete-b"], [item["job_id"] for item in body["deleted"]])
        self.assertEqual([{"job_id": "../bad", "error": "Invalid commentary job id"}], body["errors"])
        self.assertFalse(os.path.exists(first_dir))
        self.assertFalse(os.path.exists(second_dir))


if __name__ == "__main__":
    unittest.main()
