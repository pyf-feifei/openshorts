import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import commentary_style_learning as learner


class FakeProvider:
    def __init__(self, videos):
        self.videos = videos

    def fetch_videos(self, profile_url, cookie_path, progress=None, cancel_event=None):
        if progress:
            progress("Fetching fake Douyin video list...")
        return self.videos


class CommentaryStyleLearningTest(unittest.TestCase):
    def test_parse_douyin_user_url_extracts_sec_uid(self):
        sec_uid = learner.parse_douyin_user_url(
            "https://www.douyin.com/user/MS4wLjABAAAAabc?from_tab_name=main"
        )
        self.assertEqual("MS4wLjABAAAAabc", sec_uid)

    def test_rank_douyin_videos_uses_like_plus_save_and_caps_to_100(self):
        videos = [
            {
                "aweme_id": str(index),
                "statistics": {"digg_count": index, "collect_count": 200 - index},
                "create_time": index,
                "desc": f"video {index}",
            }
            for index in range(120)
        ]
        videos.append({
            "aweme_id": "top",
            "statistics": {"digg_count": 500, "collect_count": 400},
            "create_time": 1,
            "desc": "top video",
        })

        ranked = learner.rank_douyin_videos(videos, max_videos=100)

        self.assertEqual(100, len(ranked))
        self.assertEqual("top", ranked[0]["aweme_id"])
        self.assertEqual(900, ranked[0]["rank_score"])
        self.assertTrue(all(item["rank_index"] == index + 1 for index, item in enumerate(ranked)))

    def test_public_douyin_payloads_to_videos_extracts_aweme_list(self):
        videos = learner._public_douyin_payloads_to_videos([{
            "status_code": 0,
            "aweme_list": [{
                "aweme_id": "7650063885696830747",
                "desc": "夫妻俩用古法压榨葵花油",
                "create_time": 1781169304,
                "statistics": {"digg_count": 467, "collect_count": 53},
                "author": {"nickname": "微记录片"},
            }],
        }])

        self.assertEqual(1, len(videos))
        self.assertEqual("7650063885696830747", videos[0]["aweme_id"])
        self.assertEqual(467, videos[0]["like_count"])
        self.assertEqual(53, videos[0]["save_count"])
        self.assertEqual(520, videos[0]["rank_score"])

    def test_public_payloads_merge_duplicate_partial_records_with_media_urls(self):
        partial = {
            "aweme_id": "a1",
            "desc": "partial high score",
            "statistics": {"digg_count": 100, "collect_count": 25},
        }
        complete = {
            "aweme_id": "a1",
            "desc": "complete media",
            "statistics": {"digg_count": 100, "collect_count": 25},
            "video": {
                "bit_rate_audio": [{
                    "audio_meta": {
                        "url_list": {
                            "main_url": "https://v26-web.douyinvod.com/media-audio-und-mp4a/",
                        },
                    },
                }],
                "play_addr": {
                    "url_list": ["https://v26-web.douyinvod.com/video/tos/test?mime_type=video_mp4"],
                },
            },
        }

        videos = learner._public_douyin_payloads_to_videos([{
            "aweme_list": [partial, complete],
        }])

        self.assertEqual(1, len(videos))
        self.assertEqual("a1", videos[0]["aweme_id"])
        self.assertEqual(125, videos[0]["rank_score"])
        self.assertIn("media-audio", videos[0]["direct_audio_url"])
        self.assertIn("mime_type=video_mp4", videos[0]["direct_video_url"])

    def test_ensure_direct_media_uses_public_detail_when_profile_item_is_partial(self):
        partial = {
            "aweme_id": "a1",
            "video_url": "https://www.douyin.com/video/a1",
            "direct_audio_url": "",
            "direct_video_url": "",
            "like_count": 100,
            "save_count": 25,
            "rank_score": 125,
        }
        detail = {
            "aweme_id": "a1",
            "statistics": {"digg_count": 100, "collect_count": 25},
            "video": {
                "bit_rate_audio": [{
                    "audio_meta": {
                        "url_list": {
                            "main_url": "https://v26-web.douyinvod.com/media-audio-und-mp4a/",
                        },
                    },
                }],
            },
        }

        with patch.object(learner, "_fetch_douyin_public_aweme_detail_payloads", new=AsyncMock(return_value=[{"aweme_detail": detail}])):
            enriched = learner.ensure_douyin_direct_media(partial)

        self.assertEqual("a1", enriched["aweme_id"])
        self.assertIn("media-audio", enriched["direct_audio_url"])
        self.assertEqual(125, enriched["rank_score"])

    def test_ensure_direct_media_can_use_browser_media_response_url(self):
        partial = {
            "aweme_id": "a1",
            "video_url": "https://www.douyin.com/video/a1",
            "direct_audio_url": "",
            "direct_video_url": "",
            "like_count": 10,
            "save_count": 2,
            "rank_score": 12,
        }

        with patch.object(learner, "_fetch_douyin_public_aweme_detail_payloads", new=AsyncMock(return_value=[{
            "__openshorts_media_url": "https://v26-web.douyinvod.com/video/tos/test?mime_type=video_mp4&__vid=a1",
        }])):
            enriched = learner.ensure_douyin_direct_media(partial)

        self.assertEqual("a1", enriched["aweme_id"])
        self.assertIn("mime_type=video_mp4", enriched["direct_video_url"])

    def test_fetch_videos_uses_public_browser_capture_without_cookies(self):
        provider = learner.DouyinProfileProvider()
        payloads = [{
            "status_code": 0,
            "aweme_list": [{
                "aweme_id": "a1",
                "desc": "public video",
                "statistics": {"digg_count": 10, "collect_count": 4},
            }],
        }]

        with patch.object(learner, "_collect_douyin_public_post_payloads", return_value=payloads):
            videos = provider.fetch_videos(
                "https://www.douyin.com/user/MS4wLjABAAAAabc",
                cookie_path="",
            )

        self.assertEqual(1, len(videos))
        self.assertEqual("a1", videos[0]["aweme_id"])

    def test_normalize_douyin_cookies_keeps_only_douyin_domains(self):
        raw = "\n".join([
            "# Netscape HTTP Cookie File",
            ".douyin.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tabc",
            "api.douyin.com\tTRUE\t/\tTRUE\t1893456000\tttwid\tabc",
            "notdouyin.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tbad",
            ".youtube.com\tTRUE\t/\tTRUE\t1893456000\tSID\tnope",
        ])

        normalized = learner.normalize_douyin_cookies(raw)
        info = learner.inspect_douyin_cookies(normalized)

        self.assertIn(".douyin.com", normalized)
        self.assertIn("api.douyin.com", normalized)
        self.assertNotIn("notdouyin", normalized)
        self.assertNotIn("youtube", normalized)
        self.assertEqual(2, info["rows"])
        self.assertTrue(info["has_login_cookies"])

    def test_sanitize_style_prompt_preserves_structure_and_caps_larger_prompt(self):
        raw = "\n".join([
            "# 核心风格定位",
            "先说画面动作，再接判断。https://www.douyin.com/video/7641867461817617698",
            "",
            "# 结构模板",
            "不要写 MS4wLjABAAAAabc 这样的主页标识。",
            "短句推进。" * 1200,
        ])

        sanitized = learner._sanitize_style_prompt(
            raw,
            profile_url="https://www.douyin.com/user/MS4wLjABAAAAabc",
        )

        self.assertIn("# 核心风格定位\n", sanitized)
        self.assertIn("# 结构模板", sanitized)
        self.assertLessEqual(len(sanitized), learner.STYLE_PROMPT_MAX_CHARS)
        self.assertGreater(len(sanitized), 1800)
        self.assertNotIn("douyin.com", sanitized)
        self.assertNotIn("MS4wLjAB", sanitized)

    def test_run_commentary_style_learning_persists_generic_style(self):
        videos = [
            {
                "aweme_id": "a1",
                "video_url": "https://www.douyin.com/video/a1",
                "title": "first",
                "like_count": 10,
                "save_count": 2,
                "rank_score": 12,
                "timestamp": 1,
            },
            {
                "aweme_id": "a2",
                "video_url": "https://www.douyin.com/video/a2",
                "title": "second",
                "like_count": 2,
                "save_count": 20,
                "rank_score": 22,
                "timestamp": 2,
            },
        ]
        openai_calls = []

        def fake_download(video, output_dir, cookie_path, cancel_event=None):
            path = os.path.join(output_dir, f"{video['aweme_id']}.mp3")
            os.makedirs(output_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"audio")
            return {"audio_path": path, "title": video["title"]}

        def fake_transcribe(audio_path):
            stem = os.path.splitext(os.path.basename(audio_path))[0]
            return {
                "text": f"{stem} 这是一段完整解说，先抓画面动作，再给短促判断，节奏快但不编造画面外信息。" * 3,
                "segments": [],
                "language": "zh",
            }

        def fake_openai(messages, max_tokens=0, response_format=None, timeout_seconds=None):
            text = messages[0]["content"]
            openai_calls.append(text)
            if "生成一个可直接用于" in text:
                return json.dumps({
                    "label": "学到的通用风格",
                    "prompt": "\n".join([
                        "# 核心风格定位",
                        "用短句先说明当前画面动作，再给一句反差判断；保持节奏紧，所有结论必须来自当前可见画面。",
                        "# 结构模板",
                        "先交代可见动作，再交代变化结果，最后补一句谨慎判断。",
                        "# 禁止事项",
                        "不写账号名、链接或固定题材。" + "必须画面先行。" * 320,
                    ]),
                    "summary": "短句、动作先行、反差判断",
                    "style_traits": ["动作先行", "节奏紧", "反差判断"],
                    "confidence": "high",
                    "warnings": [],
                }, ensure_ascii=False)
            return json.dumps({
                "usable": True,
                "summary": "动作先行，短句推进，再补一句判断。",
                "traits": ["第三人称", "短句", "节奏快"],
                "pacing": "短促密集",
                "rhetoric": "先画面后判断",
                "avoid_copying": [],
            }, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = os.path.join(tmpdir, "douyin_cookies.txt")
            style_path = os.path.join(tmpdir, "styles.json")
            with open(cookie_path, "w", encoding="utf-8") as f:
                f.write(".douyin.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tabc\n")

            result = learner.run_commentary_style_learning(
                profile_url="https://www.douyin.com/user/MS4wLjABAAAAabc",
                output_dir=tmpdir,
                openai_config={"api_key": "k", "base_url": "https://api.example/v1", "model": "m"},
                cookie_path=cookie_path,
                style_name="学到的通用风格",
                provider=FakeProvider(videos),
                download_audio_fn=fake_download,
                transcribe_fn=fake_transcribe,
                openai_chat_fn=fake_openai,
                style_storage_path=style_path,
            )

            self.assertEqual("学到的通用风格", result["style"]["label"])
            self.assertEqual(2, result["selected_count"])
            self.assertEqual(2, result["transcript_count"])
            self.assertTrue(result["style"]["id"].startswith("custom:"))
            self.assertNotIn("douyin.com", result["style"]["prompt"])
            self.assertIn("# 核心风格定位\n", result["style"]["prompt"])
            self.assertGreater(len(result["style"]["prompt"]), 1800)
            aggregate_prompt = next(call for call in openai_calls if "生成一个可直接用于" in call)
            self.assertIn("复刻同一种“解说说话方式”", aggregate_prompt)
            self.assertIn("复刻公式", aggregate_prompt)
            self.assertIn("建议 1800 到 4200", aggregate_prompt)
            saved = learner.list_commentary_styles(style_path)
            self.assertEqual(1, len(saved))
            self.assertEqual(result["style"]["prompt"], saved[0]["prompt"])
            self.assertGreaterEqual(len(openai_calls), 3)

    def test_run_commentary_style_learning_checkpoints_ranked_videos_before_media_resolution(self):
        videos = [
            {
                "aweme_id": "a1",
                "video_url": "https://www.douyin.com/video/a1",
                "title": "first",
                "like_count": 10,
                "save_count": 2,
                "rank_score": 12,
            },
            {
                "aweme_id": "a2",
                "video_url": "https://www.douyin.com/video/a2",
                "title": "second",
                "like_count": 2,
                "save_count": 20,
                "rank_score": 22,
            },
        ]
        checkpoints = []
        progress_logs = []

        def fake_download(video, output_dir, cookie_path, cancel_event=None):
            path = os.path.join(output_dir, f"{video['aweme_id']}.mp3")
            os.makedirs(output_dir, exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"audio")
            return {"audio_path": path, "title": video["title"]}

        def fake_transcribe(audio_path):
            return {
                "text": "先描述画面动作，再补一句判断。这个结构重复出现，足够分析风格。" * 2,
                "segments": [],
                "language": "zh",
            }

        def fake_openai(messages, max_tokens=0, response_format=None, timeout_seconds=None):
            text = messages[0]["content"]
            if "生成一个可直接用于" in text:
                return json.dumps({
                    "label": "先排序再解析",
                    "prompt": "# 核心风格定位\n先说画面动作。\n# 禁止事项\n不编造画面外剧情。",
                    "summary": "动作先行",
                    "style_traits": ["动作先行"],
                    "confidence": "high",
                    "warnings": [],
                }, ensure_ascii=False)
            return json.dumps({
                "usable": True,
                "coverage": "充分",
                "summary": "动作先行",
                "voice_persona": "旁观讲述",
                "script_structure": "先动作后判断",
                "opening_hook": "先抛动作",
                "visual_grounding": "画面优先",
                "sentence_rhythm": "短句",
                "pacing": "紧凑",
                "rhetoric": "反差",
                "word_choice": "动作词",
                "emotion": "克制",
                "do_rules": ["先说画面"],
                "dont_rules": ["不编造"],
                "template": "[动作]+[变化]+[判断]",
            }, ensure_ascii=False)

        def fake_checkpoint(fields):
            checkpoints.append(dict(fields or {}))

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(learner, "ensure_douyin_direct_media", side_effect=lambda video, progress=None, cancel_event=None: video):
            learner.run_commentary_style_learning(
                profile_url="https://www.douyin.com/user/MS4wLjABAAAAabc",
                output_dir=tmpdir,
                openai_config={"api_key": "k", "base_url": "https://api.example/v1", "model": "m"},
                cookie_path="",
                provider=FakeProvider(videos),
                download_audio_fn=fake_download,
                transcribe_fn=fake_transcribe,
                openai_chat_fn=fake_openai,
                style_storage_path=os.path.join(tmpdir, "styles.json"),
                progress=progress_logs.append,
                checkpoint=fake_checkpoint,
            )

        first_selection = next(item for item in checkpoints if item.get("selected_videos"))
        self.assertEqual(2, first_selection["selected_count"])
        self.assertEqual(["a2", "a1"], [item["aweme_id"] for item in first_selection["selected_videos"]])
        self.assertLess(
            progress_logs.index("Selected 2 videos from 2 visible Douyin videos."),
            next(index for index, line in enumerate(progress_logs) if "Resolving public media URLs" in line),
        )


if __name__ == "__main__":
    unittest.main()
