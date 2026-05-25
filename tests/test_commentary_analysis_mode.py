import os
import sys
import tempfile
import types as pytypes
import unittest
from unittest.mock import ANY, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import commentary
import main


def scene_matched_block_text():
    return (
        "这一段解说紧贴当前画面，先说明这个时间段里材料、设备和工人动作的具体变化，"
        "再把这个工序放回废旧电机回收成新铜材的流程里解释，避免提前讲后面的画面。"
    ) * 6


def scene_matched_blocks(count=12, seconds=100.0, text=None):
    block_text = text or scene_matched_block_text()
    starts = [i * seconds for i in range(count)]
    if count == 12 and seconds == 100.0:
        starts = [i * 300.0 for i in range(count - 1)] + [3600.0]
    return [
        {
            "start": start,
            "end": start + seconds,
            "visual": "workers handle copper motor scrap in this exact process stage",
            "narration": block_text,
        }
        for start in starts
    ]


class CommentaryAnalysisModeTests(unittest.TestCase):
    def test_default_analysis_mode_is_gemini_video_input(self):
        self.assertEqual("video", commentary.DEFAULT_ANALYSIS_MODE)
        self.assertEqual("video", commentary._normalize_analysis_mode(None))
        self.assertEqual("video", commentary._normalize_analysis_mode(""))
        self.assertEqual("openai", commentary._normalize_analysis_mode("openai"))

    def test_openai_chat_completions_url_accepts_base_or_direct_endpoint(self):
        self.assertEqual(
            "https://provider.example.com/v1/chat/completions",
            commentary._openai_chat_completions_url("https://provider.example.com/v1/"),
        )
        self.assertEqual(
            "https://provider.example.com/v1/chat/completions",
            commentary._openai_chat_completions_url("https://provider.example.com/v1/chat/completions"),
        )

    def test_openai_script_max_tokens_defaults_to_64000(self):
        self.assertEqual(64000, commentary.OPENAI_SCRIPT_MAX_TOKENS)

    def test_prompt_mentions_mode_specific_visual_inputs(self):
        transcript = {
            "text": "A short transcript",
            "language": "en",
            "segments": [{"start": 0, "end": 3, "text": "A short transcript"}],
        }

        current_prompt = commentary._build_commentary_prompt(
            transcript=transcript,
            video_title="Demo",
            duration=12.0,
            language="zh",
            style="documentary",
            target_duration="short",
            analysis_mode="current",
        )
        video_prompt = commentary._build_commentary_prompt(
            transcript=transcript,
            video_title="Demo",
            duration=12.0,
            language="zh",
            style="documentary",
            target_duration="short",
            analysis_mode="video",
        )
        openai_prompt = commentary._build_commentary_prompt(
            transcript=transcript,
            video_title="Demo",
            duration=12.0,
            language="zh",
            style="documentary",
            target_duration="short",
            analysis_mode="openai",
            visual_analysis={"observations": [{"timestamp": 6, "visual": "worker sorts copper"}]},
        )

        self.assertIn("sampled keyframes", current_prompt)
        self.assertIn("low-resolution video", video_prompt)
        self.assertIn("original full source video timeline", video_prompt)
        self.assertIn('"edit_segments"', video_prompt)
        self.assertIn("OPENAI-COMPATIBLE MULTIMODAL VISUAL TIMELINE", openai_prompt)
        self.assertIn("worker sorts copper", openai_prompt)
        self.assertIn("title must clearly say what the video is doing", video_prompt)
        self.assertIn("specific title that says what the video does", video_prompt)
        self.assertIn("camera/meta phrasing", video_prompt)
        self.assertIn("镜头切到", video_prompt)

    def test_openai_full_prompt_warns_density_must_pass_first_response(self):
        transcript = {
            "text": "A long transcript",
            "language": "en",
            "segments": [{"start": 0, "end": 30, "text": "A long transcript"}],
        }

        prompt = commentary._build_commentary_prompt(
            transcript=transcript,
            video_title="Demo",
            duration=3935.0,
            language="zh",
            style="casual_roast",
            target_duration="full",
            analysis_mode="openai",
            visual_analysis={"observations": [{"timestamp": 6, "visual": "worker sorts copper"}]},
        )

        self.assertIn("first complete JSON response must pass every per-block density check", prompt)
        self.assertIn("request a focused local repair for the failed block", prompt)
        self.assertIn('"density_audit"', prompt)
        self.assertIn("Hard requirement before style", prompt)
        self.assertIn("planning average", prompt)
        self.assertIn("do not pad it with meaningless words", prompt)
        self.assertIn("better block design, not filler narration", prompt)
        self.assertIn("write those numbers into density_audit", prompt)
        self.assertIn("3-second retention rule", prompt)
        self.assertIn("curiosity, contrast, stakes, surprise, or payoff expectation", prompt)
        self.assertIn("matching the first visible action", prompt)

    def test_narration_rejects_camera_meta_phrasing(self):
        data = {
            "narration": "镜头切到工人把铜线从电机里抽出来。",
            "narration_blocks": [],
        }

        with self.assertRaisesRegex(Exception, "camera/meta phrasing"):
            commentary._validate_commentary_script_for_target(data, 60.0, "short", "zh")

    def test_visual_metadata_allows_camera_word_outside_narration(self):
        data = {
            "narration": "工人把铜线从电机里抽出来。",
            "narration_blocks": [{"visual": "镜头切到工人操作", "narration": "铜线被慢慢抽离出来。"}],
        }

        commentary._validate_commentary_script_for_target(data, 60.0, "short", "zh")

    def test_openai_regeneration_prompt_focuses_failed_density_block(self):
        invalid_script = {
            "narration_blocks": [
                {"start": 0, "end": 30, "visual": "opening", "narration": "足够长的开场解说" * 20, "video_speed": 1.0},
                {"start": 30, "end": 63, "visual": "worker moves copper scrap", "narration": "短解说" * 30, "video_speed": 1.0},
            ]
        }
        validation_error = Exception(
            "AI narration block is too short for its selected visual range. "
            "Block 2 has 93 chars for 33.0s of playable visuals; expected at least 121. "
            "Rewrite this block from the source visuals."
        )

        prompt = commentary._build_openai_regeneration_prompt(
            "ORIGINAL PROMPT",
            invalid_script,
            validation_error,
            duration=3935.1,
            target_duration="full",
            language="zh",
            attempt=2,
        )

        self.assertIn("FOCUSED REPAIR REQUIRED", prompt)
        self.assertIn("narration_blocks[1]", prompt)
        self.assertIn("Block 2", prompt)
        self.assertIn("93 non-whitespace characters for 33.0s", prompt)
        self.assertIn("at least 179 scene-matched characters", prompt)
        self.assertIn("worker moves copper scrap", prompt)
        self.assertIn("keep all unrelated narration_blocks unchanged", prompt)
        self.assertIn("len(non_whitespace(narration))", prompt)
        self.assertIn("* 5.4", prompt)

    def test_near_miss_density_repair_slightly_speeds_block(self):
        data = {
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 42,
                    "visual": "worker keeps handling the same process",
                    "narration": "讲" * 150,
                    "video_speed": 1.0,
                }
            ]
        }

        commentary._repair_near_miss_narration_density_blocks(data, "zh")
        block = data["narration_blocks"][0]
        repaired_duration = commentary._block_visual_duration(block)
        repaired_min_chars = commentary._accepted_minimum_narration_chars(
            commentary._minimum_spoken_block_chars(repaired_duration, "zh")
        )

        self.assertGreater(block["video_speed"], 1.0)
        self.assertLess(block["video_speed"], 1.1)
        self.assertGreaterEqual(150, repaired_min_chars)

    def test_near_miss_density_repair_handles_openai_short_block_without_request(self):
        data = {
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 35,
                    "visual": "factory process step with slightly sparse narration",
                    "narration": "讲" * 107,
                    "video_speed": 1.0,
                }
            ]
        }

        commentary._repair_near_miss_narration_density_blocks(data, "zh")
        block = data["narration_blocks"][0]
        repaired_duration = commentary._block_visual_duration(block)
        repaired_min_chars = commentary._accepted_minimum_narration_chars(
            commentary._minimum_spoken_block_chars(repaired_duration, "zh")
        )

        self.assertGreater(block["video_speed"], 1.18)
        self.assertLess(block["video_speed"], 1.35)
        self.assertGreaterEqual(107, repaired_min_chars)

    def test_near_miss_density_repair_handles_moderate_openai_short_block_without_request(self):
        data = {
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 39.6,
                    "visual": "moderately sparse process shot that can play faster",
                    "narration": "讲" * 101,
                    "video_speed": 1.0,
                }
            ]
        }

        commentary._repair_near_miss_narration_density_blocks(data, "zh")
        block = data["narration_blocks"][0]
        repaired_duration = commentary._block_visual_duration(block)
        repaired_min_chars = commentary._accepted_minimum_narration_chars(
            commentary._minimum_spoken_block_chars(repaired_duration, "zh")
        )

        self.assertGreater(block["video_speed"], 1.4)
        self.assertLess(block["video_speed"], 1.6)
        self.assertGreaterEqual(101, repaired_min_chars)

    def test_openai_sampling_options_include_default_visual_concurrency(self):
        with patch.object(commentary, "OPENAI_VISUAL_CONCURRENCY", 3):
            options = commentary.resolve_openai_sampling_options()

        self.assertEqual(3, options["visual_concurrency"])

    def test_openai_sampling_options_clamp_visual_concurrency(self):
        self.assertEqual(1, commentary.resolve_openai_sampling_options(visual_concurrency=0)["visual_concurrency"])
        self.assertEqual(
            commentary.OPENAI_VISUAL_CONCURRENCY_LIMIT,
            commentary.resolve_openai_sampling_options(visual_concurrency=99)["visual_concurrency"],
        )

    def test_commentary_block_concurrency_defaults_to_three(self):
        with patch.object(commentary, "COMMENTARY_BLOCK_CONCURRENCY", 3):
            self.assertEqual(3, commentary.resolve_commentary_block_concurrency())

    def test_commentary_block_concurrency_clamps_to_limits(self):
        self.assertEqual(1, commentary.resolve_commentary_block_concurrency(0))
        self.assertEqual(
            commentary.COMMENTARY_BLOCK_CONCURRENCY_LIMIT,
            commentary.resolve_commentary_block_concurrency(99),
        )

    def test_openai_sampling_options_allow_larger_batch_size(self):
        self.assertEqual(64, commentary.resolve_openai_sampling_options(batch_size=64)["batch_size"])
        self.assertEqual(
            commentary.OPENAI_BATCH_SIZE_LIMIT,
            commentary.resolve_openai_sampling_options(batch_size=999)["batch_size"],
        )

    def test_openai_uniform_frame_samples_respect_interval_and_max_frames(self):
        with patch.object(commentary, "OPENAI_MAX_FRAMES", 5), \
             patch.object(commentary, "OPENAI_FRAME_INTERVAL_SECONDS", 10):
            samples = commentary._select_openai_uniform_frame_samples(120)

        self.assertEqual(5, len(samples))
        self.assertEqual("uniform", samples[0]["sample_role"])
        self.assertTrue(all(0 <= sample["timestamp"] < 120 for sample in samples))

    def test_openai_scene_aware_samples_static_and_dynamic_scenes(self):
        class FakeTimecode:
            def __init__(self, seconds):
                self.seconds = seconds

            def get_seconds(self):
                return self.seconds

        scenes = [
            (FakeTimecode(0), FakeTimecode(3)),
            (FakeTimecode(3), FakeTimecode(12)),
            (FakeTimecode(12), FakeTimecode(20)),
        ]

        def fake_motion(_path, start, _end):
            if start == 3:
                return 0.3
            if start == 12:
                return 0.1
            return 0.0

        with patch.object(commentary, "detect_scenes", return_value=(scenes, 30)), \
             patch.object(commentary, "_estimate_scene_motion_score", side_effect=fake_motion), \
             patch.object(commentary, "OPENAI_MAX_FRAMES", 20):
            samples = commentary._select_openai_scene_aware_frame_samples("video.mp4", 20)

        self.assertEqual(9, len(samples))
        self.assertEqual([1, 2, 2, 2, 2, 2, 3, 3, 3], [sample["scene_index"] for sample in samples])
        self.assertIn("scene_start", samples[0])
        self.assertIn("motion_score", samples[0])

    def test_openai_scene_aware_samples_densify_long_scenes_by_interval(self):
        class FakeTimecode:
            def __init__(self, seconds):
                self.seconds = seconds

            def get_seconds(self):
                return self.seconds

        scenes = [(FakeTimecode(0), FakeTimecode(60))]

        with patch.object(commentary, "detect_scenes", return_value=(scenes, 30)), \
             patch.object(commentary, "_estimate_scene_motion_score", return_value=0.1), \
             patch.object(commentary, "OPENAI_FRAME_INTERVAL_SECONDS", 6), \
             patch.object(commentary, "OPENAI_SCENE_MAX_KEYFRAMES", 12), \
             patch.object(commentary, "OPENAI_MAX_FRAMES", 180):
            samples = commentary._select_openai_scene_aware_frame_samples("video.mp4", 60)

        self.assertEqual(10, len(samples))
        self.assertEqual([1] * 10, [sample["scene_index"] for sample in samples])
        self.assertLessEqual(max(sample["timestamp"] for sample in samples), 59.95)

    def test_openai_scene_aware_samples_respect_global_max_frames(self):
        class FakeTimecode:
            def __init__(self, seconds):
                self.seconds = seconds

            def get_seconds(self):
                return self.seconds

        scenes = [
            (FakeTimecode(index * 10), FakeTimecode((index + 1) * 10))
            for index in range(20)
        ]

        with patch.object(commentary, "detect_scenes", return_value=(scenes, 30)), \
             patch.object(commentary, "_estimate_scene_motion_score", return_value=0.3), \
             patch.object(commentary, "OPENAI_MAX_FRAMES", 10):
            samples = commentary._select_openai_scene_aware_frame_samples("video.mp4", 200)

        self.assertEqual(10, len(samples))
        self.assertEqual(sorted(sample["timestamp"] for sample in samples), [sample["timestamp"] for sample in samples])

    def test_extract_openai_analysis_frames_preserves_scene_metadata(self):
        samples = [{
            "timestamp": 5.0,
            "scene_index": 1,
            "scene_start": 0.0,
            "scene_end": 10.0,
            "scene_duration": 10.0,
            "sample_role": "middle",
            "motion_score": 0.2,
        }]

        def fake_run_command(cmd):
            with open(cmd[-1], "wb") as frame_file:
                frame_file.write(b"jpg")

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(commentary, "OPENAI_SCENE_AWARE_SAMPLING", True), \
             patch.object(commentary, "_select_openai_scene_aware_frame_samples", return_value=samples), \
             patch.object(commentary, "_run_command", side_effect=fake_run_command):
            frames = commentary._extract_openai_analysis_frames("video.mp4", tmpdir, 10)

        self.assertEqual(1, len(frames))
        self.assertEqual(1, frames[0]["scene_index"])
        self.assertEqual("middle", frames[0]["sample_role"])

    def test_openai_visual_batch_prompt_includes_scene_metadata(self):
        prompt = commentary._openai_visual_batch_prompt(
            "Demo",
            30,
            [{
                "timestamp": 5.0,
                "scene_index": 2,
                "scene_start": 4.0,
                "scene_end": 8.0,
                "sample_role": "middle",
                "motion_score": 0.2,
            }],
            1,
            1,
        )

        self.assertIn("scene_index", prompt)
        self.assertIn("scene_start", prompt)
        self.assertIn("sample_role", prompt)

    def test_openai_visual_analysis_prompt_omits_raw_batches(self):
        visual_analysis = {
            "provider": "openai_compatible",
            "model": "demo-model",
            "frame_count": 128,
            "batch_count": 4,
            "observations": [{"timestamp": 1, "visual": "worker sorts copper"}],
            "candidate_segments": [{"start": 0, "end": 10, "reason": "clear process stage"}],
            "batches": [{"raw_analysis": "x" * 100000}],
        }

        text = commentary._openai_visual_analysis_prompt_text(visual_analysis)

        self.assertIn("worker sorts copper", text)
        self.assertIn("clear process stage", text)
        self.assertNotIn("raw_analysis", text)
        self.assertNotIn("batches", text)
        self.assertLessEqual(len(text), commentary.OPENAI_VISUAL_PROMPT_MAX_CHARS)

    def test_openai_visual_analysis_compaction_preserves_full_timeline_coverage(self):
        visual_analysis = {
            "observations": [
                {"timestamp": index * 30, "visual": f"stage {index}"}
                for index in range(300)
            ],
            "candidate_segments": [
                {"start": index * 30, "end": index * 30 + 10, "reason": f"candidate {index}"}
                for index in range(240)
            ],
        }

        compact = commentary._compact_openai_visual_analysis(visual_analysis)
        observation_timestamps = [item["timestamp"] for item in compact["observations"]]
        candidate_starts = [item["start"] for item in compact["candidate_segments"]]

        self.assertEqual(220, len(compact["observations"]))
        self.assertEqual(160, len(compact["candidate_segments"]))
        self.assertEqual(300, compact["timeline_coverage"]["observations"])
        self.assertEqual(240, compact["timeline_coverage"]["candidate_segments"])
        self.assertEqual(0, observation_timestamps[0])
        self.assertEqual(8970, observation_timestamps[-1])
        self.assertEqual(0, candidate_starts[0])
        self.assertEqual(7170, candidate_starts[-1])

    def test_video_content_builder_uses_files_api_for_complete_video(self):
        with self.assertRaisesRegex(Exception, "Missing analysis video"):
            commentary._build_video_analysis_contents("prompt", "")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            upload_calls = []
            uploaded = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="ACTIVE",
            )
            def fake_upload(file):
                upload_calls.append(file)
                return uploaded

            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=fake_upload,
                    get=lambda name: uploaded,
                )
            )

            contents = commentary._build_video_analysis_contents("prompt", path, client=client, duration=12)
        finally:
            os.remove(path)

        self.assertEqual([path], upload_calls)
        self.assertEqual(1, len(contents))
        file_data = contents[0].parts[0].file_data
        self.assertEqual("https://files.example/video-1", file_data.file_uri)
        self.assertEqual("video/mp4", file_data.mime_type)
        self.assertEqual("prompt", contents[0].parts[1].text)

    def test_long_video_content_builder_adds_low_fps_metadata_to_reduce_input_tokens(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            uploaded = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="ACTIVE",
            )
            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=lambda file: uploaded,
                    get=lambda name: uploaded,
                )
            )

            contents = commentary._build_video_analysis_contents("prompt", path, client=client, duration=3935)
        finally:
            os.remove(path)

        video_part = contents[0].parts[0]
        self.assertIsNotNone(video_part.video_metadata)
        self.assertLess(video_part.video_metadata.fps, 1.0)

    def test_video_content_builder_reuses_existing_files_api_uri_without_upload(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            upload_calls = []
            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=lambda file: upload_calls.append(file),
                )
            )

            contents = commentary._build_video_analysis_contents(
                "prompt",
                path,
                client=client,
                duration=12,
                gemini_file={"uri": "https://files.example/reused", "mime_type": "video/mp4"},
            )
        finally:
            os.remove(path)

        self.assertEqual([], upload_calls)
        file_data = contents[0].parts[0].file_data
        self.assertEqual("https://files.example/reused", file_data.file_uri)
        self.assertEqual("video/mp4", file_data.mime_type)

    def test_video_upload_retries_transient_disconnect_once(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            uploaded = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="ACTIVE",
            )
            calls = []

            def fake_upload(file):
                calls.append(file)
                if len(calls) == 1:
                    raise RuntimeError("Server disconnected without sending a response")
                return uploaded

            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=fake_upload,
                    get=lambda name: uploaded,
                )
            )

            part = commentary._upload_gemini_video_part(client, path)
        finally:
            os.remove(path)

        self.assertEqual([path, path], calls)
        self.assertEqual("https://files.example/video-1", part.file_data.file_uri)

    def test_large_video_processing_timeout_scales_beyond_default_wait(self):
        timeout = commentary._gemini_file_processing_timeout(
            duration=3935,
            file_size=180 * 1024 * 1024,
        )

        self.assertGreaterEqual(timeout, 3148)
        self.assertGreaterEqual(timeout, commentary.GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS)

    def test_video_upload_waits_long_enough_for_large_file_processing(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            uploaded = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="PROCESSING",
            )
            active = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="ACTIVE",
            )
            get_calls = []

            def fake_get(name):
                get_calls.append(name)
                return active if len(get_calls) >= 75 else uploaded

            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=lambda file: uploaded,
                    get=fake_get,
                )
            )

            with patch.object(commentary.time, "sleep"):
                part = commentary._upload_gemini_video_part(client, path)
        finally:
            os.remove(path)

        self.assertGreaterEqual(len(get_calls), 75)
        self.assertEqual("https://files.example/video-1", part.file_data.file_uri)

    def test_video_content_builder_rejects_files_over_official_hard_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name

        try:
            with patch.object(os.path, "getsize", return_value=commentary.GEMINI_FILES_API_HARD_MAX_BYTES + 1):
                with self.assertRaisesRegex(Exception, "超过 Gemini Files API 单文件上限"):
                    commentary._build_video_analysis_contents("prompt", path, client=object(), duration=1)
        finally:
            os.remove(path)

    def test_prepare_analysis_video_compresses_even_files_api_sized_video_to_360p_with_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "analysis_low.mp4")
            prepared_path = os.path.join(tmpdir, "analysis_low_gemini_360p.mp4")
            with open(source_path, "wb") as f:
                f.write(b"small")

            def fake_run_command(cmd, **_kwargs):
                self.assertIn("ffmpeg", cmd[0])
                self.assertIn("scale=-2:360", cmd)
                self.assertIn("-c:a", cmd)
                self.assertIn("48k", cmd)
                with open(prepared_path, "wb") as f:
                    f.write(b"small")

            with patch.object(os.path, "getsize", return_value=commentary.GEMINI_ANALYSIS_TARGET_MAX_BYTES - 1), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command) as run_command:
                prepared = commentary._prepare_analysis_video_for_gemini(source_path, tmpdir)

        self.assertTrue(prepared.endswith("analysis_low_gemini_360p.mp4"))
        run_command.assert_called_once()

    def test_prepare_analysis_video_falls_back_to_noaudio_if_360p_audio_copy_is_too_large(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "analysis_low.mp4")
            prepared_path = os.path.join(tmpdir, "analysis_low_gemini_360p.mp4")
            no_audio_path = os.path.join(tmpdir, "analysis_low_gemini_360p_noaudio.mp4")
            with open(source_path, "wb") as f:
                f.write(b"source")

            def fake_run_command(cmd, **_kwargs):
                output_path = cmd[-1]
                with open(output_path, "wb") as f:
                    f.write(b"small")

            with patch.object(os.path, "getsize", side_effect=[
                commentary.GEMINI_ANALYSIS_TARGET_MAX_BYTES + 1,
                commentary.GEMINI_ANALYSIS_TARGET_MAX_BYTES - 1,
            ]), patch.object(commentary, "_run_command", side_effect=fake_run_command):
                prepared = commentary._prepare_analysis_video_for_gemini(source_path, tmpdir)

            self.assertEqual(os.path.normpath(no_audio_path), os.path.normpath(prepared))
            self.assertTrue(os.path.exists(prepared_path))

    def test_prepare_analysis_video_oversized_error_uses_configured_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "analysis_low.mp4")
            with open(source_path, "wb") as f:
                f.write(b"source")

            def fake_run_command(cmd, **_kwargs):
                with open(cmd[-1], "wb") as f:
                    f.write(b"still-large")

            with patch.object(os.path, "getsize", return_value=commentary.GEMINI_ANALYSIS_TARGET_MAX_BYTES + 1), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):
                with self.assertRaises(Exception) as ctx:
                    commentary._prepare_analysis_video_for_gemini(source_path, tmpdir)

        limit_mb = commentary.GEMINI_ANALYSIS_TARGET_MAX_BYTES // (1024 * 1024)
        self.assertIn(f"{limit_mb}MB", str(ctx.exception))
        self.assertNotIn("512MB", str(ctx.exception))

    def test_full_source_visual_edit_can_stream_copy_without_upscaling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            output_path = os.path.join(tmpdir, "out.mp4")
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir)
            open(source_path, "wb").close()
            commands = []

            def fake_run_command(cmd, **_kwargs):
                commands.append(cmd)
                if cmd[0] == "ffmpeg" and "-f" not in cmd:
                    part_path = cmd[-1]
                    with open(part_path, "wb") as f:
                        f.write(b"part")

            with patch.object(commentary, "_run_command", side_effect=fake_run_command):
                commentary._create_visual_edit(
                    source_path,
                    [{"start": 0.0, "end": 3935.0, "reason": "full"}],
                    output_path,
                    "16:9",
                    work_dir,
                    preserve_source_resolution=True,
                )

        first_cmd = commands[0]
        self.assertNotIn("-vf", first_cmd)
        self.assertIn("-c:v", first_cmd)
        self.assertEqual("copy", first_cmd[first_cmd.index("-c:v") + 1])
        self.assertIn("-an", first_cmd)

    def test_commentary_ffmpeg_commands_are_thread_limited(self):
        limited = commentary._limit_ffmpeg_threads(["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"])

        self.assertIn("-threads", limited)
        self.assertIn(str(commentary.FFMPEG_THREADS), limited)
        self.assertLess(limited.index("-threads"), limited.index("-i"))

    def test_commentary_ffmpeg_thread_limiter_preserves_existing_threads(self):
        cmd = ["ffmpeg", "-y", "-threads", "1", "-i", "in.mp4", "out.mp4"]

        limited = commentary._limit_ffmpeg_threads(cmd)
        self.assertEqual("-threads", limited[2])
        self.assertEqual("1", limited[3])
        self.assertEqual(cmd[4:], limited[4:])

    def test_burn_subtitles_uses_thread_limited_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            subtitle_path = os.path.join(tmpdir, "subs.ass")
            output_path = os.path.join(tmpdir, "final.mp4")
            open(video_path, "wb").close()
            open(subtitle_path, "w").close()

            with patch.object(commentary, "_run_command") as run_command, \
                patch.object(commentary.subprocess, "run") as raw_run:
                commentary._burn_subtitles(video_path, subtitle_path, output_path)

        run_command.assert_called_once()
        raw_run.assert_not_called()

    def test_ass_subtitle_header_enables_smart_wrapping(self):
        header = commentary._ass_header_lines()

        self.assertIn("WrapStyle: 0", header)
        self.assertNotIn("WrapStyle: 2", header)

    def test_ass_subtitle_header_matches_video_dimensions(self):
        header = commentary._ass_header_lines(1920, 1080)
        style = next(line for line in header if line.startswith("Style: Default"))

        self.assertIn("PlayResX: 1920", header)
        self.assertIn("PlayResY: 1080", header)
        self.assertIn(",49,", style)
        self.assertIn(",115,115,79,", style)

    def test_ass_subtitle_line_width_expands_for_wide_video(self):
        self.assertGreater(
            commentary._subtitle_max_line_units(1920, 1080),
            commentary.ASS_SUBTITLE_MAX_LINE_UNITS,
        )

    def test_long_ass_subtitle_text_is_wrapped_before_burning(self):
        lines = commentary._ass_header_lines()
        sentence = "这是一个非常长的二创解说字幕句子需要自动换行否则会超出竖屏视频的左右边界影响观看体验。"

        commentary._append_weighted_subtitle_lines(lines, [sentence], 0.0, 4.0)

        dialogue = lines[-1]
        self.assertIn(r"\N", dialogue)
        rendered_text = dialogue.rsplit(",", 1)[-1]
        for line in rendered_text.split(r"\N"):
            units = sum(commentary._subtitle_char_units(char) for char in line)
            self.assertLessEqual(units, commentary.ASS_SUBTITLE_MAX_LINE_UNITS)

    def test_downloader_quality_settings_cap_high_default_at_720p_and_keep_low_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            high = main._youtube_download_settings("My Video", tmpdir)
            low = main._youtube_download_settings("My Video", tmpdir, quality="low", filename_suffix="_analysis_low")

        self.assertIn("bestvideo[height<=720][vcodec^=avc1][ext=mp4]", high["format"])
        self.assertIn("best[height<=720][ext=mp4]", high["format"])
        self.assertTrue(high["expected_file"].endswith("My_Video.mp4"))
        self.assertIn("height<=360", low["format"])
        self.assertTrue(low["expected_file"].endswith("My_Video_analysis_low.mp4"))

    def test_video_mode_downloads_low_analysis_copy_but_renders_from_high_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            high_path = os.path.join(tmpdir, "source.mp4")
            low_path = os.path.join(tmpdir, "source_analysis_low.mp4")
            open(high_path, "wb").close()
            open(low_path, "wb").close()

            download_calls = []

            def fake_download(url, output_dir=".", quality="high", filename_suffix=""):
                download_calls.append((url, output_dir, quality, filename_suffix))
                return (low_path if quality == "low" else high_path), "Source Title"

            prepared_analysis_path = os.path.join(tmpdir, "source_analysis_low_gemini_360p.mp4")
            open(prepared_analysis_path, "wb").close()

            with patch.object(commentary, "download_youtube_video", side_effect=fake_download), \
                patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=prepared_analysis_path) as prepare_analysis, \
                patch.object(commentary, "_get_video_info", return_value={"duration": 42, "width": 1920, "height": 1080, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "_extract_keyframes") as extract_frames, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "Narration",
                    "edit_segments": [{"start": 0, "end": 10, "reason": "best part"}],
                    "chapters": [],
                    "hashtags": [],
                }) as generate_script, \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=lambda **kwargs: open(kwargs["output_path"], "wb").close()), \
                patch.object(commentary, "_create_visual_edit") as create_visual, \
                patch.object(commentary, "_fit_video_to_voiceover"), \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None) as create_ambient, \
                patch.object(commentary, "_mix_voiceover_with_video"), \
                patch.object(commentary, "_generate_commentary_covers", return_value={}):

                result = commentary.generate_commentary_video(
                    source="https://youtube.test/watch?v=1",
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="url",
                    subtitles=False,
                    analysis_mode="video",
                    gemini_model="gemini-custom",
                )

        self.assertEqual(
            [
                ("https://youtube.test/watch?v=1", tmpdir, "high", ""),
                ("https://youtube.test/watch?v=1", tmpdir, "low", "_analysis_low"),
            ],
            download_calls,
        )
        transcribe.assert_not_called()
        extract_frames.assert_not_called()
        self.assertEqual("", generate_script.call_args.kwargs["transcript"]["text"])
        self.assertEqual([], generate_script.call_args.kwargs["transcript"]["segments"])
        prepare_analysis.assert_called_once_with(low_path, tmpdir, progress=ANY)
        self.assertEqual(prepared_analysis_path, generate_script.call_args.kwargs["analysis_video_path"])
        self.assertEqual("video", generate_script.call_args.kwargs["analysis_mode"])
        create_visual.assert_called_once()
        self.assertEqual(high_path, create_visual.call_args.args[0])
        create_ambient.assert_called_once()
        self.assertEqual(high_path, create_ambient.call_args.args[0])
        self.assertEqual("video", result["analysis_mode"])
        self.assertEqual("source_analysis_low_gemini_360p.mp4", result["analysis_video"])
        self.assertEqual("source.mp4", result["source_video"])
        self.assertEqual("gemini-custom", result["gemini_model"])

    def test_video_mode_reuses_prepared_analysis_video_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            prepared_path = os.path.join(tmpdir, "source_gemini_360p.mp4")
            open(source_path, "wb").close()
            open(prepared_path, "wb").close()

            with patch.object(commentary, "_prepare_analysis_video_for_gemini") as prepare_analysis, \
                patch.object(commentary, "_get_video_info", return_value={"duration": 42, "width": 1920, "height": 1080, "fps": 30}), \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "Narration",
                    "edit_segments": [{"start": 0, "end": 10, "reason": "best part"}],
                    "chapters": [],
                    "hashtags": [],
                }) as generate_script, \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=lambda **kwargs: open(kwargs["output_path"], "wb").close()), \
                patch.object(commentary, "_create_visual_edit"), \
                patch.object(commentary, "_fit_video_to_voiceover"), \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_mix_voiceover_with_video"):

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    prepared_analysis_video_path=prepared_path,
                    gemini_file={"uri": "https://files.example/reused", "mime_type": "video/mp4"},
                )

        prepare_analysis.assert_not_called()
        self.assertEqual(prepared_path, generate_script.call_args.kwargs["analysis_video_path"])
        self.assertEqual("https://files.example/reused", generate_script.call_args.kwargs["gemini_file"]["uri"])
        self.assertEqual("source_gemini_360p.mp4", result["analysis_video"])

    def test_full_duration_creates_comprehensive_edit_instead_of_raw_full_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()

            sparse_segments = [
                {"start": 0, "end": 40, "reason": "intro"},
                {"start": 110, "end": 140, "reason": "detail"},
                {"start": 230, "end": 280, "reason": "detail"},
            ]

            def fake_create_visual(_video_path, _segments, output_path, _aspect_mode, _work_dir, **_kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"video")

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
                    "narration_blocks": scene_matched_blocks(),
                    "edit_segments": sparse_segments,
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=lambda **kwargs: open(kwargs["output_path"], "wb").close()), \
                patch.object(commentary, "_get_audio_duration", return_value=120), \
                patch.object(commentary, "_create_visual_edit", side_effect=fake_create_visual) as create_visual, \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", return_value=("ambient.m4a", [])) as create_synced, \
                patch.object(commentary, "_fit_video_to_voiceover") as fit_video, \
                patch.object(commentary, "_create_ambient_audio_bed", return_value="ambient.m4a"), \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video:

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    target_duration="full",
                )

        transcribe.assert_not_called()
        selected_seconds = sum(segment["end"] - segment["start"] for segment in result["edit_segments"])
        target_seconds = commentary._target_visual_duration_seconds(3935, "full")
        self.assertGreaterEqual(selected_seconds, target_seconds * 0.9)
        self.assertLess(selected_seconds, 3935 * 0.5)
        create_visual.assert_not_called()
        create_synced.assert_called_once()
        expected_segments = commentary._narration_blocks_to_edit_segments(scene_matched_blocks())
        self.assertEqual(result["edit_segments"], expected_segments)
        fit_video.assert_not_called()
        self.assertEqual(result["edited_visual"], result["timed_visual"])
        self.assertTrue(result["auto_video_speed"])
        self.assertGreater(result["auto_video_speed_summary"]["accelerated_count"], 0)
        self.assertGreater(result["auto_video_speed_summary"]["saved_seconds"], 0)
        mix_video.assert_called_once()
        self.assertTrue(mix_video.call_args.kwargs["trim_to_voiceover"])

    def test_full_duration_renders_ai_planned_commentary_episodes_from_final_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()
            blocks = scene_matched_blocks()
            rendered_commands = []

            def fake_mix(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"final")

            def fake_run_command(cmd, cwd=None):
                rendered_commands.append((cmd, cwd))
                with open(cmd[-1], "wb") as f:
                    f.write(b"episode")

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in blocks),
                    "narration_blocks": blocks,
                    "episode_plan": {"should_split": True, "reason": "工序分明，适合连续分集"},
                    "episodes": [
                        {"episode_number": 1, "title": "第1集", "summary": "开端", "start_block": 1, "end_block": 4},
                        {"episode_number": 2, "title": "第2集", "summary": "推进", "start_block": 5, "end_block": 8},
                        {"episode_number": 3, "title": "第3集", "summary": "结尾", "start_block": 9, "end_block": 12},
                    ],
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "_get_audio_duration", return_value=120), \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", return_value=("ambient.m4a", [10.0] * len(blocks))), \
                patch.object(commentary, "_mix_voiceover_with_video", side_effect=fake_mix), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    target_duration="full",
                )

        self.assertTrue(result["episode_plan"]["should_split"])
        self.assertEqual(3, len(result["episodes"]))
        self.assertEqual("/videos", result["episodes"][0]["video_url"][:7])
        self.assertEqual(0.0, result["episodes"][0]["output_start"])
        self.assertEqual(40.0, result["episodes"][0]["duration"])
        self.assertEqual(3, len(rendered_commands))
        self.assertEqual("-ss", rendered_commands[0][0][2])
        self.assertEqual("0.000", rendered_commands[0][0][3])
        self.assertTrue(os.path.exists(result["episodes"][2]["video_path"]))

    def test_full_duration_rejects_overlong_voiceover_before_visual_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"audio")

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
                    "narration_blocks": scene_matched_blocks(),
                    "edit_segments": commentary._narration_blocks_to_edit_segments(scene_matched_blocks()),
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=5000), \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", side_effect=lambda **kwargs: (open(kwargs["voiceover_path"], "wb").close(), (None, []))[1]), \
                patch.object(commentary, "_create_visual_edit") as create_visual:

                with self.assertRaisesRegex(Exception, "voiceover is too long"):
                    commentary.generate_commentary_video(
                        source=source_path,
                        output_dir=tmpdir,
                        gemini_key="key",
                        source_type="file",
                        subtitles=False,
                        analysis_mode="video",
                        target_duration="full",
                    )

        transcribe.assert_not_called()
        create_visual.assert_not_called()

    def test_full_duration_skips_full_video_retiming_to_avoid_heavy_reencode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"audio")

            def fake_create_visual(_video_path, _segments, output_path, _aspect_mode, _work_dir, **_kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"video")

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "Narration",
                    "edit_segments": commentary._fallback_edit_segments(3935, "full"),
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=1100), \
                patch.object(commentary, "_create_visual_edit", side_effect=fake_create_visual), \
                patch.object(commentary, "_fit_video_to_voiceover") as fit_video, \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video:

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    target_duration="full",
                )

        transcribe.assert_not_called()
        fit_video.assert_not_called()
        self.assertEqual(result["edited_visual"], result["timed_visual"])
        self.assertTrue(mix_video.call_args.kwargs["trim_to_voiceover"])

    def test_full_duration_burns_subtitles_into_final_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"audio")

            def fake_create_visual(_video_path, _segments, output_path, _aspect_mode, _work_dir, **_kwargs):
                with open(output_path, "wb") as f:
                    f.write(b"video")

            def fake_subtitles(_narration, _voiceover_path, output_path, **_kwargs):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("[Script Info]")

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video"), \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "Narration.",
                    "edit_segments": commentary._fallback_edit_segments(3935, "full"),
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=1100), \
                patch.object(commentary, "_create_visual_edit", side_effect=fake_create_visual), \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video, \
                patch.object(commentary, "_probe_video_dimensions", return_value=(854, 480)), \
                patch.object(commentary, "_write_text_timed_ass", side_effect=fake_subtitles) as write_subtitles, \
                patch.object(commentary, "_burn_subtitles") as burn_subtitles:

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=True,
                    analysis_mode="video",
                    target_duration="full",
                )

        mix_video.assert_called_once()
        write_subtitles.assert_called_once()
        self.assertEqual((854, 480), write_subtitles.call_args.kwargs["video_dimensions"])
        burn_subtitles.assert_called_once()
        self.assertEqual("Remix_final.mp4", result["video_filename"])
        self.assertEqual("Remix_commentary.ass", result["subtitle"])

    def test_full_duration_targets_comprehensive_edit_not_raw_full_source(self):
        self.assertEqual(
            1180,
            int(commentary._target_visual_duration_seconds(3935, "full")),
        )
        segments = commentary._fallback_edit_segments(3935, "full")
        total = sum(segment["end"] - segment["start"] for segment in segments)
        self.assertGreater(len(segments), 1)
        self.assertGreaterEqual(total, 1100)
        self.assertLess(total, 3935 * 0.5)
        self.assertGreaterEqual(
            commentary._minimum_narration_chars(3935, "full", "zh"),
            2000,
        )

    def test_full_duration_prompt_demands_long_scene_matched_narration(self):
        prompt = commentary._build_commentary_prompt(
            transcript={"text": "source transcript", "segments": [], "language": "en"},
            video_title="Demo",
            duration=3935,
            language="zh",
            style="documentary",
            target_duration="full",
            analysis_mode="video",
        )

        self.assertIn("comprehensive long-form commentary edit", prompt)
        self.assertIn("not a raw full-length copy", prompt)
        self.assertIn("Do not preserve the entire source", prompt)
        self.assertIn("Treat narration_blocks as the production timeline", prompt)
        self.assertIn("do not put 2 minutes of words into a 20-second visual range", prompt)
        self.assertIn("do not narrate every second like a robot", prompt)
        self.assertIn("pause=true blocks", prompt)
        self.assertIn("original source audio", prompt)
        self.assertIn("vary rate and pitch", prompt)
        self.assertIn("final payoff", prompt)
        self.assertIn("effect showcase", prompt)
        self.assertIn("video_speed", prompt)
        self.assertIn("episode_plan.should_split", prompt)
        self.assertIn("start_block", prompt)
        self.assertIn('"narration_blocks"', prompt)
        self.assertIn('"episode_plan"', prompt)
        self.assertIn('"episodes"', prompt)
        self.assertIn('"pause"', prompt)
        self.assertIn('"rate"', prompt)
        self.assertIn('"pitch"', prompt)
        self.assertIn('"video_speed"', prompt)
        self.assertIn(f"at least {commentary._minimum_narration_chars(3935, 'full', 'zh')}", prompt)

    def test_full_duration_normalizes_ai_episode_plan_by_narration_blocks(self):
        data = {
            "title": "Remix",
            "narration": "讲" * 3000,
            "narration_blocks": scene_matched_blocks(count=6, seconds=80),
            "episode_plan": {"should_split": True, "reason": "工序分明"},
            "episodes": [
                {"episode_number": 1, "title": "第1集", "summary": "前半段", "start_block": 1, "end_block": 3},
                {"episode_number": 2, "title": "第2集", "summary": "后半段", "start_block": 4, "end_block": 6},
            ],
        }

        commentary._normalize_script_timeline(data, 600, "full", "zh")

        self.assertTrue(data["episode_plan"]["should_split"])
        self.assertEqual(2, len(data["episodes"]))
        self.assertEqual((1, 3), (data["episodes"][0]["start_block"], data["episodes"][0]["end_block"]))
        self.assertEqual((4, 6), (data["episodes"][1]["start_block"], data["episodes"][1]["end_block"]))
        self.assertEqual(0.0, data["episodes"][0]["start"])
        self.assertEqual(480.0, data["episodes"][1]["end"])

    def test_funny_style_prompt_requires_visual_grounded_china_international_comparison(self):
        prompt = commentary._build_commentary_prompt(
            transcript={"text": "source transcript", "segments": [], "language": "en"},
            video_title="Demo",
            duration=3935,
            language="zh",
            style="funny",
            target_duration="full",
            analysis_mode="video",
        )

        self.assertIn("轻松吐槽风格要求", prompt)
        self.assertIn("国际工厂/海外回收流程与中国工厂/中国回收效率的对比", prompt)
        self.assertIn("必须围绕当前画面", prompt)
        self.assertIn("不要写脱离画面的国际形势", prompt)

    def test_full_duration_rejects_blocks_that_stop_before_late_timeline(self):
        blocks = [
            {"start": i * 160, "end": i * 160 + 100, "visual": "early process", "narration": "讲" * 420, "pause": False}
            for i in range(12)
        ]

        with self.assertRaisesRegex(Exception, "stopped before the end"):
            commentary._validate_commentary_script_for_target(
                {"narration": "讲" * 3000, "narration_blocks": blocks},
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_accepts_blocks_with_late_timeline_coverage(self):
        blocks = [
            {"start": i * 280, "end": i * 280 + 90, "visual": "full process", "narration": "讲" * 380, "pause": False}
            for i in range(11)
        ]
        blocks.append({"start": 3600, "end": 3690, "visual": "ending process", "narration": "讲" * 380, "pause": False})

        commentary._validate_commentary_script_for_target(
            {"narration": "讲" * 5000, "narration_blocks": blocks},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_accepts_near_threshold_narration_with_late_timeline_coverage(self):
        blocks = [
            {"start": i * 280, "end": i * 280 + 90, "visual": "full process", "narration": "讲" * 380, "pause": False}
            for i in range(11)
        ]
        blocks.append({"start": 3600, "end": 3690, "visual": "ending process", "narration": "讲" * 380, "pause": False})
        min_chars = commentary._minimum_narration_chars_for_blocks(
            commentary._normalize_narration_blocks(blocks, 3935),
            3935,
            "full",
            "zh",
        )
        old_default_required = int(commentary.math.floor(min_chars * 0.92))
        narration = "讲" * (old_default_required - 5)

        self.assertGreaterEqual(len(narration), commentary._accepted_minimum_narration_chars(min_chars))
        commentary._validate_commentary_script_for_target(
            {"narration": narration, "narration_blocks": blocks},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_accepts_target_sized_real_cut_above_static_source_retention_fraction(self):
        blocks = [
            {"start": 0, "end": 45, "visual": "opening setup", "narration": "讲" * 220, "pause": False},
            {"start": 60, "end": 105, "visual": "material sorting", "narration": "讲" * 220, "pause": False},
            {"start": 130, "end": 175, "visual": "first machine process", "narration": "讲" * 220, "pause": False},
            {"start": 205, "end": 250, "visual": "inspection step", "narration": "讲" * 220, "pause": False},
            {"start": 285, "end": 330, "visual": "middle assembly", "narration": "讲" * 220, "pause": False},
            {"start": 365, "end": 410, "visual": "machine closeup", "narration": "讲" * 220, "pause": False},
            {"start": 450, "end": 495, "visual": "quality check", "narration": "讲" * 220, "pause": False},
            {"start": 540, "end": 585, "visual": "packaging begins", "narration": "讲" * 220, "pause": False},
            {"start": 635, "end": 680, "visual": "late process", "narration": "讲" * 220, "pause": False},
            {"start": 735, "end": 780, "visual": "final machine run", "narration": "讲" * 220, "pause": False},
            {"start": 835, "end": 880, "visual": "finished parts", "narration": "讲" * 220, "pause": False},
            {"start": 935, "end": 980, "visual": "ending packaging", "narration": "讲" * 220, "pause": False},
        ]
        narration = "讲" * 2700

        self.assertGreater(
            commentary._segments_total_duration(commentary._narration_blocks_to_edit_segments(blocks)),
            1037.2 * commentary.FULL_MODE_MAX_SOURCE_RETENTION_FRACTION,
        )
        commentary._validate_commentary_script_for_target(
            {"narration": narration, "narration_blocks": blocks},
            duration=1037.2,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_can_build_narration_from_scene_matched_blocks(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        block_text = (
            "这一段解说紧贴画面，说明工人正在处理废旧电机、分离铜线、转运材料，并交代这个环节在整个回收流程里的作用。"
            "同时补充机器动作、人工配合、材料状态变化和下一道工序之间的联系。"
            "让观众知道此刻保留这段画面的原因，以及它如何推动废铜重新变成新材料。"
        ) * 4
        payload = {
            "title": "Block Based",
            "summary": "summary",
            "hook": "hook",
            "narration": "太短。",
            "narration_blocks": [
                {"start": i * 320, "end": i * 320 + 100, "narration": block_text}
                for i in range(11)
            ] + [{"start": 3600, "end": 3700, "narration": block_text}],
            "edit_segments": [
                {"start": i * 320, "end": i * 320 + 100, "reason": "process"}
                for i in range(11)
            ] + [{"start": 3600, "end": 3700, "reason": "ending process"}],
            "chapters": [],
            "hashtags": [],
        }

        class BlockModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return pytypes.SimpleNamespace(text=commentary.json.dumps(payload, ensure_ascii=False))

        fake_models = BlockModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)

        with patch.object(commentary, "create_gemini_client", return_value=fake_client):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=3935,
                gemini_key="key",
                analysis_mode="current",
                target_duration="full",
                language="zh",
            )

        self.assertEqual("Block Based", result["title"])
        self.assertEqual(1, len(fake_models.calls))
        self.assertGreaterEqual(
            len(commentary.re.sub(r"\s+", "", result["narration"])),
            commentary._minimum_narration_chars(3935, "full", "zh"),
        )
        self.assertIn(block_text, result["narration"])

    def test_full_duration_validation_counts_pause_blocks_as_non_spoken_time(self):
        blocks = [
            {"start": 0, "end": 390, "visual": "opening process", "narration": "讲" * 1700, "pause": False},
            {"start": 390, "end": 402, "visual": "machine sound reveal", "narration": "", "pause": True},
            {"start": 402, "end": 792, "visual": "main process", "narration": "讲" * 1700, "pause": False},
            {"start": 792, "end": 804, "visual": "original audio beat", "narration": "", "pause": True},
            {"start": 3400, "end": 3796, "visual": "ending process", "narration": "讲" * 1700, "pause": False},
        ]
        spoken_text = "".join(block["narration"] for block in blocks)
        normalized = commentary._normalize_narration_blocks(blocks, 3935)
        min_chars = commentary._minimum_narration_chars_for_blocks(normalized, 3935, "full", "zh")

        self.assertLess(min_chars, commentary._minimum_narration_chars(3935, "full", "zh"))
        self.assertGreaterEqual(len(spoken_text), commentary._accepted_minimum_narration_chars(min_chars))
        commentary._validate_commentary_script_for_target(
            {"narration": spoken_text, "narration_blocks": blocks},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_normalize_narration_blocks_preserves_pause_rate_pitch_and_video_speed(self):
        blocks = commentary._normalize_narration_blocks(
            [
                {"start": 0, "end": 8, "visual": "machine sound", "narration": "", "pause": True},
                {"start": 8, "end": 20, "visual": "worker action", "narration": "画面里工人正在处理铜料。", "rate": "-15%", "pitch": "+4Hz", "video_speed": 1.75},
                {"start": 20, "end": 30, "visual": "empty bad block", "narration": ""},
                {"start": 30, "end": 40, "visual": "invalid prosody", "narration": "继续讲这个画面。", "rate": "fast", "pitch": "high", "speed_up": True},
            ],
            duration=60,
        )

        self.assertEqual(3, len(blocks))
        self.assertTrue(blocks[0]["pause"])
        self.assertEqual("", blocks[0]["narration"])
        self.assertEqual("+0%", blocks[0]["rate"])
        self.assertEqual("+0Hz", blocks[0]["pitch"])
        self.assertEqual(1.0, blocks[0]["video_speed"])
        self.assertFalse(blocks[1]["pause"])
        self.assertEqual("-15%", blocks[1]["rate"])
        self.assertEqual("+4Hz", blocks[1]["pitch"])
        self.assertEqual(1.75, blocks[1]["video_speed"])
        self.assertEqual("+0%", blocks[2]["rate"])
        self.assertEqual("+0Hz", blocks[2]["pitch"])
        self.assertEqual(1.5, blocks[2]["video_speed"])

    def test_auto_video_speed_can_force_blocks_to_original_speed(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 30, "visual": "slow transport", "narration": "moving material", "video_speed": 1.75},
                {"start": 30, "end": 45, "visual": "pause", "narration": "", "pause": True, "video_speed": 1.5},
            ],
            enabled=False,
        )

        self.assertEqual([1.0, 1.0], [block["video_speed"] for block in blocks])

    def test_auto_video_speed_adds_conservative_fallback_for_repetitive_blocks(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 22, "visual": "worker repeats slow transport and loading steps", "narration": "这段是重复搬运和上料。", "video_speed": 1.0},
                {"start": 22, "end": 42, "visual": "final result showcase", "narration": "最终成品亮相。", "video_speed": 1.0},
                {"start": 42, "end": 48, "visual": "short moving shot", "narration": "短镜头。", "video_speed": 1.0},
            ],
            enabled=True,
        )

        self.assertEqual(1.5, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])
        self.assertEqual(1.0, blocks[2]["video_speed"])

    def test_auto_video_speed_preserves_ai_selected_speed(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 22, "visual": "slow transport", "narration": "运输。", "video_speed": 1.75},
                {"start": 22, "end": 48, "visual": "repetitive loading", "narration": "重复上料。", "video_speed": 1.0},
            ],
            enabled=True,
        )

        self.assertEqual(1.75, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])

    def test_edge_voiceover_passes_rate_and_pitch_to_edge_tts(self):
        calls = []

        class FakeCommunicate:
            def __init__(self, text, voice, **kwargs):
                calls.append((text, voice, kwargs))

            async def save(self, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"voice")

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.dict(sys.modules, {"edge_tts": pytypes.SimpleNamespace(Communicate=FakeCommunicate)}):
            output_path = os.path.join(tmpdir, "voice.mp3")
            commentary.generate_edge_voiceover(
                "这段要有一点节奏。",
                output_path,
                voice="zh-CN-YunjianNeural",
                rate="+18%",
                pitch="-5Hz",
            )

        self.assertEqual(1, len(calls))
        self.assertEqual("+18%", calls[0][2]["rate"])
        self.assertEqual("-5Hz", calls[0][2]["pitch"])

    def test_ambient_audio_filter_preserves_source_audio_with_volume_only(self):
        audio_filter = commentary._ambient_audio_filter(0.08)

        self.assertEqual("volume=0.08", audio_filter)
        self.assertNotIn("pan=stereo", audio_filter)
        self.assertNotIn("c0-c1", audio_filter)
        self.assertNotIn("c1-c0", audio_filter)

    def test_extract_original_audio_clip_applies_configured_volume(self):
        commands = []

        def fake_run_command(cmd, cwd=None):
            commands.append(cmd)

        with patch.object(commentary, "_run_command", side_effect=fake_run_command):
            commentary._extract_original_audio_clip("source.mp4", 1.0, 2.0, "out.m4a", volume=0.6)

        self.assertEqual("volume=0.6", commands[0][commands[0].index("-af") + 1])

    def test_extract_original_audio_clip_applies_speed_and_output_duration(self):
        commands = []

        def fake_run_command(cmd, cwd=None):
            commands.append(cmd)

        with patch.object(commentary, "_run_command", side_effect=fake_run_command):
            commentary._extract_original_audio_clip(
                "source.mp4",
                1.0,
                12.0,
                "out.m4a",
                volume=0.6,
                speed=2.0,
                output_duration=6.0,
            )

        audio_filter = commands[0][commands[0].index("-af") + 1]
        self.assertIn("volume=0.6", audio_filter)
        self.assertIn("atempo=2.000000", audio_filter)
        self.assertIn("atrim=0:6.000", audio_filter)
        self.assertEqual("6.000", commands[0][commands[0].index("-t", commands[0].index("-i")) + 1])

    def test_block_synced_render_pause_block_skips_tts_and_uses_original_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "source.mp4")
            open(video_path, "wb").close()
            timed_video_path = os.path.join(tmpdir, "timed.mp4")
            voiceover_path = os.path.join(tmpdir, "voiceover.m4a")
            ambient_audio_path = os.path.join(tmpdir, "ambient.m4a")
            commands = []
            silence_calls = []
            original_audio_calls = []

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"voice")

            def fake_fit_audio(_input_audio_path, output_audio_path, _target_duration):
                with open(output_audio_path, "wb") as f:
                    f.write(b"fit")

            def fake_silence(output_path, duration):
                silence_calls.append((os.path.basename(output_path), duration))
                with open(output_path, "wb") as f:
                    f.write(b"silence")

            def fake_original_audio(_video_path, start, duration, output_path, volume=1.0, speed=1.0, output_duration=None):
                original_audio_calls.append((start, duration, os.path.basename(output_path), volume, speed, output_duration))
                with open(output_path, "wb") as f:
                    f.write(b"original")

            def fake_run_command(cmd, cwd=None):
                commands.append(cmd)
                with open(cmd[-1], "wb") as f:
                    f.write(b"media")

            blocks = [
                {"start": 0, "end": 12, "visual": "worker action", "narration": "画面里工人正在处理铜料。", "rate": "+12%", "pitch": "+3Hz"},
                {"start": 12, "end": 20, "visual": "machine sound reveal", "narration": "", "pause": True},
            ]
            with patch.object(commentary, "_get_video_duration", return_value=20.0), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover) as voiceover, \
                patch.object(commentary, "_get_audio_duration", return_value=11.0), \
                patch.object(commentary, "_fit_audio_part_to_duration", side_effect=fake_fit_audio) as fit_audio, \
                patch.object(commentary, "_create_silent_audio_clip", side_effect=fake_silence), \
                patch.object(commentary, "_extract_original_audio_clip", side_effect=fake_original_audio), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):
                ambient, block_durations = commentary._create_block_synced_visuals_and_audio(
                    video_path=video_path,
                    narration_blocks=blocks,
                    timed_video_path=timed_video_path,
                    voiceover_path=voiceover_path,
                    ambient_audio_path=ambient_audio_path,
                    aspect_mode="16:9",
                    work_dir=tmpdir,
                    tts_provider="edge",
                    language="zh",
                    elevenlabs_key=None,
                    voice_id="voice",
                    edge_voice="zh-CN-YunjianNeural",
                    original_audio_volume=0.08,
                    pause_original_audio_volume=0.6,
                    preserve_source_resolution=True,
                )

        self.assertEqual(1, voiceover.call_count)
        self.assertEqual("+12%", voiceover.call_args.kwargs["rate"])
        self.assertEqual("+3Hz", voiceover.call_args.kwargs["pitch"])
        self.assertEqual(1, fit_audio.call_count)
        self.assertEqual(12.0, fit_audio.call_args.args[2])
        self.assertEqual(1, len(silence_calls))
        self.assertEqual(2, len(original_audio_calls))
        self.assertIn((0.0, 12.0, "block_ambient_001_s1000_src12000_dur12000.m4a", 0.08, 1.0, 12.0), original_audio_calls)
        self.assertIn((12.0, 8.0, "block_ambient_002_s1000_src8000_dur8000.m4a", 0.6, 1.0, 8.0), original_audio_calls)
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertIn("12.000", [cmd[cmd.index("-t") + 1] for cmd in video_cmds])
        self.assertEqual(ambient_audio_path, ambient)
        self.assertEqual([12.0, 8.0], block_durations)

    def test_block_synced_render_preserves_selected_visual_ranges_for_narrated_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "source.mp4")
            open(video_path, "wb").close()
            timed_video_path = os.path.join(tmpdir, "timed.mp4")
            voiceover_path = os.path.join(tmpdir, "voiceover.m4a")
            ambient_audio_path = os.path.join(tmpdir, "ambient.m4a")
            commands = []
            fit_calls = []

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"voice")

            def fake_fit_audio(input_audio_path, output_audio_path, target_duration):
                fit_calls.append((os.path.basename(input_audio_path), os.path.basename(output_audio_path), target_duration))
                with open(output_audio_path, "wb") as f:
                    f.write(b"fit")

            def fake_run_command(cmd, cwd=None):
                commands.append(cmd)
                with open(cmd[-1], "wb") as f:
                    f.write(b"media")

            blocks = scene_matched_blocks(count=2, seconds=12.0, text="画面里工人正在处理铜料。")
            with patch.object(commentary, "_get_video_duration", return_value=24.0), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover) as voiceover, \
                patch.object(commentary, "_get_audio_duration", return_value=11.0), \
                patch.object(commentary, "_fit_audio_part_to_duration", side_effect=fake_fit_audio), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):
                ambient, block_durations = commentary._create_block_synced_visuals_and_audio(
                    video_path=video_path,
                    narration_blocks=blocks,
                    timed_video_path=timed_video_path,
                    voiceover_path=voiceover_path,
                    ambient_audio_path=ambient_audio_path,
                    aspect_mode="16:9",
                    work_dir=tmpdir,
                    tts_provider="edge",
                    language="zh",
                    elevenlabs_key=None,
                    voice_id="voice",
                    edge_voice="zh-CN-YunjianNeural",
                    original_audio_volume=0.08,
                    preserve_source_resolution=True,
                )

        self.assertEqual(2, voiceover.call_count)
        self.assertEqual(2, len(fit_calls))
        self.assertEqual(12.0, fit_calls[0][2])
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertEqual(2, len(video_cmds))
        self.assertIn("-t", video_cmds[0])
        self.assertEqual("12.000", video_cmds[0][video_cmds[0].index("-t") + 1])
        self.assertIn("setsar=1", video_cmds[0])
        self.assertNotIn("setpts=PTS/", " ".join(video_cmds[0]))
        self.assertEqual(ambient_audio_path, ambient)
        self.assertEqual([12.0, 12.0], block_durations)

    def test_block_synced_render_applies_ai_video_speed_to_video_and_ambient_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "source.mp4")
            open(video_path, "wb").close()
            timed_video_path = os.path.join(tmpdir, "timed.mp4")
            voiceover_path = os.path.join(tmpdir, "voiceover.m4a")
            ambient_audio_path = os.path.join(tmpdir, "ambient.m4a")
            commands = []
            fit_calls = []
            original_audio_calls = []

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"voice")

            def fake_fit_audio(input_audio_path, output_audio_path, target_duration):
                fit_calls.append((os.path.basename(input_audio_path), os.path.basename(output_audio_path), target_duration))
                with open(output_audio_path, "wb") as f:
                    f.write(b"fit")

            def fake_original_audio(_video_path, start, duration, output_path, volume=1.0, speed=1.0, output_duration=None):
                original_audio_calls.append((start, duration, os.path.basename(output_path), volume, speed, output_duration))
                with open(output_path, "wb") as f:
                    f.write(b"original")

            def fake_run_command(cmd, cwd=None):
                commands.append(cmd)
                with open(cmd[-1], "wb") as f:
                    f.write(b"media")

            blocks = [{
                "start": 0,
                "end": 12,
                "visual": "slow transport section",
                "narration": "这段运输过程可以稍微加速，但保留完整起止动作。",
                "video_speed": 2.0,
            }]
            with patch.object(commentary, "_get_video_duration", return_value=12.0), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=5.0), \
                patch.object(commentary, "_fit_audio_part_to_duration", side_effect=fake_fit_audio), \
                patch.object(commentary, "_extract_original_audio_clip", side_effect=fake_original_audio), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):
                ambient, block_durations = commentary._create_block_synced_visuals_and_audio(
                    video_path=video_path,
                    narration_blocks=blocks,
                    timed_video_path=timed_video_path,
                    voiceover_path=voiceover_path,
                    ambient_audio_path=ambient_audio_path,
                    aspect_mode="16:9",
                    work_dir=tmpdir,
                    tts_provider="edge",
                    language="zh",
                    elevenlabs_key=None,
                    voice_id="voice",
                    edge_voice="zh-CN-YunjianNeural",
                    original_audio_volume=0.08,
                    preserve_source_resolution=True,
                )

        self.assertEqual(1, len(fit_calls))
        self.assertEqual(6.0, fit_calls[0][2])
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertEqual(1, len(video_cmds))
        self.assertEqual("12.000", video_cmds[0][video_cmds[0].index("-t") + 1])
        self.assertIn("setpts=PTS/2.000000", video_cmds[0][video_cmds[0].index("-vf") + 1])
        self.assertEqual((0.0, 12.0, "block_ambient_001_s2000_src12000_dur6000.m4a", 0.08, 2.0, 6.0), original_audio_calls[0])
        self.assertEqual(ambient_audio_path, ambient)
        self.assertEqual([6.0], block_durations)

    def test_block_synced_render_tightens_visuals_when_tts_is_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "source.mp4")
            open(video_path, "wb").close()
            timed_video_path = os.path.join(tmpdir, "timed.mp4")
            voiceover_path = os.path.join(tmpdir, "voiceover.m4a")
            ambient_audio_path = os.path.join(tmpdir, "ambient.m4a")
            commands = []
            fit_calls = []
            original_audio_calls = []
            progress_messages = []

            def fake_voiceover(**kwargs):
                with open(kwargs["output_path"], "wb") as f:
                    f.write(b"voice")

            def fake_fit_audio(input_audio_path, output_audio_path, target_duration):
                fit_calls.append((os.path.basename(input_audio_path), os.path.basename(output_audio_path), target_duration))
                with open(output_audio_path, "wb") as f:
                    f.write(b"fit")

            def fake_original_audio(_video_path, start, duration, output_path, volume=1.0, speed=1.0, output_duration=None):
                original_audio_calls.append((start, duration, os.path.basename(output_path), volume, speed, output_duration))
                with open(output_path, "wb") as f:
                    f.write(b"original")

            def fake_run_command(cmd, cwd=None):
                commands.append(cmd)
                with open(cmd[-1], "wb") as f:
                    f.write(b"media")

            blocks = [{
                "start": 0,
                "end": 12,
                "visual": "long repetitive conveyor footage",
                "narration": "这句解说很短，不能让后面十几秒都没声音。",
                "video_speed": 1.0,
            }]
            with patch.object(commentary, "_get_video_duration", return_value=12.0), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=3.0), \
                patch.object(commentary, "_fit_audio_part_to_duration", side_effect=fake_fit_audio), \
                patch.object(commentary, "_extract_original_audio_clip", side_effect=fake_original_audio), \
                patch.object(commentary, "_run_command", side_effect=fake_run_command):
                ambient, block_durations = commentary._create_block_synced_visuals_and_audio(
                    video_path=video_path,
                    narration_blocks=blocks,
                    timed_video_path=timed_video_path,
                    voiceover_path=voiceover_path,
                    ambient_audio_path=ambient_audio_path,
                    aspect_mode="16:9",
                    work_dir=tmpdir,
                    tts_provider="edge",
                    language="zh",
                    elevenlabs_key=None,
                    voice_id="voice",
                    edge_voice="zh-CN-YunjianNeural",
                    original_audio_volume=0.08,
                    preserve_source_resolution=True,
                    progress=progress_messages.append,
                )

        self.assertEqual(1, len(fit_calls))
        self.assertAlmostEqual(4.5, fit_calls[0][2], places=2)
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertEqual(1, len(video_cmds))
        self.assertEqual("12.000", video_cmds[0][video_cmds[0].index("-t") + 1])
        self.assertIn("setpts=PTS/2.667000", video_cmds[0][video_cmds[0].index("-vf") + 1])
        self.assertAlmostEqual(2.667, original_audio_calls[0][4], places=3)
        self.assertAlmostEqual(4.5, original_audio_calls[0][5], places=2)
        self.assertTrue(any("Tightening commentary block" in message for message in progress_messages))
        self.assertEqual(ambient_audio_path, ambient)
        self.assertAlmostEqual(4.5, block_durations[0], places=2)

    def test_full_duration_rejects_missing_narration_blocks(self):
        with self.assertRaisesRegex(Exception, "narration_blocks are required"):
            commentary._validate_commentary_script_for_target(
                {
                    "narration": "这些废旧电机最终会被回收成铜材。",
                    "edit_segments": [{"start": 0, "end": 700, "reason": "process"}],
                },
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_rejects_short_narration_with_timestamp_blocks(self):
        short_text = "这些废旧电机最终会被回收成铜材。"
        blocks = [
            {"start": i * 280, "end": i * 280 + 90, "visual": "full process", "narration": short_text}
            for i in range(11)
        ]
        blocks.append({"start": 3600, "end": 3690, "visual": "ending process", "narration": short_text})

        with self.assertRaisesRegex(Exception, "too short"):
            commentary._validate_commentary_script_for_target(
                {
                    "narration": short_text,
                    "narration_blocks": blocks,
                    "edit_segments": [{"start": 0, "end": 700, "reason": "process"}],
                },
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_rejects_overlong_narration_before_tts(self):
        target_seconds = commentary._target_visual_duration_seconds(3935, "full")
        overlong = "这是一段会导致语音和视频时长失控的超长解说。" * 1400

        self.assertGreater(
            len(commentary.re.sub(r"\s+", "", overlong)),
            commentary._maximum_narration_chars(3935, "full", "zh"),
        )
        with self.assertRaisesRegex(Exception, "too long for comprehensive full-mode commentary"):
            commentary._validate_commentary_script_for_target(
                {
                    "narration": overlong,
                    "narration_blocks": scene_matched_blocks(text=overlong),
                    "edit_segments": [{"start": 0, "end": target_seconds, "reason": "process"}],
                },
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_rejects_excessive_pause_ratio(self):
        narration = "讲" * 4000
        blocks = [
            {"start": 0, "end": 400, "visual": "opening process", "narration": "讲" * 1800, "pause": False},
            {"start": 400, "end": 700, "visual": "too much source audio", "narration": "", "pause": True},
            {"start": 700, "end": 1200, "visual": "ending process", "narration": "讲" * 2200, "pause": False},
        ]

        with self.assertRaisesRegex(Exception, "too much no-commentary footage"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_rejects_overlong_pause_block(self):
        narration = "讲" * 4800
        blocks = [
            {"start": 0, "end": 590, "visual": "opening process", "narration": "讲" * 2400, "pause": False},
            {"start": 590, "end": 605, "visual": "long source audio beat", "narration": "", "pause": True},
            {"start": 605, "end": 1200, "visual": "ending process", "narration": "讲" * 2400, "pause": False},
        ]

        with self.assertRaisesRegex(Exception, "overlong no-commentary pause block"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_rejects_consecutive_pause_blocks(self):
        narration = "讲" * 4800
        blocks = [
            {"start": 0, "end": 580, "visual": "opening process", "narration": "讲" * 2400, "pause": False},
            {"start": 580, "end": 588, "visual": "first source audio beat", "narration": "", "pause": True},
            {"start": 588, "end": 596, "visual": "second source audio beat", "narration": "", "pause": True},
            {"start": 596, "end": 1200, "visual": "ending process", "narration": "讲" * 2400, "pause": False},
        ]

        with self.assertRaisesRegex(Exception, "consecutive no-commentary pause blocks"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_gemini_zero_region_quota_error_is_actionable_without_retrying(self):
        transcript = {
            "text": "hello",
            "language": "en",
            "segments": [{"start": 0, "end": 1, "text": "hello"}],
        }
        quota_error = Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
            "\"Quota exceeded for quota metric 'API requests' and limit "
            "'Request limit per minute for a region'\", 'status': "
            "'RESOURCE_EXHAUSTED', 'details': [{'metadata': {"
            "'quota_limit_value': '0', 'quota_location': 'asia-east1', "
            "'quota_limit': 'ApiRequestsPerMinutePerProjectPerRegion'}}]}}"
        )

        class FailingModels:
            calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                raise quota_error

        fake_models = FailingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)

        with patch.object(commentary, "create_gemini_client", return_value=fake_client):
            with self.assertRaisesRegex(Exception, "asia-east1.*每分钟请求配额为 0"):
                commentary.generate_commentary_script(
                    transcript=transcript,
                    video_title="Demo",
                    duration=1.0,
                    gemini_key="key",
                    analysis_mode="current",
                )

        self.assertEqual(1, fake_models.calls)

    def test_full_duration_short_narration_is_retried_with_video_grounded_regeneration_prompt(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        short_payload = {
            "title": "Short",
            "summary": "summary",
            "hook": "hook",
            "narration": "太短了。" * 30,
            "edit_segments": [{"start": 0, "end": 700, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }
        long_payload = {
            "title": "Long",
            "summary": "summary",
            "hook": "hook",
            "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
            "narration_blocks": scene_matched_blocks(),
            "edit_segments": commentary._narration_blocks_to_edit_segments(scene_matched_blocks()),
            "chapters": [],
            "hashtags": [],
        }

        class RetryingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                payload = short_payload if len(self.calls) == 1 else long_payload
                return pytypes.SimpleNamespace(text=commentary.json.dumps(payload, ensure_ascii=False))

        fake_models = RetryingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)
        fake_contents = [
            pytypes.SimpleNamespace(
                parts=[
                    pytypes.SimpleNamespace(text=None, file_data=pytypes.SimpleNamespace(file_uri="file://video")),
                    pytypes.SimpleNamespace(text="original prompt", file_data=None),
                ]
            )
        ]

        with patch.object(commentary, "create_gemini_client", return_value=fake_client), \
            patch.object(commentary, "_build_video_analysis_contents", return_value=fake_contents):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=3935,
                gemini_key="key",
                analysis_mode="video",
                target_duration="full",
                language="zh",
            )

        self.assertEqual("Long", result["title"])
        self.assertEqual(2, len(fake_models.calls))
        retry_contents = fake_models.calls[1]["contents"]
        self.assertEqual("file://video", retry_contents[0].parts[0].file_data.file_uri)
        self.assertIn("REGENERATE FROM THE ATTACHED VIDEO", retry_contents[0].parts[1].text)
        self.assertNotIn("EXPAND THE SCRIPT", retry_contents[0].parts[1].text)

    def test_full_duration_short_visual_plan_is_finalized_without_resending_video(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        visual_plan = {
            "title": "Plan",
            "summary": "summary",
            "hook": "hook",
            "narration": "太短了。" * 30,
            "narration_blocks": [
                {"start": i * 40, "end": i * 40 + 40, "visual": "workers and copper", "narration": "短段落"}
                for i in range(16)
            ],
            "edit_segments": [{"start": i * 40, "end": i * 40 + 40, "reason": "process"} for i in range(16)],
            "chapters": [],
            "hashtags": [],
        }
        final_payload = {
            "title": "Final",
            "summary": "summary",
            "hook": "hook",
            "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
            "narration_blocks": scene_matched_blocks(),
            "edit_segments": commentary._narration_blocks_to_edit_segments(scene_matched_blocks()),
            "chapters": [],
            "hashtags": [],
        }

        class FinalizingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                payload = visual_plan if len(self.calls) == 1 else final_payload
                return pytypes.SimpleNamespace(text=commentary.json.dumps(payload, ensure_ascii=False))

        fake_models = FinalizingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)
        fake_contents = [
            pytypes.SimpleNamespace(
                parts=[
                    pytypes.SimpleNamespace(text=None, file_data=pytypes.SimpleNamespace(file_uri="file://video")),
                    pytypes.SimpleNamespace(text="original prompt", file_data=None),
                ]
            )
        ]

        with patch.object(commentary, "create_gemini_client", return_value=fake_client), \
            patch.object(commentary, "_build_video_analysis_contents", return_value=fake_contents):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=3935,
                gemini_key="key",
                analysis_mode="video",
                target_duration="full",
                language="zh",
            )

        self.assertEqual("Final", result["title"])
        self.assertEqual(2, len(fake_models.calls))
        self.assertIsInstance(fake_models.calls[1]["contents"][0], str)
        self.assertIn("VIDEO-DERIVED VISUAL PLAN", fake_models.calls[1]["contents"][0])
        self.assertNotIn("EXPAND THE SCRIPT", fake_models.calls[1]["contents"][0])

    def test_full_duration_visual_plan_finalization_retries_transient_503(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        visual_plan = {
            "title": "Plan",
            "summary": "summary",
            "hook": "hook",
            "narration": "太短了。" * 30,
            "narration_blocks": [
                {"start": i * 40, "end": i * 40 + 40, "visual": "workers and copper", "narration": "短段落"}
                for i in range(16)
            ],
            "edit_segments": [{"start": i * 40, "end": i * 40 + 40, "reason": "process"} for i in range(16)],
            "chapters": [],
            "hashtags": [],
        }
        final_payload = {
            "title": "Final After Retry",
            "summary": "summary",
            "hook": "hook",
            "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
            "narration_blocks": scene_matched_blocks(),
            "edit_segments": commentary._narration_blocks_to_edit_segments(scene_matched_blocks()),
            "chapters": [],
            "hashtags": [],
        }

        class FlakyFinalizingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return pytypes.SimpleNamespace(text=commentary.json.dumps(visual_plan, ensure_ascii=False))
                if len(self.calls) == 2:
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return pytypes.SimpleNamespace(text=commentary.json.dumps(final_payload, ensure_ascii=False))

        fake_models = FlakyFinalizingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)
        fake_contents = [
            pytypes.SimpleNamespace(
                parts=[
                    pytypes.SimpleNamespace(text=None, file_data=pytypes.SimpleNamespace(file_uri="file://video")),
                    pytypes.SimpleNamespace(text="original prompt", file_data=None),
                ]
            )
        ]

        with patch.object(commentary, "create_gemini_client", return_value=fake_client), \
            patch.object(commentary, "_build_video_analysis_contents", return_value=fake_contents), \
            patch.object(commentary.time, "sleep"):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=3935,
                gemini_key="key",
                analysis_mode="video",
                target_duration="full",
                language="zh",
            )

        self.assertEqual("Final After Retry", result["title"])
        self.assertEqual(3, len(fake_models.calls))

    def test_full_duration_allows_multiple_video_grounded_regeneration_attempts(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        short_payload = {
            "title": "Short",
            "summary": "summary",
            "hook": "hook",
            "narration": "还是太短。" * 80,
            "edit_segments": [{"start": 0, "end": 700, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }
        long_payload = {
            "title": "Long",
            "summary": "summary",
            "hook": "hook",
            "narration": "\n\n".join(block["narration"] for block in scene_matched_blocks()),
            "narration_blocks": scene_matched_blocks(),
            "edit_segments": commentary._narration_blocks_to_edit_segments(scene_matched_blocks()),
            "chapters": [],
            "hashtags": [],
        }

        class RetryingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                payload = long_payload if len(self.calls) == 3 else short_payload
                return pytypes.SimpleNamespace(text=commentary.json.dumps(payload, ensure_ascii=False))

        fake_models = RetryingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)
        fake_contents = [
            pytypes.SimpleNamespace(
                parts=[
                    pytypes.SimpleNamespace(text=None, file_data=pytypes.SimpleNamespace(file_uri="file://video")),
                    pytypes.SimpleNamespace(text="original prompt", file_data=None),
                ]
            )
        ]

        with patch.object(commentary, "create_gemini_client", return_value=fake_client), \
            patch.object(commentary, "_build_video_analysis_contents", return_value=fake_contents):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=3935,
                gemini_key="key",
                analysis_mode="video",
                target_duration="full",
                language="zh",
            )

        self.assertEqual("Long", result["title"])
        self.assertEqual(3, len(fake_models.calls))
        self.assertEqual("file://video", fake_models.calls[1]["contents"][0].parts[0].file_data.file_uri)
        self.assertEqual("file://video", fake_models.calls[2]["contents"][0].parts[0].file_data.file_uri)
        self.assertIn("REGENERATE FROM THE ATTACHED VIDEO", fake_models.calls[2]["contents"][0].parts[1].text)

    def test_openai_script_request_uses_longer_timeout(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        visual_analysis = {
            "provider": "openai_compatible",
            "model": "demo-model",
            "frame_count": 1,
            "batch_count": 1,
            "sampling_options": commentary.resolve_openai_sampling_options(),
            "observations": [{"timestamp": 1, "visual": "worker sorts copper"}],
        }
        payload = {
            "title": "Demo",
            "summary": "summary",
            "hook": "hook",
            "narration": "这是一段足够长的解说内容。" * 80,
            "edit_segments": [{"start": 0, "end": 60, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }
        calls = []

        def fake_call(**kwargs):
            calls.append(kwargs)
            return commentary.json.dumps(payload, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(commentary, "_load_openai_visual_analysis", return_value=visual_analysis), \
            patch.object(commentary, "_call_openai_compatible_chat", side_effect=fake_call):
            result = commentary.generate_openai_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=60,
                openai_key="key",
                openai_base_url="https://provider.example/v1",
                openai_model="demo-model",
                frame_infos=[{"path": "frame.jpg", "timestamp": 1}],
                target_duration="short",
                output_dir=tmpdir,
            )

        self.assertEqual("Demo", result["title"])
        self.assertEqual(commentary.OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS, calls[0]["timeout_seconds"])

    def test_openai_visual_analysis_cache_reused_before_batch_calls(self):
        cached = {
            "provider": "openai_compatible",
            "model": "demo-model",
            "frame_count": 1,
            "batch_count": 41,
            "sampling_options": commentary.resolve_openai_sampling_options(),
            "observations": [{"timestamp": 1, "visual": "worker sorts copper"}],
        }
        payload = {
            "title": "Demo",
            "summary": "summary",
            "hook": "hook",
            "narration": "这是一段足够长的解说内容。" * 80,
            "edit_segments": [{"start": 0, "end": 60, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            commentary._save_openai_visual_analysis(tmpdir, cached)
            with patch.object(commentary, "_analyze_openai_visual_timeline") as analyze, \
                patch.object(commentary, "_call_openai_compatible_chat", return_value=commentary.json.dumps(payload, ensure_ascii=False)):
                result = commentary.generate_openai_commentary_script(
                    transcript={"text": "factory process", "segments": []},
                    video_title="Demo",
                    duration=60,
                    openai_key="key",
                    openai_base_url="https://provider.example/v1",
                    openai_model="demo-model",
                    frame_infos=[{"path": "frame.jpg", "timestamp": 1}],
                    target_duration="short",
                    output_dir=tmpdir,
                )

        self.assertEqual("Demo", result["title"])
        analyze.assert_not_called()

    def test_openai_analysis_frames_manifest_reused_when_files_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = os.path.join(tmpdir, "frame.jpg")
            with open(frame_path, "wb") as f:
                f.write(b"jpg")
            frames = [{"path": frame_path, "timestamp": 1.0}]
            commentary._save_openai_analysis_frames(tmpdir, frames)

            with patch.object(commentary, "_run_command") as run_command:
                reused = commentary._extract_openai_analysis_frames("video.mp4", tmpdir, 60)

        self.assertEqual(frames, reused)
        run_command.assert_not_called()

    def test_openai_analysis_frames_legacy_directory_reused_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, "openai_analysis_frames")
            os.makedirs(frames_dir)
            frame_path = os.path.join(frames_dir, "frame_0001_000001500.jpg")
            with open(frame_path, "wb") as f:
                f.write(b"jpg")

            with patch.object(commentary, "_run_command") as run_command:
                reused = commentary._extract_openai_analysis_frames("video.mp4", tmpdir, 60)

            self.assertTrue(os.path.exists(commentary._openai_frames_manifest_path(tmpdir)))

        self.assertEqual([{"path": frame_path, "timestamp": 1.5}], reused)
        run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
