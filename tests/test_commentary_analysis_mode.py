import os
import re
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
    ) * 5


def repeated_scene_text(base: str, count: int) -> str:
    return (base + "。") * count


def varied_scene_text(parts):
    return "。".join(parts) + "。"


def joined_scene_text(base: str, count: int) -> str:
    return "，".join(base for _ in range(count)) + "。"


def beehive_visual_analysis():
    segments = [
        (2, 11), (11, 13), (19, 22), (46, 52), (64, 77), (84, 94),
        (114, 122), (128, 132), (153, 156), (168, 177), (216, 231),
        (247, 253), (272, 281), (307, 317), (316, 322), (336, 364),
        (419, 430), (430, 452), (465, 482), (491, 500), (518, 537),
        (563, 567), (573, 583), (592, 603), (609, 629), (654, 661),
        (665, 670), (670, 676),
    ]
    observations = []
    for index, (start, end) in enumerate(segments):
        importance = 5 if index % 4 == 0 else 4
        observations.append({
            "timestamp": (start + end) / 2,
            "visual": "beekeeper handles hive on tree with bees"
            if start < 609
            else "beekeeper has secured the hive bag and shows the final packed result near the tree",
            "process_stage": "beehive work"
            if start < 609
            else "bag secured final result packed completion",
            "importance": importance,
            "keep_candidate": True,
        })
    return {
        "observations": observations,
        "candidate_segments": [
            {"start": start, "end": end, "reason": "useful beehive process moment"}
            for start, end in segments
        ],
    }


WORKER_MATERIAL_NARRATION = (
    "工人把材料送到设备旁继续处理，机器运转时材料不断移动，"
    "这个工序最后会形成更清楚的处理结果"
)


TREE_HONEY_NARRATION = (
    "手套贴着树干继续操作，烟熏着蜂群和蜂巢，袋子、绳子和刀具都围着这块蜂蜜收尾"
)


def scene_matched_blocks(count=12, seconds=50.0, text=None):
    block_text = text or repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
    starts = [i * seconds for i in range(count)]
    if count == 12:
        starts = [i * 280.0 for i in range(count - 1)] + [3600.0]
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

    def test_prompt_includes_custom_style_prompt_as_additional_style_instruction(self):
        transcript = {
            "text": "A short transcript",
            "language": "en",
            "segments": [{"start": 0, "end": 3, "text": "A short transcript"}],
        }

        prompt = commentary._build_commentary_prompt(
            transcript=transcript,
            video_title="Demo",
            duration=12.0,
            language="zh",
            style="custom",
            target_duration="short",
            analysis_mode="current",
            custom_style_prompt="用第一人称紧张整活口吻，短句优先，必须先说画面动作。",
        )

        self.assertIn("COMMENTARY STYLE: custom", prompt)
        self.assertIn("CUSTOM STYLE PROMPT: 用第一人称紧张整活口吻", prompt)
        self.assertIn("Custom user style instruction", prompt)
        self.assertIn("as long as it does not conflict with visual grounding", prompt)
        self.assertIn("必须先说画面动作", prompt)

    def test_openai_full_prompt_uses_backend_sync_instead_of_density_audit(self):
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

        self.assertIn("Backend rendering preserves each selected source range", prompt)
        self.assertIn("will not rescue a sparse narration block by cutting or speeding the visuals", prompt)
        self.assertIn("Do not add meaningless word padding", prompt)
        self.assertIn("return the production script and timeline only", prompt)
        self.assertNotIn("density_audit", prompt)
        self.assertNotIn("per-block density check", prompt)
        self.assertIn("There is no filler word-count target", prompt)
        self.assertIn("There is no total minimum word count", prompt)
        self.assertIn("not too dense, but also not empty", prompt)
        self.assertIn("what is visible, what changes, and why that moment matters", prompt)
        self.assertIn("do not make narration sparse by writing one vague sentence over a long visual block", prompt)
        self.assertIn("do not pad it with meaningless words", prompt)
        self.assertIn('Completion words such as "finished", "packed", "done", "收工"', prompt)
        self.assertIn("3-second retention rule", prompt)
        self.assertIn("curiosity, contrast, stakes, surprise, or payoff expectation", prompt)
        self.assertIn("matching the first visible action", prompt)

    def test_full_mode_repairs_under_explained_block_without_shortening_visual_target(self):
        data = {
            "narration": "工人在处理材料。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人在处理材料",
                    "narration": "工人在处理材料。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 24,
                    "end": 360,
                    "visual": "后续流程完成并展示结果",
                    "visual_facts": ["工人继续处理材料并展示最终结果"],
                    "narration": "随后工人继续把材料整理、分离和检查，最后把处理后的结果集中展示出来。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 24, "end": 360}],
        }

        commentary._validate_commentary_script_for_target(data, 400.0, "full", "zh")

        self.assertAlmostEqual(
            400.0,
            sum(commentary._block_visual_duration(block) for block in data["narration_blocks"]),
            places=2,
        )
        self.assertIn("工人在处理材料", data["narration_blocks"][0]["narration"])
        self.assertTrue(any(block.get("pause") for block in data["narration_blocks"]))

    def test_full_mode_rejects_long_visual_range_with_sparse_narration(self):
        data = {
            "narration": "工人沿树干往上爬。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 25,
                    "visual": "工人沿树干往上爬",
                    "visual_facts": ["工人沿树干往上爬"],
                    "narration": "工人沿树干往上爬。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 25,
                    "end": 360,
                    "visual": "工人处理蜂巢并展示后续结果",
                    "visual_facts": ["工人处理蜂巢并展示后续结果"],
                    "narration": repeated_scene_text("工人继续处理蜂巢，袋子和绳子配合着把蜂巢固定住", 16),
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
        }

        with self.assertRaisesRegex(Exception, "too short for its selected visual range"):
            commentary._validate_commentary_script_for_target(data, 400.0, "full", "zh")

    def test_full_mode_trims_small_density_shortfall_when_budget_stays_valid(self):
        blocks = []
        for index in range(19):
            start = index * 24.0
            blocks.append({
                "start": start,
                "end": start + 21.0,
                "visual": f"第{index + 1}段工人贴着树干继续处理蜂巢",
                "visual_facts": [f"第{index + 1}段工人贴着树干继续处理蜂巢"],
                "narration": joined_scene_text(
                    f"第{index + 1}段工人贴着树干处理蜂巢，袋子绳子跟着调整",
                    3,
                ),
                "pause": False,
                "video_speed": 1.0,
            })
        blocks.append({
            "start": 860.0,
            "end": 880.0,
            "visual": "后段工人继续贴着树干处理蜂巢并整理袋子",
            "visual_facts": ["后段工人继续贴着树干处理蜂巢并整理袋子"],
            "narration": joined_scene_text("后段工人继续贴着树干处理蜂巢，袋子绳子跟着整理", 3),
            "pause": False,
            "video_speed": 1.0,
        })
        blocks[2]["narration"] = "工人贴着树干处理蜂巢，袋子绳子跟着调整，蜂群还在旁边飞动，手套继续稳住蜂巢位置慢慢变化下去了"
        data = {
            "narration": "工人贴着树干处理蜂巢。",
            "narration_blocks": blocks,
        }

        self.assertEqual(46, len(re.sub(r"\s+", "", blocks[2]["narration"])))
        self.assertEqual(49, commentary._expected_narration_chars_for_visual_duration(21.0, "zh"))

        commentary._validate_commentary_script_for_target(data, 1000.0, "full", "zh")

        target = commentary._target_visual_duration_seconds(1000.0, "full")
        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        self.assertGreaterEqual(
            playable,
            commentary._full_mode_min_playable_visual_seconds(1000.0, target)
            - commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        self.assertLess(data["narration_blocks"][2]["end"], 2 * 24.0 + 21.0)
        commentary._validate_narration_density_matches_visual_duration(data["narration_blocks"][2], 3, "zh")

    def test_full_mode_trims_moderate_density_shortfall_when_budget_stays_valid(self):
        blocks = []
        for index in range(19):
            start = index * 24.0
            narration = (
                f"第{index + 1}段工人贴着树干处理蜂巢，袋子绳子继续调整，"
                f"蜂群还在旁边活动，手套沿着蜂巢边缘继续推进第{index + 1}步"
            )
            blocks.append({
                "start": start,
                "end": start + 21.0,
                "visual": f"第{index + 1}段工人贴着树干继续处理蜂巢",
                "visual_facts": [f"第{index + 1}段工人贴着树干继续处理蜂巢"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            })
        blocks[0]["end"] = 28.0
        blocks[0]["narration"] = "工人贴着树干处理蜂巢，袋子绳子继续调整，蜂群还在旁边活动，手套沿着蜂巢边缘继续推进下去慢慢处理着"
        blocks.append({
            "start": 860.0,
            "end": 900.0,
            "visual": "后段工人继续处理蜂巢并整理袋子",
            "visual_facts": ["后段工人继续处理蜂巢并整理袋子"],
            "narration": joined_scene_text("后段工人继续贴着树干处理蜂巢，袋子绳子跟着整理，蜂群仍在旁边活动", 6),
            "pause": False,
            "video_speed": 1.0,
        })
        data = {
            "narration": "工人贴着树干处理蜂巢。",
            "narration_blocks": blocks,
        }

        self.assertEqual(48, len(re.sub(r"\s+", "", blocks[0]["narration"])))
        self.assertEqual(68, commentary._expected_narration_chars_for_visual_duration(28.0, "zh"))

        commentary._validate_commentary_script_for_target(data, 1000.0, "full", "zh")

        self.assertLess(data["narration_blocks"][0]["end"], 28.0)
        commentary._validate_narration_density_matches_visual_duration(data["narration_blocks"][0], 1, "zh")

    def test_full_mode_trims_small_overselected_visual_budget(self):
        narration = "工人贴着树干处理蜂巢，袋子绳子继续调整，蜂群仍在旁边活动，手套一点点稳住蜂巢位置继续往下慢慢处理着"
        blocks = []
        for index in range(25):
            start = index * 22.0
            block_narration = (
                f"第{index + 1}段工人贴着树干处理蜂巢，袋子绳子继续调整，"
                f"蜂群还在旁边活动，手套沿着蜂巢边缘慢慢处理推进第{index + 1}步"
            )
            blocks.append({
                "start": start,
                "end": start + 21.0,
                "visual": f"第{index + 1}段工人贴着树干处理蜂巢",
                "visual_facts": [f"第{index + 1}段工人贴着树干处理蜂巢"],
                "narration": block_narration,
                "pause": False,
                "video_speed": 1.0,
            })
        blocks.append({
            "start": 1540.0,
            "end": 1562.0,
            "visual": "后段工人继续贴着树干处理蜂巢并整理袋子",
            "visual_facts": ["后段工人继续贴着树干处理蜂巢并整理袋子"],
            "narration": narration,
            "pause": False,
            "video_speed": 1.0,
        })
        data = {
            "narration": "工人贴着树干处理蜂巢。",
            "narration_blocks": blocks,
        }

        target = commentary._target_visual_duration_seconds(1800.0, "full")
        self.assertGreater(
            sum(commentary._block_visual_duration(block) for block in data["narration_blocks"]),
            commentary._full_mode_max_playable_visual_seconds(1800.0, target),
        )

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        self.assertLessEqual(
            playable,
            commentary._full_mode_max_playable_visual_seconds(1800.0, target)
            + commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        self.assertGreaterEqual(
            max(block["end"] for block in data["narration_blocks"]),
            1800.0 * commentary.FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION,
        )

    def test_full_mode_min_playable_visual_window_allows_ten_percent_shortfall(self):
        target = commentary._target_visual_duration_seconds(1800.0, "full")

        self.assertEqual(480.0, target)
        self.assertAlmostEqual(432.0, commentary._full_mode_min_playable_visual_seconds(1800.0, target))

    def test_full_mode_strips_camera_meta_phrasing_before_validation(self):
        data = {
            "narration": "镜头切到工人贴着树干处理蜂巢。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 20,
                    "visual": "工人贴着树干处理蜂巢",
                    "visual_facts": ["工人贴着树干处理蜂巢"],
                    "narration": "镜头切到工人贴着树干处理蜂巢，袋子绳子跟着调整，蜂群还在旁边活动，手套沿着蜂巢边缘继续处理。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 420,
                    "end": 650,
                    "visual": "中段工人继续处理蜂巢并调整袋子",
                    "visual_facts": ["中段工人继续处理蜂巢并调整袋子"],
                    "narration": "中段工人继续贴着树干处理蜂巢，袋子绳子跟着整理，蜂群仍在旁边活动，手套把蜂巢边缘慢慢稳住继续处理，整段动作顺着树干一点点往下推进。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 860,
                    "end": 1000,
                    "visual": "后段工人继续处理蜂巢并整理袋子",
                    "visual_facts": ["后段工人继续处理蜂巢并整理袋子"],
                    "narration": "后段工人继续贴着树干处理蜂巢，袋子绳子跟着整理，蜂群仍在旁边活动，手套把蜂巢边缘慢慢收住继续固定位置，最后的动作还在稳稳往下收。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
        }

        commentary._strip_camera_meta_phrasing(data)

        self.assertNotIn("镜头", data["narration"])
        self.assertNotIn("镜头", data["narration_blocks"][0]["narration"])

    def test_full_mode_allows_slight_density_shortfall_with_grounded_narration(self):
        block = {
            "start": 0,
            "end": 40.0,
            "visual": "工人贴着树干处理蜂巢，蜂群和袋子都在旁边",
            "visual_facts": ["工人贴着树干处理蜂巢，蜂群和袋子都在旁边"],
            "narration": "手套贴着树干处理蜂巢，蜂群围着树干飞动，工人一点点把蜂巢位置稳住，袋子和绳子还在旁边跟着继续调整变化",
            "pause": False,
            "video_speed": 1.0,
        }

        self.assertEqual(50, len(re.sub(r"\s+", "", block["narration"])))
        self.assertEqual(60, commentary._expected_narration_chars_for_visual_duration(40.0, "zh"))
        commentary._validate_narration_density_matches_visual_duration(block, 3, "zh")

    def test_full_mode_rejects_severe_sparse_narration_for_long_range(self):
        block = {
            "start": 0,
            "end": 52.9,
            "visual": "工人贴着树干继续处理蜂巢，蜂群还在旁边活动",
            "visual_facts": ["工人贴着树干继续处理蜂巢，蜂群还在旁边活动"],
            "narration": "蜂巢还在处理。",
            "pause": False,
            "video_speed": 1.0,
        }

        with self.assertRaisesRegex(Exception, "too short for its selected visual range"):
            commentary._validate_narration_density_matches_visual_duration(block, 3, "zh")

    def test_full_mode_expands_sparse_zh_narration_from_english_visual_concepts(self):
        block = {
            "start": 0,
            "end": 40.0,
            "visual": "worker hands on tree trunk while bees swarm around hive and bag rope stay nearby",
            "visual_facts": ["gloved worker handles beehive on tree trunk as bees move around honeycomb"],
            "narration": "手套贴着树干处理蜂巢，蜂群围着树干飞动，工人一点点把蜂巢位置稳住，袋子和绳子还在旁边跟着继续调整变化",
            "pause": False,
            "video_speed": 1.0,
        }

        repaired = commentary._repair_short_narration_visual_ranges([block], "zh")
        repaired_text = repaired[0]["narration"]

        self.assertEqual(1, len(repaired))
        self.assertFalse(repaired[0]["pause"])
        self.assertGreaterEqual(
            len(re.sub(r"\s+", "", repaired_text)),
            commentary._expected_narration_chars_for_visual_duration(40.0, "zh"),
        )
        self.assertIn("蜂群", repaired_text)

    def test_full_mode_allows_only_brief_auto_silence_tail(self):
        block = {
            "start": 0,
            "end": 13.0,
            "visual": "",
            "visual_facts": [],
            "narration": "短句",
            "pause": False,
            "video_speed": 1.0,
        }

        repaired = commentary._repair_short_narration_visual_ranges([block], "zh")

        self.assertEqual(2, len(repaired))
        self.assertFalse(repaired[0]["pause"])
        self.assertTrue(repaired[1]["pause"])
        self.assertLessEqual(
            commentary._block_visual_duration(repaired[1]),
            commentary.FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS + commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )

    def test_full_mode_repairs_small_underselected_visual_budget_with_pause_bridges(self):
        starts = [index * 30.0 for index in range(20)] + [1540.0 + index * 30.0 for index in range(6)]
        blocks = [
            {
                "start": start,
                "end": start + 15.0,
                "visual": f"第{index + 1}处工人处理材料并调整设备位置",
                "visual_facts": [f"第{index + 1}处工人处理材料并调整设备位置"],
                "narration": (
                    f"第{index + 1}处工人围着材料和设备调整位置，工具、手部和材料边缘都在同步变化，"
                    "处理节奏保持清楚，后续动作也能顺着这个工序继续推进。"
                ),
                "pause": False,
                "video_speed": 1.0,
            }
            for index, start in enumerate(starts)
        ]
        data = {
            "narration": "工人处理材料并调整设备位置。",
            "narration_blocks": blocks,
        }

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        target = commentary._target_visual_duration_seconds(1800.0, "full")
        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        added_pauses = [block for block in data["narration_blocks"] if block.get("auto_filled_visual_budget")]

        self.assertGreaterEqual(
            playable,
            commentary._full_mode_min_playable_visual_seconds(1800.0, target)
            - commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        self.assertTrue(added_pauses or any(block["end"] > block["start"] + 15.0 for block in data["narration_blocks"]))
        self.assertTrue(all(block.get("pause") for block in added_pauses))
        self.assertTrue(all(not block.get("narration") for block in added_pauses))
        self.assertTrue(
            all(
                commentary._block_visual_duration(block)
                <= commentary.FULL_MODE_MAX_PAUSE_SECONDS + commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS
                for block in added_pauses
            )
        )
        self.assertEqual(data["edit_segments"], commentary._narration_blocks_to_edit_segments(data["narration_blocks"]))

        data["narration_blocks"] = commentary._strip_auto_filled_user_visible_fields(data["narration_blocks"])
        data["edit_segments"] = commentary._narration_blocks_to_edit_segments(data["narration_blocks"])
        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")
        self.assertFalse(any(block.get("auto_filled_visual_budget") for block in data["narration_blocks"]))

    def test_full_mode_repairs_underselected_budget_with_leading_pause_bridge(self):
        blocks = []
        for index in range(20):
            start = 18.0 + index * 21.0
            narration = (
                f"第{index + 1}段工人贴着树干处理蜂巢，袋子绳子跟着调整，"
                f"蜂群还在旁边活动，手套沿着蜂巢边缘继续推进第{index + 1}步"
            )
            blocks.append({
                "start": start,
                "end": start + 20.0,
                "visual": f"第{index + 1}段工人贴着树干处理蜂巢",
                "visual_facts": [f"第{index + 1}段工人贴着树干处理蜂巢"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            })
        blocks.append({
            "start": 1540.0,
            "end": 1560.0,
            "visual": "后段工人继续处理蜂巢并整理袋子",
            "visual_facts": ["后段工人继续处理蜂巢并整理袋子"],
            "narration": "后段工人继续贴着树干处理蜂巢，袋子绳子跟着整理，蜂群仍在旁边活动，手套把蜂巢边缘慢慢收住继续固定位置",
            "pause": False,
            "video_speed": 1.0,
        })
        data = {
            "narration": "工人贴着树干处理蜂巢。",
            "narration_blocks": blocks,
        }

        target = commentary._target_visual_duration_seconds(1800.0, "full")
        self.assertLess(
            sum(commentary._block_visual_duration(block) for block in data["narration_blocks"]),
            commentary._full_mode_min_playable_visual_seconds(1800.0, target),
        )

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        added_pauses = [block for block in data["narration_blocks"] if block.get("auto_filled_visual_budget")]
        extended_blocks = [
            block
            for block in data["narration_blocks"]
            if not block.get("pause") and block.get("end", 0) > block.get("start", 0) + 20.0
        ]
        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        self.assertTrue(added_pauses or extended_blocks)
        self.assertGreaterEqual(
            playable,
            commentary._full_mode_min_playable_visual_seconds(1800.0, target)
            - commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        if added_pauses:
            self.assertLessEqual(
                max(commentary._block_visual_duration(block) for block in added_pauses),
                commentary.FULL_MODE_MAX_PAUSE_SECONDS + commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
            )

    def test_full_mode_repairs_underselected_budget_by_extending_dense_narrated_blocks(self):
        blocks = []
        starts = [index * 20.0 for index in range(20)] + [1540.0]
        for index, start in enumerate(starts):
            blocks.append({
                "start": start,
                "end": start + 20.0,
                "visual": f"第{index + 1}段工人连续处理材料和设备",
                "visual_facts": [f"第{index + 1}段工人连续处理材料和设备"],
                "narration": (
                    f"第{index + 1}段工人处理材料和设备，工具位置变化，"
                    "手部动作与蜂群位置同步推进，材料状态也继续变化下去。"
                ),
                "pause": False,
                "video_speed": 1.0,
            })
        data = {
            "narration": "工人连续处理材料和设备。",
            "narration_blocks": blocks,
        }

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        target = commentary._target_visual_duration_seconds(1800.0, "full")
        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])

        self.assertGreaterEqual(
            playable,
            commentary._full_mode_min_playable_visual_seconds(1800.0, target)
            - commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        self.assertGreater(data["narration_blocks"][-1]["end"], 420.0)

    def test_full_mode_allows_subsecond_visual_budget_shortfall_after_repair(self):
        target = commentary._target_visual_duration_seconds(1800.0, "full")
        min_visual_seconds = commentary._full_mode_min_playable_visual_seconds(1800.0, target)
        playable_seconds = min_visual_seconds - 0.4
        blocks = []
        for index in range(20):
            duration = 21.0
            start = index * 30.0
            if index == 19:
                duration = playable_seconds - (21.0 * 19)
                start = 1800.0 - duration
            expected_chars = commentary._expected_narration_chars_for_visual_duration(duration, "zh")
            prefix = f"第{index + 1}段"
            narration = prefix + ("工" * max(0, expected_chars - len(prefix)))
            blocks.append({
                "start": start,
                "end": start + duration,
                "visual": f"第{index + 1}段工人处理材料并调整设备",
                "visual_facts": [f"第{index + 1}段工人处理材料并调整设备"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            })
        data = {
            "narration": "\n\n".join(block["narration"] for block in blocks),
            "narration_blocks": blocks,
        }

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        self.assertLess(playable, min_visual_seconds)
        self.assertGreaterEqual(
            playable,
            min_visual_seconds - commentary._full_mode_visual_budget_tolerance_seconds(target),
        )

    def test_full_mode_repairs_small_underselected_budget_by_extending_trailing_pause(self):
        blocks = []
        for index in range(20):
            start = index * 20.0
            blocks.append({
                "start": start,
                "end": start + 20.0,
                "visual": f"第{index + 1}段工人连续处理材料和设备",
                "visual_facts": [f"第{index + 1}段工人连续处理材料和设备"],
                "narration": joined_scene_text(
                    f"第{index + 1}段工人连续处理材料和设备，工具位置和材料状态继续变化",
                    2,
                ),
                "pause": False,
                "video_speed": 1.0,
            })
        blocks.append({
            "start": 1540.0,
            "end": 1556.0,
            "visual": "后段工人继续处理材料，原片环境声承接",
            "visual_facts": ["后段工人继续处理材料，原片环境声承接"],
            "narration": "",
            "pause": True,
            "video_speed": 1.0,
        })
        data = {
            "narration": "工人连续处理材料和设备。",
            "narration_blocks": blocks,
        }

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        target = commentary._target_visual_duration_seconds(1800.0, "full")
        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        added_pauses = [block for block in data["narration_blocks"] if block.get("auto_filled_visual_budget")]
        extended_blocks = [
            block
            for index, block in enumerate(data["narration_blocks"][:20])
            if block.get("end", 0) > index * 20.0 + 20.0
        ]
        pause_blocks = [block for block in data["narration_blocks"] if block.get("pause")]

        self.assertGreaterEqual(
            playable,
            commentary._full_mode_min_playable_visual_seconds(1800.0, target)
            - commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS,
        )
        self.assertTrue(added_pauses or extended_blocks)
        self.assertTrue(all(block.get("pause") for block in added_pauses))
        self.assertTrue(all(not block.get("narration") for block in added_pauses))
        self.assertTrue(
            all(
                commentary._block_visual_duration(block)
                <= commentary.FULL_MODE_MAX_PAUSE_SECONDS + commentary.FULL_MODE_VALIDATION_EPSILON_SECONDS
                for block in pause_blocks
            )
        )

    def test_full_mode_rejects_auto_fill_placeholder_narration_as_ai_selection(self):
        placeholder = "这一段把前后工序之间的衔接补上，材料处理、设备运转和人员操作继续往前推进，流程不是突然跳过去，而是顺着这里进入下一步"
        data = {
            "narration": f"前面正常解说。\n\n{placeholder}\n\n后面正常解说。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 120,
                    "visual": "工人整理材料并准备进入下一步",
                    "visual_facts": ["工人整理材料并准备进入下一步"],
                    "narration": "工人先把材料整理好，准备进入下一步处理。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 120,
                    "end": 240,
                    "visual": "自动补齐的作业流程过渡",
                    "visual_facts": ["这段补齐范围用于承接前后已经选中的作业流程"],
                    "narration": placeholder,
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 600,
                    "end": 720,
                    "visual": "后段流程展示最终处理结果",
                    "visual_facts": ["后段流程展示最终处理结果"],
                    "narration": "后段流程把最终处理结果交代出来。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
        }

        with self.assertRaisesRegex(Exception, "auto-filled placeholder phrase"):
            commentary._validate_commentary_script_for_target(data, 720.0, "full", "zh")

        self.assertFalse(any(block.get("auto_filled_visual_budget") for block in data["narration_blocks"]))

    def test_full_mode_rewrites_unsupported_completion_claim_instead_of_rejecting(self):
        data = {
            "narration": "工人继续处理蜂巢，袋子还在旁边。蜂巢装好收工。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 230,
                    "visual": "工人继续处理蜂巢，袋子和绳子还在旁边调整",
                    "visual_facts": ["工人继续处理蜂巢，袋子和绳子还在旁边调整"],
                    "narration": joined_scene_text("工人继续处理蜂巢，袋子和绳子还在旁边调整", 46),
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 1540,
                    "end": 1770,
                    "visual": "工人继续沿树干移动，蜂群仍在周围活动",
                    "visual_facts": ["工人继续沿树干移动，蜂群仍在周围活动"],
                    "narration": joined_scene_text("工人继续沿树干移动，蜂群仍在周围活动，蜂巢装好收工", 39),
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
        }

        commentary._validate_commentary_script_for_target(data, 1800.0, "full", "zh")

        self.assertNotIn("收工", data["narration"])
        self.assertNotIn("装好", data["narration"])
        self.assertNotIn("收工", data["narration_blocks"][1]["narration"])
        self.assertIn("继续处理", data["narration_blocks"][1]["narration"])

    def test_medium_mode_rewrites_unsupported_completion_claim_instead_of_rejecting(self):
        data = {
            "title": "蜂巢处理",
            "summary": "summary",
            "hook": "hook",
            "narration": "工人继续处理蜂巢，蜂群还在周围。蜂巢装好收工。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 20,
                    "visual": "工人继续处理蜂巢，蜂群还在周围飞动",
                    "visual_facts": ["工人继续处理蜂巢，蜂群还在周围飞动"],
                    "narration": "工人继续处理蜂巢，蜂群还在周围飞动，蜂巢装好收工。",
                    "pause": False,
                },
            ],
            "edit_segments": [{"start": 0, "end": 20, "reason": "蜂巢处理"}],
            "chapters": [],
            "hashtags": [],
        }

        commentary._validate_commentary_script_for_target(data, 120.0, "medium", "zh")

        self.assertNotIn("收工", data["narration"])
        self.assertNotIn("装好", data["narration"])
        self.assertNotIn("收工", data["narration_blocks"][0]["narration"])
        self.assertIn("继续处理", data["narration_blocks"][0]["narration"])

    def test_medium_mode_rewrites_unsupported_container_loading_claim_instead_of_rejecting(self):
        data = {
            "title": "蜂巢处理",
            "summary": "summary",
            "hook": "hook",
            "narration": "工人继续处理蜂巢，蜂巢块被放进袋子里。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 20,
                    "visual": "工人继续处理蜂巢，袋子和绳子还在旁边调整",
                    "visual_facts": ["工人继续处理蜂巢，袋子和绳子还在旁边调整"],
                    "narration": "工人继续处理蜂巢，手套贴着树干往下动，蜂巢块被放进袋子里。",
                    "pause": False,
                },
            ],
            "edit_segments": [{"start": 0, "end": 20, "reason": "蜂巢处理"}],
            "chapters": [],
            "hashtags": [],
        }

        commentary._validate_commentary_script_for_target(data, 120.0, "medium", "zh")

        self.assertNotIn("放进袋", data["narration"])
        self.assertNotIn("放进袋", data["narration_blocks"][0]["narration"])
        self.assertIn("继续处理", data["narration_blocks"][0]["narration"])

    def test_medium_mode_rewrites_unsupported_bag_mouth_claim_instead_of_rejecting(self):
        data = {
            "title": "蜂巢处理",
            "summary": "summary",
            "hook": "hook",
            "narration": "蜂巢被放入袋中，袋口被拧紧。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 20,
                    "visual": "工人继续处理蜂巢，袋子和绳子还在旁边调整",
                    "visual_facts": ["工人继续处理蜂巢，袋子和绳子还在旁边调整"],
                    "narration": "工人继续处理蜂巢，蜂巢被放入袋中，袋口被拧紧。",
                    "pause": False,
                },
            ],
            "edit_segments": [{"start": 0, "end": 20, "reason": "蜂巢处理"}],
            "chapters": [],
            "hashtags": [],
        }

        commentary._validate_commentary_script_for_target(data, 120.0, "medium", "zh")

        self.assertNotIn("放入袋", data["narration"])
        self.assertNotIn("袋口被拧紧", data["narration"])
        self.assertNotIn("放入袋", data["narration_blocks"][0]["narration"])
        self.assertNotIn("袋口被拧紧", data["narration_blocks"][0]["narration"])

    def test_auto_filled_blocks_do_not_expose_placeholder_visuals(self):
        blocks = [
            {
                "start": 0,
                "end": 12,
                "visual": "自动补齐的作业流程过渡，承接前后选段里的材料处理、设备运转和人员操作",
                "visual_facts": ["这段补齐范围用于承接前后已经选中的作业流程"],
                "narration": "",
                "pause": True,
                "auto_filled_visual_budget": True,
                "video_speed": 1.5,
            },
            {
                "start": 12,
                "end": 36,
                "visual": "工人把材料送进机器",
                "visual_facts": ["工人把材料送进机器"],
                "narration": "工人把材料送进机器，处理流程继续推进。",
                "pause": False,
                "video_speed": 1.0,
            },
        ]

        cleaned = commentary._strip_auto_filled_user_visible_fields(blocks)
        segments = commentary._narration_blocks_to_edit_segments(cleaned)
        summary = commentary._summarize_auto_video_speed(cleaned, True)

        self.assertEqual(cleaned[0]["visual"], commentary.COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL)
        self.assertEqual(cleaned[0]["visual_facts"], [])
        self.assertEqual(segments[0]["reason"], commentary.COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL)
        self.assertEqual(summary["accelerated_blocks"][0]["visual"], "")
        self.assertFalse(any("自动补齐" in str(item) for item in [cleaned, segments, summary]))

    def test_full_mode_does_not_append_english_visual_facts_to_zh_narration(self):
        data = {
            "narration": "工人在处理材料。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人把材料送进机器，机器开始挤压",
                    "visual_facts": ["worker feeds material into machine", "machine starts pressing"],
                    "narration": "工人在处理材料。画面里worker feeds material into machine，machine starts pressing。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 26,
                    "end": 32,
                    "visual": "材料被压平后继续向前推出",
                    "visual_facts": ["flattened material exits the machine"],
                    "narration": "压平后的材料继续往前送，下一步处理就接上了。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 26, "end": 32}],
        }

        commentary._validate_commentary_script_for_target(data, 34.0, "full", "zh")

        narration = data["narration_blocks"][0]["narration"]
        self.assertIn("工人在处理材料。工人把材料送进机器，机器开始挤压", narration)
        self.assertNotIn("画面里", narration)
        self.assertNotIn("worker feeds material", narration)
        self.assertNotIn("machine starts pressing", narration)

    def test_full_mode_inserts_separator_before_cached_zh_visual_fact_tail(self):
        data = {
            "narration": "这一步很关键画面里工人把材料送进机器，机器开始挤压。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人把材料送进机器，机器开始挤压",
                    "narration": "这一步很关键画面里工人把材料送进机器，机器开始挤压。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 26,
                    "end": 32,
                    "visual": "材料被压平后继续向前推出",
                    "narration": "压平后的材料继续往前送，下一步处理就接上了。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 26, "end": 32}],
        }

        commentary._finalize_full_mode_narration_blocks_for_render(data, 34.0, "full", "zh")

        narration = data["narration_blocks"][0]["narration"]
        self.assertIn("这一步很关键。工人把材料送进机器，机器开始挤压", narration)
        self.assertNotIn("画面里", narration)

    def test_full_mode_sanitizes_cached_detached_visual_phrasing(self):
        data = {
            "narration": "画面里工人把材料送进机器，机器开始挤压。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人把材料送进机器，机器开始挤压",
                    "narration": "画面里工人把材料送进机器，机器开始挤压。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 26,
                    "end": 32,
                    "visual": "材料被压平后继续向前推出",
                    "narration": "视频中材料继续往前送，下一步处理就接上了。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 26, "end": 32}],
        }

        commentary._finalize_full_mode_narration_blocks_for_render(data, 34.0, "full", "zh")

        narration = "\n".join(block["narration"] for block in data["narration_blocks"])
        self.assertNotIn("画面里", narration)
        self.assertNotIn("视频中", narration)
        self.assertIn("工人把材料送进机器，机器开始挤压", narration)

    def test_full_mode_sanitizes_visual_analysis_tone_from_zh_narration(self):
        data = {
            "narration": "当前画面可以看到工人把材料送进机器，画面显示机器开始挤压，最后展示结果。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人把材料送进机器，机器开始挤压",
                    "visual_facts": ["画面显示工人把材料送进机器", "展示最终结果"],
                    "narration": "当前画面可以看到工人把材料送进机器，画面显示机器开始挤压，最后展示结果。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 26,
                    "end": 34,
                    "visual": "材料被压平后继续向前推出",
                    "narration": "压平后的材料继续往前送，下一步处理就接上了。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 26, "end": 34}],
        }

        commentary._finalize_full_mode_narration_blocks_for_render(data, 36.0, "full", "zh")

        narration = "\n".join(block["narration"] for block in data["narration_blocks"])
        self.assertNotRegex(narration, r"当前画面|画面显示|画面展示|可以看到|能看到")
        self.assertIn("工人把材料送进机器", narration)
        self.assertIn("最后呈现结果", narration)

    def test_scene_fact_sentence_uses_natural_zh_visual_fact(self):
        block = {
            "visual": "当前画面显示工人把材料送进机器",
            "visual_facts": ["画面显示工人把材料送进机器", "展示最终结果"],
        }

        sentence = commentary._scene_fact_sentence(block, "zh")

        self.assertEqual("工人把材料送进机器，呈现最终结果。", sentence)

    def test_full_mode_drops_trailing_pause_for_complete_commentary(self):
        data = {
            "narration": "工人把材料处理完成。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 18,
                    "visual": "工人把材料处理完成",
                    "narration": "工人把材料处理完成。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 18,
                    "end": 30,
                    "visual": "普通空镜等待",
                    "narration": "",
                    "pause": True,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 18}, {"start": 18, "end": 30}],
        }

        commentary._finalize_full_mode_narration_blocks_for_render(data, 32.0, "full", "zh")

        self.assertEqual(1, len(data["narration_blocks"]))
        self.assertFalse(data["narration_blocks"][-1]["pause"])
        self.assertEqual([{"start": 0.0, "end": 18.0, "reason": "工人把材料处理完成"}], data["edit_segments"])

    def test_full_mode_does_not_append_editorial_closing_to_narration(self):
        data = {
            "narration": "这场高空赌局终于收场。身穿灰色防护服的人站在树下。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 18,
                    "visual": "身穿灰色防护服的人站在树下",
                    "narration": "这场高空赌局终于收场。身穿灰色防护服的人站在树下。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 18}],
        }

        commentary._finalize_full_mode_narration_blocks_for_render(data, 18.0, "full", "zh")

        narration = data["narration_blocks"][-1]["narration"]
        self.assertEqual("这场高空赌局终于收场。身穿灰色防护服的人站在树下", narration)
        self.assertNotIn("这段解说", narration)
        self.assertNotIn("自然收住", narration)

    def test_full_mode_rejects_editorial_meta_narration_before_tts(self):
        with self.assertRaisesRegex(Exception, "editorial phrasing|banned phrase"):
            commentary._validate_commentary_script_for_target(
                {
                    "narration": "工人把袋子绑紧。到这里，前面的动作和结果已经交代完整，这段解说也自然收住。",
                    "narration_blocks": [
                        {
                            "start": 0,
                            "end": 18,
                            "visual": "工人把袋子绑紧",
                            "narration": "工人把袋子绑紧。到这里，前面的动作和结果已经交代完整，这段解说也自然收住。",
                            "pause": False,
                            "video_speed": 1.0,
                        },
                    ],
                },
                18.0,
                "full",
                "zh",
            )

    def test_narration_rejects_editorial_meta_variants_before_tts(self):
        variants = [
            "工人把袋子绑紧，到此整个过程也算告一段落。",
            "工人把袋子绑紧，该交代的都交代完了，后面就不再展开。",
            "工人把袋子绑紧，这段旁白到这里完成收尾。",
            "工人把袋子绑紧，作为收尾一句，结果已经很清楚。",
            "The worker tightens the bag, and this narration wraps up the segment.",
        ]
        for text in variants:
            with self.subTest(text=text):
                with self.assertRaisesRegex(Exception, "editorial phrasing|banned phrase"):
                    commentary._validate_commentary_script_for_target(
                        {
                            "narration": text,
                            "narration_blocks": [
                                {
                                    "start": 0,
                                    "end": 18,
                                    "visual": "工人把袋子绑紧",
                                    "narration": text,
                                    "pause": False,
                                    "video_speed": 1.0,
                                },
                            ],
                        },
                        18.0,
                        "full",
                        "zh",
                    )

    def test_narration_allows_real_visible_tightening_action(self):
        commentary._validate_commentary_script_for_target(
            {
                "narration": "工人把袋口收紧，再用绳子绕一圈固定住，袋子里的蜂巢不会继续晃。",
                "narration_blocks": [
                    {
                        "start": 0,
                        "end": 18,
                        "visual": "工人把袋口收紧并用绳子固定袋子",
                        "visual_facts": ["工人把袋口收紧", "绳子固定袋子"],
                        "narration": "工人把袋口收紧，再用绳子绕一圈固定住，袋子里的蜂巢不会继续晃。",
                        "pause": False,
                        "video_speed": 1.0,
                    },
                ],
            },
            18.0,
            "short",
            "zh",
        )

    def test_full_mode_accepts_concise_scene_matched_long_narrated_block(self):
        data = {
            "narration": "工人先把材料摊开检查，能看到不同碎料被分到两侧，方便后面继续挑出有价值的部分。",
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 24,
                    "visual": "工人把材料摊开检查，并把不同碎料分到两侧",
                    "visual_facts": ["材料被摊开", "不同碎料被分到两侧"],
                    "evidence_timestamps": [4.0, 16.0],
                    "narration": "工人先把材料摊开检查，能看到不同碎料被分到两侧，方便后面继续挑出有价值的部分。",
                    "pause": False,
                    "video_speed": 1.0,
                },
                {
                    "start": 24,
                    "end": 360,
                    "visual": "后续流程完成并展示结果",
                    "visual_facts": ["工人继续处理材料并展示最终结果"],
                    "narration": "随后工人继续把材料整理、分离和检查，最后把处理后的结果集中展示出来。",
                    "pause": False,
                    "video_speed": 1.0,
                },
            ],
            "edit_segments": [{"start": 0, "end": 24}, {"start": 24, "end": 360}],
        }

        commentary._validate_commentary_script_for_target(data, 400.0, "full", "zh")

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

    def test_openai_regeneration_prompt_handles_historical_density_error_without_padding(self):
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
        self.assertIn("Do not pad any block with repeated generic filler", prompt)
        self.assertIn("Fix timing and scene match", prompt)
        self.assertIn("worker moves copper scrap", prompt)
        self.assertIn("Keep unrelated valid narration_blocks unchanged", prompt)
        self.assertNotIn("len(non_whitespace(narration))", prompt)
        self.assertNotIn("density_audit", prompt)

    def test_previous_error_note_sanitizes_historical_density_failure(self):
        note = commentary._retry_correction_note(
            "AI narration block is too short for its selected visual range. "
            "Block 14 has 54 chars for 37.5s of playable visuals; expected at least 98."
        )

        self.assertIn("too sparse for its selected visual range", note)
        self.assertIn("Do not add filler just to satisfy a word-count target", note)
        self.assertIn("Long selected source ranges still need matching spoken detail", note)
        self.assertNotIn("54 chars", note)
        self.assertNotIn("expected at least 98", note)

    def test_medium_retry_includes_previous_completion_claim_error_note(self):
        transcript = {
            "text": "beekeeper process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "beekeeper process"}],
        }
        payload = {
            "title": "Medium",
            "summary": "summary",
            "hook": "hook",
            "narration": "手套贴着树干继续操作，蜂群还在旁边飞，镜头只看到当前处理动作。" * 12,
            "narration_blocks": [
                {
                    "start": 0,
                    "end": 30,
                    "visual": "beekeeper handles hive on tree with bees still active",
                    "visual_facts": ["beekeeper handles hive on tree"],
                    "evidence_timestamps": [10],
                    "narration": "手套贴着树干继续操作，蜂群还在旁边飞，镜头只看到当前处理动作。",
                    "pause": False,
                }
            ],
            "edit_segments": [{"start": 0, "end": 30, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }

        class CapturingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return pytypes.SimpleNamespace(text=commentary.json.dumps(payload, ensure_ascii=False))

        fake_models = CapturingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)
        previous_error = (
            "AI narration block claims a completed packing/ending action that is not supported by its selected visual range. "
            "Block 13 says the work is finished, completed, or 收工, but the block visual description does not show a final result."
        )

        with patch.object(commentary, "create_gemini_client", return_value=fake_client):
            commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=120,
                gemini_key="key",
                analysis_mode="current",
                target_duration="medium",
                language="zh",
                previous_error=previous_error,
            )

        prompt = fake_models.calls[0]["contents"][0]
        self.assertIn("Retry correction note", prompt)
        self.assertIn("completed packing/ending action", prompt)
        self.assertIn("describe only what is visible", prompt)
        self.assertNotIn("full-mode target duration", prompt)

    def test_previous_error_invalidates_cached_script_only_for_script_failures(self):
        self.assertFalse(commentary._previous_error_invalidates_cached_script(
            "ffmpeg failed while burning subtitles: Error writing trailer: No space left on device"
        ))
        self.assertFalse(commentary._previous_error_invalidates_cached_script(
            "Error muxing a packet. Conversion failed!"
        ))
        self.assertTrue(commentary._previous_error_invalidates_cached_script(
            "AI narration_blocks do not match the selected full-mode edit target."
        ))
        self.assertTrue(commentary._previous_error_invalidates_cached_script(
            "Gemini returned invalid JSON for the commentary script."
        ))

    def test_medium_gemini_invalid_json_is_repaired_before_failing(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        repaired_payload = {
            "title": "Repaired",
            "summary": "summary",
            "hook": "hook",
            "narration": "这是一段修复后的解说内容。" * 30,
            "edit_segments": [{"start": 0, "end": 60, "reason": "process"}],
            "chapters": [],
            "hashtags": [],
        }

        class RepairingModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return pytypes.SimpleNamespace(text='{"title":"Broken" "summary":"missing comma"}')
                return pytypes.SimpleNamespace(text=commentary.json.dumps(repaired_payload, ensure_ascii=False))

        fake_models = RepairingModels()
        fake_client = pytypes.SimpleNamespace(models=fake_models)

        with patch.object(commentary, "create_gemini_client", return_value=fake_client):
            result = commentary.generate_commentary_script(
                transcript=transcript,
                video_title="Demo",
                duration=120,
                gemini_key="key",
                analysis_mode="current",
                target_duration="medium",
                language="zh",
            )

        self.assertEqual("Repaired", result["title"])
        self.assertEqual(2, len(fake_models.calls))
        self.assertIn("PREVIOUS RESPONSE WAS NOT VALID JSON", fake_models.calls[1]["contents"][0])

    def test_audio_fit_allows_render_sync_speedup_for_short_visual_block(self):
        commands = []

        with patch.object(commentary, "_get_audio_duration", return_value=21.3), \
            patch.object(commentary, "_run_command", side_effect=lambda cmd: commands.append(cmd)):
            commentary._fit_audio_part_to_duration("voice.mp3", "fit.m4a", 13.8)

        self.assertEqual(1, len(commands))
        audio_filter = commands[0][commands[0].index("-af") + 1]
        self.assertIn("atempo=1.543478", audio_filter)
        self.assertIn("atrim=0:13.800", audio_filter)

    def test_audio_concat_resets_part_timestamps_before_aac_encode(self):
        commands = []

        with patch.object(commentary, "_run_command", side_effect=lambda cmd: commands.append(cmd)):
            commentary._concat_media_parts(
                ["one.m4a", "two.m4a"],
                "joined.m4a",
                "/tmp",
                codec="aac",
                media_type="audio",
            )

        self.assertEqual(1, len(commands))
        command = commands[0]
        self.assertNotIn("-f", command)
        self.assertNotIn("concat_commentary", " ".join(command))
        self.assertIn("-filter_complex", command)
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("asetpts=N/SR/TB[a0]", filter_complex)
        self.assertIn("asetpts=N/SR/TB[a1]", filter_complex)
        self.assertIn("[a0][a1]concat=n=2:v=0:a=1[aout]", filter_complex)
        self.assertIn("-map", command)
        self.assertEqual("[aout]", command[command.index("-map") + 1])
        self.assertIn("-c:a", command)
        self.assertEqual("aac", command[command.index("-c:a") + 1])

    def test_concat_media_parts_writes_absolute_paths_for_relative_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir, exist_ok=True)
            part_path = os.path.join(tmpdir, "part.mp4")
            output_path = os.path.join(tmpdir, "out.mp4")
            open(part_path, "wb").close()
            captured = {}

            def fake_run_command(cmd, cwd=None):
                captured["cmd"] = cmd

            current_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.object(commentary, "_run_command", side_effect=fake_run_command):
                    commentary._concat_media_parts(["part.mp4"], output_path, work_dir)
            finally:
                os.chdir(current_dir)

            list_path = captured["cmd"][captured["cmd"].index("-i") + 1]
            with open(list_path, "r", encoding="utf-8") as f:
                concat_list = f.read()

        self.assertIn(os.path.abspath(part_path).replace("\\", "/"), concat_list)

    def test_synced_block_slows_accelerated_video_when_tts_is_long(self):
        fit_calls = []
        video_commands = []
        progress = []

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(commentary, "_get_video_duration", return_value=30.0), \
            patch.object(commentary, "generate_commentary_voiceover"), \
            patch.object(commentary, "_get_audio_duration", return_value=25.0), \
            patch.object(commentary, "_fit_audio_part_to_duration", side_effect=lambda src, dst, dur: fit_calls.append((src, dst, dur))), \
            patch.object(commentary, "_run_command", side_effect=lambda cmd: video_commands.append(cmd)), \
            patch.object(commentary, "_concat_media_parts"), \
            patch.object(commentary, "_force_audio_clip_duration"), \
            patch.object(commentary, "FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED", 2.0):
            _, durations = commentary._create_block_synced_visuals_and_audio(
                video_path="source.mp4",
                narration_blocks=[{
                    "start": 0,
                    "end": 30,
                    "visual": "slow repeated action",
                    "narration": "A long but useful explanation for this accelerated visual block.",
                    "video_speed": 3.0,
                }],
                timed_video_path=os.path.join(tmpdir, "timed.mp4"),
                voiceover_path=os.path.join(tmpdir, "voice.m4a"),
                ambient_audio_path=os.path.join(tmpdir, "ambient.m4a"),
                aspect_mode="original",
                work_dir=tmpdir,
                tts_provider="edge",
                language="en",
                elevenlabs_key=None,
                voice_id="voice",
                edge_voice=None,
                original_audio_volume=0.0,
                block_concurrency=1,
                progress=progress.append,
            )

        self.assertAlmostEqual(12.5, durations[0], places=1)
        self.assertAlmostEqual(12.5, fit_calls[0][2], places=1)
        video_filter = video_commands[0][video_commands[0].index("-vf") + 1]
        self.assertIn("setpts=PTS/2.4", video_filter)
        self.assertTrue(any("Slowing commentary block" in item for item in progress))

    def test_synced_block_shortens_and_regenerates_tts_when_video_cannot_slow_enough(self):
        spoken_texts = []
        fit_calls = []

        def fake_voiceover(text, output_path, **kwargs):
            spoken_texts.append(text)
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir, \
            patch.object(commentary, "_get_video_duration", return_value=10.0), \
            patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
            patch.object(commentary, "_get_audio_duration", side_effect=[25.0, 18.0]), \
            patch.object(commentary, "_fit_audio_part_to_duration", side_effect=lambda src, dst, dur: fit_calls.append((src, dst, dur))), \
            patch.object(commentary, "_run_command"), \
            patch.object(commentary, "_concat_media_parts"), \
            patch.object(commentary, "_force_audio_clip_duration"), \
            patch.object(commentary, "FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED", 2.0), \
            patch.object(commentary, "FULL_MODE_RENDER_SYNC_TTS_REWRITE_ATTEMPTS", 2):
            commentary._create_block_synced_visuals_and_audio(
                video_path="source.mp4",
                narration_blocks=[{
                    "start": 0,
                    "end": 10,
                    "visual": "dense visual moment",
                    "narration": "This narration is intentionally too long for the available footage and must be shortened automatically.",
                    "video_speed": 1.0,
                }],
                timed_video_path=os.path.join(tmpdir, "timed.mp4"),
                voiceover_path=os.path.join(tmpdir, "voice.m4a"),
                ambient_audio_path=os.path.join(tmpdir, "ambient.m4a"),
                aspect_mode="original",
                work_dir=tmpdir,
                tts_provider="edge",
                language="en",
                elevenlabs_key=None,
                voice_id="voice",
                edge_voice=None,
                original_audio_volume=0.0,
                block_concurrency=1,
            )

        self.assertEqual(2, len(spoken_texts))
        self.assertLess(len(spoken_texts[1]), len(spoken_texts[0]))
        self.assertAlmostEqual(10.0, fit_calls[0][2], places=1)

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

    def test_video_upload_uses_ascii_safe_temp_path_for_emoji_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ATTACKED!_Brave_Honey_Hunter_vs__Giant_Angry_Bees_😱.mp4")
            with open(path, "wb") as f:
                f.write(b"video")

            uploaded = pytypes.SimpleNamespace(
                name="files/video-1",
                uri="https://files.example/video-1",
                mime_type="video/mp4",
                state="ACTIVE",
            )
            upload_calls = []

            def fake_upload(file):
                upload_calls.append(file)
                file.encode("ascii")
                return uploaded

            client = pytypes.SimpleNamespace(
                files=pytypes.SimpleNamespace(
                    upload=fake_upload,
                    get=lambda name: uploaded,
                )
            )

            part = commentary._upload_gemini_video_part(client, path)

            self.assertEqual("https://files.example/video-1", part.file_data.file_uri)
            self.assertEqual(1, len(upload_calls))
            self.assertNotEqual(path, upload_calls[0])
            upload_calls[0].encode("ascii")
            self.assertFalse(os.path.exists(upload_calls[0]))

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

    def test_visual_edit_requires_ai_selected_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            output_path = os.path.join(tmpdir, "out.mp4")
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir)
            open(source_path, "wb").close()

            with self.assertRaisesRegex(Exception, "AI-selected edit_segments"):
                commentary._create_visual_edit(
                    source_path,
                    [],
                    output_path,
                    "16:9",
                    work_dir,
                )

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

    def test_downloader_quality_settings_cap_high_default_at_1080p_and_keep_low_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            high = main._youtube_download_settings("My Video", tmpdir)
            low = main._youtube_download_settings("My Video", tmpdir, quality="low", filename_suffix="_analysis_low")

        self.assertIn("bestvideo[height<=1080][vcodec^=avc1][ext=mp4]", high["format"])
        self.assertIn("best[height<=1080][ext=mp4]", high["format"])
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

    def test_video_mode_retry_after_proxy_oss_timeout_falls_back_to_current_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            prepared_path = os.path.join(tmpdir, "source_gemini_360p.mp4")
            frame_path = os.path.join(tmpdir, "frame.jpg")
            open(source_path, "wb").close()
            open(prepared_path, "wb").close()
            open(frame_path, "wb").close()
            transcript = {
                "text": "cached transcript",
                "segments": [{"start": 0, "end": 2, "text": "cached transcript"}],
                "language": "zh",
            }
            previous_error = (
                "500 INTERNAL. {'error': {'code': 500, 'message': "
                "'Response timeout for 60000ms, please increase the timeout, "
                "see more details at https://github.com/ali-sdk/ali-oss#responsetimeouterror'}}"
            )

            with patch.object(commentary, "_prepare_analysis_video_for_gemini") as prepare_analysis, \
                patch.object(commentary, "_get_video_info", return_value={"duration": 42, "width": 1920, "height": 1080, "fps": 30}), \
                patch.object(commentary, "transcribe_video", return_value=transcript) as transcribe, \
                patch.object(commentary, "_extract_keyframes", return_value=[frame_path]) as extract_frames, \
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
                patch.object(commentary, "_mix_voiceover_with_video"), \
                patch.object(commentary, "_generate_commentary_covers", return_value={}):

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    prepared_analysis_video_path=prepared_path,
                    gemini_file={"uri": "files/old-video", "mime_type": "video/mp4"},
                    previous_error=previous_error,
                )

        prepare_analysis.assert_not_called()
        transcribe.assert_called_once()
        extract_frames.assert_called_once()
        self.assertEqual("current", generate_script.call_args.kwargs["analysis_mode"])
        self.assertEqual(transcript, generate_script.call_args.kwargs["transcript"])
        self.assertEqual([frame_path], generate_script.call_args.kwargs["frame_paths"])
        self.assertIsNone(generate_script.call_args.kwargs["analysis_video_path"])
        self.assertIsNone(generate_script.call_args.kwargs["gemini_file"])
        self.assertEqual("current", result["analysis_mode"])
        self.assertEqual("video", result["requested_analysis_mode"])
        self.assertIn("ali-oss", result["analysis_fallback_reason"])

    def test_current_mode_reuses_cached_transcript_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()
            cached_transcript = {
                "text": "cached transcript",
                "segments": [{"start": 0, "end": 2, "text": "cached transcript"}],
                "language": "zh",
            }
            commentary._save_commentary_transcript_cache(tmpdir, cached_transcript)
            checkpoints = []

            with patch.object(commentary, "_get_video_info", return_value={"duration": 42, "width": 1920, "height": 1080, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "_extract_keyframes", return_value=[]), \
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
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_mix_voiceover_with_video"), \
                patch.object(commentary, "_generate_commentary_covers", return_value={}):

                commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="current",
                    checkpoint=checkpoints.append,
                )

        transcribe.assert_not_called()
        self.assertEqual(cached_transcript, generate_script.call_args.kwargs["transcript"])
        create_visual.assert_called_once()
        self.assertFalse(any("transcript_path" in item for item in checkpoints))

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

            ai_blocks = scene_matched_blocks()
            ai_blocks[1]["video_speed"] = 1.35
            ai_blocks[1]["speed_reason"] = "AI chose light acceleration for a repeated process range"

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in ai_blocks),
                    "narration_blocks": ai_blocks,
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
        expected_segments = commentary._narration_blocks_to_edit_segments(ai_blocks)
        self.assertEqual(result["edit_segments"], expected_segments)
        fit_video.assert_not_called()
        self.assertEqual(result["edited_visual"], result["timed_visual"])
        self.assertTrue(result["auto_video_speed"])
        self.assertGreater(result["auto_video_speed_summary"]["accelerated_count"], 0)
        self.assertGreater(result["auto_video_speed_summary"]["saved_seconds"], 0)
        mix_video.assert_called_once()
        self.assertTrue(mix_video.call_args.kwargs["trim_to_voiceover"])

    def test_medium_duration_uses_narration_blocks_for_block_synced_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.mp4")
            open(source_path, "wb").close()

            blocks = [
                {
                    "start": 0,
                    "end": 20,
                    "visual": "worker walks into the field with a tool",
                    "narration": "她先带着工具走进山路，准备开始今天的采集。",
                    "video_speed": 1.0,
                },
                {
                    "start": 80,
                    "end": 100,
                    "visual": "worker chops into a fallen trunk",
                    "narration": "接着劈开倒下的树干，寻找藏在里面的食材。",
                    "video_speed": 1.0,
                },
                {
                    "start": 120,
                    "end": 140,
                    "visual": "worker picks larvae into a bucket",
                    "narration": "画面切到她把虫子一只只捡进桶里。",
                    "video_speed": 1.0,
                },
            ]

            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3600, "width": 1920, "height": 1080, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in blocks),
                    "narration_blocks": blocks,
                    "edit_segments": [{"start": 0, "end": 180, "reason": "legacy long edit"}],
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover") as voiceover, \
                patch.object(commentary, "_create_visual_edit") as create_visual, \
                patch.object(commentary, "_fit_video_to_voiceover") as fit_video, \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", return_value=("ambient.m4a", [4.0, 5.0, 4.5])) as create_synced, \
                patch.object(commentary, "_create_ambient_audio_bed") as create_ambient, \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video:

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    target_duration="medium",
                )

        transcribe.assert_not_called()
        voiceover.assert_not_called()
        create_visual.assert_not_called()
        fit_video.assert_not_called()
        create_ambient.assert_not_called()
        create_synced.assert_called_once()
        self.assertEqual(commentary._narration_blocks_to_edit_segments(blocks), result["edit_segments"])
        self.assertEqual(result["edited_visual"], result["timed_visual"])
        self.assertEqual([4.0, 5.0, 4.5], result["subtitle_block_durations"])
        mix_video.assert_called_once()

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
        episode_commands = [
            item for item in rendered_commands
            if item[0][-1].endswith(".mp4") and "_episode_" in os.path.basename(item[0][-1])
        ]
        self.assertEqual(3, len(episode_commands))
        self.assertEqual("-ss", episode_commands[0][0][2])
        self.assertEqual("0.000", episode_commands[0][0][3])
        self.assertEqual(episode_commands[2][0][-1], result["episodes"][2]["video_path"])

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

            blocks = scene_matched_blocks()
            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video") as transcribe, \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in blocks),
                    "narration_blocks": blocks,
                    "edit_segments": commentary._narration_blocks_to_edit_segments(blocks),
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=1100), \
                patch.object(commentary, "_create_visual_edit", side_effect=fake_create_visual), \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", return_value=("ambient.m4a", [100.0] * len(blocks))), \
                patch.object(commentary, "_fit_video_to_voiceover") as fit_video, \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_resolve_background_music_track", return_value={
                    "id": "aodebiao_caravan",
                    "label": "默认 奥德彪专属音乐",
                    "title": "Caravan",
                    "artist": "a_hisa",
                    "path": "caravan.mp3",
                }) as resolve_music, \
                patch.object(commentary, "_create_background_music_bed", return_value=os.path.join(tmpdir, "Remix_background_music.m4a")) as create_music, \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video:

                result = commentary.generate_commentary_video(
                    source=source_path,
                    output_dir=tmpdir,
                    gemini_key="key",
                    source_type="file",
                    subtitles=False,
                    analysis_mode="video",
                    target_duration="full",
                    background_music_enabled=True,
                    background_music_track="aodebiao_caravan",
                    background_music_volume=0.22,
                )

        transcribe.assert_not_called()
        fit_video.assert_not_called()
        self.assertEqual(result["edited_visual"], result["timed_visual"])
        self.assertEqual(commentary._narration_blocks_to_edit_segments(blocks), result["edit_segments"])
        self.assertTrue(mix_video.call_args.kwargs["trim_to_voiceover"])
        resolve_music.assert_called_once_with("aodebiao_caravan")
        create_music.assert_called_once()
        self.assertEqual("Remix_background_music.m4a", result["background_music_audio"])
        self.assertEqual("默认 奥德彪专属音乐", result["background_music_label"])
        self.assertEqual(0.22, result["background_music_volume"])
        self.assertEqual(0.22, create_music.call_args.args[3])
        self.assertEqual(os.path.join(tmpdir, "Remix_background_music.m4a"), mix_video.call_args.kwargs["background_music_path"])

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

            def fake_subtitles(_blocks, output_path, _block_durations=None, **_kwargs):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("[Script Info]")

            blocks = scene_matched_blocks()
            with patch.object(commentary, "_prepare_analysis_video_for_gemini", return_value=source_path), \
                patch.object(commentary, "_get_video_info", return_value={"duration": 3935, "width": 854, "height": 480, "fps": 30}), \
                patch.object(commentary, "transcribe_video"), \
                patch.object(commentary, "generate_commentary_script", return_value={
                    "title": "Remix",
                    "summary": "summary",
                    "hook": "hook",
                    "narration": "\n\n".join(block["narration"] for block in blocks),
                    "narration_blocks": blocks,
                    "edit_segments": commentary._narration_blocks_to_edit_segments(blocks),
                    "chapters": [],
                    "hashtags": [],
                }), \
                patch.object(commentary, "generate_commentary_voiceover", side_effect=fake_voiceover), \
                patch.object(commentary, "_get_audio_duration", return_value=1100), \
                patch.object(commentary, "_create_visual_edit", side_effect=fake_create_visual), \
                patch.object(commentary, "_create_block_synced_visuals_and_audio", return_value=("ambient.m4a", [100.0] * len(blocks))), \
                patch.object(commentary, "_create_ambient_audio_bed", return_value=None), \
                patch.object(commentary, "_mix_voiceover_with_video") as mix_video, \
                patch.object(commentary, "_probe_video_dimensions", return_value=(854, 480)), \
                patch.object(commentary, "_write_block_timed_ass", side_effect=fake_subtitles) as write_subtitles, \
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
            629,
            int(commentary._target_visual_duration_seconds(3935, "full")),
        )
        self.assertEqual(2644, commentary._maximum_narration_chars(3935, "full", "zh"))
        self.assertLess(commentary._target_visual_duration_seconds(3600, "full"), 900)
        self.assertEqual([], commentary._resolve_edit_segments_for_target([], 3935, "full"))
        with self.assertRaisesRegex(Exception, "will not invent fallback edit_segments"):
            commentary._require_ai_selected_edit_segments({"edit_segments": []}, 3935, "full")

    def test_full_duration_compacts_medium_complete_process_sources(self):
        self.assertEqual(
            284,
            int(commentary._target_visual_duration_seconds(678.4, "full")),
        )
        self.assertEqual(
            378,
            int(commentary._target_visual_duration_seconds(900, "full")),
        )
        self.assertEqual(
            504,
            int(commentary._target_visual_duration_seconds(1200, "full")),
        )
        self.assertEqual([], commentary._resolve_edit_segments_for_target([], 678.4, "full"))

    def test_full_duration_uses_visual_analysis_to_choose_content_fit_runtime(self):
        visual_analysis = beehive_visual_analysis()

        target = commentary._target_visual_duration_seconds_for_analysis(678.4, "full", visual_analysis)

        self.assertEqual(284, int(commentary._target_visual_duration_seconds(678.4, "full")))
        self.assertEqual(284, int(target))
        self.assertLess(target, 360)
        self.assertGreater(target, commentary._visual_candidate_duration_seconds(visual_analysis, 678.4))
        self.assertLess(target, 678.4 * 0.5)

    def test_full_duration_rejects_ten_minute_result_when_visual_analysis_prefers_shorter_cut(self):
        narration = repeated_scene_text("工人继续处理蜂巢，蜂群围着树干飞动，袋子和工具跟着动作调整", 20)
        blocks = []
        start = 0.0
        for _index in range(16):
            end = start + 39.25
            blocks.append({
                "start": start,
                "end": end,
                "visual": "工人在树干上处理蜂巢，蜂群持续围绕",
                "visual_facts": ["工人在树干上处理蜂巢，蜂群持续围绕"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            })
            start = end

        with self.assertRaisesRegex(Exception, "selected too much source footage|do not match the selected full-mode edit target"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                678.4,
                "full",
                "zh",
                visual_analysis=beehive_visual_analysis(),
            )

    def test_full_duration_rejects_underfilled_openai_content_cut_instead_of_padding(self):
        visual_analysis = beehive_visual_analysis()
        narration = repeated_scene_text("工人继续处理蜂巢，蜂群围着树干飞动，袋子和工具跟着动作调整", 8)
        blocks = [
            {
                "start": 0,
                "end": 115,
                "visual": "工人开始沿树干处理蜂巢，蜂群围着树干飞动",
                "visual_facts": ["工人开始沿树干处理蜂巢，蜂群围着树干飞动"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 180,
                "end": 325,
                "visual": "工人切开蜂巢，把蜂巢往袋子方向处理",
                "visual_facts": ["工人切开蜂巢，把蜂巢往袋子方向处理"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 420,
                "end": 520,
                "visual": "袋子和绳子固定蜂巢，蜂群还在周围飞动",
                "visual_facts": ["袋子和绳子固定蜂巢，蜂群还在周围飞动"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 600,
                "end": 620,
                "visual": "工人下到树干下方准备收尾",
                "visual_facts": ["工人下到树干下方准备收尾"],
                "narration": "人已经从树干上下来，蜂巢切开一大块，装好收工。",
                "pause": False,
                "video_speed": 1.0,
            },
        ]
        data = {"narration": narration, "narration_blocks": blocks}

        with self.assertRaisesRegex(Exception, "do not match the selected full-mode edit target"):
            commentary._validate_commentary_script_for_target(
                data,
                678.4,
                "full",
                "zh",
                visual_analysis=visual_analysis,
            )

    def test_full_duration_rejects_six_nineteen_result_when_visual_analysis_prefers_shorter_cut(self):
        visual_analysis = beehive_visual_analysis()
        narration = repeated_scene_text("工人继续处理蜂巢，蜂群围着树干飞动，袋子和工具跟着动作调整", 3)
        blocks = []
        start = 0.0
        for _index in range(16):
            end = start + (379.7 / 16)
            blocks.append({
                "start": start,
                "end": end,
                "visual": "工人在树干上处理蜂巢，蜂群持续围绕",
                "visual_facts": ["工人在树干上处理蜂巢，蜂群持续围绕"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            })
            start = end

        with self.assertRaisesRegex(Exception, "do not match the selected full-mode edit target"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                678.4,
                "full",
                "zh",
                visual_analysis=visual_analysis,
            )

    def test_visual_budget_repair_prompt_tells_ai_to_recover_useful_time_not_backend_fill(self):
        narration = repeated_scene_text("工人继续处理蜂巢，蜂群围着树干飞动，袋子和工具跟着动作调整", 3)
        blocks = [
            {
                "start": index * 30.0,
                "end": index * 30.0 + 28.0,
                "visual": "工人在树干上处理蜂巢，蜂群持续围绕",
                "visual_facts": ["工人在树干上处理蜂巢，蜂群持续围绕"],
                "evidence_timestamps": [index * 30.0 + 10.0],
                "narration": narration,
                "pause": False,
                "video_speed": 1.2 if index % 2 == 0 else 1.0,
                "speed_reason": "AI chose mild acceleration for repeated visible movement" if index % 2 == 0 else "",
            }
            for index in range(16)
        ]
        script = {"narration": narration, "narration_blocks": blocks}
        validation_error = Exception(
            "AI narration_blocks do not match the selected full-mode edit target. "
            "Got 351.2s of block-matched visuals for a 391.8s target; expected between 360.5s and 431.0s. "
            "AI must select enough useful scene-matched source ranges from the visual evidence; OpenShorts will not invent filler ranges or evenly sample the timeline."
        )

        repair = commentary._focused_validation_repair_instruction(
            validation_error,
            script,
            "zh",
            16,
            duration=678.4,
            target_duration="full",
        )
        openai_prompt = commentary._build_openai_regeneration_prompt(
            "ORIGINAL",
            script,
            validation_error,
            678.4,
            "full",
            "zh",
            5,
            visual_analysis=beehive_visual_analysis(),
        )
        gemini_prompt = commentary._build_regeneration_prompt(
            "ORIGINAL",
            script,
            678.4,
            "full",
            "zh",
            attempt=5,
            validation_error=validation_error,
        )
        finalization_prompt = commentary._build_visual_plan_finalization_prompt(
            script,
            678.4,
            "full",
            "zh",
            attempt=5,
            validation_error=validation_error,
        )
        retry_note = commentary._retry_correction_note(str(validation_error))

        self.assertIn("under-selected", repair)
        self.assertIn("Add at least", repair)
        self.assertIn("AI-selected, useful, scene-matched source ranges", repair)
        self.assertIn("lowering over-aggressive video_speed", repair)
        self.assertIn("Do not invent filler ranges", repair)
        self.assertIn("Return exactly 16 narration_blocks", openai_prompt)
        self.assertIn("Narration must be at most 1645", openai_prompt)
        self.assertIn("Narration must be at most 1645", gemini_prompt)
        self.assertIn("The final narration must be at most 1645", finalization_prompt)
        self.assertIn("selected playable visual time should be near 391 seconds", openai_prompt)
        self.assertIn("selected playable visual time should be near 391 seconds", gemini_prompt)
        self.assertIn("391.8s target", retry_note)
        self.assertIn("The AI must decide keep/cut/splice/speed", retry_note)
        self.assertIn("Do not use backend filler", retry_note)
        self.assertNotIn("fallback", repair.lower())

    def test_density_repair_prompt_preserves_full_edit_visual_budget(self):
        narration = repeated_scene_text("工人继续处理蜂巢，蜂群围着树干飞动，袋子和工具跟着动作调整", 2)
        blocks = [
            {
                "start": index * 27.0,
                "end": index * 27.0 + 25.0,
                "visual": "工人在树干上处理蜂巢，蜂群持续围绕",
                "visual_facts": ["工人在树干上处理蜂巢，蜂群持续围绕"],
                "evidence_timestamps": [index * 27.0 + 8.0],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            }
            for index in range(16)
        ]
        blocks[10]["start"] = 270.0
        blocks[10]["end"] = 329.3
        blocks[10]["narration"] = "蜂巢继续处理，袋子还在旁边。"
        validation_error = Exception(
            "AI narration block is too short for its selected visual range. "
            "Block 11 has 60 chars for 59.3s of playable visuals; expected at least 89. "
            "Shorten this block's source range, split it, add concrete scene-matched narration, or mark a brief pause=true moment. "
            "Do not rely on render-time speedup or trimming because that makes the commentary and visuals drift out of sync."
        )

        repair = commentary._focused_validation_repair_instruction(
            validation_error,
            {"narration": narration, "narration_blocks": blocks},
            "zh",
            16,
            duration=678.4,
            target_duration="full",
            target_seconds=391.8,
        )

        self.assertIn("Full-edit timing must remain inside the target window", repair)
        self.assertIn("391.8s target", repair)
        self.assertIn("352.6-431.0s", repair)
        self.assertIn("Do not let a local density fix break the full-edit visual target", repair)
        self.assertIn("recover the removed playable time", repair)
        self.assertIn("Preserve exactly 16 total narration_blocks", repair)

    def test_density_repair_prompt_lists_all_current_sparse_blocks(self):
        healthy_narration = repeated_scene_text("工人贴着树干继续处理蜂巢，蜂群围着手套飞动，袋子和刀具跟着动作调整", 3)
        blocks = [
            {
                "start": index * 31.0,
                "end": index * 31.0 + 30.0,
                "visual": "工人在树干旁处理蜂巢，蜂群持续围绕",
                "visual_facts": ["工人在树干旁处理蜂巢，蜂群持续围绕"],
                "evidence_timestamps": [index * 31.0 + 12.0],
                "narration": healthy_narration,
                "pause": False,
                "video_speed": 1.0,
            }
            for index in range(16)
        ]
        blocks[6]["start"] = 190.0
        blocks[6]["end"] = 246.2
        blocks[6]["visual"] = "蜂巢、树干、手套和袋子都在画面里继续移动"
        blocks[6]["narration"] = "蜂巢还在处理，袋子在旁边。"
        blocks[11]["start"] = 330.0
        blocks[11]["end"] = 377.5
        blocks[11]["visual"] = "工人用刀处理蜂巢并把蜂蜜往袋子里放"
        blocks[11]["narration"] = "刀贴着蜂巢往下切，蜂蜜被放进袋子，蜂群还在周围飞。"
        validation_error = Exception(
            "AI narration block is too short for its selected visual range. "
            "Block 12 has 52 chars for 47.5s of playable visuals; expected at least 72. "
            "Shorten this block's source range, split it, add concrete scene-matched narration, or mark a brief pause=true moment."
        )

        repair = commentary._focused_validation_repair_instruction(
            validation_error,
            {"narration": healthy_narration, "narration_blocks": blocks},
            "zh",
            16,
            duration=678.4,
            target_duration="full",
            target_seconds=479.8,
        )
        prompt = commentary._build_openai_regeneration_prompt(
            "ORIGINAL PROMPT",
            {"narration": healthy_narration, "narration_blocks": blocks},
            validation_error,
            duration=678.4,
            target_duration="full",
            language="zh",
            attempt=7,
        )

        self.assertIn("Current density-risk blocks", repair)
        self.assertIn("Block 12 / narration_blocks[11]", repair)
        self.assertIn("Block 7 / narration_blocks[6]", repair)
        self.assertIn("Fix all listed density-risk blocks in one pass", repair)
        self.assertIn("audit every non-pause block", repair)
        self.assertIn("backend rendering preserves the selected visual ranges", prompt)
        self.assertNotIn("backend rendering handles trailing visual sync", prompt)

    def test_full_mode_validation_attempt_default_allows_budget_followup_after_density_repairs(self):
        self.assertGreaterEqual(commentary.GEMINI_SCRIPT_VALIDATION_ATTEMPTS, 8)

    def test_full_duration_rejects_near_full_source_for_medium_complete_process_source(self):
        narration = repeated_scene_text("工人继续处理蜂巢，袋子和绳子配合着把流程往后推进", 20)
        blocks = [
            {
                "start": 0,
                "end": 220,
                "visual": "工人开始沿树干处理蜂巢",
                "visual_facts": ["工人开始沿树干处理蜂巢"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 220,
                "end": 460,
                "visual": "工人切蜂巢并把蜂巢装进袋子",
                "visual_facts": ["工人切蜂巢并把蜂巢装进袋子"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 460,
                "end": 678.4,
                "visual": "工人继续处理蜂巢并在地面收尾",
                "visual_facts": ["工人继续处理蜂巢并在地面收尾"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
        ]

        with self.assertRaisesRegex(Exception, "selected too much source footage|do not match the selected full-mode edit target"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                678.4,
                "full",
                "zh",
            )

    def test_full_duration_allows_compact_complete_process_cut_for_medium_source(self):
        narration = repeated_scene_text("工人沿着树干处理蜂巢，蜂群和袋子的位置变化继续推进", 8)
        blocks = [
            {
                "start": 0,
                "end": 90,
                "visual": "工人开始沿树干处理蜂巢",
                "visual_facts": ["工人开始沿树干处理蜂巢"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 180,
                "end": 300,
                "visual": "工人切蜂巢并把蜂巢装进袋子",
                "visual_facts": ["工人切蜂巢并把蜂巢装进袋子"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 575,
                "end": 665,
                "visual": "工人继续在树干和地面附近处理蜂巢收尾",
                "visual_facts": ["工人继续在树干和地面附近处理蜂巢收尾"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.25,
                "speed_reason": "AI chose light acceleration for a slow continuation range while preserving the ending context",
            },
        ]
        data = {"narration": narration, "narration_blocks": blocks}

        commentary._validate_commentary_script_for_target(data, 678.4, "full", "zh")

        playable = sum(commentary._block_visual_duration(block) for block in data["narration_blocks"])
        target = commentary._target_visual_duration_seconds(678.4, "full")
        self.assertGreaterEqual(playable, target * commentary.FULL_MODE_MIN_PLAYABLE_TARGET_RATIO)
        self.assertLessEqual(playable, target * commentary.FULL_MODE_MAX_PLAYABLE_TARGET_RATIO)

    def test_full_duration_rejects_ai_speed_without_reason(self):
        narration = repeated_scene_text("工人沿着树干处理蜂巢，蜂群和袋子的位置变化继续推进", 8)
        blocks = [
            {
                "start": 0,
                "end": 130,
                "visual": "工人开始沿树干处理蜂巢",
                "visual_facts": ["工人开始沿树干处理蜂巢"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.35,
            },
            {
                "start": 180,
                "end": 340,
                "visual": "工人切蜂巢并把蜂巢装进袋子",
                "visual_facts": ["工人切蜂巢并把蜂巢装进袋子"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 420,
                "end": 640,
                "visual": "工人继续在树干和地面附近处理蜂巢收尾",
                "visual_facts": ["工人继续在树干和地面附近处理蜂巢收尾"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.0,
            },
        ]

        with self.assertRaisesRegex(Exception, "missing a visual reason"):
            commentary._validate_commentary_script_for_target(
                {"narration": narration, "narration_blocks": blocks},
                678.4,
                "full",
                "zh",
            )

    def test_full_duration_rejects_ai_speed_that_makes_voiceover_too_tight(self):
        long_narration = repeated_scene_text("工人沿着树干持续处理蜂巢，蜂群和工具的变化都要跟当前画面对上", 18)
        blocks = [
            {
                "start": 0,
                "end": 40,
                "visual": "工人开始沿树干处理蜂巢",
                "visual_facts": ["工人开始沿树干处理蜂巢"],
                "narration": long_narration,
                "pause": False,
                "video_speed": 2.5,
                "speed_reason": "AI says this long range is repeated setup and can be very fast",
            },
            {
                "start": 80,
                "end": 260,
                "visual": "工人切蜂巢并把蜂巢装进袋子",
                "visual_facts": ["工人切蜂巢并把蜂巢装进袋子"],
                "narration": repeated_scene_text("工人切蜂巢并把蜂巢装进袋子", 8),
                "pause": False,
                "video_speed": 1.0,
            },
            {
                "start": 320,
                "end": 660,
                "visual": "工人继续在树干和地面附近处理蜂巢收尾",
                "visual_facts": ["工人继续在树干和地面附近处理蜂巢收尾"],
                "narration": repeated_scene_text("工人继续在树干和地面附近处理蜂巢收尾", 8),
                "pause": False,
                "video_speed": 1.0,
            },
        ]

        with self.assertRaisesRegex(Exception, "makes narration too tight"):
            commentary._validate_commentary_script_for_target(
                {"narration": long_narration, "narration_blocks": blocks},
                678.4,
                "full",
                "zh",
            )

    def test_full_duration_allows_ai_compressed_medium_complete_process_after_speed(self):
        narration = repeated_scene_text("工人沿着树干处理蜂巢，画面里的动作和蜂群变化继续推进", 3)
        starts = [0, 90, 180, 270, 360, 450, 540, 625]
        blocks = [
            {
                "start": start,
                "end": start + 50,
                "visual": "工人在树干上处理蜂巢 slow repetitive process",
                "visual_facts": ["工人在树干上处理蜂巢，蜂群围着树干飞动"],
                "narration": narration,
                "pause": False,
                "video_speed": 1.45,
                "speed_reason": "AI chose moderate acceleration because the selected range is repeated slow beehive handling",
            }
            for start in starts
        ]
        visual_seconds = sum(commentary._block_visual_duration(block) for block in blocks)
        self.assertAlmostEqual(275.862, visual_seconds, places=2)

        data = {"narration": narration, "narration_blocks": blocks}
        commentary._validate_commentary_script_for_target(data, 678.4, "full", "zh")

        self.assertGreaterEqual(
            sum(commentary._block_visual_duration(block) for block in data["narration_blocks"]),
            commentary._target_visual_duration_seconds(678.4, "full") * commentary.FULL_MODE_MIN_PLAYABLE_TARGET_RATIO,
        )

    def test_full_duration_long_sources_keep_at_least_ten_minutes(self):
        self.assertEqual(
            480,
            int(commentary._target_visual_duration_seconds(1233.7, "full")),
        )
        self.assertEqual(
            480,
            int(commentary._target_visual_duration_seconds(1800, "full")),
        )
        self.assertEqual(
            576,
            int(commentary._target_visual_duration_seconds(3600, "full")),
        )
        self.assertLess(
            commentary._target_visual_duration_seconds(3600, "full"),
            600,
        )

    def test_full_duration_prompt_demands_breathable_scene_matched_narration(self):
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
        self.assertIn("do not talk over every second", prompt)
        self.assertIn("25% of selected visual time", prompt)
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
        self.assertIn("There is no filler word-count target", prompt)
        self.assertIn("There is no total minimum word count", prompt)
        self.assertNotIn("Narration must be at least", prompt)
        self.assertNotIn("total narration must be at least", prompt)

    def test_full_duration_prompt_compacts_medium_complete_process_source(self):
        prompt = commentary._build_commentary_prompt(
            transcript={"text": "source transcript", "segments": [], "language": "en"},
            video_title="Demo",
            duration=678.4,
            language="zh",
            style="documentary",
            target_duration="full",
            analysis_mode="openai",
            visual_analysis=beehive_visual_analysis(),
        )

        self.assertIn("select about 284 seconds", prompt)
        self.assertIn("preserve the complete process arc", prompt)
        self.assertIn("should total about 284 playable seconds after video_speed", prompt)
        self.assertIn("compress repeated hammering", prompt)
        self.assertIn("video_speed", prompt)
        self.assertIn("suggested_speed", prompt)
        self.assertNotIn("select only about 360 seconds", prompt)
        self.assertNotIn("preserve roughly 90-100% of the source workflow", prompt)

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
        block_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        blocks = [
            {
                "start": i * 160,
                "end": i * 160 + 100,
                "visual": "workers handle material during early process",
                "narration": block_text,
                "pause": False,
            }
            for i in range(12)
        ]

        with self.assertRaisesRegex(Exception, "stopped before the end"):
            commentary._validate_commentary_script_for_target(
                {"narration": block_text, "narration_blocks": blocks},
                duration=3935,
                target_duration="full",
                language="zh",
            )

    def test_full_duration_accepts_blocks_with_late_timeline_coverage(self):
        block_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        blocks = [
            {
                "start": i * 280,
                "end": i * 280 + 50,
                "visual": "workers handle material during full process",
                "narration": block_text,
                "pause": False,
            }
            for i in range(11)
        ]
        blocks.append({
            "start": 3600,
            "end": 3650,
            "visual": "ending process with final material result",
            "narration": block_text,
            "pause": False,
        })

        commentary._validate_commentary_script_for_target(
            {"narration": block_text, "narration_blocks": blocks, "_skip_repeat_validation": True},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_accepts_concise_narration_with_late_timeline_coverage(self):
        block_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        blocks = [
            {
                "start": i * 280,
                "end": i * 280 + 50,
                "visual": "workers handle material during full process",
                "narration": block_text,
                "pause": False,
            }
            for i in range(11)
        ]
        blocks.append({
            "start": 3600,
            "end": 3650,
            "visual": "ending process with final material result",
            "narration": block_text,
            "pause": False,
        })
        narration = "工人处理废旧材料，设备运转后最终会形成更清楚的回收结果。"
        data = {"narration": narration, "narration_blocks": blocks}
        data["_skip_repeat_validation"] = True
        data["narration_blocks"] = commentary._normalize_narration_blocks(data["narration_blocks"], 3935)
        data["narration"] = narration

        commentary._validate_commentary_script_for_target(
            data,
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_rejects_over_retained_source_for_medium_source(self):
        worker_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        blocks = [
            {"start": 0, "end": 45, "visual": "worker opening setup with material", "narration": worker_text, "pause": False},
            {"start": 60, "end": 105, "visual": "material sorting", "narration": worker_text, "pause": False},
            {"start": 130, "end": 175, "visual": "first machine process", "narration": worker_text, "pause": False},
            {"start": 205, "end": 250, "visual": "worker inspection step", "narration": worker_text, "pause": False},
            {"start": 285, "end": 330, "visual": "middle material assembly", "narration": worker_text, "pause": False},
            {"start": 365, "end": 410, "visual": "machine closeup", "narration": worker_text, "pause": False},
            {"start": 450, "end": 495, "visual": "worker quality check on material", "narration": worker_text, "pause": False},
            {"start": 540, "end": 585, "visual": "packaging begins with material", "narration": worker_text, "pause": False},
            {"start": 635, "end": 680, "visual": "late process with workers and material", "narration": worker_text, "pause": False},
            {"start": 735, "end": 780, "visual": "final machine run", "narration": worker_text, "pause": False},
            {"start": 835, "end": 880, "visual": "finished material result", "narration": worker_text, "pause": False},
            {"start": 935, "end": 980, "visual": "ending packaging of material", "narration": worker_text, "pause": False},
        ]
        narration = worker_text

        self.assertGreater(
            commentary._segments_total_duration(commentary._narration_blocks_to_edit_segments(blocks)),
            1037.2 * commentary.FULL_MODE_MAX_SOURCE_RETENTION_FRACTION,
        )
        with self.assertRaisesRegex(Exception, "selected too much source footage|do not match the selected full-mode edit target"):
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
        block_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        payload = {
            "title": "Block Based",
            "summary": "summary",
            "hook": "hook",
            "narration": "太短。",
            "narration_blocks": [
                {"start": i * 280, "end": i * 280 + 50, "narration": block_text}
                for i in range(11)
            ] + [{"start": 3600, "end": 3650, "narration": block_text}],
            "edit_segments": [
                {"start": i * 280, "end": i * 280 + 50, "reason": "process"}
                for i in range(11)
            ] + [{"start": 3600, "end": 3650, "reason": "ending process"}],
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
        self.assertLessEqual(
            len(commentary.re.sub(r"\s+", "", result["narration"])),
            commentary._maximum_narration_chars(3935, "full", "zh"),
        )
        self.assertIn(block_text.rstrip("。"), result["narration"])

    def test_full_duration_validation_counts_pause_blocks_as_non_spoken_time(self):
        long_worker_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 18)
        blocks = [
            {"start": 0, "end": 220, "visual": "opening process", "visual_facts": ["opening process shows workers handling material"], "narration": long_worker_text, "pause": False},
            {"start": 220, "end": 232, "visual": "machine sound reveal", "narration": "", "pause": True},
            {"start": 520, "end": 740, "visual": "main process", "visual_facts": ["main process shows the machine and workers continuing"], "narration": long_worker_text, "pause": False},
            {"start": 740, "end": 752, "visual": "original audio beat", "narration": "", "pause": True},
            {"start": 3400, "end": 3600, "visual": "ending process", "visual_facts": ["ending process shows the final material result"], "narration": long_worker_text, "pause": False},
        ]
        spoken_text = "".join(block["narration"] for block in blocks)

        commentary._validate_commentary_script_for_target(
            {"narration": spoken_text, "narration_blocks": blocks},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_openai_visual_timeline_rejects_post_harvest_climbing_regression(self):
        blocks = [
            {
                "start": 741.104,
                "end": 785.0,
                "visual": "smoking leafy torch held while tying green bag straps to limb",
                "visual_facts": ["green bag is secured to a tree limb while smoke drifts"],
                "narration": repeated_scene_text("火把继续冒烟，我把袋子和绳子绑在树枝上固定住蜂巢", 8),
                "pause": False,
            },
            {
                "start": 809.0,
                "end": 855.0,
                "visual": "climber feet and hands gripping trunk with beehive and swarm visible lower right",
                "visual_facts": ["climber uses trunk and pegs while bees swarm near the hive"],
                "narration": repeated_scene_text("袋子已经绑好，我继续拉紧绳子把蜂蜜固定在树枝上", 8),
                "pause": False,
            },
        ]
        visual_analysis = {
            "observations": [
                {"timestamp": 751.143, "process_stage": "securing equipment", "visual": "Both pink-gloved hands tie and secure straps of green bag to tree limb"},
                {"timestamp": 809.928, "process_stage": "climbing ascent", "visual": "POV looking up tall trunk, boots on bark, metal pegs visible, distant beehive with bees"},
                {"timestamp": 811.783, "process_stage": "climbing toward hive", "visual": "Climber's feet and hands gripping trunk, beehive and bee swarm visible lower right"},
            ]
        }

        with self.assertRaisesRegex(Exception, "regress to an earlier source action"):
            commentary._validate_commentary_script_for_target(
                {"narration": "\n".join(block["narration"] for block in blocks), "narration_blocks": blocks},
                duration=1026.0,
                target_duration="full",
                language="zh",
                visual_analysis=visual_analysis,
            )

    def test_openai_visual_timeline_rejects_unsupported_completion_claim(self):
        block = {
            "start": 609.2,
            "end": 635.0,
            "visual": "gloved hands prying bark, removing hive material while bees approach",
            "visual_facts": ["gloved hands prying at trunk edge", "bees flying near boots"],
            "narration": "蜂巢切开一大块，装好收工",
            "pause": False,
            "video_speed": 1.5,
            "speed_reason": "repeated hive material removal remains understandable at mild acceleration",
        }
        visual_analysis = {
            "observations": [
                {
                    "timestamp": 609.227,
                    "visual": "Bee flies directly past gloved hand on trunk; motion blur on insect",
                    "process_stage": "bee activity",
                },
                {
                    "timestamp": 623.04,
                    "visual": "Gloved hand pulls small dark object from trunk base while second hand holds tool",
                    "process_stage": "hive material removal",
                },
                {
                    "timestamp": 634.88,
                    "visual": "Multiple bees flying near boots and lower trunk; active swarm close to worker",
                    "process_stage": "bee defense",
                },
            ]
        }

        with self.assertRaisesRegex(Exception, "completed packing/ending action"):
            commentary._validate_scene_matched_narration_blocks(
                {"narration": block["narration"], "narration_blocks": [block]},
                visual_analysis=visual_analysis,
            )

    def test_openai_visual_timeline_allows_visible_completion_claim(self):
        block = {
            "start": 418.5,
            "end": 445.0,
            "visual": "knife cutting more comb then placing pieces into bag on leg",
            "visual_facts": ["yellow comb in bag", "bees swarming"],
            "narration": "蜂巢切下来塞进袋子，袋口扎紧这才算装好",
            "pause": False,
            "video_speed": 1.0,
        }
        visual_analysis = {
            "observations": [
                {
                    "timestamp": 430.383,
                    "visual": "Large yellow honeycomb pieces are placed into a plastic bag on worker's leg",
                    "process_stage": "bagging honeycomb",
                },
                {
                    "timestamp": 442.437,
                    "visual": "Bag is held close while bees swarm around the packed honeycomb",
                    "process_stage": "packing",
                },
            ]
        }

        commentary._validate_scene_matched_narration_blocks(
            {"narration": block["narration"], "narration_blocks": [block]},
            visual_analysis=visual_analysis,
        )

    def test_openai_visual_timeline_allows_noisy_concept_labels_when_scene_is_grounded(self):
        block = {
            "start": 92.0,
            "end": 118.0,
            "visual": "worker climbing on tree trunk while bees swarm near hive",
            "visual_facts": ["worker hands stay on the trunk while bees move around the hive"],
            "narration": "工人抱住树干继续往上爬，手套贴着树皮一点点靠近蜂巢。",
            "pause": False,
            "video_speed": 1.0,
        }
        visual_analysis = {
            "observations": [
                {
                    "timestamp": 96.2,
                    "visual": "Worker hands grip tree trunk while bees swarm around hive; bright metal-looking artifact is visible from lighting",
                    "process_stage": "climbing near hive with possible copper/metal highlight",
                },
                {
                    "timestamp": 112.4,
                    "visual": "Person in gloves stays on branch and trunk as honeycomb and bees remain close",
                    "process_stage": "approaching hive",
                },
            ]
        }

        commentary._validate_scene_matched_narration_blocks(
            {"narration": block["narration"], "narration_blocks": [block]},
            visual_analysis=visual_analysis,
        )

    def test_full_mode_output_alignment_rejects_final_longer_than_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "final.mp4")
            voice_path = os.path.join(tmpdir, "voice.m4a")
            open(final_path, "wb").close()
            open(voice_path, "wb").close()

            def fake_duration(path):
                if path == final_path:
                    return 712.0
                if path == voice_path:
                    return 712.0
                return None

            with patch.object(commentary, "_probe_media_format_duration", side_effect=fake_duration):
                with self.assertRaisesRegex(Exception, "longer than the source video"):
                    commentary._assert_full_mode_output_alignment(
                        final_path,
                        voice_path,
                        None,
                        [712.0],
                        source_duration=678.4,
                    )

    def test_full_mode_output_alignment_allows_subtitles_to_end_before_pause_tail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "final.mp4")
            voice_path = os.path.join(tmpdir, "voice.m4a")
            subtitle_path = os.path.join(tmpdir, "subs.ass")
            open(final_path, "wb").close()
            open(voice_path, "wb").close()
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write("\n".join([
                    "Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,第一段",
                    "Dialogue: 0,0:00:10.00,0:00:20.00,Default,,0,0,0,,第二段",
                ]))

            def fake_duration(path):
                if path == final_path:
                    return 24.0
                if path == voice_path:
                    return 24.0
                return None

            with patch.object(commentary, "_probe_media_format_duration", side_effect=fake_duration):
                commentary._assert_full_mode_output_alignment(
                    final_path,
                    voice_path,
                    subtitle_path,
                    [10.0, 10.0, 4.0],
                    source_duration=60.0,
                    narration_blocks=[
                        {"narration": "第一段", "pause": False},
                        {"narration": "第二段", "pause": False},
                        {"narration": "", "pause": True},
                    ],
                )

    def test_rendered_cached_full_mode_script_allows_split_pause_tails(self):
        data = {
            "narration": "第一段解说。第二段解说。",
            "narration_blocks": [
                {"start": 0.0, "end": 20.0, "visual": "worker climbs", "narration": "第一段解说。", "pause": False, "rendered_duration": 20.0},
                {"start": 20.0, "end": 22.0, "visual": "worker climbs", "narration": "", "pause": True, "rendered_duration": 2.0},
                {"start": 22.0, "end": 40.0, "visual": "worker lowers bag", "narration": "第二段解说。", "pause": False, "rendered_duration": 18.0},
                {"start": 40.0, "end": 42.0, "visual": "worker lowers bag", "narration": "", "pause": True, "rendered_duration": 2.0},
                {"start": 42.0, "end": 45.0, "visual": "ambient ending", "narration": "", "pause": True, "rendered_duration": 3.0},
            ],
        }

        commentary._validate_rendered_cached_full_mode_script(data, 120.0, "full", "zh")

        unrendered_data = {
            "narration": data["narration"],
            "narration_blocks": [
                {key: value for key, value in block.items() if key != "rendered_duration"}
                for block in data["narration_blocks"]
            ],
        }
        with self.assertRaises(Exception):
            commentary._validate_commentary_script_for_target(unrendered_data, 120.0, "full", "zh")

    def test_normalize_narration_blocks_preserves_pause_rate_pitch_and_video_speed(self):
        blocks = commentary._normalize_narration_blocks(
            [
                {"start": 0, "end": 8, "visual": "machine sound", "narration": "", "pause": True},
                {"start": 8, "end": 20, "visual": "worker action", "narration": "工人正在处理铜料。", "rate": "-15%", "pitch": "+4Hz", "video_speed": 1.75},
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

    def test_auto_video_speed_preserves_ai_speed_without_backend_guessing(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 22, "visual": "worker repeats slow transport and loading steps", "narration": "这段是重复搬运和上料。", "video_speed": 1.35, "speed_reason": "AI chose moderate speed for repeated loading"},
                {"start": 22, "end": 42, "visual": "final result showcase", "narration": "最终成品亮相。", "video_speed": 1.0},
                {"start": 42, "end": 48, "visual": "short moving shot", "narration": "短镜头。", "video_speed": 1.0},
            ],
            enabled=True,
        )

        self.assertEqual(1.35, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])
        self.assertEqual(1.0, blocks[2]["video_speed"])

    def test_auto_video_speed_does_not_supplement_ai_selected_speed(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 22, "visual": "slow transport", "narration": "运输。", "video_speed": 1.75, "speed_reason": "AI chose speed for slow transport"},
                {"start": 22, "end": 48, "visual": "repetitive loading", "narration": "重复上料。", "video_speed": 1.0},
            ],
            enabled=True,
        )

        self.assertEqual(1.75, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])

    def test_auto_video_speed_keeps_ai_speed_and_protected_blocks(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {"start": 0, "end": 24, "visual": "slow transport", "narration": "运输。", "video_speed": 1.25, "speed_reason": "AI chose light acceleration"},
                {"start": 24, "end": 48, "visual": "repetitive loading and waiting", "narration": "重复上料等待。", "video_speed": 1.0},
                {"start": 48, "end": 72, "visual": "final result showcase", "narration": "最终成果展示。", "video_speed": 1.0},
            ],
            enabled=True,
        )

        self.assertEqual(1.25, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])
        self.assertEqual(1.0, blocks[2]["video_speed"])

    def test_auto_video_speed_does_not_use_backend_visual_keyword_guessing(self):
        blocks = commentary._apply_auto_video_speed_to_blocks(
            [
                {
                    "start": 48.1,
                    "end": 74.0,
                    "visual": "粉色手套继续处理树干，蜜蜂围着手臂",
                    "narration": "钉子敲进树干形成梯子，往下看腿都软。",
                    "video_speed": 1.0,
                },
                {
                    "start": 418.5,
                    "end": 452.0,
                    "visual": "蜂巢块块装袋，蜜蜂还跟着进去",
                    "narration": "蜂巢放入袋中，袋口被拧紧。",
                    "video_speed": 1.0,
                },
            ],
            enabled=True,
            visual_analysis={
                "candidate_segments": [
                    {"start": 48.1, "end": 54.7, "reason": "Hands inserting pegs with downward height perspective and bee movement"},
                    {"start": 66.4, "end": 72.9, "reason": "Repeated hammering action showing peg ladder construction progress"},
                    {"start": 418.5, "end": 452.0, "reason": "Honeycomb pieces placed into bag and secured"},
                ],
            },
        )

        self.assertEqual(1.0, blocks[0]["video_speed"])
        self.assertEqual(1.0, blocks[1]["video_speed"])

    def test_full_duration_rejects_ignored_ai_speed_evidence(self):
        narration = repeated_scene_text("工人沿着树干重复调整位置，蜂群和蜂巢状态继续变化", 5)
        block = {
            "start": 48.0,
            "end": 80.0,
            "visual": "粉色手套反复敲钉，脚窝沿着树干一点点往上铺",
            "visual_facts": ["手套反复敲钉", "树干上形成脚窝"],
            "narration": narration,
            "pause": False,
            "video_speed": 1.0,
        }
        visual_analysis = {
            "candidate_segments": [
                {
                    "start": 48.0,
                    "end": 80.0,
                    "reason": "repeated peg hammering stays understandable when accelerated",
                    "suggested_speed": 1.5,
                    "speed_reason": "repeated hammering is slow but useful setup",
                },
            ],
        }

        with self.assertRaisesRegex(Exception, "ignores visual speed evidence"):
            commentary._validate_ai_video_speed_decisions([block], "zh", visual_analysis=visual_analysis, duration=120.0)

        block["video_speed"] = 1.35
        block["speed_reason"] = "AI chose mild acceleration because the exact range repeats peg hammering on the trunk"
        commentary._validate_ai_video_speed_decisions([block], "zh", visual_analysis=visual_analysis, duration=120.0)

    def test_full_mode_speed_budget_reverts_acceleration_that_makes_visuals_too_short(self):
        narration = "这一段解说紧贴实际动作，把工人处理材料、设备运转和流程推进讲清楚。"
        blocks = [
            {
                "start": i * 70,
                "end": i * 70 + 70,
                "visual": "slow repetitive process transport",
                "visual_facts": ["workers keep the process moving"],
                "narration": narration,
                "video_speed": 2.5,
                "pause": False,
            }
            for i in range(9)
        ]
        sped_seconds = sum(commentary._block_visual_duration(block) for block in blocks)
        self.assertLess(sped_seconds, 360.0 * 0.9)

        protected = commentary._protect_full_mode_visual_budget_after_speed(blocks, 700.0, "full")

        self.assertGreaterEqual(
            sum(commentary._block_visual_duration(block) for block in protected),
            commentary._target_visual_duration_seconds(700.0, "full") * 0.9,
        )
        self.assertGreater(
            len([block for block in protected if commentary._safe_video_speed(block.get("video_speed")) == 1.0]),
            0,
        )

    def test_full_mode_speed_budget_preserves_short_complete_process_runtime(self):
        narration = "这一段解说紧贴实际动作，把工人处理蜂巢、树干位置和蜂群变化讲清楚。"
        blocks = [
            {
                "start": i * 67.84,
                "end": (i + 1) * 67.84,
                "visual": "slow repetitive beehive work on tree trunk",
                "visual_facts": ["worker keeps handling the beehive on the tree trunk"],
                "narration": narration,
                "video_speed": 1.45,
                "pause": False,
            }
            for i in range(10)
        ]
        self.assertLess(
            sum(commentary._block_visual_duration(block) for block in blocks),
            678.4 * 0.75,
        )

        protected = commentary._protect_full_mode_visual_budget_after_speed(blocks, 678.4, "full")

        self.assertGreaterEqual(
            sum(commentary._block_visual_duration(block) for block in protected),
            commentary._target_visual_duration_seconds(678.4, "full") * 0.9,
        )

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

    def test_background_music_bed_loops_and_fades_to_target_duration(self):
        commands = []

        def fake_run_command(cmd, cwd=None):
            commands.append(cmd)

        with patch.object(commentary, "_run_command", side_effect=fake_run_command):
            result = commentary._create_background_music_bed("music.mp3", "bgm.m4a", 12.0, volume=0.16)

        self.assertEqual("bgm.m4a", result)
        command = commands[0]
        self.assertIn("-stream_loop", command)
        self.assertEqual("-1", command[command.index("-stream_loop") + 1])
        self.assertEqual("12.000", command[command.index("-t") + 1])
        audio_filter = command[command.index("-af") + 1]
        self.assertIn("volume=0.16", audio_filter)
        self.assertIn("afade=t=out:st=10.500:d=1.500", audio_filter)

    def test_mix_voiceover_with_video_can_include_background_music(self):
        commands = []

        def fake_run_command(cmd, cwd=None):
            commands.append(cmd)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(commentary, "_run_command", side_effect=fake_run_command):
            voice_path = os.path.join(tmpdir, "voice.m4a")
            ambient_path = os.path.join(tmpdir, "ambient.m4a")
            background_path = os.path.join(tmpdir, "background.m4a")
            for path in (voice_path, ambient_path, background_path):
                open(path, "wb").close()
            commentary._mix_voiceover_with_video(
                video_path="video.mp4",
                voiceover_path=voice_path,
                output_path="out.mp4",
                ambient_audio_path=ambient_path,
                background_music_path=background_path,
            )

        command = commands[0]
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("amix=inputs=3", filter_complex)
        self.assertIn(background_path, command)

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
                {"start": 0, "end": 12, "visual": "worker action", "narration": "工人正在处理铜料。", "rate": "+12%", "pitch": "+3Hz"},
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

            blocks = scene_matched_blocks(count=2, seconds=12.0, text="工人正在处理铜料。")
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

    def test_block_synced_render_preserves_visuals_when_tts_is_short(self):
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
                "end": 37.5,
                "visual": "long repetitive conveyor footage",
                "narration": "这句解说很短，不能让后面十几秒都没声音。",
                "video_speed": 1.0,
            }]
            with patch.object(commentary, "_get_video_duration", return_value=37.5), \
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

        expected_narrated_duration = commentary._max_visual_seconds_for_actual_voiceover(3.0)
        expected_pause_duration = 37.5 - expected_narrated_duration
        self.assertEqual(1, len(fit_calls))
        self.assertAlmostEqual(expected_narrated_duration, fit_calls[0][2], places=2)
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertEqual(2, len(video_cmds))
        self.assertEqual(f"{expected_narrated_duration:.3f}", video_cmds[0][video_cmds[0].index("-t") + 1])
        self.assertEqual(f"{expected_narrated_duration:.3f}", video_cmds[1][video_cmds[1].index("-ss") + 1])
        self.assertEqual(f"{expected_pause_duration:.3f}", video_cmds[1][video_cmds[1].index("-t") + 1])
        self.assertNotIn("setpts=PTS/", video_cmds[0][video_cmds[0].index("-vf") + 1])
        self.assertAlmostEqual(1.0, original_audio_calls[0][4], places=3)
        self.assertAlmostEqual(expected_narrated_duration, original_audio_calls[0][5], places=2)
        self.assertAlmostEqual(0.6, original_audio_calls[1][3], places=3)
        self.assertTrue(any("Splitting short-TTS commentary block" in message for message in progress_messages))
        self.assertEqual(ambient_audio_path, ambient)
        self.assertEqual(2, len(block_durations))
        self.assertAlmostEqual(expected_narrated_duration, block_durations[0], places=2)
        self.assertAlmostEqual(expected_pause_duration, block_durations[1], places=2)
        self.assertFalse(blocks[0]["pause"])
        self.assertTrue(blocks[1]["pause"])

    def test_block_synced_render_can_trim_short_tts_tails_for_compact_edits(self):
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
                "start": 120,
                "end": 140,
                "visual": "worker picks larvae into a bucket",
                "narration": "她把虫子一只只捡进桶里。",
                "video_speed": 1.0,
            }]
            with patch.object(commentary, "_get_video_duration", return_value=3600.0), \
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
                    trim_short_tts_tails=True,
                    progress=progress_messages.append,
                )

        expected_duration = commentary._max_visual_seconds_for_actual_voiceover(3.0)
        self.assertEqual(1, len(fit_calls))
        self.assertAlmostEqual(expected_duration, fit_calls[0][2], places=2)
        video_cmds = [cmd for cmd in commands if cmd[-1].endswith(".mp4") and "-ss" in cmd]
        self.assertEqual(1, len(video_cmds))
        self.assertEqual("120.000", video_cmds[0][video_cmds[0].index("-ss") + 1])
        self.assertEqual(f"{expected_duration:.3f}", video_cmds[0][video_cmds[0].index("-t") + 1])
        self.assertEqual(1, len(original_audio_calls))
        self.assertAlmostEqual(expected_duration, original_audio_calls[0][1], places=2)
        self.assertAlmostEqual(expected_duration, original_audio_calls[0][5], places=2)
        self.assertTrue(any("Trimming short-TTS commentary block" in message for message in progress_messages))
        self.assertEqual(ambient_audio_path, ambient)
        self.assertEqual(1, len(block_durations))
        self.assertAlmostEqual(expected_duration, block_durations[0], places=2)
        self.assertFalse(blocks[0]["pause"])
        self.assertAlmostEqual(120 + expected_duration, blocks[0]["end"], places=2)

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

    def test_full_duration_accepts_short_narration_with_timestamp_blocks(self):
        short_text = repeated_scene_text(WORKER_MATERIAL_NARRATION, 2)
        blocks = [
            {"start": i * 280, "end": i * 280 + 50, "visual": "full process", "narration": short_text}
            for i in range(11)
        ]
        blocks.append({"start": 3600, "end": 3650, "visual": "ending process", "narration": short_text})

        commentary._validate_commentary_script_for_target(
            {
                "narration": short_text,
                "narration_blocks": blocks,
                "edit_segments": [{"start": 0, "end": 700, "reason": "process"}],
                "_skip_repeat_validation": True,
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

    def test_full_duration_repairs_excessive_pause_ratio(self):
        opening_text = joined_scene_text("工人分拣材料，设备继续运转并把材料送到下一步", 33)
        ending_text = joined_scene_text("处理后的材料出现最终结果，工人继续检查成品状态", 37)
        late_text = joined_scene_text("后段时间线呈现最终结果和收尾状态", 47)
        narration = opening_text + ending_text + late_text
        blocks = [
            {"start": 0, "end": 190, "visual": "opening process with visible sorting and machine movement", "visual_facts": ["workers sort material while the machine keeps moving"], "narration": opening_text, "pause": False},
            {"start": 190, "end": 250, "visual": "too much source audio", "narration": "", "pause": True},
            {"start": 700, "end": 920, "visual": "ending process with visible finished material and result", "visual_facts": ["finished material is shown after the process"], "narration": ending_text, "pause": False},
            {"start": 3400, "end": 3600, "visual": "late source result reveal with visible final output", "visual_facts": ["the later timeline shows the final output"], "narration": late_text, "pause": False},
        ]
        data = {"narration": narration, "narration_blocks": blocks}

        commentary._validate_commentary_script_for_target(
            data,
            duration=3935,
            target_duration="full",
            language="zh",
        )
        repaired_pauses = [block for block in data["narration_blocks"] if block.get("pause")]
        self.assertEqual(1, len(repaired_pauses))
        self.assertLessEqual(commentary._block_visual_duration(repaired_pauses[0]), commentary.FULL_MODE_MAX_PAUSE_SECONDS)

    def test_full_duration_repairs_overlong_pause_block(self):
        opening_text = joined_scene_text("工人分拣材料，设备继续运转并把材料送到下一步", 38)
        ending_text = joined_scene_text("处理后的材料出现最终结果，工人继续检查成品状态", 37)
        late_text = joined_scene_text("后段时间线呈现最终结果和收尾状态", 47)
        narration = opening_text + ending_text + late_text
        blocks = [
            {"start": 0, "end": 220, "visual": "opening process with visible sorting and machine movement", "visual_facts": ["workers sort material while the machine keeps moving"], "narration": opening_text, "pause": False},
            {"start": 220, "end": 257, "visual": "long source audio beat", "narration": "", "pause": True},
            {"start": 605, "end": 825, "visual": "ending process with visible finished material and result", "visual_facts": ["finished material is shown after the process"], "narration": ending_text, "pause": False},
            {"start": 3400, "end": 3600, "visual": "late source result reveal with visible final output", "visual_facts": ["the later timeline shows the final output"], "narration": late_text, "pause": False},
        ]
        data = {"narration": narration, "narration_blocks": blocks}

        commentary._validate_commentary_script_for_target(
            data,
            duration=3935,
            target_duration="full",
            language="zh",
        )
        repaired_pause = [block for block in data["narration_blocks"] if block.get("pause")][0]
        self.assertEqual(220.0, repaired_pause["start"])
        self.assertLessEqual(
            commentary._block_visual_duration(repaired_pause),
            commentary.FULL_MODE_MAX_PAUSE_SECONDS,
        )

    def test_normalize_narration_blocks_merges_adjacent_pause_blocks(self):
        blocks = commentary._normalize_narration_blocks(
            [
                {"start": 0, "end": 8, "visual": "first source audio beat", "narration": "", "pause": True},
                {"start": 8, "end": 16, "visual": "second source audio beat", "narration": "", "pause": True},
                {"start": 16, "end": 30, "visual": "worker resumes", "narration": "工人继续往前推进。", "pause": False},
            ],
            60,
        )

        self.assertEqual(2, len(blocks))
        self.assertTrue(blocks[0]["pause"])
        self.assertEqual(0.0, blocks[0]["start"])
        self.assertEqual(16.0, blocks[0]["end"])
        self.assertIn("first source audio beat", blocks[0]["visual"])
        self.assertIn("second source audio beat", blocks[0]["visual"])

    def test_full_duration_allows_adjacent_short_pause_blocks_after_normalization(self):
        opening_text = repeated_scene_text("工人处理材料并把材料送到下一步", 45)
        ending_text = repeated_scene_text("后段流程呈现最终材料结果", 45)
        narration = opening_text + ending_text
        blocks = [
            {"start": 0, "end": 280, "visual": "opening process", "visual_facts": ["opening process shows workers handling material"], "narration": opening_text, "pause": False},
            {"start": 280, "end": 288, "visual": "first source audio beat", "narration": "", "pause": True},
            {"start": 288, "end": 296, "visual": "second source audio beat", "narration": "", "pause": True},
            {"start": 3400, "end": 3700, "visual": "ending process", "visual_facts": ["ending process shows the final material result"], "narration": ending_text, "pause": False},
        ]

        commentary._validate_commentary_script_for_target(
            {"narration": narration, "narration_blocks": blocks},
            duration=3935,
            target_duration="full",
            language="zh",
        )

    def test_full_duration_repairs_adjacent_pause_blocks_when_merged_pause_is_too_long(self):
        opening_text = joined_scene_text("工人分拣材料，设备继续运转并把材料送到下一步", 49)
        ending_text = joined_scene_text("处理后的材料出现最终结果，工人继续检查成品状态", 51)
        narration = opening_text + ending_text
        blocks = [
            {"start": 0, "end": 280, "visual": "opening process with visible sorting and machine movement", "visual_facts": ["workers sort material while the machine keeps moving"], "narration": opening_text, "pause": False},
            {"start": 280, "end": 288, "visual": "first source audio beat", "narration": "", "pause": True},
            {"start": 288, "end": 296, "visual": "second source audio beat", "narration": "", "pause": True},
            {"start": 296, "end": 304, "visual": "third source audio beat", "narration": "", "pause": True},
            {"start": 3400, "end": 3700, "visual": "ending process with visible finished material and result", "visual_facts": ["finished material is shown after the process"], "narration": ending_text, "pause": False},
        ]
        data = {"narration": narration, "narration_blocks": blocks}

        commentary._validate_commentary_script_for_target(
            data,
            duration=3935,
            target_duration="full",
            language="zh",
        )
        repaired_pause = [block for block in data["narration_blocks"] if block.get("pause")][0]
        self.assertEqual(280.0, repaired_pause["start"])
        self.assertLessEqual(
            commentary._block_visual_duration(repaired_pause),
            commentary.FULL_MODE_MAX_PAUSE_SECONDS,
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

    def test_full_duration_missing_narration_blocks_is_retried_with_video_grounded_regeneration_prompt(self):
        transcript = {
            "text": "factory process",
            "language": "en",
            "segments": [{"start": 0, "end": 10, "text": "factory process"}],
        }
        short_payload = {
            "title": "Short",
            "summary": "summary",
            "hook": "hook",
            "narration": "这些废旧电机最终会被回收成铜材。",
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
