import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import commentary


CUSTOM_OPERATION_STYLE = """
# 核心风格定位
工业机械操作解说风格。必须解释操作逻辑、这一步目的、动作层和结果层。
"""


def _script_with_narration(narrations):
    return {
        "title": "轮胎回收流程",
        "summary": "",
        "narration": "\n".join(narrations),
        "narration_blocks": [
            {
                "start": index * 6,
                "end": index * 6 + 6,
                "visual": "工人操作机械设备处理轮胎橡胶和纤维",
                "visual_facts": ["机械设备处理材料", "材料状态发生变化"],
                "narration": narration,
                "pause": False,
                "rate": "+0%",
                "pitch": "+0Hz",
                "video_speed": 1.0,
            }
            for index, narration in enumerate(narrations)
        ],
    }


class CustomStyleOperationLogicTest(unittest.TestCase):
    def test_custom_operation_style_rejects_action_only_process_blocks(self):
        script = _script_with_narration([
            "机械爪夹住胎面转动调整。",
            "液压刀刃压进胎侧橡胶里切开。",
            "刷辊把纤维从混合料里挑出来。",
            "工人按下控制屏确认批次完成。",
        ])

        with self.assertRaisesRegex(Exception, "Custom style operation logic validation failed"):
            commentary._validate_custom_style_operation_logic(
                script,
                "zh",
                CUSTOM_OPERATION_STYLE,
                duration=30,
            )

    def test_custom_operation_style_accepts_visible_action_with_logic(self):
        script = _script_with_narration([
            "机械爪先夹住胎面转动调整。这一步是为了把切口位置固定住。",
            "液压刀刃压进胎侧橡胶里切开。这样能先把厚边分离出来。",
            "刷辊把纤维从混合料里挑出来。主要是让橡胶颗粒更干净。",
            "工人按下控制屏确认批次完成。这个动作保证后续输送按批次走。",
        ])

        commentary._validate_custom_style_operation_logic(
            script,
            "zh",
            CUSTOM_OPERATION_STYLE,
            duration=30,
        )

    def test_source_commentary_transcript_is_detected_and_formatted(self):
        transcript = {
            "text": " ".join(["original narration"] * 40),
            "segments": [
                {"start": 1.0, "end": 7.0, "text": "The original narrator explains the first hunting step."},
                {"start": 10.0, "end": 16.0, "text": "The narrator describes tools and why they matter."},
                {"start": 20.0, "end": 28.0, "text": "The speaker explains how the group tracks animals."},
                {"start": 32.0, "end": 38.0, "text": "The old narration explains the result of the hunt."},
                {"start": 40.0, "end": 48.0, "text": "The narrator summarizes the survival lesson."},
            ],
            "language": "en",
        }

        self.assertTrue(commentary._transcript_has_source_commentary(transcript))
        timeline = commentary._format_source_commentary_timeline(transcript)
        self.assertIn("1.00-7.00", timeline)
        self.assertIn("original narrator", timeline)

        local_timeline = commentary._format_source_commentary_timeline(
            transcript,
            start=18.0,
            end=30.0,
            margin=1.0,
        )
        self.assertIn("20.00-28.00", local_timeline)
        self.assertNotIn("1.00-7.00", local_timeline)


if __name__ == "__main__":
    unittest.main()
