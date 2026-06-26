import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import commentary


CUSTOM_OPERATION_STYLE = """
# 核心风格定位
工业机械操作解说风格。必须解释操作逻辑、这一步目的、动作层和结果层。
"""

CUSTOM_VALUE_STYLE = """
# 核心风格定位
民间猎奇与反差，适合废旧物品回收、手工改造、民间手工艺制作。

# 内容结构模板
- 动作层：用生动、夸张的动词描述手工过程。
- 价值层：穿插说明材料便宜和成品性价比。
- 互动层：关键步骤后抛出问题。
- 禁止遗漏变废为宝、性价比、老板赚钱等商业/价值逻辑。
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

    def test_custom_value_style_does_not_trigger_operation_logic_validation(self):
        script = _script_with_narration([
            "小伙用金属刀霍霍刮木箭杆，把废木刮直做箭身。",
            "接着用纤维丝滑绑在箭尖上，固定箭头不松动。",
            "三人带箭出门踩着三万六千步找猎物。",
            "低角度追击瞄准，箭在手中快速调整。",
        ])

        self.assertFalse(commentary._custom_style_requires_operation_logic(CUSTOM_VALUE_STYLE, "zh"))
        commentary._validate_custom_style_operation_logic(
            script,
            "zh",
            CUSTOM_VALUE_STYLE,
            duration=30,
        )

    def test_chinese_operation_logic_detects_same_range_result_wording(self):
        self.assertTrue(commentary._narration_has_operation_logic(
            "小伙用金属刀霍霍刮木箭杆，把废木刮直做箭身，这样箭杆笔直射击更准。",
            "zh",
        ))
        self.assertTrue(commentary._narration_has_operation_logic(
            "纤维丝绑在箭尖上固定箭头，这样让整支箭飞出去更稳。",
            "zh",
        ))

    def test_custom_style_instruction_only_adds_operation_rule_when_explicit(self):
        value_instruction = commentary._custom_style_instruction(CUSTOM_VALUE_STYLE, "zh")
        operation_instruction = commentary._custom_style_instruction(CUSTOM_OPERATION_STYLE, "zh")

        self.assertNotIn("explicitly asks for process/operation/step logic", value_instruction)
        self.assertIn("explicitly asks for process/operation/step logic", operation_instruction)

    def test_custom_style_instruction_keeps_long_learned_prompt(self):
        marker = "复刻公式必须保留到长提示词后半段"
        custom_prompt = "短句画面先行。" * 180 + marker + "不要编造画面外剧情。" * 80

        instruction = commentary._custom_style_instruction(custom_prompt, "zh")

        self.assertIn(marker, instruction)

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
