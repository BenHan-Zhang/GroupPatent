import unittest

from agent_v2.utils.patent_quality import (
    normalize_claims,
    parse_claims,
    quality_issues,
    repair_patent,
)


class PatentQualityTests(unittest.TestCase):
    def test_unnumbered_claims_are_numbered_continuously(self):
        source = (
            "一种数据处理方法，其特征在于，包括：\n"
            "获取待处理数据并进行规范化处理。\n\n"
            "根据权利要求1所述的方法，其特征在于，所述规范化处理包括去除重复数据。"
        )

        normalized = normalize_claims(source)
        self.assertEqual([part.original_number for part in parse_claims(normalized)], [1, 2])
        self.assertTrue(normalized.startswith("1. "))
        self.assertIn("\n\n2. 根据权利要求1所述的方法", normalized)

    def test_restarted_numbers_and_duplicate_claims_are_removed(self):
        first = "一种数据处理方法，其特征在于，包括：获取数据并输出处理结果。"
        second = "根据权利要求1所述的方法，其特征在于，所述处理结果写入数据库。"
        source = f"1. {first}\n\n2. {second}\n\n1. {first}"

        normalized = normalize_claims(source)
        self.assertEqual(len(parse_claims(normalized)), 2)
        self.assertEqual(
            [part.original_number for part in parse_claims(normalized)],
            [1, 2],
        )
        self.assertEqual(normalized.count(first), 1)

    def test_repair_removes_local_and_cross_field_repetition(self):
        repeated = "对目标数据进行校验并输出校验结果。"
        patent = {
            "标题": "一种数据处理方法",
            "摘要": f"{repeated}本发明能够提高处理效率。",
            "权利要求书": f"1. 一种数据处理方法，其特征在于，包括：{repeated}",
            "技术领域": "本申请涉及数据处理技术领域。",
            "背景技术": "现有方法存在效率低的问题。现有方法存在效率低的问题。",
            "发明内容": f"{repeated}本发明还提供一种装置。",
            "具体实施方式": f"{repeated}具体执行数据处理。",
        }

        repaired = repair_patent(patent, draft="采用数字串进行查询。")
        self.assertNotIn("本申请", "".join(repaired.values()))
        self.assertEqual(repaired["背景技术"].count("现有方法存在效率低的问题。"), 1)
        self.assertLessEqual(repaired["发明内容"].count(repeated), 1)
        self.assertLessEqual(repaired["具体实施方式"].count(repeated), 1)
        self.assertFalse(
            any("权利要求存在未编号" in issue for issue in quality_issues(repaired))
        )


if __name__ == "__main__":
    unittest.main()

