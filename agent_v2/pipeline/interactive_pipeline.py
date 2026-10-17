"""三层交互式专利生成流水线

架构:
  Layer 1 (问题层) → background_writer → [反思·层内]
  Layer 2 (方案/法律层) → invention_writer ↔ claim_writer (交替逐点) → [反思·层间 L2→L1]
  Layer 3 (表达层) → description_writer → abstract_writer → title_writer → technical_field_writer → [反思·层间 L3→L2]
"""

import json
import re
from typing import Tuple, List

from agent_v2.memory.global_memory import GlobalMemory
from agent_v2.prompts.prompt_manager import PromptManager
from agent_v2.agents.short_component_writer import ShortComponentWriterAgent
from agent_v2.agents.reflection_agent import ReflectionAgent
from agent_v2.models.qwen_client import chat
from agent_v2.utils.patent_quality import (
    normalize_claims,
    quality_issues,
    repair_patent,
)
from agent_v2.utils.text_parser import extract_xml_tag
from loguru import logger


MAX_REFLECTION_RETRIES = 3
MAX_PROTECTION_POINTS = 9
DEFAULT_SUMMARY_TARGET = 2200
DEFAULT_CLAIMS_TARGET = 2200
DEFAULT_DESCRIPTION_TARGET = 5200
TARGET_MIN_RATIO = 0.95
TARGET_MAX_RATIO = 1.05
PRE_FINAL_LENGTH_RETRIES = 3
FINAL_LENGTH_RETRIES = 3
BACKGROUND_LENGTH_RETRIES = 4


class InteractivePipeline:
    """三层交互式专利生成流水线"""

    def __init__(self, model: str = "qwen3.6-35b"):
        self.global_memory = GlobalMemory()
        self.prompt_manager = PromptManager()
        self.model = model

        self.background_writer = ShortComponentWriterAgent(
            "background_writer", self.prompt_manager, self.global_memory, "writer_background", model
        )
        self.summary_writer = ShortComponentWriterAgent(
            "summary_writer", self.prompt_manager, self.global_memory, "writer_summary", model
        )
        self.claims_writer = ShortComponentWriterAgent(
            "claims_writer", self.prompt_manager, self.global_memory, "writer_claims", model
        )
        self.description_writer = ShortComponentWriterAgent(
            "description_writer", self.prompt_manager, self.global_memory, "writer_description", model
        )
        self.abstract_writer = ShortComponentWriterAgent(
            "abstract_writer", self.prompt_manager, self.global_memory, "writer_abstract", model
        )
        self.title_writer = ShortComponentWriterAgent(
            "title_writer", self.prompt_manager, self.global_memory, "writer_title", model
        )
        self.technical_field_writer = ShortComponentWriterAgent(
            "technical_field_writer", self.prompt_manager, self.global_memory, "writer_technical_field", model
        )

        self.reflection = ReflectionAgent(self.prompt_manager, self.global_memory, model)

        logger.info("InteractivePipeline 初始化完成")

    def run(self, input_draft: str, target_lengths: dict | None = None) -> dict:
        logger.info("=" * 50)
        logger.info("开始执行 三层交互式专利生成流程")
        logger.info("=" * 50)

        self.global_memory.reset()
        self.global_memory.write("input_draft", input_draft)
        self.global_memory.write("target_lengths", target_lengths or {})

        self._run_layer1()
        self._run_layer2()
        self._run_layer3()
        patent = self._assemble_final()
        patent = self._apply_quality_gate(patent, "assemble")
        patent = self._normalize_final_lengths(patent)
        # Length rewriting is another model boundary and can reintroduce
        # numbering or duplicate-content defects, so inspect the final text
        # again after all length operations have completed.
        patent = self._apply_quality_gate(patent, "post_length")

        logger.info("=" * 50)
        logger.info("三层交互式专利生成流程执行完成")
        logger.info("=" * 50)

        return patent

    def _reflect_loop(self, reflect_fn, max_retries: int = MAX_REFLECTION_RETRIES) -> bool:
        """通用反思循环，最多重试 max_retries 次"""
        for attempt in range(max_retries):
            is_pass, feedback = reflect_fn()
            if is_pass:
                logger.info(f"反思通过 (第{attempt+1}次)")
                return True
            logger.warning(f"反思不通过 (第{attempt+1}/{max_retries}次): {feedback[:100]}...")
        logger.error(f"反思 {max_retries} 次均未通过，跳过")
        return False

    # ─── Layer 1 ────────────────────────────────────────────────

    def _run_layer1(self):
        logger.info("── 第一层：背景技术生成 ──")
        self.background_writer.execute()
        background = self.global_memory.read("short_components.background") or ""
        background = self._refine_background_to_target(background, self.global_memory.read("input_draft") or "")
        self.global_memory.write("short_components.background", background)
        self._reflect_loop(lambda: self.reflection.reflect_intra_layer1())

    def _refine_background_to_target(self, background: str, draft_D: str) -> str:
        target = self._target_length("背景技术", 320)
        if self._within_target(len(background), target):
            return background

        current = background
        best = background
        best_distance = self._target_distance(len(background), target)

        for attempt in range(BACKGROUND_LENGTH_RETRIES):
            lower = int(target * TARGET_MIN_RATIO)
            upper = int(target * TARGET_MAX_RATIO)
            direction = "压缩" if len(current) > upper else "扩写"
            rendered = f"""# 专利草稿
{draft_D}

# 当前背景技术
{current}

# 长度要求
目标长度：{target} 字左右
合格区间：{lower}-{upper} 字
当前长度：{len(current)} 字
本轮任务：{direction}

请重写“背景技术”字段，使长度落入合格区间并尽量接近目标长度。
要求：
1. 只输出背景技术正文，不要输出标题、标签、Markdown、解释或字数统计；
2. 只能介绍与本申请相关的现有技术、应用场景、业务流程、数据处理方式、常见实现和存在的问题；
3. 不得披露本发明的具体技术方案、步骤编号、模块结构、权利要求式限定或实施例细节；
4. 如果需要扩写，请沿现有技术链条补充技术环境、处理流程、数据来源、人工处理负担、准确性、实时性、兼容性、可扩展性和维护成本等问题；
5. 如果需要压缩，请合并重复背景、删去泛化铺陈，保留现有技术缺陷和发明动机；
6. 末尾应自然落在“因此需要改进”一类的客观需求表述上，不能给出本发明方案。
"""
            rendered += self._length_balance_note(current, target)
            result = chat(
                prompt=rendered,
                system_prompt="你是一个严格控制篇幅的中国专利代理人。",
                model=self.model,
                max_tokens=self._max_tokens_for_final_target("背景技术", target, len(current)),
            )
            candidate = self._clean_component(result)
            if not candidate:
                break

            candidate_complete = self._field_complete_enough("背景技术", candidate)
            candidate_distance = self._target_distance(len(candidate), target)
            logger.info(f"背景技术长度修订: {len(current)} -> {len(candidate)}，目标 {target}")

            if candidate_complete and candidate_distance < best_distance:
                best = candidate
                best_distance = candidate_distance
            if candidate_complete and self._within_target(len(candidate), target):
                return candidate
            current = candidate if candidate_complete else current

        return best

    # ─── Layer 2 ────────────────────────────────────────────────

    def _target_length(self, field: str, default: int) -> int:
        target_lengths = self.global_memory.read("target_lengths") or {}
        try:
            return int(target_lengths.get(field) or default)
        except (TypeError, ValueError):
            return default

    def _make_layer2_plan(self) -> list[dict]:
        draft_D = self.global_memory.read("input_draft")
        summary_target = self._target_length("发明内容", DEFAULT_SUMMARY_TARGET)
        claims_target = self._target_length("权利要求书", DEFAULT_CLAIMS_TARGET)
        max_points = self._max_points_for_targets(summary_target, claims_target)
        rendered = self.prompt_manager.render("layer2_plan", {
            "D": draft_D,
            "summary_target": str(summary_target),
            "claims_target": str(claims_target),
        })
        result = chat(prompt=rendered, model=self.model, max_tokens=4096)
        plan_text = extract_xml_tag(result, "Plan") or result
        items = self._parse_layer2_plan(plan_text)
        if not items:
            logger.warning("保护规划解析为空，回退到保护单元拆分")
            points = self._split_points()
            if len(points) > max_points:
                points = points[:max_points - 1] + ["；".join(points[max_points - 1:])]
            summary_each = max(180, summary_target // max(len(points), 1))
            claims_each = max(120, claims_target // max(len(points), 1))
            items = [
                {
                    "topic": point,
                    "summary_target": summary_each,
                    "claim_type": self._claim_type(idx, point),
                    "claim_target": claims_each,
                }
                for idx, point in enumerate(points, 1)
            ]
        if len(items) > max_points:
            logger.warning(f"保护单元过多: {len(items)}，根据目标长度合并为 {max_points} 个")
            items = self._merge_plan_items(items, max_points)
        logger.info(f"保护规划完成，共 {len(items)} 个保护单元")
        return items

    def _max_points_for_targets(self, summary_target: int, claims_target: int) -> int:
        minimum = 3 if summary_target < 1600 else 4
        summary_cap = max(minimum, summary_target // 350)
        claims_cap = max(minimum, claims_target // 300)
        return max(minimum, min(MAX_PROTECTION_POINTS, min(summary_cap, claims_cap)))

    def _merge_plan_items(self, items: list[dict], max_points: int) -> list[dict]:
        if len(items) <= max_points:
            return items
        head = items[:max_points - 1]
        tail = items[max_points - 1:]
        merged_topics = self._dedupe_lines([item.get("topic", "") for item in tail])
        merged = {
            "topic": "；".join(merged_topics),
            "summary_target": sum(int(item.get("summary_target") or 0) for item in tail),
            "claim_type": tail[-1].get("claim_type") or "从属权利要求",
            "claim_target": sum(int(item.get("claim_target") or 0) for item in tail),
        }
        return head + [merged]

    def _parse_layer2_plan(self, text: str) -> list[dict]:
        items = []
        current = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^\d+[\.\、\)]\s*(?:保护主题[:：])?\s*(.+)", line)
            if match:
                if current:
                    items.append(current)
                current = {
                    "topic": self._clean_component(match.group(1)),
                    "summary_target": 260,
                    "claim_type": "从属权利要求",
                    "claim_target": 180,
                }
                continue
            if current is None:
                continue
            key_match = re.match(r"^(保护主题|发明内容目标字数|权利要求类型|权利要求目标字数)[:：]\s*(.+)", line)
            if not key_match:
                continue
            key, value = key_match.groups()
            value = self._clean_component(value)
            if key == "保护主题":
                current["topic"] = value
            elif key == "发明内容目标字数":
                current["summary_target"] = self._extract_int(value, current["summary_target"])
            elif key == "权利要求类型":
                current["claim_type"] = value
            elif key == "权利要求目标字数":
                current["claim_target"] = self._extract_int(value, current["claim_target"])
        if current:
            items.append(current)
        return [item for item in items if item["topic"]]

    def _extract_int(self, text: str, default: int) -> int:
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else default

    def _split_points(self) -> List[str]:
        """将草稿拆分为技术点列表"""
        draft_D = self.global_memory.read("input_draft")
        rendered = self.prompt_manager.render("point_splitter", {"D": draft_D})
        result = chat(prompt=rendered, model=self.model, max_tokens=2048)
        points_text = extract_xml_tag(result, "Points") or result

        lines = []
        for line in points_text.strip().split("\n"):
            line = line.strip()
            match = re.match(r'^\d+[\.\、\)]\s*(.+)', line)
            if match:
                lines.append(match.group(1).strip())
            elif line and not line.startswith("<"):
                lines.append(line)

        lines = self._dedupe_lines(lines)
        if len(lines) > MAX_PROTECTION_POINTS:
            logger.warning(f"保护单元过多: {len(lines)}，仅保留前 {MAX_PROTECTION_POINTS} 个")
            lines = lines[:MAX_PROTECTION_POINTS]

        if not lines:
            logger.warning("技术点拆分为空，回退为单个技术点")
            lines = [draft_D]

        logger.info(f"技术点拆分完成，共 {len(lines)} 个点")
        return lines

    def _run_layer2(self):
        logger.info("── 第二层：发明内容 & 权利要求交替生成 ──")

        plan_items = self._make_layer2_plan()
        total = len(plan_items)
        draft_D = self.global_memory.read("input_draft")
        background = self.global_memory.read("short_components.background")

        invention_parts = []
        claims_parts = []

        for idx, item in enumerate(plan_items, 1):
            point = item["topic"]
            logger.info(f"处理技术点 {idx}/{total}")

            inv_text = self._generate_invention_point(
                idx, total, point, claims_parts, draft_D, background,
                target_chars=item["summary_target"],
            )
            claim_no = len(claims_parts) + 1
            claim_type = item.get("claim_type") or self._claim_type(idx, point)
            claim_text = self._generate_claim_point(
                inv_text, claims_parts, draft_D, claim_no, claim_type,
                target_chars=item["claim_target"],
            )

            is_pass, feedback = self.reflection.reflect_intra_layer2_point(inv_text, claim_text)
            if not is_pass:
                logger.warning(f"技术点 {idx} 反思不通过: {feedback[:100]}...")
                refined_inv = self._generate_invention_point(
                    idx, total, point, claims_parts, draft_D, background,
                    feedback=f"上一轮反馈: {feedback[:600]}",
                    target_chars=item["summary_target"],
                )
                refined_claim = self._generate_claim_point(
                    refined_inv, claims_parts, draft_D, claim_no, claim_type,
                    target_chars=item["claim_target"],
                )
                is_pass2, _ = self.reflection.reflect_intra_layer2_point(refined_inv, refined_claim)
                if is_pass2:
                    inv_text, claim_text = refined_inv, refined_claim
                logger.info(f"技术点 {idx} 反思{'通过' if is_pass2 else '跳过'}")

            invention_parts.append(inv_text)
            claims_parts.append(claim_text)

        full_invention = "\n\n".join(invention_parts)
        full_claims = "\n\n".join(self._dedupe_and_renumber_claims(claims_parts))
        full_invention = self._refine_summary_to_target(full_invention, draft_D, full_claims)
        full_claims = normalize_claims(self._refine_claims_to_target(full_claims, draft_D))

        self.global_memory.write("short_components.summary", full_invention)
        self.global_memory.write("short_components.claims", full_claims)
        logger.info(f"第二层完成: 发明内容 {len(full_invention)} 字符, 权利要求 {len(full_claims)} 字符")

        self._reflect_loop(lambda: self.reflection.reflect_inter_L2L1())

    def _generate_invention_point(self, idx: int, total: int, point: str,
                                   claims_parts: list, draft_D: str, background: str,
                                   feedback: str = "", target_chars: int = 260) -> str:
        existing_claims = self._format_claim_summaries(claims_parts)
        best_text = ""
        best_distance = None
        current_feedback = feedback.strip()
        current_source = ""

        for attempt in range(2):
            rendered = self.prompt_manager.render("writer_invention_point", {
                "D": draft_D, "B": background,
                "total": str(total), "idx": str(idx), "point": point, "C": existing_claims,
                "target_chars": str(target_chars),
            })
            extra_feedback = current_feedback
            if current_source:
                if len(current_source) < target_chars * TARGET_MIN_RATIO:
                    length_note = (
                        f"上一版仅 {len(current_source)} 字，距离目标 {target_chars} 字仍短 "
                        f"{target_chars - len(current_source)} 字左右。"
                        "请在保留原有技术特征和表达顺序的基础上适度扩写，补充技术问题承接、步骤关系、模块作用、"
                        "与前后段落的衔接以及有益效果。上一版内容如下："
                        f"\n{current_source}"
                    )
                else:
                    length_note = (
                        f"上一版 {len(current_source)} 字，已经超过目标 {target_chars} 字。"
                        "请压缩冗余表述，合并相近技术效果和衔接语，只保留当前保护单元的必要技术特征。上一版内容如下："
                        f"\n{current_source}"
                    )
                extra_feedback = f"{extra_feedback}\n{length_note}" if extra_feedback else length_note
            if extra_feedback:
                rendered += f"\n\n# 修改要求：{extra_feedback}"

            result = chat(
                prompt=rendered,
                model=self.model,
                max_tokens=max(256, min(3072, int(target_chars * 0.65))),
            )
            candidate = self._clean_component(extract_xml_tag(result, "InventionPoint") or result)
            if not candidate:
                continue

            candidate_distance = self._target_distance(len(candidate), target_chars)
            if best_distance is None or candidate_distance < best_distance:
                best_text = candidate
                best_distance = candidate_distance

            if self._within_target(len(candidate), target_chars):
                return candidate

            current_source = candidate
            if len(candidate) < target_chars * TARGET_MIN_RATIO:
                current_feedback = (
                    f"请继续扩写上一版，当前长度 {len(candidate)} 字，目标 {target_chars} 字左右。"
                    "优先补充当前保护单元的技术作用、实现路径和效果，不要压缩已有内容。"
                )
            else:
                current_feedback = (
                    f"请压缩上一版，当前长度 {len(candidate)} 字，目标 {target_chars} 字左右。"
                    "优先删除重复衔接语、泛化效果说明和非必要实施细节。"
                )

        return best_text

    def _generate_claim_point(
        self,
        invention_point: str,
        claims_parts: list,
        draft_D: str,
        claim_no: int,
        claim_type: str,
        target_chars: int = 180,
    ) -> str:
        existing_claims = self._format_claim_summaries(claims_parts)
        rendered = self.prompt_manager.render("writer_claim_point", {
            "D": draft_D,
            "I": invention_point,
            "C": existing_claims,
            "claim_no": str(claim_no),
            "claim_type": claim_type,
            "target_chars": str(target_chars),
        })
        result = chat(
            prompt=rendered,
            model=self.model,
            max_tokens=max(256, min(2048, int(target_chars * 0.75))),
        )
        claim = self._clean_component(extract_xml_tag(result, "ClaimPoint") or result)
        return self._extract_single_claim(claim, claim_no)

    def _refine_summary_to_target(self, summary: str, draft_D: str, claims: str = "") -> str:
        target = self._target_length("发明内容", DEFAULT_SUMMARY_TARGET)
        if self._within_target(len(summary), target):
            return summary
        current = summary
        best = summary
        best_distance = self._target_distance(len(summary), target)

        for attempt in range(PRE_FINAL_LENGTH_RETRIES):
            rendered = self.prompt_manager.render("refine_summary_to_target", {
                "D": draft_D,
                "S": current,
                "C": claims,
                "target": str(target),
                "current_length": str(len(current)),
                "missing_chars": str(max(target - len(current), 0)),
            })
            rendered += self._length_balance_note(current, target)
            if attempt > 0 and len(current) < target:
                rendered += (
                    "\n\n# 追加要求：上一版仍未达到目标长度或结构不够完整。"
                    "请保留上一版已经正确的技术特征，继续按草稿中的方法步骤、可选实现、装置模块和有益效果展开，"
                    "优先复用草稿和权利要求中的术语及顺序，使长度更接近目标。"
                )

            result = chat(
                prompt=rendered,
                model=self.model,
                max_tokens=self._max_tokens_for_rewrite_target(target, len(current), 6144),
            )
            refined = self._clean_component(result)
            if not refined:
                break

            refined_complete = self._field_complete_enough("发明内容", refined)
            refined_distance = self._target_distance(len(refined), target)
            logger.info(f"发明内容长度修订: {len(current)} -> {len(refined)}，目标 {target}")

            if refined_complete and refined_distance < best_distance:
                best = refined
                best_distance = refined_distance

            if refined_complete and self._within_target(len(refined), target):
                return refined

            current = refined if refined_complete else current

        return best

    def _refine_claims_to_target(self, claims: str, draft_D: str) -> str:
        target = self._target_length("权利要求书", DEFAULT_CLAIMS_TARGET)
        if self._within_target(len(claims), target):
            return claims

        current = claims
        best = claims
        best_distance = self._target_distance(len(claims), target)

        for attempt in range(PRE_FINAL_LENGTH_RETRIES):
            rendered = self.prompt_manager.render("refine_claims_to_target", {
                "D": draft_D,
                "C": current,
                "target": str(target),
            })
            rendered += self._length_balance_note(current, target)
            current_claim_count = self._claim_count(current)
            rendered += (
                f"\n\n# 当前权利要求数量：{current_claim_count} 项。"
                "除非当前长度距离目标超过 20% 且草稿明确支持新增保护类型，否则应保持现有权利要求数量，"
                "只通过补充或压缩现有权利要求的限定语来调整篇幅。"
            )
            if attempt > 0 and len(current) < target:
                rendered += (
                    "\n\n# 追加要求：上一版仍未接近目标长度。"
                    "请在不重复完整主流程、不新增多个权利要求的前提下，补充现有权利要求中的可选实现、规则、判断分支或装置模块限定，"
                    "并保持权利要求编号连续。"
                )

            result = chat(
                prompt=rendered,
                model=self.model,
                max_tokens=self._max_tokens_for_claims_target(target, len(current)),
            )
            refined = self._clean_component(result)
            if not refined:
                break

            refined_complete = self._field_complete_enough("权利要求书", refined)
            refined_distance = self._target_distance(len(refined), target)
            logger.info(f"权利要求书长度修订: {len(current)} -> {len(refined)}，目标 {target}")

            if refined_complete and refined_distance < best_distance:
                best = refined
                best_distance = refined_distance

            if refined_complete and self._within_target(len(refined), target):
                return refined

            current = refined if refined_complete and refined_distance <= best_distance else best

        if best is claims:
            logger.warning("权利要求书修订后更偏离目标长度，保留修订前版本")
        return best

    def _claim_count(self, claims: str) -> int:
        return len(re.findall(r"(?m)^\s*\d+[\.\、]\s*", claims or ""))

    def _max_tokens_for_claims_target(self, target: int, current_length: int) -> int:
        if current_length > target * TARGET_MAX_RATIO:
            return max(1024, min(8192, int(target * 1.15)))
        return max(1024, min(8192, int(target * 1.35)))

    def _max_tokens_for_rewrite_target(self, target: int, current_length: int, cap: int) -> int:
        if current_length > target * TARGET_MAX_RATIO:
            if cap > 10000:
                return max(2048, min(cap, int(target * 1.25)))
            return max(768, min(cap, int(target * 1.20)))
        return max(1024, min(cap, int(target * 2.0)))

    def _within_target(self, length: int, target: int) -> bool:
        return target * TARGET_MIN_RATIO <= length <= target * TARGET_MAX_RATIO

    def _target_distance(self, length: int, target: int) -> int:
        return abs(length - target)

    def _length_balance_note(self, current: str, target: int) -> str:
        diff = len(current) - target
        if diff > 0:
            return (
                f"\n\n# 追加要求：当前版本比目标长度长 {diff} 字。"
                "请优先压缩冗余、合并重复表述、删去铺陈性解释和无关扩展，"
                "在保留核心技术特征、必要步骤顺序和法律限定的前提下，把篇幅收回到目标区间内，"
                "不要继续扩写，也不要基本照搬当前长文本。"
            )
        if diff < 0:
            return (
                f"\n\n# 追加要求：当前版本比目标长度短 {-diff} 字。"
                "请在不引入无关内容的前提下适度扩写，补足必要的技术步骤、判断条件、可选实现或效果说明。"
            )
        return ""

    def _dedupe_lines(self, lines: list[str]) -> list[str]:
        seen = set()
        result = []
        for line in lines:
            clean = self._clean_component(line)
            if not clean:
                continue
            signature = re.sub(r"\s+", "", clean)
            if signature in seen:
                continue
            seen.add(signature)
            result.append(clean)
        return result

    def _clean_component(self, text: str) -> str:
        text = re.sub(r"</?\w+>", "", text or "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _claim_type(self, idx: int, point: str) -> str:
        if any(keyword in point for keyword in ("装置", "系统", "介质", "设备")):
            return "装置独立权利要求"
        if idx == 1:
            return "方法独立权利要求"
        return "从属权利要求"

    def _format_claim_summaries(self, claims_parts: list[str]) -> str:
        if not claims_parts:
            return "（暂无）"
        summaries = []
        for claim in claims_parts:
            one_line = re.sub(r"\s+", " ", claim).strip()
            summaries.append(one_line[:180])
        return "\n".join(summaries)

    def _extract_single_claim(self, text: str, claim_no: int) -> str:
        matches = list(re.finditer(r"(?m)^\s*(\d+)[\.\、]\s*", text))
        if matches:
            selected_idx = None
            for idx, match in enumerate(matches):
                if int(match.group(1)) == claim_no:
                    selected_idx = idx
                    break
            if selected_idx is None:
                logger.warning(f"未找到编号 {claim_no} 的权利要求，使用模型输出的第一项")
                selected_idx = 0
            start = matches[selected_idx].start()
            end = matches[selected_idx + 1].start() if selected_idx + 1 < len(matches) else len(text)
            text = text[start:end].strip()

        text = re.sub(r"^\s*\d+[\.\、]\s*", f"{claim_no}. ", text, count=1)
        if not re.match(rf"^\s*{claim_no}\.", text):
            text = f"{claim_no}. {text}"
        return self._clean_component(text)

    def _dedupe_and_renumber_claims(self, claims_parts: list[str]) -> list[str]:
        unique_bodies = []
        seen = set()
        for claim in claims_parts:
            clean = self._clean_component(claim)
            body = re.sub(r"^\s*\d+[\.\、]\s*", "", clean, count=1).strip()
            signature = re.sub(r"\s+", "", body)
            if not body or signature in seen:
                logger.warning(f"跳过重复权利要求: {body[:80]}...")
                continue
            seen.add(signature)
            unique_bodies.append(body)
        return [f"{idx}. {body}" for idx, body in enumerate(unique_bodies, 1)]

    # ─── Layer 3 ────────────────────────────────────────────────

    def _run_layer3(self):
        logger.info("── 第三层：表达层生成 ──")

        writers_config = [
            (self.description_writer, "description", "具体实施方式"),
            (self.abstract_writer, "abstract", "摘要"),
            (self.title_writer, "title", "标题"),
            (self.technical_field_writer, "technical_field", "技术领域"),
        ]

        for writer, memory_key, label in writers_config:
            logger.info(f"生成: {label}")
            writer.execute()
            content = self.global_memory.read(f"short_components.{memory_key}") or ""
            if memory_key == "description":
                content = self._refine_description_to_target(content, self.global_memory.read("input_draft"))
                self.global_memory.write(f"short_components.{memory_key}", content)
            self._reflect_loop(
                lambda label=label, content=content: self.reflection.reflect_inter_L3L2(label, content)
            )

        logger.info("第三层完成")

    def _refine_description_to_target(self, description: str, draft_D: str) -> str:
        target = self._target_length("具体实施方式", DEFAULT_DESCRIPTION_TARGET)
        if self._within_target(len(description), target):
            return description

        current = description
        best = description
        best_distance = self._target_distance(len(description), target)
        invention = self.global_memory.read("short_components.summary") or ""
        claims = self.global_memory.read("short_components.claims") or ""

        for attempt in range(PRE_FINAL_LENGTH_RETRIES):
            rendered = self.prompt_manager.render("refine_description_to_target", {
                "D": draft_D,
                "I": invention,
                "C": claims,
                "V": current,
                "target": str(target),
                "current_length": str(len(current)),
                "missing_chars": str(max(target - len(current), 0)),
            })
            rendered += self._length_balance_note(current, target)
            if attempt > 0 and len(current) < target:
                rendered += (
                    "\n\n# 追加要求：上一版仍未接近目标长度。"
                    "请继续扩写具体实施方式中的步骤实现、条件判断、示例数据、数据库表更新、装置模块配合和附图说明，"
                    "不要改成摘要或权利要求，不要使用 Markdown。"
                )

            result = chat(
                prompt=rendered,
                model=self.model,
                max_tokens=self._max_tokens_for_rewrite_target(target, len(current), 16384),
            )
            refined = self._clean_component(result)
            if not refined:
                break

            refined_complete = self._field_complete_enough("具体实施方式", refined)
            refined_distance = self._target_distance(len(refined), target)
            logger.info(f"具体实施方式长度修订: {len(current)} -> {len(refined)}，目标 {target}")

            if refined_complete and refined_distance < best_distance:
                best = refined
                best_distance = refined_distance

            if refined_complete and self._within_target(len(refined), target):
                return refined

            current = refined if refined_complete else current

        return best

    def _normalize_final_lengths(self, patent: dict) -> dict:
        target_lengths = self.global_memory.read("target_lengths") or {}
        if not target_lengths:
            return patent

        draft_D = self.global_memory.read("input_draft") or ""
        normalized = dict(patent)
        for field, text in list(normalized.items()):
            target = self._target_length(field, len(text))
            if target <= 0 or self._within_target(len(text), target):
                continue
            logger.info(f"最终长度校正: {field} 当前 {len(text)}，目标 {target}")
            normalized[field] = self._refine_field_to_target(field, text, draft_D, target)

        self.global_memory.write("final_patent", json.dumps(normalized, ensure_ascii=False))
        return normalized

    def _refine_field_to_target(self, field: str, text: str, draft_D: str, target: int) -> str:
        best = text
        best_distance = self._target_distance(len(text), target)
        current = text

        for attempt in range(FINAL_LENGTH_RETRIES):
            lower = int(target * TARGET_MIN_RATIO)
            upper = int(target * TARGET_MAX_RATIO)
            direction = "压缩" if len(current) > upper else "扩写"
            rendered = self._build_final_length_prompt(
                field=field,
                draft_D=draft_D,
                current=current,
                target=target,
                lower=lower,
                upper=upper,
                direction=direction,
            )
            result = chat(
                prompt=rendered,
                system_prompt="你是一个严格控制篇幅的中国专利代理人。",
                model=self.model,
                max_tokens=self._max_tokens_for_final_target(field, target, len(current)),
            )
            candidate = self._clean_component(result)
            if not candidate:
                break

            candidate_complete = self._field_complete_enough(field, candidate)
            candidate_distance = self._target_distance(len(candidate), target)
            logger.info(
                f"最终长度校正: {field} 第{attempt + 1}次 {len(current)} -> {len(candidate)}，目标 {target}"
            )
            if not candidate_complete:
                logger.warning(f"最终长度校正: {field} 第{attempt + 1}次输出疑似未完整结束，忽略为最佳候选")
            if candidate_complete and candidate_distance < best_distance:
                best = candidate
                best_distance = candidate_distance
            if candidate_complete and self._within_target(len(candidate), target):
                return candidate
            current = candidate if candidate_complete else current

        lower = int(target * TARGET_MIN_RATIO)
        upper = int(target * TARGET_MAX_RATIO)
        if len(best) > upper:
            if field == "权利要求书":
                trimmed = self._trim_claims_to_length(best, upper)
            else:
                trimmed = self._trim_to_length(best, upper, lower)
            if trimmed and self._field_complete_enough(field, trimmed):
                logger.info(f"最终长度句界裁剪: {field} {len(best)} -> {len(trimmed)}，目标 {target}")
                return trimmed

        return best

    def _build_final_length_prompt(
        self,
        *,
        field: str,
        draft_D: str,
        current: str,
        target: int,
        lower: int,
        upper: int,
        direction: str,
    ) -> str:
        claim_count_note = ""
        if field == "权利要求书":
            claim_count_note = (
                f"\n7. 当前权利要求书共有 {self._claim_count(current)} 项。"
                "如果当前长度距离目标不超过 20%，必须保持现有权利要求数量，只微调各项的限定语；"
                "不得新增多个电子设备、存储介质或系统权利要求。"
            )
        field_rules = {
            "标题": "输出一个专利标题，不要句号、书名号或解释。",
            "摘要": "保留技术问题、核心方案和有益效果，写成一个摘要自然段。",
            "权利要求书": "保持权利要求编号连续，每项权利要求必须是完整句子；压缩时优先删去重复限定，不能破坏必要技术特征。",
            "技术领域": "通常输出一句话，采用“本发明涉及……技术领域，尤其涉及……”的表达。",
            "背景技术": "只写现有技术背景和缺陷，不披露具体实施例。",
            "发明内容": "合并相近保护点，保留目的、总体技术方案、可选实现、装置/系统对应方案和有益效果；不要逐一铺开成具体实施方式。",
            "具体实施方式": "保留步骤编号、关键处理规则、示例数据和附图/模块说明；压缩时删去重复铺陈和泛化说明。",
        }
        strict_length_note = ""
        if direction == "压缩":
            strict_length_note = (
                f"\n附加压缩要求：当前正文超过合格上限，重写后必须不超过 {upper} 字。"
                "可以删除重复有益效果、次要示例、重复模块说明和铺垫性语句；"
                "不得仅替换同义词后保持原长度。"
            )
        return f"""# 专利草稿
{draft_D}

# 字段
{field}

# 当前正文
{current}

# 长度要求
目标长度：{target} 字左右
合格区间：{lower}-{upper} 字
当前长度：{len(current)} 字
本轮任务：{direction}

请重写“{field}”字段，使最终长度落入合格区间，并尽量贴近目标长度。
要求：
1. 只输出重写后的“{field}”正文，不要输出标题、标签、Markdown、解释或字数统计；
2. 目标是接近参考文本长度，不是越长越好；
3. 必须忠实于专利草稿和当前正文中的核心技术特征，不得加入草稿没有的新技术；
4. 如果本轮任务是压缩，必须合并相近表述、删除重复效果和泛化铺陈，不要继续扩写；
5. 如果本轮任务是扩写，只能补充草稿已有的步骤关系、判断条件、可选实现、模块配合或技术效果；
6. {field_rules.get(field, "使用规范专利中文，保持字段语体。")}
{claim_count_note}
{strict_length_note}
"""

    def _max_tokens_for_final_target(self, field: str, target: int, current_length: int) -> int:
        if field == "标题":
            return 128
        if field == "权利要求书":
            multiplier = 1.15 if current_length > target * TARGET_MAX_RATIO else 1.35
            return max(1024, min(8192, int(target * multiplier)))
        if field == "具体实施方式":
            multiplier = 1.25 if current_length > target * TARGET_MAX_RATIO else 2.0
            return max(2048, min(16384, int(target * multiplier)))
        if field in {"摘要", "技术领域", "背景技术"}:
            multiplier = 1.20 if current_length > target * TARGET_MAX_RATIO else 3
            return max(256, min(2048, int(target * multiplier)))
        multiplier = 1.20 if current_length > target * TARGET_MAX_RATIO else 2.0
        return max(1024, min(16384, int(target * multiplier)))

    def _field_complete_enough(self, field: str, text: str) -> bool:
        clean = (text or "").strip()
        if not clean:
            return False
        if field == "标题":
            return True
        if field == "权利要求书":
            return self._claims_complete_enough(clean)
        return self._ends_cleanly(clean)

    def _claims_complete_enough(self, claims: str) -> bool:
        matches = list(re.finditer(r"(?m)^\s*\d+[\.\、]\s*", claims or ""))
        if not matches:
            return self._ends_cleanly(claims)
        last_start = matches[-1].start()
        last_claim = claims[last_start:].strip()
        return self._ends_cleanly(last_claim)

    def _ends_cleanly(self, text: str) -> bool:
        clean = (text or "").strip()
        if not clean:
            return False
        clean = clean.rstrip("）】》〉」』’”'\"")
        return clean.endswith(("。", "；", ";", "！", "？"))

    def _trim_to_length(self, text: str, limit: int, lower: int | None = None) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit]
        min_pos = lower if lower is not None else int(limit * 0.85)
        break_chars = "。；;\n"
        pos = max(cut.rfind(ch) for ch in break_chars)
        if pos >= min_pos:
            return cut[:pos + 1].strip()
        return ""

    def _trim_claims_to_length(self, claims: str, limit: int) -> str:
        matches = list(re.finditer(r"(?m)^\s*\d+[\.\、]\s*", claims or ""))
        if not matches:
            return self._trim_to_length(claims, limit)

        parts = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(claims)
            part = claims[start:end].strip()
            if part:
                parts.append(part)

        kept = []
        for part in parts:
            candidate = "\n".join(kept + [part])
            if len(candidate) <= limit:
                kept.append(part)
                continue
            prefix = "\n".join(kept)
            remaining = limit - len(prefix) - (1 if prefix else 0)
            lower = max(80, int(remaining * 0.70))
            trimmed_part = self._trim_to_length(part, remaining, lower)
            if trimmed_part:
                kept.append(trimmed_part)
            break

        if kept:
            return "\n".join(kept).strip()
        return self._trim_to_length(claims, limit)

    # ─── 输出 ────────────────────────────────────────────────────

    def _apply_quality_gate(self, patent: dict, stage: str, draft: str | None = None) -> dict:
        """Run deterministic repairs and retain an auditable issue snapshot."""
        draft_text = draft if draft is not None else (self.global_memory.read("input_draft") or "")
        before = quality_issues(patent, draft_text)
        repaired = repair_patent(patent, draft_text)
        after = quality_issues(repaired, draft_text)
        if before or after:
            logger.warning(
                f"终稿质检[{stage}]发现 {len(before)} 项问题，"
                f"修复后剩余 {len(after)} 项"
            )
            for issue in after[:8]:
                logger.warning(f"终稿质检[{stage}]: {issue}")
        self.global_memory.write(
            f"quality_report.{stage}",
            json.dumps(
                {"before": before, "after": after},
                ensure_ascii=False,
            ),
        )
        return repaired

    def _assemble_final(self) -> dict:
        sc = self.global_memory.read("short_components")
        patent = {
            "标题": sc["title"],
            "摘要": sc["abstract"],
            "权利要求书": sc["claims"],
            "技术领域": sc["technical_field"],
            "背景技术": sc["background"],
            "发明内容": sc["summary"],
            "具体实施方式": sc["description"],
        }
        self.global_memory.write("final_patent", json.dumps(patent, ensure_ascii=False))
        return patent
