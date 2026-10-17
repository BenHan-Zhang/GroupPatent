"""反思智能体：执行层内反思和层间反思"""
from typing import Tuple

from agent_v2.agents.base_agent import BaseAgent
from agent_v2.prompts.prompt_manager import PromptManager
from agent_v2.memory.global_memory import GlobalMemory
from agent_v2.utils.text_parser import extract_xml_tag
from loguru import logger


class ReflectionAgent(BaseAgent):
    """反思智能体：层内反思 + 层间反思"""

    def __init__(self, prompt_manager: PromptManager, global_memory: GlobalMemory, model: str = "qwen3.6-35b"):
        super().__init__(
            agent_name="reflection_agent",
            prompt_manager=prompt_manager,
            global_memory=global_memory,
            default_prompt_id="reflection_intra_layer2",
            model=model
        )

    def _call_and_parse(self, prompt_id: str, input_vars: dict) -> Tuple[bool, str]:
        rendered = self.render_prompt(prompt_id, input_vars)
        result = self.call_llm(rendered)
        is_pass = "<Result>Pass</Result>" in result
        feedback = extract_xml_tag(result, "Advice") or "无建议"
        return is_pass, feedback

    def reflect_intra_layer1(self) -> Tuple[bool, str]:
        """层内反思—第一层：背景技术 vs draft"""
        draft_D = self.read_global("input_draft")
        background = self.read_global("short_components.background")
        return self._call_and_parse("reflection_intra_layer1", {"D": draft_D, "B": background})

    def reflect_intra_layer2_point(self, invention_point: str, claim_point: str) -> Tuple[bool, str]:
        """层内反思—第二层：逐点审查 发明内容点 ↔ 权利要求条款"""
        draft_D = self.read_global("input_draft")
        return self._call_and_parse("reflection_intra_layer2", {
            "D": draft_D, "I": invention_point, "C": claim_point
        })

    def reflect_inter_L2L1(self) -> Tuple[bool, str]:
        """层间反思：Layer 2 全文 vs Layer 1"""
        draft_D = self.read_global("input_draft")
        background = self.read_global("short_components.background")
        invention = self.read_global("short_components.summary")
        claims = self.read_global("short_components.claims")
        return self._call_and_parse("reflection_inter_L2L1", {
            "D": draft_D, "B": background, "I": invention, "C": claims
        })

    def reflect_inter_L3L2(self, component_name: str, content: str) -> Tuple[bool, str]:
        """层间反思：Layer 3 单个组件 vs Layer 2 + Layer 1"""
        draft_D = self.read_global("input_draft")
        background = self.read_global("short_components.background")
        invention = self.read_global("short_components.summary")
        claims = self.read_global("short_components.claims")
        return self._call_and_parse("reflection_inter_L3L2", {
            "D": draft_D, "B": background, "I": invention, "C": claims,
            "component_name": component_name, "V": content
        })

    def execute(self, *args, **kwargs):
        pass