"""审查智能体：负责草稿审核和子章节内容审核"""

from typing import Tuple
from agent_v2.agents.base_agent import BaseAgent
from agent_v2.prompts.prompt_manager import PromptManager
from agent_v2.memory.global_memory import GlobalMemory
from agent_v2.utils.text_parser import extract_xml_tag
from loguru import logger


class ExaminerAgent(BaseAgent):
    """审查智能体：负责草稿审核和子章节内容审核"""
    
    def __init__(self, prompt_manager: PromptManager, global_memory: GlobalMemory, model: str = "qwen-plus"):
        super().__init__(
            agent_name="examiner_agent",
            prompt_manager=prompt_manager,
            global_memory=global_memory,
            default_prompt_id="examiner_content",
            model=model
        )
    
    def review_draft(self, draft_D: str) -> Tuple[bool, str]:
        """
        审核输入草稿的质量
        :param draft_D: 发明人草稿
        :return: (是否通过, 审核结果/原因)
        """
        rendered_prompt = self.render_prompt("draft_review", {"D": draft_D})
        result = self.call_llm(rendered_prompt)
        is_pass = "Pass" in result
        return is_pass, result
    
    def review_subsection(self, draft_D: str, nij: str, dij: str) -> Tuple[bool, str]:
        """
        审核子章节内容
        :param draft_D: 发明人草稿
        :param nij: 子章节撰写指引
        :param dij: 子章节内容
        :return: (是否通过, 修改建议)
        """
        rendered_prompt = self.render_prompt(
            self.default_prompt_id,
            {"D": draft_D, "nij": nij, "dij": dij}
        )
        result = self.call_llm(rendered_prompt)
        is_pass = "<Result>Pass</Result>" in result
        feedback = extract_xml_tag(result, "Advice") or "无建议"
        return is_pass, feedback
    
    def execute(self, review_type: str, **kwargs) -> Tuple[bool, str]:
        """智能体核心执行入口：根据review_type分发到不同方法"""
        if review_type == "draft":
            return self.review_draft(kwargs["draft_D"])
        elif review_type == "subsection":
            return self.review_subsection(kwargs["draft_D"], kwargs["nij"], kwargs["dij"])
        else:
            raise ValueError(f"未知的审核类型: {review_type}")