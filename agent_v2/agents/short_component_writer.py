"""短组件撰写智能体：通用类，可实例化5个专属智能体"""

from agent_v2.agents.base_agent import BaseAgent
from agent_v2.prompts.prompt_manager import PromptManager
from agent_v2.memory.global_memory import GlobalMemory
from agent_v2.utils.text_parser import extract_xml_tag
from loguru import logger


PLACEHOLDER_TEXTS = {
    "生成的具体实施方式",
    "生成的专利摘要",
    "生成的专利标题",
    "生成的背景技术内容",
    "生成的发明内容",
    "生成的专利权利要求书",
    "生成的技术领域内容",
}


class ShortComponentWriterAgent(BaseAgent):
    """短组件撰写智能体：通用类，可实例化5个专属智能体"""
    
    def __init__(
        self,
        agent_name: str,
        prompt_manager: PromptManager,
        global_memory: GlobalMemory,
        bind_prompt_id: str,
        model: str = "qwen-plus"
    ):
        super().__init__(
            agent_name=agent_name,
            prompt_manager=prompt_manager,
            global_memory=global_memory,
            default_prompt_id=bind_prompt_id,
            model=model
        )
    
    def generate_component(self) -> str:
        """
        生成对应专利短组件
        :return: 组件内容
        """
        draft_D = self.read_global("input_draft")
        memory_key = self._get_memory_key()
        target_chars = self._target_chars(memory_key)
        rendered_prompt = self.render_prompt(
            prompt_id=self.default_prompt_id,
            input_vars={
                "D": draft_D,
                "target_chars": target_chars,
                "B": self.read_global("short_components.background") or "",
                "I": self.read_global("short_components.summary") or "",
                "C": self.read_global("short_components.claims") or "",
            }
        )
        component_content = self.call_llm(rendered_prompt, max_tokens=self._max_tokens_for_target(target_chars))
        if "<" in component_content and ">" in component_content:
            component_content = extract_xml_tag(component_content, self._get_tag_name()) or component_content
        component_content = component_content.strip()
        self._validate_component(component_content)
        return component_content

    def _target_chars(self, memory_key: str) -> int:
        target_lengths = self.read_global("target_lengths") or {}
        field_map = {
            "title": "标题",
            "abstract": "摘要",
            "technical_field": "技术领域",
            "background": "背景技术",
            "summary": "发明内容",
            "claims": "权利要求书",
            "description": "具体实施方式",
        }
        default_map = {
            "title": 18,
            "abstract": 300,
            "technical_field": 40,
            "background": 320,
            "summary": 2200,
            "claims": 2200,
            "description": 5200,
        }
        try:
            return int(target_lengths.get(field_map.get(memory_key, ""), 0) or default_map.get(memory_key, 0))
        except (TypeError, ValueError):
            return default_map.get(memory_key, 0)

    def _max_tokens_for_target(self, target_chars: int) -> int:
        if target_chars <= 0:
            return 16384
        return max(2048, min(16384, int(target_chars * 3)))

    def _validate_component(self, component: str):
        if component in PLACEHOLDER_TEXTS:
            raise ValueError(f"{self.agent_name} 返回了占位文本: {component}")
        if self.default_prompt_id == "writer_description" and len(component) < 1000:
            raise ValueError(f"{self.agent_name} 生成内容过短: {len(component)} 字符")
    
    def _get_tag_name(self) -> str:
        """根据提示词ID获取对应的XML标签名"""
        tag_map = {
            "writer_title": "Title",
            "writer_abstract": "Abstract",
            "writer_technical_field": "TechnicalField",
            "writer_background": "Background",
            "writer_summary": "Summary",
            "writer_claims": "Claims",
            "writer_description": "Description",
        }
        return tag_map.get(self.default_prompt_id, "")
    
    def execute(self) -> str:
        """智能体核心执行入口"""
        component = self.generate_component()
        memory_key = self._get_memory_key()
        if memory_key:
            self.write_global(f"short_components.{memory_key}", component)
            logger.info(f"{self.agent_name} 生成完成，已写入全局记忆")
        return component
    
    def _get_memory_key(self) -> str:
        """根据提示词ID获取对应的记忆key"""
        key_map = {
            "writer_title": "title",
            "writer_abstract": "abstract",
            "writer_technical_field": "technical_field",
            "writer_background": "background",
            "writer_summary": "summary",
            "writer_claims": "claims",
            "writer_description": "description",
        }
        return key_map.get(self.default_prompt_id, "")
