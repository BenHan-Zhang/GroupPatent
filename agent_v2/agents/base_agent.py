"""智能体抽象基类：所有具体智能体必须继承此类"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from agent_v2.models.qwen_client import chat
from agent_v2.prompts.prompt_manager import PromptManager
from agent_v2.memory.global_memory import GlobalMemory
from agent_v2.memory.session_memory import SessionMemory
from loguru import logger


class BaseAgent(ABC):
    """智能体抽象基类：所有具体智能体必须继承此类"""
    
    def __init__(
        self,
        agent_name: str,
        prompt_manager: PromptManager,
        global_memory: GlobalMemory,
        default_prompt_id: str,
        model: str = "qwen-plus"
    ):
        """
        依赖注入：所有核心依赖通过构造函数传入，方便测试和替换
        :param agent_name: 智能体唯一名称
        :param prompt_manager: 提示词管理器实例
        :param global_memory: 全局共享记忆实例
        :param default_prompt_id: 智能体默认绑定的提示词ID
        :param model: 使用的模型名称
        """
        self.agent_name = agent_name
        self.prompt_manager = prompt_manager
        self.global_memory = global_memory
        self.default_prompt_id = default_prompt_id
        self.session_memory = SessionMemory()
        self.model = model
        logger.info(f"初始化智能体: {agent_name}")
    
    def render_prompt(self, prompt_id: str, input_vars: Dict[str, Any]) -> str:
        """渲染提示词：调用提示词管理器填充变量"""
        return self.prompt_manager.render(prompt_id, input_vars)
    
    def call_llm(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """调用LLM：自动记录会话记忆"""
        if system_prompt is None:
            system_prompt = "你是一个专业的专利代理人。"
        response = chat(prompt=prompt, system_prompt=system_prompt, model=self.model, **kwargs)
        self.session_memory.write("llm_input", prompt)
        self.session_memory.write("llm_output", response)
        logger.debug(f"{self.agent_name} 调用LLM完成，输出长度: {len(response)}")
        return response
    
    def read_global(self, key_path: str) -> Any:
        """读取全局记忆：支持嵌套key路径"""
        return self.global_memory.read(key_path)
    
    def write_global(self, key_path: str, value: Any):
        """写入全局记忆"""
        self.global_memory.write(key_path, value)
        logger.debug(f"{self.agent_name} 写入全局记忆: {key_path}")
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        智能体核心执行逻辑：所有子类必须实现此方法
        不同智能体的execute参数和返回值不同，在子类中定义
        """
        pass
