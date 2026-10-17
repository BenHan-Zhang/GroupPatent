"""提示词管理器：加载、渲染提示词模板"""

import os
from typing import Dict, Any
from loguru import logger


class PromptManager:
    """提示词管理器：加载、渲染提示词模板"""
    
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = os.path.dirname(__file__)
        self.prompts_dir = prompts_dir
        self._prompt_cache: Dict[str, str] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """从目录加载所有提示词模板到缓存"""
        if not os.path.exists(self.prompts_dir):
            raise FileNotFoundError(f"提示词目录不存在: {self.prompts_dir}")
        
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".txt"):
                prompt_id = filename[:-4]
                file_path = os.path.join(self.prompts_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    self._prompt_cache[prompt_id] = f.read().strip()
                logger.debug(f"加载提示词: {prompt_id}")
        logger.info(f"提示词管理器初始化完成，共加载 {len(self._prompt_cache)} 个提示词")
    
    def render(self, prompt_id: str, input_vars: Dict[str, Any]) -> str:
        """
        渲染提示词：将input_vars中的变量填充到模板中
        :param prompt_id: 提示词ID（对应文件名）
        :param input_vars: 变量字典，如 {"D": "草稿内容", "nij": "撰写指引"}
        :return: 渲染后的提示词
        """
        if prompt_id not in self._prompt_cache:
            raise ValueError(f"提示词ID不存在: {prompt_id}")
        
        template = self._prompt_cache[prompt_id]
        rendered = template
        for var_name, var_value in input_vars.items():
            rendered = rendered.replace(f"{{{var_name}}}", str(var_value))
        return rendered
    
    def get_prompt(self, prompt_id: str) -> str:
        """获取原始提示词模板"""
        if prompt_id not in self._prompt_cache:
            raise ValueError(f"提示词ID不存在: {prompt_id}")
        return self._prompt_cache[prompt_id]