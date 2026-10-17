"""全局共享记忆：单例模式，所有智能体共享"""

from typing import Any
from loguru import logger


class GlobalMemory:
    """全局共享记忆：单例模式，所有智能体共享"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_memory"):
            self._memory = {
                "input_draft": None,
                "short_components": {
                    "title": None, "abstract": None, "technical_field": None,
                    "background": None, "summary": None, "claims": None
                },
                "reference_lib_R": None,
                "pgtree_W": None,
                "description_subsections": {},
                "full_description": None,
                "final_patent": None
            }
            logger.info("全局共享记忆初始化完成")
    
    def read(self, key_path: str) -> Any:
        """读取记忆：支持嵌套key路径，如 'short_components.title'"""
        keys = key_path.split(".")
        value = self._memory
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"全局记忆读取失败，key路径不存在: {key_path}")
            return None
    
    def write(self, key_path: str, value: Any):
        """写入记忆：支持嵌套key路径"""
        keys = key_path.split(".")
        target = self._memory
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        logger.debug(f"全局记忆写入: {key_path}")
    
    def get_full_memory(self) -> dict:
        """获取完整记忆字典（用于归档）"""
        return self._memory.copy()
    
    def reset(self):
        self._memory = {
            "input_draft": None,
            "short_components": {
                "title": None, "abstract": None, "technical_field": None,
                "background": None, "summary": None, "claims": None
            },
            "reference_lib_R": None,
            "pgtree_W": None,
            "description_subsections": {},
            "full_description": None,
            "final_patent": None
        }
        logger.info("全局共享记忆已重置")