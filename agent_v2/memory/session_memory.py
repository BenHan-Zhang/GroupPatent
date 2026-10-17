"""智能体会话记忆：单任务临时存储"""

from typing import Any, List, Dict
from loguru import logger


class SessionMemory:
    """智能体会话记忆：单任务临时存储，任务结束后归档/销毁"""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def write(self, key: str, value: Any):
        """写入会话记忆"""
        self.history.append({"key": key, "value": value})
    
    def read(self, key: str) -> Any:
        """读取会话记忆：返回最后一次写入的值"""
        for item in reversed(self.history):
            if item["key"] == key:
                return item["value"]
        return None
    
    def clear(self):
        """清空会话记忆"""
        self.history = []
        logger.debug("会话记忆已清空")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整历史记录"""
        return self.history.copy()