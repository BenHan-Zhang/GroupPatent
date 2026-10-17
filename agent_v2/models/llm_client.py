from abc import ABC, abstractmethod
from typing import Optional, Callable


class LLMClient(ABC):
    """LLM客户端抽象基类：所有具体模型客户端必须继承此类并实现generate方法"""

    @abstractmethod
    def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            on_token: Callable[[str], None] = lambda x: None
    ) -> str:
        """
        调用LLM生成文本
        :param prompt: 输入提示词
        :param max_tokens: 最大生成token数
        :param temperature: 采样温度
        :param on_token: 每个生成的token调用一次，用于实时处理生成的token。可为空。输入 '\0' 表示调用失败，需要清空输出缓冲区
        :return: 生成的文本
        """
        pass
