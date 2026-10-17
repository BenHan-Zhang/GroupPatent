"""文本解析工具：提取XML标签内容"""

import re
from typing import Optional


def extract_xml_tag(text: str, tag_name: str) -> Optional[str]:
    """
    从文本中提取指定XML标签的内容
    :param text: 包含XML标签的文本
    :param tag_name: 标签名称
    :return: 标签内容，如果未找到则返回None
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_all_xml_tags(text: str) -> dict:
    """
    提取文本中所有XML标签及其内容
    :param text: 包含XML标签的文本
    :return: 标签名到内容的字典
    """
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    return {tag_name: content.strip() for tag_name, content in matches}