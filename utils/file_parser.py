from typing import Optional
from astrbot.api import logger
import os
import aiofiles
import asyncio
from markitdown import MarkItDown
from openai import AsyncOpenAI, OpenAI

class FileParser:
    def __init__(self, llm_api_url: str, llm_api_key: str, llm_model_name: str):
        # 初始化 MarkItDown

        self.api_key = llm_api_key
        self.api_url = llm_api_url
        self.model_name = llm_model_name
        if self.api_key is None or self.api_url is None or self.model_name is None:
            logger.warning("未配置 LLM API 密钥、基础地址和模型名称，图片解析可能失败")
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        self.sync_client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        self.md_converter = MarkItDown(enable_plugins=True, llm_client=self.async_client, llm_model=self.model_name)
        self.image_converter = MarkItDown(enable_plugins=True, llm_client=self.sync_client, llm_model=self.model_name)

        self.text_extensions = {".txt", ".md"}
        self.image_extensions = {".jpg", ".jpeg", ".png"}

        # MarkItDown 支持的文件扩展名
        self.markitdown_extensions = {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".json",
            ".xml",
            ".csv",
            ".epub",
            # ".mp3",
            # ".wav",
        }

    async def parse_file_content(self, file_path: str) -> Optional[str]:
        """
        异步读取并解析文件内容。

        Args:
            file_path: 文件路径。

        Returns:
            文件文本内容，如果解析失败则返回 None。
        """
        try:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # 处理普通文本文件
            if extension in self.text_extensions:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                return content

            # 使用 MarkItDown 处理其他支持的文件格式
            elif extension in self.image_extensions:
                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.image_converter.convert(file_path)
                    )
                    content = result.text_content
                    return content
                except Exception as e:
                    logger.error(f"图片转换失败 {file_path}: {e}")
                    return None
            elif extension in self.markitdown_extensions:
                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.md_converter.convert(file_path)
                    )
                    content = result.text_content
                    return content
                except Exception as e:
                    logger.error(f"MarkItDown 转换文件失败 {file_path}: {e}")
                    return None

            else:
                logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {e}")
            return None

    def get_supported_extensions(self) -> set:
        """
        获取所有支持的文件扩展名

        Returns:
            包含所有支持的文件扩展名的集合
        """
        return self.text_extensions.union(self.markitdown_extensions)
