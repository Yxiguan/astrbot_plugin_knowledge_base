# astrbot_plugin_knowledge_base/main.py
import os
import asyncio
from typing import Optional, Union

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import (
    session_waiter,
    SessionController,
)
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import StarTools


from .core import constants
from .utils.installation import ensure_vector_db_dependencies
from .utils.embedding import EmbeddingUtil
from .utils.text_splitter import TextSplitterUtil
from .utils.file_parser import FileParser
from .vector_store.base import VectorDBBase
from .vector_store.faiss_store import FaissStore
from .vector_store.milvus_lite_store import MilvusLiteStore
from .vector_store.milvus_store import MilvusStore
from .core.user_prefs_handler import UserPrefsHandler
from .core.llm_enhancer import clean_contexts_from_kb_content, enhance_request_with_kb
from .commands import (
    general_commands,
    add_commands,
    search_commands,
    manage_commands,
)


@register(
    constants.PLUGIN_REGISTER_NAME,
    "lxfight",
    "一个支持多种向量数据库的知识库插件",
    "0.4.0",
    "https://github.com/lxfight/astrbot_plugin_knowledge_base",
)
class KnowledgeBasePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._initialize_basic_paths()

        self.vector_db: Optional[VectorDBBase] = None
        self.embedding_util: Optional[Union[EmbeddingUtil, Star]] = None
        self.text_splitter: Optional[TextSplitterUtil] = None
        self.file_parser: Optional[FileParser] = None
        self.user_prefs_handler: Optional[UserPrefsHandler] = None

        ensure_vector_db_dependencies(self.config.get("vector_db_type", "faiss"))
        self.init_task = asyncio.create_task(self._initialize_components())

    def _initialize_basic_paths(self):
        self.plugin_name_for_path = constants.PLUGIN_REGISTER_NAME
        self.persistent_data_root_path = StarTools.get_data_dir(
            self.plugin_name_for_path
        )
        os.makedirs(self.persistent_data_root_path, exist_ok=True)
        logger.info(f"知识库插件的持久化数据目录: {self.persistent_data_root_path}")
        self.user_prefs_path = os.path.join(
            self.persistent_data_root_path, "user_collection_prefs.json"
        )

    async def _initialize_components(self):
        try:
            logger.info("知识库插件开始初始化...")
            # Embedding Util
            try:
                embedding_plugin = self.context.get_registered_star(
                    "astrbot_plugin_embedding_adapter"
                )
                if embedding_plugin:
                    self.embedding_util = embedding_plugin.star_cls
                    dim = self.embedding_util.get_dim()
                    model_name = self.embedding_util.get_model_name()
                    if dim is not None and model_name is not None:
                        self.config["embedding_dimension"] = dim
                        self.config["embedding_model_name"] = model_name
                    logger.info("成功加载并使用 astrbot_plugin_embedding_adapter。")
            except Exception as e:
                logger.warning(f"嵌入服务适配器插件加载失败: {e}", exc_info=True)
                self.embedding_util = None  # Fallback

            if self.embedding_util is None:  # If adapter failed or not found
                self.embedding_util = EmbeddingUtil(
                    api_url=self.config.get("embedding_api_url"),
                    api_key=self.config.get("embedding_api_key"),
                    model_name=self.config.get("embedding_model_name"),
                )
            logger.info("Embedding 工具初始化完成。")

            # Text Splitter
            self.text_splitter = TextSplitterUtil(
                chunk_size=self.config.get("text_chunk_size"),
                chunk_overlap=self.config.get("text_chunk_overlap"),
            )
            logger.info("文本分割工具初始化完成。")

            # File Parser
            self.file_parser = FileParser(context=self.context)
            logger.info("文件解析器初始化完成。")

            # Vector DB
            db_type = self.config.get("vector_db_type", "faiss")
            dimension = self.config.get("embedding_dimension", 1024)

            if db_type == "faiss":
                faiss_subpath = self.config.get("faiss_db_subpath", "faiss_data")
                faiss_full_path = os.path.join(
                    self.persistent_data_root_path, faiss_subpath
                )
                self.vector_db = FaissStore(
                    self.embedding_util, dimension, faiss_full_path
                )
            elif db_type == "milvus_lite":
                milvus_lite_subpath = self.config.get(
                    "milvus_lite_db_subpath", "milvus_lite_data/milvus_lite.db"
                )
                milvus_lite_full_path = os.path.join(
                    self.persistent_data_root_path, milvus_lite_subpath
                )
                os.makedirs(os.path.dirname(milvus_lite_full_path), exist_ok=True)
                self.vector_db = MilvusLiteStore(
                    self.embedding_util, dimension, milvus_lite_full_path
                )
            elif db_type == "milvus":
                self.vector_db = MilvusStore(
                    self.embedding_util,
                    dimension,
                    data_path="",
                    host=self.config.get("milvus_host"),
                    port=self.config.get("milvus_port"),
                    user=self.config.get("milvus_user"),
                    password=self.config.get("milvus_password"),
                )
            else:
                logger.error(f"不支持的向量数据库类型: {db_type}，请检查配置。")
                return

            if self.vector_db:
                await self.vector_db.initialize()
                logger.info(f"向量数据库 '{db_type}' 初始化完成。")

            self.user_prefs_handler = UserPrefsHandler(
                self.user_prefs_path, self.vector_db, self.config
            )
            await self.user_prefs_handler.load_user_preferences()

            logger.info("知识库插件初始化成功。")

        except Exception as e:
            print("出现问题")
            logger.error(f"知识库插件初始化失败: {e}", exc_info=True)
            self.vector_db = None

    async def _ensure_initialized(self) -> bool:
        if self.init_task and not self.init_task.done():
            await self.init_task
        if (
            not self.vector_db
            or not self.embedding_util
            or not self.text_splitter
            or not self.user_prefs_handler
        ):
            logger.error("知识库插件未正确初始化，请检查日志和配置。")
            return False
        return True

    async def _load_user_preferences(self):
        try:
            if os.path.exists(self.user_prefs_path):
                with open(self.user_prefs_path, "r", encoding="utf-8") as f:
                    self.user_collection_preferences = json.load(f)
                logger.info(f"从 {self.user_prefs_path} 加载了用户知识库偏好。")
            else:
                logger.info(
                    f"用户知识库偏好文件 {self.user_prefs_path} 未找到，将使用默认值。"
                )
        except Exception as e:
            logger.error(f"加载用户知识库偏好失败: {e}")
            self.user_collection_preferences = {}

    async def _save_user_preferences(self):
        try:
            with open(self.user_prefs_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.user_collection_preferences, f, ensure_ascii=False, indent=4
                )
            logger.info(f"用户知识库偏好已保存到 {self.user_prefs_path}。")
        except Exception as e:
            logger.error(f"保存用户知识库偏好失败: {e}")

    def _get_user_default_collection(self, event: AstrMessageEvent) -> str:
        """获取用户/群组的默认知识库，如果没有设置则返回全局默认"""
        user_key = event.unified_msg_origin  # 使用 unified_msg_origin 作为唯一标识
        return self.user_collection_preferences.get(
            user_key, self.config.get("default_collection_name", "general")
        )

    async def _set_user_default_collection(
        self, event: AstrMessageEvent, collection_name: str
    ):
        """设置用户/群组的默认知识库"""
        if not await self.vector_db.collection_exists(collection_name):
            # 如果配置了自动创建，并且集合不存在
            if self.config.get("auto_create_collection", True):
                try:
                    await self.vector_db.create_collection(collection_name)
                    logger.info(f"自动创建知识库 '{collection_name}' 成功。")
                except Exception as e:
                    logger.error(f"自动创建知识库 '{collection_name}' 失败: {e}")
                    yield event.plain_result(
                        f"自动创建知识库 '{collection_name}' 失败: {e}"
                    )
                    return
            else:
                yield event.plain_result(
                    f"知识库 '{collection_name}' 不存在，且未配置自动创建。"
                )
                return

        user_key = event.unified_msg_origin
        self.user_collection_preferences[user_key] = collection_name
        await self._save_user_preferences()
        yield event.plain_result(f"当前会话默认知识库已设置为: {collection_name}")

    # --- 辅助函数：下载文件 ---
    async def _download_file(self, url: str, destination_folder: str) -> Optional[str]:
        """
        异步下载文件到指定文件夹。
        返回下载后的文件路径，如果失败则返回 None。
        """
        try:
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True
            ) as client:  # 增加超时和重定向
                response = await client.get(url)
                response.raise_for_status()  # 检查 HTTP 错误

                # 从 URL 获取文件名，或生成一个
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename:  # 如果路径为空，例如 "http://example.com/"
                    # 尝试从 Content-Disposition header 获取文件名
                    content_disposition = response.headers.get("Content-Disposition")
                    if content_disposition:
                        import re

                        match = re.search(r'filename="?([^"]+)"?', content_disposition)
                        if match:
                            filename = match.group(1)
                    if not filename:  # 仍然没有文件名，生成一个
                        filename = (
                            f"downloaded_file_{tempfile._RandomNameSequence().next()}"
                        )

                # 限制文件名，防止路径遍历等问题
                filename = "".join(
                    c for c in filename if c.isalnum() or c in [".", "_", "-"]
                ).strip()
                if not filename:
                    filename = "untitled_download"  # 最终回退

                # 简单的文件大小限制 (例如 50MB)
                max_size = 50 * 1024 * 1024
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_size:
                    logger.error(
                        f"文件下载失败：文件过大 ({int(content_length) / (1024 * 1024):.2f} MB > {max_size / (1024 * 1024)} MB)。URL: {url}"
                    )
                    return None

                # 简单的文件类型嗅探或基于扩展名过滤
                _, extension = os.path.splitext(filename)
                allowed_extensions = [
                    ".txt",
                    ".md",
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
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".mp3",
                    ".wav",
                ]
                if extension.lower() not in allowed_extensions:
                    logger.error(
                        f"文件下载失败：不支持的文件类型 '{extension}'. URL: {url}"
                    )
                    return None

                temp_file_path = os.path.join(destination_folder, filename)

                # 分块写入，处理大文件
                with open(temp_file_path, "wb") as f:
                    downloaded_size = 0
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if (
                            downloaded_size > max_size
                        ):  # 再次检查，以防 Content-Length 未提供或不准确
                            f.close()
                            os.remove(temp_file_path)  # 删除不完整的文件
                            logger.error(
                                f"文件下载失败：文件在下载过程中超出大小限制。URL: {url}"
                            )
                            return None

                logger.info(f"文件已成功下载到: {temp_file_path} 从 URL: {url}")
                return temp_file_path
        except httpx.HTTPStatusError as e:
            logger.error(
                f"文件下载 HTTP 错误: {e.response.status_code} - {e.response.text}. URL: {url}"
            )
            return None
        except Exception as e:
            logger.error(f"文件下载失败: {e}. URL: {url}", exc_info=True)
            return None

    def _clean_contexts_from_kb_content(self, req: ProviderRequest):
        """
        自动删除 req.contexts 里面由知识库补充的历史对话内容。
        使用 ###KBDATA_START### 和 ###KBDATA_END### 标记进行识别。
        """
        if not req.contexts:
            return

        cleaned_contexts = []
        initial_context_count = len(req.contexts)

        for message in req.contexts:
            role = message.get("role")
            content = message.get("content", "")

            # 1. 清理作为 system 消息插入的知识库内容
            # 如果 system 消息包含知识库开始标记，则认为它是知识库注入的消息，直接删除
            if role == "system" and KB_START_MARKER in content:
                logger.debug(
                    f"从历史对话中检测到并删除知识库 system 消息 (通过标记识别): {content[:100]}..."
                )
                continue  # 不将此消息添加到 cleaned_contexts，实现删除

            # 2. 清理作为 user 消息插入的知识库内容 (当使用 prepend_prompt 方法时)
            # 这种情况下，知识库内容会被包裹在标记中，并拼接在用户原始问题之前
            elif role == "user" and KB_START_MARKER in content:
                start_marker_idx = content.find(KB_START_MARKER)
                end_marker_idx = content.find(KB_END_MARKER, start_marker_idx)

                if start_marker_idx != -1 and end_marker_idx != -1:
                    # 知识库内容结束后，后面跟着的是 "用户的原始问题是：" 分隔符和原始用户问题
                    original_prompt_delimiter_idx = content.find(
                        USER_PROMPT_DELIMITER_IN_HISTORY,
                        end_marker_idx + len(KB_END_MARKER),
                    )

                    if original_prompt_delimiter_idx != -1:
                        # 提取原始用户问题部分
                        original_user_prompt = content[
                            original_prompt_delimiter_idx
                            + len(USER_PROMPT_DELIMITER_IN_HISTORY) :
                        ].strip()
                        message["content"] = (
                            original_user_prompt  # 更新消息内容为原始用户问题
                        )
                        cleaned_contexts.append(message)  # 将清理后的消息添加到列表
                        logger.debug(
                            f"从历史对话 user 消息中清理知识库标记和内容，保留原用户问题: {original_user_prompt[:100]}..."
                        )
                    else:
                        # 理论上不会发生，如果原始问题分隔符丢失，则删除该消息
                        logger.warning(
                            f"用户消息中检测到知识库标记但缺少原始用户问题分隔符，删除该消息: {content[:100]}..."
                        )
                        continue
                else:
                    # 如果找到了开始标记但没有找到结束标记，说明消息不完整或格式错误，也删除
                    logger.warning(
                        f"用户消息中检测到知识库起始标记但缺少结束标记，删除该消息: {content[:100]}..."
                    )
                    continue
            else:
                # 保留所有其他消息（例如：助手回复、工具调用结果、以及未被知识库修改的用户/系统消息）
                cleaned_contexts.append(message)

        req.contexts = cleaned_contexts
        if len(req.contexts) < initial_context_count:
            logger.info(
                f"成功从历史对话中删除了 {initial_context_count - len(req.contexts)} 条知识库补充消息。"
            )

    @filter.on_llm_request()
    async def kb_on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not await self._ensure_initialized():
            logger.warning("LLM 请求时知识库插件未初始化，跳过知识库增强。")
            return

        clean_contexts_from_kb_content(req)

        await enhance_request_with_kb(
            event, req, self.vector_db, self.user_prefs_handler, self.config
        )

    # --- Command Groups & Commands ---
    @filter.command_group("kb", alias={"knowledge", "知识库"})
    def kb_group(self):
        """知识库管理指令集"""
        pass

    @kb_group.command("help", alias={"帮助"})
    async def kb_help(self, event: AstrMessageEvent):
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_help(self, event):
            yield result

    @kb_group.group("add")
    def kb_add_group(self, event: AstrMessageEvent):
        """添加内容到知识库的子指令组"""
        pass

    @kb_add_group.command("text")
    async def kb_add_text(
        self,
        event: AstrMessageEvent,
        content: str,
        collection_name: Optional[str] = None,
    ):
        """添加文本内容到知识库。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in add_commands.handle_add_text(
            self, event, content, collection_name
        ):
            yield result

    @kb_add_group.command("file")
    async def kb_add_file(
        self,
        event: AstrMessageEvent,
        path_or_url: str,
        collection_name: Optional[str] = None,
    ):
        """从本地路径或 URL 添加文件内容到知识库。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in add_commands.handle_add_file(
            self, event, path_or_url, collection_name
        ):
            yield result

    @kb_group.command("search", alias={"搜索", "find", "查找"})
    async def kb_search(
        self,
        event: AstrMessageEvent,
        query: str,
        collection_name: Optional[str] = None,
        top_k_str: Optional[str] = None,
    ):
        """在知识库中搜索内容。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in search_commands.handle_search(
            self, event, query, collection_name, top_k_str
        ):
            yield result

    @kb_group.command("list", alias={"列表", "showall"})
    async def kb_list_collections(self, event: AstrMessageEvent):
        """列出所有可用的知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_list_collections(self, event):
            yield result

    @kb_group.command("current", alias={"当前"})
    async def kb_current_collection(self, event: AstrMessageEvent):
        """查看当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_current_collection(self, event):
            yield result

    @kb_group.command("use", alias={"使用", "set"})
    async def kb_use_collection(self, event: AstrMessageEvent, collection_name: str):
        """设置当前会话的默认知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in general_commands.handle_kb_use_collection(
            self, event, collection_name
        ):
            yield result

    @kb_group.command("create", alias={"创建"})
    async def kb_create_collection(self, event: AstrMessageEvent, collection_name: str):
        """创建一个新的知识库"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_create_collection(
            self, event, collection_name
        ):
            yield result

    @filter.permission_type(filter.PermissionType.ADMIN)
    @kb_group.command("delete", alias={"删除"})
    async def kb_delete_collection(self, event: AstrMessageEvent, collection_name: str):
        """删除一个知识库及其所有内容 (危险操作! 仅管理员)。"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return

        if not collection_name:
            yield event.plain_result(
                "请输入要删除的知识库名称。用法: /kb delete <知识库名>"
            )
            return

        if not await self.vector_db.collection_exists(collection_name):
            yield event.plain_result(f"知识库 '{collection_name}' 不存在。")
            return

        confirmation_phrase = f"确认删除{collection_name}"
        yield event.plain_result(
            f"警告：你确定要删除知识库 '{collection_name}' 及其所有内容吗？此操作不可恢复！\n"
            f"请在 60 秒内回复 '{confirmation_phrase}' 来执行。"
        )

        # The session_waiter needs to be defined within the scope where `self` (plugin instance)
        # and `collection_name` are accessible.
        # The actual logic will be in manage_commands.
        @session_waiter(timeout=60, record_history_chains=False)
        async def delete_confirmation_waiter(
            controller: SessionController, confirm_event: AstrMessageEvent
        ):
            user_input = confirm_event.message_str.strip()
            if user_input == confirmation_phrase:
                # Call the handler logic
                await manage_commands.handle_delete_collection_logic(
                    self, confirm_event, collection_name
                )
                controller.stop()
            elif user_input.lower() in ["取消", "cancel"]:
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"已取消删除知识库 '{collection_name}'。"
                    )
                )
                controller.stop()
            else:
                await confirm_event.send(
                    confirm_event.plain_result(
                        f"输入无效。如需删除，请回复 '{confirmation_phrase}'；如需取消，请回复 '取消'。"
                    )
                )
                controller.keep(timeout=60, reset_timeout=True)

        try:
            await delete_confirmation_waiter(event)
        except TimeoutError:
            yield event.plain_result(
                f"删除知识库 '{collection_name}' 操作超时，已自动取消。"
            )
        except Exception as e_sess:
            logger.error(f"删除知识库确认会话发生错误: {e_sess}", exc_info=True)
            yield event.plain_result(f"删除确认过程中发生错误: {e_sess}")
        finally:
            event.stop_event()

    @kb_group.command("count", alias={"数量"})
    async def kb_count_documents(
        self, event: AstrMessageEvent, collection_name: Optional[str] = None
    ):
        """查看指定知识库的文档数量"""
        if not await self._ensure_initialized():
            yield event.plain_result("知识库插件未初始化，请联系管理员。")
            return
        async for result in manage_commands.handle_count_documents(
            self, event, collection_name
        ):
            yield result

    # --- Termination ---
    async def terminate(self):
        logger.info("知识库插件正在终止...")
        if hasattr(self, "init_task") and self.init_task and not self.init_task.done():
            logger.info("等待初始化任务完成...")
            try:
                await asyncio.wait_for(self.init_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("初始化任务超时，尝试取消。")
                self.init_task.cancel()
            except Exception as e:
                logger.error(f"等待初始化任务完成时出错: {e}")

        if (
            self.embedding_util
            and hasattr(self.embedding_util, "close")
            and not isinstance(self.embedding_util, Star)
        ):
            await self.embedding_util.close()
            logger.info("Embedding 工具已关闭。")

        if self.vector_db:
            await self.vector_db.close()
            logger.info("向量数据库已关闭。")

        if self.user_prefs_handler:
            await self.user_prefs_handler.save_user_preferences()

        logger.info("知识库插件终止完成。")
