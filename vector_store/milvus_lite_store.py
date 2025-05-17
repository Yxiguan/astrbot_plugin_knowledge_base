# -*- coding: utf-8 -*-
# /vector_store/milvus_lite_store.py
from typing import List, Optional, Tuple
from .base import VectorDBBase, Document
from astrbot.api import logger
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema


class MilvusLiteStore(VectorDBBase):
    def __init__(self, embedding_util, dimension: int, data_path: str, **kwargs):
        super().__init__(embedding_util, dimension, data_path)
        self.client: Optional[MilvusClient] = None
        self.metric_type = kwargs.get("metric_type", "L2")
        # 定义 VARCHAR 字段的默认最大长度
        self.varchar_max_length = kwargs.get("varchar_max_length", 65535)
        self.pk_max_length = kwargs.get("pk_max_length", 36)

    async def initialize(self):
        logger.info(f"初始化 Milvus Lite 客户端，数据文件: {self.data_path}...")
        try:
            self.client = MilvusClient(uri=self.data_path)
            logger.info(f"Milvus Lite 客户端初始化成功，使用数据文件: {self.data_path}")
        except Exception as e:
            logger.error(f"Milvus Lite 客户端初始化失败: {e}")
            self.client = None
            raise

    async def _ensure_index_and_load(self, collection_name: str):
        """辅助函数：确保指定集合的 embedding 字段有索引并已加载"""
        if not self.client:
            raise ConnectionError("Milvus Lite client 未初始化。")

        try:
            # 1. 检查索引是否存在
            indexes = self.client.list_indexes(collection_name)
            has_embedding_index = any(
                idx.get("field_name") == "embedding"
                for idx in indexes
                if isinstance(idx, dict)
            )

            if not has_embedding_index:
                logger.warning(
                    f"集合 '{collection_name}' 的 embedding 字段上没有索引。将尝试创建 AUTOINDEX。"
                )
                # 为 embedding 字段创建索引
                # 使用 client.prepare_index_params() 来构建 IndexParams 对象
                # 然后传递给 client.create_index()
                index_to_create = self.client.prepare_index_params(
                    field_name="embedding",
                    index_type="AUTOINDEX",  # 或者 "FLAT"
                    metric_type=self.metric_type,
                    # params={} # AUTOINDEX 和 FLAT 通常不需要额外参数
                )
                self.client.create_index(
                    collection_name=collection_name,
                    index_params=index_to_create,  # 传递 IndexParams 对象
                )
                logger.info(
                    f"为集合 '{collection_name}' 的 embedding 字段创建 AUTOINDEX 索引成功。"
                )
                indexes = self.client.list_indexes(collection_name)  # 重新检查
                logger.info(f"创建后，集合 '{collection_name}' 的索引: {indexes}")
            else:
                logger.info(f"集合 '{collection_name}' 的 embedding 字段上已存在索引。")

            # 2. 确保集合已加载
            self.client.load_collection(collection_name)
            logger.info(f"Milvus Lite 集合 '{collection_name}' 已尝试加载。")

        except Exception as e:
            logger.error(
                f"检查/创建索引或加载集合 '{collection_name}' 失败: {e}", exc_info=True
            )
            raise

    async def create_collection(self, collection_name: str):
        if not self.client:
            raise ConnectionError("Milvus Lite client 未初始化。")

        if await self.collection_exists(collection_name):
            logger.info(
                f"Milvus Lite 集合 '{collection_name}' 已存在。将检查索引并加载。"
            )
            await self._ensure_index_and_load(collection_name)
            return

        try:
            pk_field = FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=self.pk_max_length,
            )
            vector_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            )
            text_content_field = FieldSchema(
                name="text_content",
                dtype=DataType.VARCHAR,
                max_length=self.varchar_max_length,
            )
            schema = CollectionSchema(
                fields=[pk_field, vector_field, text_content_field],
                description=f"Knowledge base collection: {collection_name}",
                enable_dynamic_field=True,
            )

            index_for_creation = self.client.prepare_index_params(
                field_name="embedding",  # 必须与向量字段名一致
                index_type="AUTOINDEX",  # 或者 "FLAT"
                metric_type=self.metric_type,
                # params={} # AUTOINDEX 和 FLAT 通常不需要参数，如果需要，在这里指定
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                metric_type=self.metric_type,  # 集合级别的 metric
                index_params=index_for_creation,  # 传递单个 IndexParams 对象
            )
            logger.info(
                f"Milvus Lite 集合 '{collection_name}' 使用详细 Schema 和内联索引创建成功。"
            )

            # 创建后，调用 _ensure_index_and_load 以进行加载和最终确认
            await self._ensure_index_and_load(collection_name)

        except Exception as e:
            logger.error(
                f"创建 Milvus Lite 集合 '{collection_name}' 失败: {e}", exc_info=True
            )
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            return False
        try:
            return self.client.has_collection(collection_name=collection_name)
        except Exception as e:
            logger.error(
                f"检查 Milvus Lite 集合 '{collection_name}' 是否存在时出错: {e}"
            )
            return False

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not self.client:
            raise ConnectionError("Milvus Lite client 未初始化。")

        # 确保集合存在、有索引并已加载
        if not await self.collection_exists(collection_name):
            logger.warning(
                f"Milvus Lite 集合 '{collection_name}' 不存在。将尝试自动创建。"
            )
            await self.create_collection(
                collection_name
            )  # create_collection 内部会调用 _ensure_index_and_load
        else:
            # 如果集合已存在，也确保索引和加载
            await self._ensure_index_and_load(collection_name)

        data_to_insert = []

        texts_to_embed = [
            doc.text_content for doc in documents if doc.embedding is None
        ]
        embeddings_list = []
        if texts_to_embed:
            embeddings_list = await self.embedding_util.get_embeddings(texts_to_embed)

        embed_idx = 0
        for doc in documents:
            embedding_to_use = None
            if doc.embedding is not None:
                embedding_to_use = doc.embedding
            elif (
                embed_idx < len(embeddings_list)
                and embeddings_list[embed_idx] is not None
            ):
                embedding_to_use = embeddings_list[embed_idx]
                embed_idx += 1

            if embedding_to_use:
                entity = {
                    "embedding": embedding_to_use,  # 对应 schema 中的 'embedding' 字段
                    "text_content": doc.text_content,  # 对应 schema 中的 'text_content' 字段
                }
                # 处理动态元数据字段
                for key, value in doc.metadata.items():
                    if key not in entity:  # 避免覆盖固定字段
                        if (
                            isinstance(value, str)
                            and len(value) > self.varchar_max_length
                        ):
                            logger.warning(
                                f"元数据字段 '{key}' 的值过长 ({len(value)} > {self.varchar_max_length})，将被截断。"
                            )
                            entity[key] = value[: self.varchar_max_length]
                        else:
                            entity[key] = value
                data_to_insert.append(entity)
            else:
                logger.warning(
                    f"未能为文档 '{doc.text_content[:50]}...' 获取 embedding，将跳过。"
                )

        if not data_to_insert:
            logger.info("没有有效的实体可供插入。")
            return []

        try:
            insert_result = self.client.insert(
                collection_name=collection_name, data=data_to_insert
            )
            logger.info(
                f"向 Milvus Lite 集合 '{collection_name}' 添加了 {len(insert_result['ids'])} 个文档。"
            )
            return [str(pk) for pk in insert_result["ids"]]
        except Exception as e:
            logger.error(
                f"向 Milvus Lite 集合 '{collection_name}' 插入数据失败: {e}",
                exc_info=True,
            )
            return []

    async def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if not self.client:
            raise ConnectionError("Milvus Lite client 未初始化。")
        if not await self.collection_exists(collection_name):
            logger.warning(f"Milvus Lite 集合 '{collection_name}' 不存在。")
            return []

        num_docs = await self.count_documents(collection_name)
        if num_docs == 0:
            logger.info(f"Milvus Lite 集合 '{collection_name}' 为空，无法搜索。")
            return []

        query_embedding = await self.embedding_util.get_embedding(query_text)
        if query_embedding is None:
            logger.error("无法为查询文本生成 embedding。")
            return []

        # 动态获取可检索的字段列表
        fields_to_request = []
        try:
            collection_info = self.client.describe_collection(
                collection_name
            )  # 返回的是 dict
            # 正确从字典中访问 'fields'键，它是一个列表
            schema_fields = collection_info.get("fields", [])

            # MilvusClient.search 默认返回 id 和 distance/score.
            # output_fields 用于指定额外的字段。
            for field_dict in schema_fields:
                field_name = field_dict.get("name")
                field_type = field_dict.get("type")  # 或者 field_dict.get('dtype')
                is_primary = field_dict.get("is_primary", False)

                # 我们需要 text_content 和其他元数据。
                # 排除主键和向量字段
                # DataType.FLOAT_VECTOR 对应的值可能是字符串 "FloatVector"
                if (
                    field_name
                    and field_name not in ["pk", "embedding"]
                    and not is_primary
                    and (
                        isinstance(field_type, DataType)
                        and field_type != DataType.FLOAT_VECTOR
                        or isinstance(field_type, str)
                        and "vector" not in field_type.lower()
                    )
                ):  # 简单判断
                    fields_to_request.append(field_name)

            # 确保 text_content 在其中 (如果它作为固定字段存在)
            # 由于我们在 create_collection 中显式定义了 text_content，它应该是固定字段
            if "text_content" not in fields_to_request and any(
                f.get("name") == "text_content" for f in schema_fields
            ):
                fields_to_request.append("text_content")

        except Exception as e_desc:
            logger.warning(
                f"获取集合 '{collection_name}' schema 失败: {e_desc}。将默认请求 'text_content'。"
            )
            fields_to_request = ["text_content"]  # 回退

        if not fields_to_request:
            # 如果 text_content 存在于 schema 中但未被加入
            if any(
                f.get("name") == "text_content"
                for f in collection_info.get("fields", [])
            ):
                fields_to_request.append("text_content")
            else:  # 极端情况，如果 text_content 字段都不确定
                logger.warning(
                    f"无法确定集合 '{collection_name}' 中的输出字段，搜索可能不返回元数据。将尝试不指定 output_fields。"
                )
                # fields_to_request = None # 或者保持为空列表，让 search 决定

        try:
            search_results_raw = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                anns_field="embedding",  # 向量字段名
                filter=filter_expr if filter_expr else "",
                limit=min(top_k, num_docs)
                if num_docs > 0
                else top_k,  # 确保 limit 不超过文档数
                output_fields=fields_to_request
                if fields_to_request
                else None,  # 如果列表为空，则不指定，让其返回默认
            )

            processed_results = []
            if search_results_raw and search_results_raw[0]:
                for hit in search_results_raw[0]:
                    doc_id = str(
                        hit.get("id")
                    )  # 'id' 是 MilvusClient.search 返回的主键字段名
                    distance = hit.get("distance")
                    similarity_score: float
                    if self.metric_type == "IP":  # Inner Product
                        similarity_score = float(distance)
                    else:  # L2 or other distance-based
                        similarity_score = 1.0 / (1.0 + float(distance))  # 简单转换

                    text_content = hit.get("text_content", "")
                    # 将 hit 中除了 id, distance, embedding 之外的都作为 metadata
                    metadata = {
                        k: v
                        for k, v in hit.items()
                        if k not in ["id", "distance", "embedding", "score"]
                    }
                    if (
                        "text_content" in metadata and not text_content
                    ):  # 如果 text_content 在 metadata 里但外面没取到
                        text_content = metadata.pop("text_content")

                    doc = Document(
                        id=doc_id, text_content=text_content, metadata=metadata
                    )
                    processed_results.append((doc, similarity_score))
            return processed_results
        except Exception as e:
            logger.error(
                f"在 Milvus Lite 集合 '{collection_name}' 中搜索失败: {e}",
                exc_info=True,
            )
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        if not self.client:
            raise ConnectionError("Milvus Lite client 未初始化。")
        if not await self.collection_exists(collection_name):
            logger.info(f"Milvus Lite 集合 '{collection_name}' 不存在，无需删除。")
            return False
        try:
            self.client.drop_collection(collection_name=collection_name)
            logger.info(f"Milvus Lite 集合 '{collection_name}' 已删除。")
            return True
        except Exception as e:
            logger.error(f"删除 Milvus Lite 集合 '{collection_name}' 失败: {e}")
            return False

    async def list_collections(self) -> List[str]:
        if not self.client:
            return []
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"列出 Milvus Lite 集合失败: {e}")
            return []

    async def count_documents(self, collection_name: str) -> int:
        if not self.client:
            return 0
        if not await self.collection_exists(collection_name):
            return 0
        try:
            stats = self.client.get_collection_stats(collection_name=collection_name)
            return int(stats.get("row_count", 0))
        except Exception as e:
            logger.warning(
                f"获取 Milvus Lite 集合 '{collection_name}' 文档数 (get_collection_stats) 失败: {e}。尝试 query count(*)..."
            )
            try:
                query_res = self.client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["count(*)"],
                )
                if (
                    query_res
                    and isinstance(query_res, list)
                    and query_res[0]
                    and "count(*)" in query_res[0]
                ):
                    return int(query_res[0]["count(*)"])
                else:  # 如果 count(*) 结果格式不对，尝试查询所有 pk
                    logger.warning(
                        f"count(*) for collection '{collection_name}' returned unexpected result: {query_res}. Trying to count by querying all pks."
                    )
                    all_pks_res = self.client.query(
                        collection_name=collection_name, filter="", output_fields=["pk"]
                    )
                    return len(all_pks_res) if all_pks_res else 0

            except Exception as query_err:
                logger.error(
                    f"通过 query 获取 Milvus Lite 集合 '{collection_name}' 文档数也失败: {query_err}"
                )
                return 0

    async def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("Milvus Lite 客户端已关闭。")
            except Exception as e:
                logger.error(f"关闭 Milvus Lite 客户端失败: {e}")
        self.client = None
