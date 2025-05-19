import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple
from .base import VectorDBBase, Document
from astrbot.api import logger
import asyncio

# Faiss 是同步库，通过 asyncio.to_thread 包装其操作以适应异步环境


class FaissStore(VectorDBBase):
    def __init__(self, embedding_util, dimension: int, data_path: str):
        super().__init__(embedding_util, dimension, data_path)
        self.indexes: Dict[str, faiss.Index] = {}
        self.doc_storages: Dict[str, List[Document]] = {}  # 存储原始 Document 对象
        os.makedirs(self.data_path, exist_ok=True)

    async def initialize(self):
        logger.info("初始化 Faiss 存储...")
        await asyncio.to_thread(self._load_all_collections)
        logger.info(f"Faiss 存储初始化完成。已加载集合: {list(self.indexes.keys())}")

    def _load_collection(self, collection_name: str):
        index_path = os.path.join(self.data_path, f"{collection_name}.index")
        storage_path = os.path.join(self.data_path, f"{collection_name}.docs")

        if os.path.exists(index_path) and os.path.exists(storage_path):
            try:
                self.indexes[collection_name] = faiss.read_index(index_path)
                with open(storage_path, "rb") as f:
                    self.doc_storages[collection_name] = pickle.load(f)
                logger.info(f"成功加载 Faiss 集合: {collection_name}")
            except Exception as e:
                logger.error(f"加载 Faiss 集合 {collection_name} 失败: {e}")
                # 如果加载失败，确保清理不完整的状态
                if collection_name in self.indexes:
                    del self.indexes[collection_name]
                if collection_name in self.doc_storages:
                    del self.doc_storages[collection_name]
        else:
            logger.info(f"Faiss 集合 {collection_name} 的文件不存在，将不会加载。")

    def _load_all_collections(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith(".index"):
                collection_name = filename[: -len(".index")]
                self._load_collection(collection_name)

    def _save_collection(self, collection_name: str):
        if collection_name in self.indexes and collection_name in self.doc_storages:
            index_path = os.path.join(self.data_path, f"{collection_name}.index")
            storage_path = os.path.join(self.data_path, f"{collection_name}.docs")
            try:
                faiss.write_index(self.indexes[collection_name], index_path)
                with open(storage_path, "wb") as f:
                    pickle.dump(self.doc_storages[collection_name], f)
                logger.info(f"成功保存 Faiss 集合: {collection_name}")
            except Exception as e:
                logger.error(f"保存 Faiss 集合 {collection_name} 失败: {e}")

    async def create_collection(self, collection_name: str):
        if await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 已存在。")
            return

        def _create_sync():
            self.indexes[collection_name] = faiss.IndexFlatL2(self.dimension)  # L2 距离
            # self.indexes[collection_name] = faiss.IndexIDMap(index_flat) # 如果需要自定义ID
            self.doc_storages[collection_name] = []
            self._save_collection(collection_name)
            logger.info(f"Faiss 集合 '{collection_name}' 创建成功。")

        await asyncio.to_thread(_create_sync)

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.indexes

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not await self.collection_exists(collection_name):
            # raise ValueError(f"Faiss 集合 '{collection_name}' 不存在。请先创建。")
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。将尝试自动创建。")
            await self.create_collection(collection_name)

        texts_to_embed = [
            doc.text_content for doc in documents if doc.embedding is None
        ]
        embeddings_list = []
        if texts_to_embed:
            embeddings_list = await self.embedding_util.get_embeddings_async(texts_to_embed)

        valid_embeddings = []
        processed_documents = []
        doc_ids = []

        embed_idx = 0
        for doc in documents:
            if doc.embedding is None:
                if (
                    embed_idx < len(embeddings_list)
                    and embeddings_list[embed_idx] is not None
                ):
                    doc.embedding = embeddings_list[embed_idx]
                    valid_embeddings.append(doc.embedding)
                    processed_documents.append(doc)
                else:
                    logger.warning(
                        f"未能为文档 '{doc.text_content[:50]}...' 生成 embedding，将跳过。"
                    )
                embed_idx += 1
            else:  # 如果文档已包含 embedding
                valid_embeddings.append(doc.embedding)
                processed_documents.append(doc)

        if not valid_embeddings:
            logger.info("没有有效的 embedding 可供添加。")
            return []

        def _add_sync():
            index = self.indexes[collection_name]
            storage = self.doc_storages[collection_name]

            new_embeddings_np = np.array(valid_embeddings).astype("float32")
            faiss.normalize_L2(
                new_embeddings_np
            )  # Faiss 通常与归一化向量配合使用效果更好，尤其是内积索引

            start_id = index.ntotal
            index.add(new_embeddings_np)

            current_ids = []
            for i, doc in enumerate(processed_documents):
                doc_id = str(start_id + i)  # Faiss 内部使用从0开始的整数ID
                doc.id = doc_id  # 将内部ID存回Document对象
                storage.append(doc)
                current_ids.append(doc_id)

            self._save_collection(collection_name)
            return current_ids

        doc_ids = await asyncio.to_thread(_add_sync)
        logger.info(f"向 Faiss 集合 '{collection_name}' 添加了 {len(doc_ids)} 个文档。")
        return doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not await self.collection_exists(collection_name):
            logger.warning(f"Faiss 集合 '{collection_name}' 不存在。")
            return []

        query_embedding = await self.embedding_util.get_embedding_async(query_text)
        if query_embedding is None:
            logger.error("无法为查询文本生成 embedding。")
            return []

        def _search_sync():
            index = self.indexes[collection_name]
            storage = self.doc_storages[collection_name]

            if index.ntotal == 0:
                logger.info(f"Faiss 集合 '{collection_name}' 为空，无法搜索。")
                return []

            query_embedding_np = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_embedding_np)

            # 实际的 top_k 不应超过集合中的文档数
            actual_top_k = min(top_k, index.ntotal)
            if actual_top_k == 0:
                return []

            distances, indices = index.search(query_embedding_np, actual_top_k)

            results = []
            for i in range(len(indices[0])):
                doc_index = indices[0][i]
                dist = distances[0][i]
                if 0 <= doc_index < len(storage):  # 确保索引有效
                    # Faiss 返回的是距离 (L2 distance)，可以转换为相似度 (1 - distance/max_dist or 1 / (1 + distance))
                    # 对于归一化向量，L2距离的平方 D^2 = 2 - 2 * cos_sim。所以 cos_sim = 1 - D^2 / 2
                    # 或者简单地使用距离的倒数或负数作为排序依据，越小越好
                    similarity_score = 1.0 - (
                        dist / 2.0
                    )  # 估算余弦相似度，范围 [-1, 1]
                    results.append((storage[doc_index], float(similarity_score)))
                else:
                    logger.warning(f"搜索结果中的索引 {doc_index} 超出范围。")
            return results

        return await asyncio.to_thread(_search_sync)

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 不存在，无需删除。")
            return False

        def _delete_sync():
            del self.indexes[collection_name]
            del self.doc_storages[collection_name]

            index_path = os.path.join(self.data_path, f"{collection_name}.index")
            storage_path = os.path.join(self.data_path, f"{collection_name}.docs")

            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(storage_path):
                    os.remove(storage_path)
                logger.info(f"Faiss 集合 '{collection_name}' 已删除。")
                return True
            except Exception as e:
                logger.error(f"删除 Faiss 集合 '{collection_name}' 文件时出错: {e}")
                return False

        return await asyncio.to_thread(_delete_sync)

    async def list_collections(self) -> List[str]:
        return await asyncio.to_thread(lambda: list(self.indexes.keys()))

    async def count_documents(self, collection_name: str) -> int:
        if not await self.collection_exists(collection_name):
            return 0
        return await asyncio.to_thread(lambda: self.indexes[collection_name].ntotal)

    async def close(self):
        logger.info("关闭 Faiss 存储 (实际上是保存所有集合)...")
        # Faiss 不需要显式关闭连接，但我们可以在此保存所有更改
        # for collection_name in self.indexes.keys():
        #     await asyncio.to_thread(self._save_collection, collection_name)
        # 在每次操作后已经保存，这里可以什么都不做，或者做一个最终的健全性检查
        logger.info("Faiss 存储已处理关闭请求。")
