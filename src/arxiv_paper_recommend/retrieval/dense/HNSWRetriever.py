"""基于 FAISS HNSW 的稠密检索；调用方按类型注解提供输入。"""

from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect
from arxiv_paper_recommend.config import INDEX_PATH, DATABASE_PATH
from arxiv_paper_recommend.retrieval import SearchResult
from pathlib import Path
import faiss
import numpy as np
from numpy.typing import ArrayLike

default_result_limit = 10


class HNSWRetriever:
    """通过 search 检索论文，复用已加载的 HNSW 索引和只读数据库连接。"""

    def __init__(self, index_path: str | Path = INDEX_PATH, database_path: str | Path = DATABASE_PATH) -> None:
        """加载向量索引并建立只读连接；索引或数据库不存在时立即失败。"""
        self.index = faiss.read_index(str(index_path))
        database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        self.conn = connect(database_uri, uri=True)

    def search(self, query_vector: ArrayLike, limit: int=default_result_limit) -> list[SearchResult]:
        """按单个查询向量检索，返回最多 limit 条论文记录。

        查询向量需为一维数组，维度与索引一致；转换为 float32 并归一化后检索。
        返回 SearchResult 列表，包含 id、doc_id、title、abstract 和 score，
        保留 FAISS 返回的相似度顺序；使用内积索引时，分数越大表示相关性越高。
        空索引返回空列表，无效 ID 和数据库中不存在的文档不计入结果。
        结果数量非正数或查询向量形状不符时抛出 ValueError。
        """
        if limit <= 0:
            raise ValueError("Result limit must be positive.")

        # normalize_L2 原地修改数组；复制以免改变调用方的查询向量。
        query = np.array(query_vector, dtype=np.float32, order="C", copy=True)
        if query.ndim != 1 or query.size != self.index.d:
            raise ValueError(f"Query vector must have shape ({self.index.d},).")
        query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        limit = min(limit, self.index.ntotal)
        if limit == 0:
            return []

        scores, ids = self.index.search(query, limit)
        document_ids = ids[0].tolist()
        documents = {
            document[0]: document
            for document in self._retrieve_documents(document_ids)
        }
        # SQL 的 IN 查询不保证顺序，使用 FAISS 的 ID 和分数顺序组装结果。
        return [
            SearchResult(*documents[document_id], score=score)
            for document_id, score in zip(document_ids, scores[0].tolist())
            if document_id in documents
        ]

    def _retrieve_documents(self, ids: list[int]) -> list[tuple[int, str, str, str | None]]:
        """按内部文档 ID 批量读取论文记录。

        此方法不对外开放，调用方负责提供符合内部约定的整数 ID 列表。
        过滤 FAISS 的无效 ID 并去重，按 SQLite 参数上限分批查询。
        返回 (id, doc_id, title, abstract) 元组列表，不保证与输入 ID 顺序一致。
        """
        document_ids = list(dict.fromkeys(
            document_id for document_id in ids if document_id >= 0
        ))
        if not document_ids:
            return []

        batch_size = self.conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)
        documents = []
        for start in range(0, len(document_ids), batch_size):
            batch = document_ids[start:start + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self.conn.execute(f"""
                SELECT id, doc_id, title, abstract
                FROM documents
                WHERE id IN ({placeholders})
            """, batch).fetchall()
            documents.extend(rows)
        return documents
