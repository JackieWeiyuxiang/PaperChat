"""基于 SQLite FTS5 的稀疏检索；调用方按类型注解提供输入。"""

from sqlite3 import connect
from arxiv_paper_recommend.config import DATABASE_PATH
from arxiv_paper_recommend.retrieval import SearchResult
from itertools import combinations
from math import ceil, comb
from pathlib import Path
from re import findall
from types import TracebackType
from typing import Self
from unicodedata import category

# 关键词数量按清理、去重后的结果计算。
min_keyword_count = 1
max_keyword_count = 7
default_result_limit = 10
max_result_limit = 100
# 最终生成的筛选查询长度上限；评分查询不限制长度。
max_query_length = 1347
keyword_match_ratio = 0.75


class BM25Retriever:
    """通过 term_search 检索论文，支持 with 或使用结束后手动调用 close。"""

    def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
        """建立可供多次检索复用的只读连接；数据库不存在时立即失败。"""
        database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        self.conn = connect(database_uri, uri=True)

    def term_search(self, terms: list[str], limit: int=default_result_limit, keyword_match_ratio: float=keyword_match_ratio) -> list[SearchResult]:
        """按关键词或短语检索，返回最多 limit 条论文记录。

        统一连续空白并去重后需有 min_keyword_count～max_keyword_count 个不同的关键词；每个多词关键词按连续短语匹配。
        候选论文至少匹配 ceil(关键词数量 * keyword_match_ratio) 项，默认比例为 0.75。
        返回 SearchResult 列表，包含 id、doc_id、title、abstract 和 score，
        按 SQLite BM25 分数升序排列，分数越小表示相关性越高。
        关键词不含可检索字符、结果数量、关键词数量或匹配比例超出范围，
        以及筛选查询超过长度上限时抛出 ValueError。
        """
        if not is_legal_result_limit(limit):
            raise ValueError(f"Result limit must be between 1 and {max_result_limit}.")
        terms = _process_keywords(terms)

        if not _is_legal_keywords(terms):
            raise ValueError(f"Keyword count must be between {min_keyword_count} and {max_keyword_count}.")

        return self._query_search(_build_search_query(terms, keyword_match_ratio), limit, terms=terms)

    def _query_search(
        self,
        query: str,
        limit: int = default_result_limit,
        terms: list[str] | None = None,
    ) -> list[SearchResult]:
        """执行内部 FTS 查询；query 用于筛选，terms 优先用于构建评分查询。

        此方法不对外开放，调用方负责提供符合内部约定的查询和关键词。
        仅限制最终生成的筛选查询长度，评分查询不限制长度。
        """
        if not _is_legal_query(query):
            raise ValueError(f"Search query length must be at most {max_query_length} characters.")

        score_query = _build_score_query(terms, query)

        # 子查询按匹配门槛筛选候选；外层用独立的 OR 查询计算 BM25，
        # 避免组合展开中的重复关键词影响评分。
        rows = self.conn.execute("""
            SELECT documents_fts.rowid, doc_id, title, abstract,
                rank AS score
            FROM documents_fts
            JOIN (
                SELECT rowid
                FROM documents_fts
                WHERE documents_fts MATCH ?
            ) AS candidates
            ON candidates.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, score_query, limit)).fetchall()
        return [SearchResult(*row) for row in rows]


    def __enter__(self) -> Self:
        """进入上下文并返回当前检索器。"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文时关闭连接，不抑制异常。"""
        self.close()

    def close(self) -> None:
        """关闭连接；关闭后该实例不能继续执行检索。"""
        self.conn.close()


def is_legal_result_limit(limit: int) -> bool:
    """在调用方保证整数类型的前提下，检查结果数量是否在允许范围内。"""
    return 0 < limit <= max_result_limit

def _process_keywords(keywords: list[str]) -> list[str]:
    """统一空白、过滤空项，拒绝无可检索字符的项，转为小写并去重。"""
    processed_keywords = set()
    for keyword in keywords:
        keyword = " ".join(keyword.split()).lower()
        if not keyword:
            continue
        # 默认 unicode61 将 L*、N* 和 Co（私用字符）类别作为词元字符。
        if not any(
            char_category[0] in ("L", "N") or char_category == "Co"
            for char_category in map(category, keyword)
        ):
            raise ValueError("Each keyword must contain at least one searchable token character.")
        processed_keywords.add(keyword)
    return list(processed_keywords)

def _is_legal_keywords(keywords: list[str]) -> bool:
    """检查清理后的关键词数量。"""
    return min_keyword_count <= len(keywords) <= max_keyword_count

def _is_legal_query(query: str) -> bool:
    """检查最终生成的筛选查询是否未超过长度上限。"""
    return len(query) <= max_query_length

def _build_search_query(terms: list[str], keyword_match_ratio: float) -> str:
    """按比例构建候选筛选查询；在转义和组合展开前检查最终查询长度。"""
    if not (0 < keyword_match_ratio <= 1):
        raise ValueError("keyword_match_ratio must be in (0, 1].")

    required_match_count = ceil(len(terms) * keyword_match_ratio)

    # 每项长度包含外围引号及内嵌双引号转义，避免先构建超长查询。
    term_count = len(terms)
    quoted_terms_length = sum(len(term) + term.count('"') + 2 for term in terms)
    if required_match_count == term_count:
        query_length = quoted_terms_length + len(" AND ") * (term_count - 1)
    elif required_match_count == 1:
        query_length = quoted_terms_length + len(" OR ") * (term_count - 1)
    else:
        group_count = comb(term_count, required_match_count)
        # 每个关键词出现在 C(n-1, k-1) 个组合中；每组另有括号和 AND。
        query_length = (
            comb(term_count - 1, required_match_count - 1) * quoted_terms_length
            + group_count * (2 + len(" AND ") * (required_match_count - 1))
            + len(" OR ") * (group_count - 1)
        )
    if query_length > max_query_length:
        raise ValueError(f"Search query length must be at most {max_query_length} characters.")

    # FTS5 使用双引号表示短语，内嵌双引号需写成两个双引号。
    terms = ['"' + t.replace('"', '""') + '"' for t in terms]

    # 全部匹配或任一匹配时，无需组合展开。
    if required_match_count == len(terms):
        return " AND ".join(terms)
    if required_match_count == 1:
        return " OR ".join(terms)
    # 每组包含所需的最少匹配项，任意一组满足即可成为候选论文。
    return " OR ".join(
        "(" + " AND ".join(group) + ")"
        for group in combinations(terms, required_match_count)
    )

def _build_score_query(terms: list[str]|None=None, search_query: str|None=None) -> str:
    """优先用关键词构建 OR 评分查询，否则从内部筛选查询提取引用项。"""
    if terms is not None:
        # 保留每个关键词的短语语义，并转义内嵌双引号。
        return " OR ".join(f'"{term.replace('"', '""')}"' for term in terms)
    if search_query is not None:
        return _to_score_query(search_query)
    raise ValueError("Either 'terms' or 'search_query' must be provided.")

def _to_score_query(search_query: str) -> str:
    """提取引用项，支持内嵌双引号转义，去重后构建小写 OR 表达式。

    此正则只提取引用项，不是通用的 FTS 查询语法解析器。
    """
    # 将连续两个双引号视为引用项内容，保留其 FTS 转义形式。
    return " OR ".join(f'"{term.lower()}"' for term in set(findall(r'"((?:[^"]|"")+)"', search_query)))

