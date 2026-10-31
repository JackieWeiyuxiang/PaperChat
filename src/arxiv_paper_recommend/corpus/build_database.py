"""将论文 JSONL 导入 SQLite，并分别构建标题与摘要的 FTS5 索引。"""

import json
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from sqlite3 import connect

from arxiv_paper_recommend.config import DATABASE_PATH, RAW_DATA_DIR


class DatabaseBuilder:
	"""导入论文并保留已有记录的整数 ID，便于与向量索引关联。"""

	def __init__(
		self,
		source_path: str | Path = RAW_DATA_DIR / "papers.jsonl",
		database_path: str | Path = DATABASE_PATH,
	) -> None:
		self.source_path = Path(source_path)
		self.database_path = Path(database_path)

	def _read_documents(self) -> Iterator[tuple[str, str, str | None]]:
		"""逐行读取，规范化论文 ID，并在数据错误中标明行号。"""
		seen_doc_ids: set[str] = set()
		with self.source_path.open(encoding="utf-8-sig") as source:
			for line_number, line in enumerate(source, start=1):
				if not line.strip():
					continue
				try:
					paper = json.loads(line)
					if not isinstance(paper, dict):
						raise TypeError("Expected a JSON object.")
					doc_id = paper["doc_id"]
					title = paper["title"]
					abstract = paper.get("abstract")
					if not isinstance(doc_id, str):
						raise TypeError("doc_id must be a string.")
					doc_id = doc_id.removeprefix("arxiv:")
					if not doc_id.strip():
						raise ValueError("doc_id must not be empty.")
					if not isinstance(title, str):
						raise TypeError("title must be a string.")
					if not title.strip():
						raise ValueError("title must be a non-empty string.")
					if abstract is not None and not isinstance(abstract, str):
						raise TypeError("abstract must be a string or null.")
					if doc_id in seen_doc_ids:
						raise ValueError(f"Duplicate doc_id: {doc_id}")
				except (ValueError, TypeError, KeyError) as error:
					raise ValueError(
						f"{self.source_path}:{line_number}: {error}"
					) from error
				seen_doc_ids.add(doc_id)
				yield doc_id, title, abstract

	def build(self) -> int:
		"""在同一事务中导入并重建索引；失败时回滚，返回导入记录数。"""
		if not self.source_path.is_file():
			raise FileNotFoundError(self.source_path)
		self.database_path.parent.mkdir(parents=True, exist_ok=True)
		with closing(connect(self.database_path)) as conn, conn:
			conn.execute("BEGIN")
			conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT
                )
            """)
			cursor = conn.executemany(
				"""
                INSERT INTO documents (doc_id, title, abstract)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title = excluded.title,
                    abstract = excluded.abstract
            """,
				self._read_documents(),
			)
			imported_count = cursor.rowcount
			for field in ("title", "abstract"):
				index_name = f"documents_{field}_fts"
				conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {index_name}
                    USING fts5(
                        {field},
                        content='documents',
                        content_rowid='id',
                        tokenize='porter'
                    )
                """)
				conn.execute(
					f"INSERT INTO {index_name}({index_name}) VALUES ('rebuild')"
				)
			conn.execute("DROP TABLE IF EXISTS documents_fts")
		return imported_count


def main() -> None:
	"""提供默认项目路径及可选的输入、输出路径。"""
	parser = ArgumentParser(description=__doc__)
	parser.add_argument("--source", type=Path, default=RAW_DATA_DIR / "papers.jsonl")
	parser.add_argument("--database", type=Path, default=DATABASE_PATH)
	args = parser.parse_args()
	builder = DatabaseBuilder(args.source, args.database)
	count = builder.build()
	print(f"Imported {count} documents into {builder.database_path}")


if __name__ == "__main__":
	main()
