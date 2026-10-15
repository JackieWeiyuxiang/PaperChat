"""Build a FAISS HNSW index from an embedding archive and a matching database."""

from argparse import ArgumentParser
from contextlib import closing
import os
from pathlib import Path
from sqlite3 import SQLITE_LIMIT_VARIABLE_NUMBER, connect
from tempfile import NamedTemporaryFile

import faiss
import numpy as np
from numpy.typing import NDArray

from arxiv_paper_recommend.config import DATABASE_PATH, INDEX_PATH, RAW_DATA_DIR


class IndexBuilder:
    """Validate and map archived vectors before replacing the output index."""

    def __init__(
        self,
        source_path: str | Path = RAW_DATA_DIR / "vectors.npz",
        database_path: str | Path = DATABASE_PATH,
        index_path: str | Path = INDEX_PATH,
    ) -> None:
        self.source_path = Path(source_path)
        self.database_path = Path(database_path)
        self.index_path = Path(index_path)

    def _read_vectors(self) -> tuple[NDArray[np.float32], list[str]]:
        with np.load(self.source_path, allow_pickle=False) as archive:
            vectors = archive["vectors"]
            doc_ids = archive["doc_ids"]
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("Vectors must be a non-empty two-dimensional matrix.")
        if not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError("Vectors must contain floating-point values.")
        if doc_ids.ndim != 1 or len(doc_ids) != len(vectors):
            raise ValueError("doc_ids must be one-dimensional with one ID per vector.")
        if doc_ids.dtype.kind != "U":
            raise ValueError("doc_ids must contain strings.")
        normalized_ids = [doc_id.removeprefix("arxiv:") for doc_id in doc_ids.tolist()]
        if any(not doc_id.strip() for doc_id in normalized_ids):
            raise ValueError("Document IDs must not be empty.")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Document IDs must be unique after removing the arxiv: prefix.")
        # Validate after conversion as float64 input may overflow float32.
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must contain finite float32 values.")
        norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64))
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Vectors must have finite, non-zero norms.")
        # Normalize in float64 before storing float32, avoiding norm overflow.
        vectors /= norms[:, None]
        return vectors, normalized_ids

    def _map_ids(self, doc_ids: list[str]) -> NDArray[np.int64]:
        database_uri = self.database_path.resolve().as_uri() + "?mode=ro"
        mapping: dict[str, int] = {}
        with closing(connect(database_uri, uri=True)) as conn:
            batch_size = conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)
            for start in range(0, len(doc_ids), batch_size):
                batch = doc_ids[start:start + batch_size]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT doc_id, id FROM documents WHERE doc_id IN ({placeholders})", batch,
                ).fetchall()
                mapping.update(rows)
        missing = [doc_id for doc_id in doc_ids if doc_id not in mapping]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{len(missing)} archived document IDs are missing from the database: {preview}")
        return np.asarray([mapping[doc_id] for doc_id in doc_ids], dtype=np.int64)

    def build(self) -> int:
        """Build using the existing HNSW defaults; return the indexed vector count.

        Reading the database never creates or modifies it. Validation and index
        construction finish before replacing an existing index.
        """
        if self.index_path.resolve() in {self.source_path.resolve(), self.database_path.resolve()}:
            raise ValueError("Index output must differ from the vector archive and database paths.")
        vectors, doc_ids = self._read_vectors()
        ids = self._map_ids(doc_ids)
        base = faiss.IndexHNSWFlat(vectors.shape[1], 128, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = 800
        base.hnsw.efSearch = 256
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(vectors, ids)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self.index_path.parent, prefix=self.index_path.name + ".", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            faiss.write_index(index, str(temporary_path))
            os.replace(temporary_path, self.index_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return int(index.ntotal)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATA_DIR / "vectors.npz")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    builder = IndexBuilder(args.source, args.database, args.index)
    count = builder.build()
    print(f"Indexed {count} vectors into {builder.index_path}")


if __name__ == "__main__":
    main()
