from numpy import load, array
from arxiv_paper_recommend.config import RAW_DATA_DIR, DATA_DIR
import faiss
from sqlite3 import connect

with load(RAW_DATA_DIR / "vectors.npz", allow_pickle=False) as data:
    vectors = data["vectors"]
    doc_ids = [id[id.find(':')+1:] for id in data["doc_ids"]]
with connect(DATA_DIR / "documents.db") as conn:
    mapping = dict(conn.execute(
        f"SELECT doc_id, id FROM documents WHERE doc_id IN ({",".join("?" for _ in doc_ids)})",
        doc_ids,
    ).fetchall())
ids = array([mapping[doc_id] for doc_id in doc_ids])

faiss.normalize_L2(vectors)

if __name__ == "__main__":
    base = faiss.IndexHNSWFlat(
        vectors.shape[1], 128, faiss.METRIC_INNER_PRODUCT
    )
    base.hnsw.efConstruction = 800
    base.hnsw.efSearch = 256

    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, ids)

    faiss.write_index(index, str(DATA_DIR / "documents.faiss"))
