"""Build and load the BGE-M3 dense vector library for title/abstract retrieval."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import tempfile

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[4]
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_LENGTH = 8192


def read_papers(path):
    papers = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not papers:
        raise ValueError("The corpus is empty.")
    for key in ("doc_id", "chunk_id", "text"):
        if any(not isinstance(p.get(key), str) or not p[key].strip() for p in papers):
            raise ValueError(f"Missing or empty {key}.")
    for key in ("doc_id", "chunk_id"):
        if len({p[key] for p in papers}) != len(papers):
            raise ValueError(f"Duplicate {key}.")
    return papers


def corpus_fingerprint(papers):
    """Hash the ordered IDs and input text, independent of JSONL line endings."""
    payload = [[p["doc_id"], p["chunk_id"], p["text"]] for p in papers]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_encoder(revision, device=None):
    model = SentenceTransformer(MODEL, revision=revision, device=device, trust_remote_code=False)
    model.max_seq_length = MAX_LENGTH
    return model


def encode_texts(model, texts, batch_size=8, show_progress=False):
    """Use the model's CLS pooling and normalized float32 output; never truncate."""
    if batch_size < 1 or not texts or any(not t.strip() for t in texts):
        raise ValueError("Provide nonempty texts and a positive batch size.")
    for start in range(0, len(texts), 64):
        tokens = model.tokenizer(texts[start:start + 64], truncation=False, padding=False)["input_ids"]
        if any(len(row) > MAX_LENGTH for row in tokens):
            raise ValueError(f"Input exceeds {MAX_LENGTH} tokens; no text was silently truncated.")
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=show_progress)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
        raise ValueError("Unexpected embedding shape or nonfinite vectors.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Embeddings are not unit normalized.")
    return vectors


def build(input_path, output_path, revision=REVISION, device=None, batch_size=8):
    from huggingface_hub import HfApi

    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from the corpus path.")
    papers = read_papers(input_path)
    # Resolve once so the tokenizer and weights come from the same immutable commit.
    revision = HfApi().model_info(MODEL, revision=revision).sha
    model = load_encoder(revision, device)
    vectors = encode_texts(model, [p["text"] for p in papers], batch_size, True)
    metadata = {
        "format_version": 1, "model": MODEL, "revision": revision,
        "count": len(papers), "dimension": 1024, "dtype": "float32",
        "normalized": True, "similarity": "cosine", "pooling": "CLS",
        "input": "title + complete abstract (text)", "max_length": MAX_LENGTH,
        "corpus_sha256": corpus_fingerprint(papers),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packages": {p: version(p) for p in ("sentence-transformers", "transformers", "torch", "numpy")},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, vectors=vectors,
                                doc_ids=np.array([p["doc_id"] for p in papers]),
                                chunk_ids=np.array([p["chunk_id"] for p in papers]),
                                metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def load_library(path, corpus_path=None):
    """Return vectors, doc IDs, chunk IDs, metadata; optionally check corpus alignment."""
    with np.load(path, allow_pickle=False) as data:
        vectors, doc_ids, chunk_ids = data["vectors"], data["doc_ids"], data["chunk_ids"]
        metadata = json.loads(str(data["metadata"].item()))
    if metadata["format_version"] != 1 or metadata["model"] != MODEL:
        raise ValueError("Unsupported vector library.")
    if vectors.shape != (metadata["count"], 1024) or vectors.dtype != np.float32:
        raise ValueError("Invalid vector dimensions or dtype.")
    if any(ids.shape != (len(vectors),) or len(set(ids)) != len(vectors) for ids in (doc_ids, chunk_ids)):
        raise ValueError("Invalid document ID mapping.")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid or unnormalized vectors.")
    if corpus_path is not None:
        papers = read_papers(corpus_path)
        if (corpus_fingerprint(papers) != metadata["corpus_sha256"]
                or not np.array_equal(doc_ids, [p["doc_id"] for p in papers])
                or not np.array_equal(chunk_ids, [p["chunk_id"] for p in papers])):
            raise ValueError("Corpus changed or IDs are misaligned; rebuild the vectors.")
    return vectors, doc_ids, chunk_ids, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/papers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/vectors.npz")
    parser.add_argument("--revision", default=REVISION, help="Hugging Face model commit or ref")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detect by default")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.revision, args.device, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
