"""FAISS vector index - fast "find the stored vectors closest to this query vector".

Key ideas:
- An embedding is a list of floats (here a numpy float32 array). Texts with similar meaning
  get vectors pointing in similar directions.
- Similarity = cosine of the angle between two vectors (1.0 = same direction, ~0 = unrelated).
  If every vector is first scaled to length 1 ("L2-normalized"), cosine similarity is just the
  dot product, which FAISS computes very fast. That's why we normalize everything.
- IndexFlatIP = exact search by Inner Product (dot product): compares the query with every
  vector. Perfectly accurate and fast enough up to ~1M vectors. (Approximate indexes such as
  HNSW are for bigger collections.)
- IndexIDMap lets us attach our own int64 ids (= SQLite chunk ids) to vectors, so a hit can be
  looked up in SQLite and so a document's vectors can be removed by id.

numpy primer: `vecs` is a 2-D array of shape (n_vectors, dimension); `vecs[i]` is one row.
"""

import os
from pathlib import Path

import faiss
import numpy as np


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)  # length of each row
    return vecs / np.where(norms == 0, 1, norms)  # divide each row by its length (avoid /0)


class VectorIndex:
    """FAISS IndexIDMap over inner product, keyed by SQLite chunk id. Rebuildable from SQLite."""

    def __init__(self, index: faiss.Index | None = None):
        self.index = index  # created lazily on first add, once the dimension is known

    # @classmethod = alternative constructor, like a C# static factory `VectorIndex.Load(path)`.
    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        if not path.exists():
            return cls()
        # (de)serialize via bytes: faiss.read_index can't handle non-ASCII paths on Windows
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        return cls(faiss.deserialize_index(data))

    def save(self, path: Path) -> None:
        if self.index is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, then swap it in: os.replace is atomic, so a reader (e.g. another
        # API request) never sees a half-written index file.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(faiss.serialize_index(self.index).tobytes())
        os.replace(tmp, path)

    @property
    def ids(self) -> set[int]:
        """All chunk ids currently in the index (used to detect drift from SQLite)."""
        if self.index is None:
            return set()
        return set(faiss.vector_to_array(self.index.id_map).tolist())

    def add(self, ids: list[int], vecs: np.ndarray) -> None:
        if not ids:
            return
        if self.index is None:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))  # shape[1] = dims
        # FAISS is a C++ library: it needs contiguous float32 vectors and int64 ids.
        self.index.add_with_ids(
            np.ascontiguousarray(vecs, dtype=np.float32), np.asarray(ids, dtype=np.int64)
        )

    def remove(self, ids: list[int]) -> None:
        if ids and self.index is not None:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def vectors(self, ids: list[int]) -> dict[int, np.ndarray]:
        """Stored vectors by chunk id (used to give BM25-only hybrid hits a cosine score).

        IndexIDMap keeps a list of ids in insertion order (id_map) next to the inner flat index,
        so the vector for id X sits at the position of X in that list.
        """
        if self.index is None or not ids:
            return {}
        position = {int(i): p for p, i in enumerate(faiss.vector_to_array(self.index.id_map))}
        inner = faiss.downcast_index(self.index.index)  # the IndexFlatIP holding the vectors
        return {i: inner.reconstruct(position[i]) for i in ids if i in position}

    def search(self, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_id, cosine score) pairs, best first. `vec` must be normalized."""
        if self.index is None or self.index.ntotal == 0:
            return []
        # FAISS searches a batch of queries; we pass a batch of one (reshape to 1 x dims)
        # and read row 0 of the results. Missing results come back as id -1.
        scores, ids = self.index.search(np.asarray(vec, dtype=np.float32).reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]
