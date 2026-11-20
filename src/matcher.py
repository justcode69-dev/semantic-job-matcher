"""Semantic job matcher: embeds a resume and job descriptions, searches with FAISS,
and explains each match through shared skills."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss  # fast vector search
except Exception:  # not installed, or its DLL is blocked (e.g. Windows Application Control)
    faiss = None


class NumpyFlatIP:
    """Drop-in replacement for faiss.IndexFlatIP: exact inner-product search with NumPy."""

    def __init__(self, dim: int):
        self.vecs = np.zeros((0, dim), dtype="float32")

    def add(self, x: np.ndarray):
        self.vecs = np.vstack([self.vecs, x.astype("float32")])

    def search(self, q: np.ndarray, k: int):
        scores = q @ self.vecs.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids


def make_index(dim: int):
    return faiss.IndexFlatIP(dim) if faiss is not None else NumpyFlatIP(dim)


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)

# Skills we look for when explaining a match. Lowercase; multi-word phrases allowed.
SKILLS = [
    "python", "java", "c++", "kotlin", "sql", "javascript", "typescript",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy", "scipy", "networkx",
    "machine learning", "deep learning", "transfer learning", "cnn", "computer vision",
    "nlp", "embeddings", "semantic search", "vector search", "faiss", "llm", "rag", "ai agents",
    "rest api", "flask", "fastapi", "android", "mysql", "postgresql", "sqlite", "git",
    "optimization", "simulated annealing", "multi-objective", "metaheuristic",
    "wireless sensor network", "data augmentation", "model evaluation", "power bi", "tableau",
    "matplotlib", "seaborn", "data structures", "algorithms", "graph algorithms",
]
# Alternate spellings that should count as a skill.
ALIASES = {
    "cnns": "cnn", "rest apis": "rest api", "apis": "rest api", "sklearn": "scikit-learn",
    "large language models": "llm", "retrieval augmented generation": "rag",
    "sentence embeddings": "embeddings", "wireless sensor networks": "wireless sensor network",
    "data structures and algorithms": "data structures", "dsa": "data structures",
    "augmentation": "data augmentation",
}


def extract_skills(text: str) -> set[str]:
    t = text.lower()
    found = set()
    for phrase in list(SKILLS) + list(ALIASES):
        if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", t):
            found.add(ALIASES.get(phrase, phrase))
    return found


class SentenceEmbedder:
    """Dense embeddings from a sentence-transformers model (default: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def fit(self, texts):  # nothing to learn
        return self

    def encode(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True).astype("float32")


class Model2VecEmbedder:
    """Static sentence embeddings distilled from a sentence-transformer (no PyTorch needed)."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel
        self.model = StaticModel.from_pretrained(model_name, force_download=False)
        self.name = model_name

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        return l2_normalize(np.asarray(self.model.encode(list(texts)), dtype="float32")).astype("float32")


class TfidfEmbedder:
    """Offline fallback: TF-IDF vectors, L2-normalised. Keyword-based, not semantic."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.name = "tfidf"

    def fit(self, texts):
        self.vec.fit(list(texts))
        return self

    def encode(self, texts) -> np.ndarray:
        return l2_normalize(self.vec.transform(list(texts)).toarray()).astype("float32")


def make_embedder(kind: str = "auto"):
    """auto: try sentence-transformers, then model2vec, then TF-IDF."""
    if kind == "tfidf":
        return TfidfEmbedder()
    options = {"sentence": [SentenceEmbedder], "model2vec": [Model2VecEmbedder],
               "auto": [SentenceEmbedder, Model2VecEmbedder]}[kind]
    for cls in options:
        try:
            return cls()
        except Exception as e:  # not installed, DLL blocked, or no internet
            if kind != "auto":
                raise
            print(f"[matcher] {cls.__name__} unavailable ({e.__class__.__name__}: {e})")
    print("[matcher] falling back to TF-IDF (keyword-based)")
    return TfidfEmbedder()


@dataclass
class Match:
    job: dict
    score: float          # cosine similarity, 0..1 for normalised vectors
    shared_skills: list
    missing_skills: list


class JobMatcher:
    def __init__(self, jobs: list[dict], embedder=None):
        self.jobs = jobs
        self.embedder = embedder or make_embedder()
        texts = [self._job_text(j) for j in jobs]
        self.embedder.fit(texts)
        vecs = self.embedder.encode(texts)
        # Inner product on L2-normalised vectors == cosine similarity
        self.index = make_index(vecs.shape[1])
        self.backend = "faiss" if faiss is not None else "numpy"
        self.index.add(vecs)
        self.job_skills = [extract_skills(t) for t in texts]

    @staticmethod
    def _job_text(job: dict) -> str:
        return f"{job['title']}. {job['description']}"

    @classmethod
    def from_json(cls, path: str | Path, embedder=None):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), embedder)

    def search(self, resume_text: str, k: int = 5) -> list[Match]:
        q = self.embedder.encode([resume_text])
        scores, ids = self.index.search(q, min(k, len(self.jobs)))
        resume_skills = extract_skills(resume_text)
        out = []
        for s, i in zip(scores[0], ids[0]):
            js = self.job_skills[i]
            out.append(Match(self.jobs[i], float(s), sorted(js & resume_skills), sorted(js - resume_skills)))
        return out


def read_resume(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")
