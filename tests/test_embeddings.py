import numpy as np
import pytest
from src.models.embeddings import MultimodalEmbedder


@pytest.fixture(scope="module")
def embedder():
    return MultimodalEmbedder(device="cpu")


def test_embed_text_dimension(embedder):
    vec = embedder.embed_text("tejido cartilaginoso hialino")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert 0.99 < float(np.linalg.norm(vec)) < 1.01


def test_embed_texts(embedder):
    texts = ["tejido oseo", "tejido muscular", "tejido nervioso"]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert v.shape == (512,)


def test_embed_text_plip(embedder):
    vec = embedder.embed_text("cartilago")
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 0


def test_compute_similarity(embedder):
    a = np.random.randn(512)
    a = a / np.linalg.norm(a)
    b = a.copy()
    sim = embedder.compute_similarity(a, b)
    assert sim > 0.99


def test_compute_similarities(embedder):
    query = np.random.randn(512)
    query = query / np.linalg.norm(query)
    candidates = [query.copy(), -query.copy()]
    sims = embedder.compute_similarities(query, candidates)
    assert sims[0] > 0.99
    assert sims[1] < -0.99
