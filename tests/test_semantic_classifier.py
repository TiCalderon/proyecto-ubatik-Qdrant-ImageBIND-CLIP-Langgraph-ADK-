import pytest
from src.search.semantic_classifier import ClasificadorSemantico
from src.models.embeddings import MultimodalEmbedder


@pytest.fixture(scope="module")
def clasificador():
    embedder = MultimodalEmbedder(device="cpu")
    return ClasificadorSemantico(embedder)


def test_clasificar_dentro_del_dominio(clasificador):
    result = clasificador.clasificar("Que tipos de tejido cartilaginoso existen?")
    assert result is True


def test_clasificar_fuera_del_dominio(clasificador):
    result = clasificador.clasificar("Cual es la capital de Francia?")
    assert result is False


def test_clasificar_con_embedding(clasificador):
    from src.models.embeddings import MultimodalEmbedder
    emb = MultimodalEmbedder(device="cpu")
    vec = emb.embed_text("tejido oseo osteonas osteocitos")
    result = clasificador.clasificar("tejido oseo", query_embedding=vec.tolist())
    assert result is True
