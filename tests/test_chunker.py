from src.ingestion.chunker import TextChunker


def test_chunk_text_simple():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Este es un texto de prueba para verificar que el chunker funciona correctamente con textos en espanol."
    chunks = chunker.chunk_text(text, source="test.pdf", page_num=1)
    assert len(chunks) > 0
    for c in chunks:
        assert "texto" in c
        assert c["fuente"] == "test.pdf"
        assert c["pagina"] == 1


def test_chunk_text_empty():
    chunker = TextChunker()
    chunks = chunker.chunk_text("", source="test.pdf", page_num=1)
    assert chunks == []


def test_chunk_text_short():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text("Hola mundo", source="test.pdf", page_num=1)
    assert len(chunks) == 0


def test_chunk_pages():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"texto": "Tejido cartilaginoso hialino. Los condrocitos se encuentran en lagunas.", "fuente": "histo.pdf", "numero": 1},
        {"texto": "Tejido oseo compacto. Las osteonas son la unidad estructural.", "fuente": "histo.pdf", "numero": 2},
    ]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["fuente"] == "histo.pdf"
        assert c["pagina"] in (1, 2)
