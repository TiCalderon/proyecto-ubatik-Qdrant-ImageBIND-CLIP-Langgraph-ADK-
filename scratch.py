from src.ingestion.indexer import QdrantIndexer
from src.models.embeddings import MultimodalEmbedder

embedder = MultimodalEmbedder()
indexer = QdrantIndexer(embedder=embedder)

query = "Me puedes mostrar una imagen del manual que muestre un Tejido Conectivo Especializado, Cartílago Elástico?"
vec = embedder.embed_text(query)

print("Text Search (in text collection):")
res = indexer.text_search(vec.tolist(), top_k=3, threshold=0.1)
for score, p in res:
    print(f"Score: {score:.3f}, text: {p['texto'][:50]}")

print("\nImage Search (in image collection, using PLIP):")
res = indexer.image_search(vec.tolist(), top_k=3, threshold=0.1, using="plip")
for score, p in res:
    print(f"Score: {score:.3f}, file: {p.get('nombre_archivo')}, caption: {p.get('caption')}")

print("\nImage Search with english query:")
query_en = "Elastic cartilage, specialized connective tissue"
vec_en = embedder.embed_text(query_en)
res = indexer.image_search(vec_en.tolist(), top_k=3, threshold=0.1, using="plip")
for score, p in res:
    print(f"Score: {score:.3f}, file: {p.get('nombre_archivo')}, caption: {p.get('caption')}")

