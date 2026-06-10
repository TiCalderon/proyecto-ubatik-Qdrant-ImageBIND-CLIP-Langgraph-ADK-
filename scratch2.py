from src.ingestion.indexer import QdrantIndexer
indexer = QdrantIndexer()
res = indexer.client.scroll(
    collection_name=indexer.col_imagenes,
    limit=100
)
for p in res[0]:
    if p.payload.get('caption') and 'Cartí' in p.payload.get('caption'):
        print(f"Found image: {p.payload.get('nombre_archivo')} -> {p.payload.get('caption')}")
