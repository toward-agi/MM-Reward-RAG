from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

embedder = JinaClipV2Embedder()
indexer = FaissIndexer(mode="jina-clip-v2")
# indexer.load("textual_index_gpu/infoseek")
indexer.load("textual_index_gpu/encyclopedia/part")

query = "the highest mountain in the world"
query_vector = embedder.embed_text(query)
print(query_vector.shape)
results = indexer.search(query_vector, top_k=5)
print(results)

