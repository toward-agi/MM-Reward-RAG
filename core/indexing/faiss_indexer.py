from operator import index
import faiss
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path
from tqdm import tqdm

from core.indexing.base import BaseIndexer
from config.config import settings

class FaissIndexer(BaseIndexer):
    """FAISS-based vector indexer."""
    
    def __init__(self, mode: str = "jina-clip-v2"):
        """Initialize FAISS indexer.
        
        Args:
            mode: Either "image" or "text" to determine which dimension to use, or "jina-clip-v2" for CLIP embeddings.
        """
        self.config = settings.indexers
        self.mode = mode
        self.dimension = self.config["faiss"]["dimension"][mode]
        self.index = None
        self.metadata = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on configuration."""
        config = self.config["faiss"]
        # res = faiss.StandardGpuResources()
        
        if config["index_type"] == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif config["index_type"] == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif config["index_type"] == "Cosine":

            self.index = faiss.IndexFlatIP(self.dimension)
            # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            # For cosine similarity, we need to normalize vectors
            self.normalize = True
        elif config["index_type"] == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, config["nlinks"])
            # 设置HNSW的参数
            # 构建参数
            self.index.efConstruction = config["ef_construction"]
            # 搜索参数
            self.index.efSearch = config["ef_search"]
            self.normalize = True
            # self.index = faiss.index_cpu_to_gpu(res, 7, self.index) not supported
        else:
            raise ValueError(f"Unsupported index type: {config['index_type']}")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        return faiss.normalize_L2(vector)
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """Add vectors to the index with optional metadata."""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        # print(f"vector shape: {vectors.shape}")
            
        if hasattr(self, 'normalize') and self.normalize:
            self._normalize_vector(vectors)
            
        if self.index is None:
            self._initialize_index()

        self.index.add(vectors)

        if metadata is not None:
            self.metadata.extend(metadata)
    
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[float], List[int], List[Dict[str, Any]]]:
        """Search for nearest neighbors of query vector.
        
        Returns:
            Tuple containing:
            - List of distances/scores
            - List of indices in the original index
            - List of metadata for each result
        """
        if hasattr(self, 'normalize') and self.normalize:
            self._normalize_vector(query_vector)
            
        scores, indices = self.index.search(
            query_vector.reshape(1, -1),
            min(top_k, len(self.metadata))
        )
        
        results = []
        result_scores = []
        result_indices = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for empty slots
                results.append(self.metadata[idx])
                result_scores.append(float(score))
                result_indices.append(int(idx))
                
        return result_scores, result_indices, results
    
    def search_batch(self, query_vectors: np.ndarray, top_k: int) -> List[Tuple[List[float], List[int], List[Dict[str, Any]]]]:
        """Search for nearest neighbors of query vectors.
        
        Returns:
            Tuple containing:
            - List of lists of distances/scores
            - List of lists of indices in the original index
            - List of lists of metadata for each result
        """
        if hasattr(self, 'normalize') and self.normalize:
            self._normalize_vector(query_vectors)
            
        scores, indices = self.index.search(
            query_vectors,
            min(top_k, len(self.metadata))
        )
        
        results = []
        result_scores = []
        result_indices = []

        for i in range(len(query_vectors)):
            results.append([])
            result_scores.append([])
            result_indices.append([])
            for score, idx in zip(scores[i], indices[i]):
                if idx != -1:
                    results[i].append(self.metadata[idx])
                    result_scores[i].append(float(score))
                    result_indices[i].append(int(idx))

        return list(zip(result_scores, result_indices, results))


    
    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        if isinstance(self.index, faiss.GpuIndex):
            self.index = faiss.index_gpu_to_cpu(self.index)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
    def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
            
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))

        # move index to GPU
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

def test_faiss_indexer():
    """Test FAISS indexer functionality."""
    # Initialize indexer
    indexer = FaissIndexer(mode="image")  # Using image dimension from config
    
    # Create some test data
    num_vectors = 1500  # Increased from 10 to 1000
    vectors = np.random.random((num_vectors, indexer.dimension)).astype('float32')
    metadata = [{"id": i, "text": f"test_{i}"} for i in range(num_vectors)]
    
    # Test adding vectors
    print("Testing add_vectors...")
    indexer.add_vectors(vectors, metadata)
    print(f"Added {len(indexer.metadata)} vectors")
    
    # Test searching
    print("\nTesting search...")
    query = np.random.random((1, indexer.dimension)).astype('float32')
    k = 5
    distances, indices, meta = indexer.search(query, k)
    print(f"Search results for k={k}:")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    print(f"First metadata: {meta[0]}")
    
    # Test save and load
    print("\nTesting save and load...")
    save_path = "test_index"
    indexer.save(save_path)
    print(f"Saved index to {save_path}")
    
    new_indexer = FaissIndexer(mode="image")
    new_indexer.load(save_path)
    print(f"Loaded index from {save_path}")
    print(f"Loaded {len(new_indexer.metadata)} vectors")
    
    # Verify search works with loaded index
    print("\nTesting search with loaded index...")
    new_distances, new_indices, new_meta = new_indexer.search(query, k)
    print(f"Search results match: {np.allclose(distances, new_distances)}")
    
    # Clean up
    import shutil
    shutil.rmtree(save_path, ignore_errors=True)
    print("\nTest completed successfully!")

def test_faiss_indexer_with_clip():
    from core.embeddings.clip_embedder import CLIPEmbedder
    print("Testing FAISS indexer with CLIP embedder...")
    embedder = CLIPEmbedder()
    print(embedder.model)
    print("Testing add_vectors...")
    image = Image.open("test/images/hk.png")
    image_cat = Image.open("test/images/cat.png")
    image_dog = Image.open("test/images/dog.png")
    print("Embedding images...")
    image_features = embedder.embed_images([image, image_cat, image_dog])
    print("PASS")
    text = ["Hong Kong", "Cat", "Dog"]
    text_features = embedder.embed_text(["Dog"])
    print("Adding vectors...")
    print(image_features.shape)

    indexer = FaissIndexer(mode="image")
    indexer.add_vectors(image_features, [{"id": i, "text": text[i]} for i in range(len(image_features))])
    query = text_features[0]
    k = 5
    distances, indices, meta = indexer.search(query, k)
    print(f"Search results for k={k}:")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    print(f"First metadata: {meta[0]}")

if __name__ == "__main__":
    test_faiss_indexer_with_clip()
