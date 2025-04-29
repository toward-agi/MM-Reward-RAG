from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("/data/track/mm-RewardRAG")
from core.indexing.faiss_indexer import FaissIndexer

# Path to the original CSV file (metadata)
input_data_file = "/data/zhiyuan/okvqa-gs122k/corpus/okvqa_full_corpus.csv"
# Directory containing the extracted embeddings
textual_feature_dir = "/data/dataset/mm_RewardRAG/gs112k/jina-clip/full/textual_features"

# Load metadata
metadata = pd.read_csv(input_data_file)

# Initialize FAISS indexer
indexer = FaissIndexer(mode="jina-clip-v2")

# Get all batch files sorted numerically
batch_files = sorted([f for f in os.listdir(textual_feature_dir) if f.startswith("batch_")],
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

metadata_start = 0
# Start adding vectors to the indexer
for batch_file in tqdm(batch_files):
    batch_path = os.path.join(textual_feature_dir, batch_file)
    textual_features = np.load(batch_path)
    
    # Get corresponding metadata for this batch
    batch_metadata = metadata.iloc[metadata_start:metadata_start+textual_features.shape[0]].to_dict('records')
    
    # Add vectors to the indexer
    indexer.add_vectors(textual_features, batch_metadata)
    
    # Update metadata index for next batch
    metadata_start += textual_features.shape[0]

# Create directory for saving the index
save_dir = "/data/track/mm-RewardRAG/indexs/jina-clip-v2/textual_index_gpu/gs112k-full"
os.makedirs(save_dir, exist_ok=True)

# Save the index
indexer.save(save_dir)

print(f"Index saved to {save_dir}")
