import sys
sys.path.append("/data/track/mm-RewardRAG")
# from core.embeddings.bge_text_embedding import BGETextEmbedder
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

import json
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

text_embedder = JinaClipV2Embedder()

# Update the input file path to point to your CSV file
input_data_file = "/data/zhiyuan/okvqa-gs122k/corpus/okvqa_full_corpus.csv"
text_feature_dir = "/data/dataset/mm_RewardRAG/gs112k/jina-clip/full/textual_features"

# Load the CSV file instead of JSON
text_data = pd.read_csv(input_data_file)

# Embed text
batch_size = 1024

os.makedirs(text_feature_dir, exist_ok=True)

# Calculate the total number of batches for the progress bar
total_batches = (len(text_data) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(text_data), batch_size), total=total_batches):
    # Get the batch of data
    batch = text_data.iloc[i:i + batch_size]
    
    # Extract the text content from the 'text' column instead of 'wikipedia_content'
    texts = batch['text'].tolist()
    
    # Embed the texts
    text_features = text_embedder.embed_text(texts)

    # Save the features
    batch_file = os.path.join(text_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, text_features)
    # the feature has already been normalized
