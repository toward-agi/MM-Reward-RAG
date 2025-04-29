import pickle
import json
import numpy as np
from tqdm import tqdm
import os
import re
import bm25s  # Using bm25s instead of rank_bm25
import Stemmer  # Optional for better results
from collections import defaultdict

# Define paths
KB_PATH = "/data/track/mm-RewardRAG/indexs/jina-clip-v2/textual_index_gpu/webqa_caption/metadata.pkl"
VAL_PATH = "/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json"
OUTPUT_DIR = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/webqa/bm25s/q2img_caption"
INDEX_PATH = os.path.join(OUTPUT_DIR, "bm25s_index")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Load validation data
print("Loading validation data...")
with open(VAL_PATH, 'r') as f:
    val_data = json.load(f)

# Define k values for evaluation
ks = [1, 5, 10, 20, 50, 100]

# Check if retriever index already exists
if os.path.exists(INDEX_PATH):
    print(f"Found existing BM25S index at {INDEX_PATH}, loading...")
    try:
        retriever = bm25s.BM25.load(INDEX_PATH, load_corpus=False)
        
        # Try to activate numba for faster retrieval
        try:
            retriever.activate_numba_scorer()
            print("Numba backend activated for faster scoring.")
        except Exception as e:
            print(f"Could not activate numba: {e}")
            
        # Need to load the document IDs mapping separately
        doc_ids_path = os.path.join(OUTPUT_DIR, "doc_ids.pkl")
        if os.path.exists(doc_ids_path):
            with open(doc_ids_path, 'rb') as f:
                doc_ids = pickle.load(f)
            print(f"Loaded {len(doc_ids)} document IDs")
        else:
            print("Document IDs file not found, rebuilding index...")
            # Set flag to rebuild
            index_exists = False
    except Exception as e:
        print(f"Error loading existing index: {e}")
        print("Will rebuild the index.")
        index_exists = False
else:
    print("No existing index found, building from scratch.")
    index_exists = False

# Initialize stemmer (for query tokenization)
stemmer = Stemmer.Stemmer("english")

# If we need to build the index
if not os.path.exists(INDEX_PATH) or 'index_exists' not in locals() or not index_exists:
    # Load knowledge base
    print("Loading knowledge base...")
    with open(KB_PATH, 'rb') as f:
        kb_data = pickle.load(f)

    # Prepare documents for BM25S
    print("Preparing corpus...")
    corpus = []
    doc_ids = []
    for doc in kb_data:
        # Combine title and fact for better retrieval
        text = f"{doc['title']} {doc['caption']}"
        corpus.append(clean_text(text))
        doc_ids.append(doc['image_id'])

    # Save doc_ids for future use
    doc_ids_path = os.path.join(OUTPUT_DIR, "doc_ids.pkl")
    with open(doc_ids_path, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"Saved {len(doc_ids)} document IDs to {doc_ids_path}")

    # Tokenize corpus
    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)

    # Create BM25S index
    print("Creating BM25S index...")
    retriever = bm25s.BM25(corpus=None, backend='numba', method='lucene')
    retriever.index(corpus_tokens)

    # Activate numba for faster retrieval if available
    try:
        retriever.activate_numba_scorer()
        print("Numba backend activated for faster scoring.")
    except Exception as e:
        print(f"Could not activate numba: {e}")
        
    # Save the BM25S index for future use
    print(f"Saving BM25S index to {INDEX_PATH}")
    retriever.save(INDEX_PATH)

# Run retrieval and calculate recall for each k
for k in ks:
    result_file = f"{OUTPUT_DIR}/retrieved_results_{k}.json"
    
    # Check if results already exist for this k
    if os.path.exists(result_file):
        print(f"Found existing results for k={k}, loading from {result_file}")
        with open(result_file, 'r') as f:
            id_to_retrieved = json.load(f)
            
        # Calculate recall metrics from existing results
        total_relevant = 0
        total_retrieved_relevant = 0
        query_recalls = []
        
        for data_id, item in id_to_retrieved.items():
            # Get supporting context ids from the cached results
            supporting_context_ids = [str(fact["image_id"]) for fact in item["supporting_context"]]
            #TODO: 所以这里是必须要有的不能空啊
            retrieved_ids = [str(doc["image_id"]) for doc in item["retrieved"]]
            
            # Compute recall for this query
            relevant_count = len(supporting_context_ids)
            total_relevant += relevant_count
            
            retrieved_relevant_count = 0
            for support_id in supporting_context_ids:
                if support_id in retrieved_ids:
                    retrieved_relevant_count += 1
                    total_retrieved_relevant += 1
            
            # Calculate query recall
            query_recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 1.0
            query_recalls.append(query_recall)
    else:
        print(f"Evaluating BM25S with k={k}")
        
        # Store results
        id_to_retrieved = {}
        
        # Micro recall tracking
        total_relevant = 0
        total_retrieved_relevant = 0
        
        # Macro recall tracking
        query_recalls = []
        
        # Process each query
        for data_id, item in tqdm(val_data.items()):
            # Get query
            query = clean_text(item["Q"])
            
            # Get supporting context ids
            # print(item["img_posFacts"])
            supporting_context_ids = [fact["image_id"] for fact in item["img_posFacts"]]
            # print(supporting_context_ids)

            if supporting_context_ids is []:
                # print(f"Skipping query {data_id} with no supporting image context.")
                continue
            
            # Tokenize query
            query_tokens = bm25s.tokenize(query, stemmer=stemmer)
            
            # Get top-k doc ids using BM25S
            docs, scores = retriever.retrieve(query_tokens, k=k)
            
            # Extract the retrieved document IDs and scores
            # BM25S returns a 2D array with shape (n_queries, k)
            retrieved_ids = [doc_ids[int(idx)] for idx in docs[0]]
            retrieved_scores = scores[0].tolist()
            
            # Store retrieved results
            retrieved_docs = []
            for i, doc_id in enumerate(retrieved_ids):
                retrieved_docs.append({
                    "image_id": doc_id,
                    "score": float(retrieved_scores[i])
                })
            
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": retrieved_docs,
                "supporting_context": item["img_posFacts"]
            }
            
            # Compute recall for this query
            relevant_count = len(supporting_context_ids)
            total_relevant += relevant_count
            
            retrieved_relevant_count = 0
            for support_id in supporting_context_ids:
                if support_id in retrieved_ids:
                    retrieved_relevant_count += 1
                    total_retrieved_relevant += 1
            
            # Calculate query recall
            query_recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 1.0
            query_recalls.append(query_recall)
        
        # Save results
        with open(result_file, "w") as f:
            json.dump(id_to_retrieved, f, indent=2)
        print(f"Saved results to {result_file}")
    
    # Calculate micro recall
    micro_recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
    
    # Calculate macro recall
    macro_recall = sum(query_recalls) / len(query_recalls) if query_recalls else 0
    
    # Print recall metrics
    print(f"Top-{k} Results:")
    print(f"  Micro-Recall@{k}: {micro_recall:.4f} ({total_retrieved_relevant}/{total_relevant})")
    print(f"  Macro-Recall@{k}: {macro_recall:.4f}")
    print("-" * 50)

print("Evaluation completed.")