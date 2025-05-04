import json
import os
from tqdm import tqdm
import sys

# Load validation dataset
valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))

# Define the k values to evaluate
ks = [1, 3, 5, 10, 20, 50, 100]
max_k = max(ks)  # Get the maximum k value

# Set up output directory
output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/webqa/bge/q2text"
retrieved_results_file = f"{output_dir}/retrieved_results_{max_k}.json"
os.makedirs(output_dir, exist_ok=True)

# Perform retrieval only once with the maximum k value
if not os.path.exists(retrieved_results_file):
    sys.path.append("/data/track/mm-RewardRAG/")
    from core.indexing.faiss_indexer import FaissIndexer
    # from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
    from core.embeddings.bge_text_embedding import BGETextEmbedder
    embedder = BGETextEmbedder()
    indexer = FaissIndexer(mode="bge")
    
    # Load index
    indexer.load("/data/track/mm-RewardRAG/indexs/bge/webqa")
    
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["Q"] for item in batch]
        supporting_contexts = [item[1]["txt_posFacts"] for item in batch]
        
        query_vectors = embedder.embed_text(questions)
        results = indexer.search_batch(query_vectors, top_k=max_k)  # Retrieve results only once with max_k
        
        for data_id, result, supporting_context in zip(data_ids, results, supporting_contexts):
            # Skip entries with empty supporting contexts
            if not supporting_context:
                continue
                
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": result,
                "supporting_context": supporting_context,
            }
            
    # Save the retrieval results
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    # Load existing retrieval results
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# Calculate recall@k for each k value
for k in ks:
    # For micro-average
    total_relevant_micro = 0
    total_retrieved_relevant_micro = 0
    
    # For macro-average
    all_recalls = []
    
    for id, item in id_to_retrieved.items():
        # For each k value, only consider the top-k results from the max_k retrievals
        retrieved_id = [i['snippet_id'] for i in item["retrieved"][2][:k]]
        supporting_context_id = [i['snippet_id'] for i in item["supporting_context"]]
        
        # Skip entries with no relevant documents (safety check)
        if not supporting_context_id:
            continue
            
        # Count total relevant documents (for micro-average)
        total_relevant_micro += len(supporting_context_id)
        
        # Count relevant documents that were retrieved in top-k
        retrieved_relevant = 0
        for support_id in supporting_context_id:
            if support_id in retrieved_id:
                retrieved_relevant += 1
                total_retrieved_relevant_micro += 1
        
        # Calculate per-query recall (for macro-average)
        query_recall = retrieved_relevant / len(supporting_context_id)
        all_recalls.append(query_recall)
    
    # Calculate micro-average recall (global)
    micro_recall = total_retrieved_relevant_micro / total_relevant_micro if total_relevant_micro > 0 else 0
    
    # Calculate macro-average recall (average of per-query recalls)
    macro_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    
    print(f"Recall@{k}:")
    print(f"  Micro-average: {micro_recall:.4f} ({total_retrieved_relevant_micro}/{total_relevant_micro})")
    print(f"  Macro-average: {macro_recall:.4f}")