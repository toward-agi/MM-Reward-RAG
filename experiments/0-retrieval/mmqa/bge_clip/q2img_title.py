# import json
# import os
# from tqdm import tqdm
# import sys
# sys.path.append("/data/track/mm-RewardRAG/")
# from core.indexing.faiss_indexer import FaissIndexer
# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

# valid_dataset = {}
# ks = [1, 5, 10, 20, 50, 100]
# for k in ks:
#     with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
#         for line in f:
#             item = json.loads(line)
#             valid_dataset[item['qid']] = item
#     output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/q2img"
#     retrieved_results_file = f"{output_dir}/retrieved_results_{k}.json"
#     os.makedirs(output_dir, exist_ok=True)
#     if not os.path.exists(retrieved_results_file):
#         embedder = JinaClipV2Embedder() #将question或者image转换为向量
#         indexer = FaissIndexer(mode="jina-clip-v2") #建的索引
        
#         # 加载index
#         # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
#         indexer.load("/data/track/mm-RewardRAG/indexs/jina-clip-v2/visual_index_gpu/mm_qa")
#         bs = 512
#         id_to_retrieved = {}
#         valid_dataset_list = list(valid_dataset.items())
#         for i in tqdm(range(0, len(valid_dataset_list), bs)):
#             batch = valid_dataset_list[i:i+bs]
#             data_ids = [item[0] for item in batch]
#             questions = [item[1]["question"] for item in batch]
#             types = [item[1]['metadata']['type'] for item in batch]
#             supporting_contexts = [item[1]["supporting_context"] for item in batch]
#             query_vectors = embedder.embed_text(questions)
#             results = indexer.search_batch(query_vectors, top_k=k) #检索,这里top-k就是返回的检索结果的数量
#             for data_id, result, supporting_context, type in zip(data_ids, results, supporting_contexts, types):
#                 id_to_retrieved[data_id] = {
#                     "data_id": data_id,
#                     "retrieved": result,
#                     "supporting_context": supporting_context,
#                     "type": type
#                 }
#         with open(retrieved_results_file, "w") as f:
#             json.dump(id_to_retrieved, f, indent=2)
#     else:
#         id_to_retrieved = json.load(open(retrieved_results_file, "r"))
#         total_relevant = 0
#         total_retrieved_relevant = 0
#         for id, item in id_to_retrieved.items():
#             type = item["type"]
#             # if type != "ImageQ":
#             if "ImageQ" not in type:
#                 continue
#             retrieved_id = [i['metadata']['id'] for i in item["retrieved"][2]]
#             supporting_context_id = [i['doc_id'] for i in item["supporting_context"]]
#             # Count total relevant documents
#             total_relevant += len(supporting_context_id)
            
#             # Count relevant documents that were retrieved
#             for support_id in supporting_context_id:
#                 if support_id in retrieved_id:
#                     total_retrieved_relevant += 1
                    
#         recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
#         print(f"Recall@{k}: {recall:.4f} ({total_retrieved_relevant}/{total_relevant})")


"""
----------------------------------------------------------------------------------------------
"""

import json
import os
from tqdm import tqdm
import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.indexing.faiss_indexer import FaissIndexer
# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.embeddings.bge_text_embedding import BGETextEmbedder

# Define k values and get max k
ks = [1, 5, 10, 20, 50, 100]
max_k = max(ks)

# Load validation dataset (only load once)
valid_dataset = {}
with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        valid_dataset[item['qid']] = item

# Setup output directory
output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/bge/q2img_title"
retrieved_results_file = f"{output_dir}/retrieved_results_{max_k}.json"
os.makedirs(output_dir, exist_ok=True)

# Perform retrieval for max k
if not os.path.exists(retrieved_results_file):
    # embedder = JinaClipV2Embedder()
    embedder = BGETextEmbedder()
    # indexer = FaissIndexer(mode="jina-clip-v2")
    indexer = FaissIndexer(mode="bge")
    
    # Load index  
    indexer.load("/data/track/mm-RewardRAG/indexs/bge/mm_qa_title")
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["question"] for item in batch]
        types = [item[1]['metadata']['type'] for item in batch]
        supporting_contexts = [item[1]["supporting_context"] for item in batch]
        query_vectors = embedder.embed_text(questions)
        results = indexer.search_batch(query_vectors, top_k=max_k)
        for data_id, result, supporting_context, type in zip(data_ids, results, supporting_contexts, types):
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": result,
                "supporting_context": supporting_context,
                "type": type
            }
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# Evaluate recall for each k
for k in ks:
    # Initialize metrics
    total_relevant_micro = 0
    total_retrieved_relevant_micro = 0
    query_recalls = []
    image_queries_count = 0
    
    for id, item in id_to_retrieved.items():
        type = item["type"]
        # Skip non-ImageQ queries
        if "ImageQ" not in type:
            continue
        
        image_queries_count += 1
        # For each k, only consider top-k results from max_k retrieval
        retrieved_id = [i['metadata']['id'] for i in item["retrieved"][2][:k]]
        supporting_context_id = [i['doc_id'] for i in item["supporting_context"]]
        
        # Skip if no relevant documents
        if not supporting_context_id:
            continue
        
        # Count total relevant documents (for micro-average)
        relevant_count = len(supporting_context_id)
        total_relevant_micro += relevant_count
        
        # Count relevant documents that were retrieved in top-k
        retrieved_relevant_count = 0
        for support_id in supporting_context_id:
            if support_id in retrieved_id:
                retrieved_relevant_count += 1
                total_retrieved_relevant_micro += 1
        
        # Calculate per-query recall (for macro-average)
        query_recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 1.0
        query_recalls.append(query_recall)
    
    # Calculate micro and macro recall
    micro_recall = total_retrieved_relevant_micro / total_relevant_micro if total_relevant_micro > 0 else 0
    macro_recall = sum(query_recalls) / len(query_recalls) if query_recalls else 0
    
    # Print results
    print(f"Top-{k} Results for ImageQ queries ({image_queries_count}):")
    print(f"  Micro-Recall@{k}: {micro_recall:.4f} ({total_retrieved_relevant_micro}/{total_relevant_micro})")
    print(f"  Macro-Recall@{k}: {macro_recall:.4f}")
    print("-" * 50)
