import json
import os
from tqdm import tqdm
import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.indexing.faiss_indexer import FaissIndexer
# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.embeddings.bge_text_embedding import BGETextEmbedder

valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))
ks = [1, 5, 10, 20, 50, 100]
for k in ks:
    output_dir = "results/webqa/bge_clip/text_kb"
    retrieved_results_file = f"{output_dir}/retrieved_results_{k}.json"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(retrieved_results_file):
        embedder = BGETextEmbedder() #将question或者image转换为向量
        indexer = FaissIndexer(mode="bge") #建的索引
        
        # 加载index
        # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
        indexer.load("indexs/bge/webqa")
        bs = 512
        id_to_retrieved = {}
        valid_dataset_list = list(valid_dataset.items())
        for i in tqdm(range(0, len(valid_dataset_list), bs)):
            batch = valid_dataset_list[i:i+bs]
            data_ids = [item[0] for item in batch]
            questions = [item[1]["Q"] for item in batch]
            supporting_contexts = [item[1]["txt_posFacts"] for item in batch] #这里处理的没问题，和mmqa的还不一样；
            query_vectors = embedder.embed_text(questions)
            results = indexer.search_batch(query_vectors, top_k=k) #检索,这里top-k就是返回的检索结果的数量
            for data_id, result, supporting_context in zip(data_ids, results, supporting_contexts):
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": result,
                    "supporting_context": supporting_context,
                }
        with open(retrieved_results_file, "w") as f:
            json.dump(id_to_retrieved, f, indent=2)
    else:
        id_to_retrieved = json.load(open(retrieved_results_file, "r"))
        
        # Micro recall calculation (original)
        total_relevant = 0
        total_retrieved_relevant = 0
        
        # Add variables for macro recall
        query_recalls = []
        
        for id, item in id_to_retrieved.items():
            retrieved_id = [i['snippet_id'] for i in item["retrieved"][2]]
            supporting_context_id = [i['snippet_id'] for i in item["supporting_context"]]
            
            # For micro recall (original code)
            total_relevant += len(supporting_context_id)
            
            # For macro recall - count relevant docs retrieved for this query
            query_retrieved_relevant = 0
            
            # Count relevant documents that were retrieved
            for support_id in supporting_context_id:
                if support_id in retrieved_id:
                    total_retrieved_relevant += 1
                    query_retrieved_relevant += 1
            
            # Calculate recall for this individual query
            query_recall = query_retrieved_relevant / len(supporting_context_id) if len(supporting_context_id) > 0 else 1.0
            query_recalls.append(query_recall)
        
        # Calculate micro recall (original)
        micro_recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
        
        # Calculate macro recall (new)
        macro_recall = sum(query_recalls) / len(query_recalls) if query_recalls else 0
        
        # Print both metrics
        print(f"Micro-Recall@{k}: {micro_recall:.4f} ({total_retrieved_relevant}/{total_relevant})")
        print(f"Macro-Recall@{k}: {macro_recall:.4f}")