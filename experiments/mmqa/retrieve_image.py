import json
import os
from tqdm import tqdm
import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

valid_dataset = {}
ks = [1, 5, 10, 20, 50, 100]
for k in ks:
    with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            valid_dataset[item['qid']] = item
    output_dir = "results/mmqa/image_kb"
    retrieved_results_file = f"{output_dir}/retrieved_results_{k}.json"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(retrieved_results_file):
        embedder = JinaClipV2Embedder() #将question或者image转换为向量
        indexer = FaissIndexer(mode="jina-clip-v2") #建的索引
        
        # 加载index
        # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
        indexer.load("visual_index_gpu/mm_qa")
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
            results = indexer.search_batch(query_vectors, top_k=k) #检索,这里top-k就是返回的检索结果的数量
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
        total_relevant = 0
        total_retrieved_relevant = 0
        for id, item in id_to_retrieved.items():
            type = item["type"]
            if type != "ImageQ":
                continue
            retrieved_id = [i['metadata']['id'] for i in item["retrieved"][2]]
            supporting_context_id = [i['doc_id'] for i in item["supporting_context"]]
            # Count total relevant documents
            total_relevant += len(supporting_context_id)
            
            # Count relevant documents that were retrieved
            for support_id in supporting_context_id:
                if support_id in retrieved_id:
                    total_retrieved_relevant += 1
                    
        recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
        print(f"Recall@{k}: {recall:.4f} ({total_retrieved_relevant}/{total_relevant})")
