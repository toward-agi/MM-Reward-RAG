from tqdm import tqdm
import os
import json
# 这里是加载用来测试的数据集
import sys
sys.path.append("/data/track/mm-RewardRAG/")

from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
import json
from tqdm import tqdm
from PIL import Image


valid_dataset = json.load(open("/data/dataset/mm_RewardRAG/InfoSeek/valid.json", "r"))

output_dir = "results/infoseek/text_kb/jina_clip_v2"
retrieved_results_file = f"{output_dir}/retrieved_results.json"
import os
os.makedirs(output_dir, exist_ok=True)

# 先做检索
if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder() #将question或者image转换为向量
    indexer = FaissIndexer(mode="jina-clip-v2") #建的索引
    
    # 加载index
    # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
    indexer.load("/data/track/mm-RewardRAG/indexs/jina-clip-v2/textual_index_gpu/infoseek")
    bs = 512
    id_to_retrieved = []
    for i in tqdm(range(0, len(valid_dataset), bs)):
        batch = valid_dataset[i:i+bs]
        data_ids = [item["data_id"] for item in batch]
        questions = [item["question"] for item in batch]
        query_vectors = embedder.embed_text(questions)
        print(query_vectors.shape)
        results = indexer.search_batch(query_vectors, top_k=100) #检索,这里top-k就是返回的检索结果的数量
        for data_id, result in zip(data_ids, results):
            id_to_retrieved.append({
                "data_id": data_id,
                "retrieved": result
            })

    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))