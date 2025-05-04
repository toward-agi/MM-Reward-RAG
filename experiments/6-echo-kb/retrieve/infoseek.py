import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.indexing.echo_index import ClipRetriever

from PIL import Image
import os

retriever = ClipRetriever(model='eva-clip', device='cuda')

kb_json_path = '/data/dataset/echo/infoseek/wiki_100_dict_v4.json'
faiss_index_load_path = '/data/dataset/echo/infoseek/'

retriever.load_knowledge_base(knowledge_base_path=kb_json_path)

retriever.load_faiss_index(load_index_path=faiss_index_load_path)
print("FAISS index loaded.")

test_file_path = "/data/dataset/mm_RewardRAG/InfoSeek/valid.json"

# 对evqa的话，就按照这个做一遍1，3，5，10，20，50，100的检索，直接检索100，然后挂个7b的模型去看结果；
# TODO: prompt那里是不是其实还可以优化的，我还是比较随意的感觉；
import json
test_data = json.load(open(test_file_path, "r"))

output_dir = "/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/results/infoseek"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "retrieved.pkl")
import pickle

batch_size = 512

from tqdm import tqdm
results = []
for i in tqdm(range(0, len(test_data), batch_size)):
    batch_data = test_data[i:i + batch_size]
    image_list = [Image.open(item["image_path"]).convert("RGB") for item in batch_data]
    retrieved_results = retriever.retrieve_image_faiss_batch(image_list, top_k=100, return_entry_list=True)
    results.extend(retrieved_results)

with open(output_file, "wb") as f:
    pickle.dump(results, f)
print(f"Results saved to {output_file}")
