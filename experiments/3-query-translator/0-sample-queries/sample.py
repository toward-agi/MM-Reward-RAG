import json
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

evqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/evqa_train_1k.json"
infoseek_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/infoseek_train_1k.json"
mmqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k.json"
webqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k.json"

with open(evqa_path, "r") as f:
    evqa_data = json.load(f)
with open(infoseek_path, "r") as f:
    infoseek_data = json.load(f)
with open(mmqa_path, "r") as f:
    mmqa_data = json.load(f)
with open(webqa_path, "r") as f:
    webqa_data = json.load(f)

evqa_queries = [
    {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_paths"][0]).convert("RGB").resize((512, 512)),
    }
    for item in evqa_data
]

logger.info(f"Loaded {len(evqa_queries)} queries from EVQA dataset.")

infoseek_queries = [
    {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_path"]).convert("RGB").resize((512, 512)),
    }
    for item in infoseek_data
]

logger.info(f"Loaded {len(infoseek_queries)} queries from InfoSeek dataset.")

mmqa_queries = [
    {
        "type": "t",
        "text": item["question"]
    }
    for item in mmqa_data
]

logger.info(f"Loaded {len(mmqa_queries)} queries from MMQA dataset.")

webqa_queries = [
    {
        "type": "t",
        "text": item["Q"]
    }
    for item in webqa_data
]

logger.info(f"Loaded {len(webqa_queries)} queries from WebQA dataset.")

import sys

sys.path.append("/data/track/mm-RewardRAG")
from core.query_translator.openai_sample_translator import VLLMSampleTranslator

translator = VLLMSampleTranslator(generator_name="vllm_sample_translator_3b")

output_dir = "/data/track/mm-RewardRAG/experiments/3-query-translator/0-sample-queries/results/3b"
import os
os.makedirs(output_dir, exist_ok=True)

evqa_results = translator.generate(evqa_queries)
with open(f"{output_dir}/evqa_results.json", "w") as f:
    json.dump(evqa_results, f, indent=4)
logger.info(f"Saved EVQA results to {output_dir}/evqa_results.json")

infoseek_results = translator.generate(infoseek_queries)
with open(f"{output_dir}/infoseek_results.json", "w") as f:
    json.dump(infoseek_results, f, indent=4)
logger.info(f"Saved InfoSeek results to {output_dir}/infoseek_results.json")

mmqa_results = translator.generate(mmqa_queries)
with open(f"{output_dir}/mmqa_results.json", "w") as f:
    json.dump(mmqa_results, f, indent=4)
logger.info(f"Saved MMQA results to {output_dir}/mmqa_results.json")

webqa_results = translator.generate(webqa_queries)
with open(f"{output_dir}/webqa_results.json", "w") as f:
    json.dump(webqa_results, f, indent=4)
logger.info(f"Saved WebQA results to {output_dir}/webqa_results.json")
