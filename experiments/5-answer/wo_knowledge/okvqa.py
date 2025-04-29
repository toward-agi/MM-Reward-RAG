from datasets import load_dataset

okvqa = load_dataset("lmms-lab/OK-VQA")["val2014"]


import json
import os
from PIL import Image
from tqdm import tqdm


import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge

generator = VLLMAnswerGenerator_wo_Knowledge()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/wo_knowledge/results/okvqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

batch_size = 2048
for i in tqdm(range(0, len(okvqa), batch_size)):
    batch = okvqa[i:i + batch_size]
    keys = batch.keys()
    values = list(zip(*batch.values()))
    buffer = [dict(zip(keys, value_tuple)) for value_tuple in values]
    queries = [
        {
            "type": "ti",
            "text": item["question"],
            "image": item["image"].convert("RGB"),
            "id": item["question_id"],
        }
        for item in buffer
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")


