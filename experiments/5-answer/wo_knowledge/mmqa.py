input_data_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl"
"""
{
    "qid":...,
    "question":...,
    "answer":...,
    "metadata":...,
    "supporting_context":...,
}
"""

import json
import os
from PIL import Image
from tqdm import tqdm

infoseek_data = [
    json.loads(line)
    for line in open(input_data_file, "r")
]

import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge

generator = VLLMAnswerGenerator_wo_Knowledge()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/wo_knowledge/results/mmqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "t",
            "text": item["question"],
            "id": item["qid"],
        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
