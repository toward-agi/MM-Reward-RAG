input_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid.json"
"""
{
    "data_id": "infoseek_val_00000000",
    "image_id": "oven_04990048",
    "question": "What is the objective of this object?",
    "answer": [
        "energy transformation"
    ],
    "answer_eval": [
        "energy conversion",
        "energy transformation",
        "conversion"
    ],
    "data_split": "val_unseen_question",
    "entity_id": "Q178185",
    "image_path": "/wikipedia_images_full/Q178/Q178185.jpg"
},
"""

import json
import os
from PIL import Image
from tqdm import tqdm
import pickle

infoseek_data = json.load(open(input_data_file, "r"))

infoseek_document = pickle.load(open("/data/track/mm-RewardRAG/indexs/bge/infoseek/metadata.pkl","rb"))
id_to_content = {
    item['entity_id']: item['wikipedia_content'] for item in infoseek_document
}

import sys
sys.path.append("/data/track/mm-RewardRAG/")
# from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge
from core.answer_generator.generator_w_gt import VLLMAnswerGenerator_w_GT

generator = VLLMAnswerGenerator_w_GT()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/w_gt/results/infoseek"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "ti",
            "text": item["question"],
            "image": Image.open(item["image_path"]).resize((512, 512)).convert("RGB"),
            "id": item["data_id"],
            "document": id_to_content[item["entity_id"]],
        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
