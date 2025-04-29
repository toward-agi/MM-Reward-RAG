input_data_file = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json"
"""
{'wikipedia_title': 'Smilax bona-nox',
'wikipedia_url': 'https://en.wikipedia.org/wiki/Smilax_bona-nox',
'question_original': 'What kind of medical usage has this plant?',
'question': 'What kind of medical usage has this plant?',
'question_type': 'templated',
'answer': 'help urinary tract infections and as an antioxidant.|help urinary tract infections and as an antioxidant.|urinary tract infections, antioxidant',
'evidence': 'The roots of this plant are known to help urinary tract infections and as an antioxidant.|The roots of this plant are known to help urinary tract infections and as an antioxidant.|The roots of this plant are known to help urinary tract infections and as an antioxidant.',
'evidence_section_id': '4',
'evidence_section_title': 'Ethnobotany|Ethnobotany|Ethnobotany',
'dataset_name': 'inaturalist',
'dataset_category_id': 6214,
'wikipedia_url_used_in_train': 1,
'encyclopedic_vqa_split': 'test',
'dataset_image_ids': '2715027',
'entity_id': 'entity_1677263',
'question_id': 'test_0',
'dataset_image_id': '2715027',
'image_path': 'iNaturalist2021/val/06214_Plantae_Tracheophyta_Liliopsida_Liliales_Smilacaceae_Smilax_bona-nox/09dd9c22-8cbc-4b19-bbdd-1003986c048c.jpg'}
"""

import json
import os
from PIL import Image
from tqdm import tqdm

infoseek_data = json.load(open(input_data_file, "r"))

import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge

generator = VLLMAnswerGenerator_wo_Knowledge()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/wo_knowledge/results/evqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "ti",
            "text": item["question"],
            "image": Image.open( f"/data/mingyan/encyclopedic/{item['image_path']}").resize((512, 512)).convert("RGB"),
            "id": item["question_id"],
        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
