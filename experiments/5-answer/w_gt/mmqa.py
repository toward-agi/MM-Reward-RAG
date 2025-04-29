input_data_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl"
"""
{
    "qid":...,
    "question":...,
    "answer":...,
    "metadata":...,
    "supporting_context":...,
}
>>> pprint(data[0])
{'answers': [{'answer': "Breeders' Cup Mile",
              'image_instances': [],
              'modality': 'text',
              'table_indices': [],
              'text_instances': [{'doc_id': 'a7e8300eb39c0cd7326d8dce5e096099',
                                  'part': 'text',
                                  'start_byte': 235,
                                  'text': "Breeders' Cup Mile"}],
              'type': 'string'}],
 'metadata': {'image_doc_ids': ['117d500aaa630023c4038b8268b309c0',
                                '42c88eafedfab5203c72e1d2ec1c2f37',
                                '5588a1833b8fa95fe1b0ed520e447d64'],
              'intermediate_answers': [],
              'modalities': ['text'],
              'table_id': '58a217d68016e334ee624bddaa6c93ec',
              'text_doc_ids': ['f6096bbed40d6d8f233b23ac660aeba8',
                               '03038ac6f16beadcc957558de566cdad',
                               'd5627da740ba433ce4d9bc5f4c9108da',
                               '99461417ae928d8fa041783372a8985b',
                               '879c392dfa9f4a4c6655f479b598b7a0',
                               'a1bf7477d5bd41f15f30f0c3ad71159b',
                               'a7e8300eb39c0cd7326d8dce5e096099',
                               '0baec69396d3ff898a370d40df9f3db1',
                               'a9223c0243cc390eb59a9e35420b9460',
                               '2d2a46f132c1e261e7c6d8b1e058a210'],
              'type': 'TextQ',
              'wiki_entities_in_answers': [{'text': "Breeders' Cup Mile",
                                            'url': "https://en.wikipedia.org/wiki/Breeders'_Cup_Mile",
                                            'wiki_title': "Breeders' Cup "
                                                          'Mile'}],
              'wiki_entities_in_question': [{'text': 'Churchill Downs',
                                             'url': 'https://en.wikipedia.org/wiki/Churchill_Downs',
                                             'wiki_title': 'Churchill Downs'},
                                            {'text': 'Barathea (horse)',
                                             'url': 'https://en.wikipedia.org/wiki/Barathea_(horse)',
                                             'wiki_title': 'Barathea (horse)'},
                                            {'text': 'Churchill Downs',
                                             'url': 'https://en.wikipedia.org/wiki/Churchill_Downs',
                                             'wiki_title': 'Churchill Downs'}]},
 'qid': '5fb253761e72a6048b8b759f04c3b784',
 'question': 'What award did Barathea win at Churchill Downs in 1994?',
 'supporting_context': [{'doc_id': '0baec69396d3ff898a370d40df9f3db1',
                         'doc_part': 'text'},
                        {'doc_id': 'a7e8300eb39c0cd7326d8dce5e096099',
                         'doc_part': 'text'}]}
>>> """

"""
>>> data = pickle.load(open("/data/track/mm-RewardRAG/indexs/bge/mm_qa/metadata.pkl","rb"))
>>> data[0]
{'title': 'Monica Lewinsky', 'url': 'https://en.wikipedia.org/wiki/Monica_Lewinsky', 'id': '9dc9fabf3a930c7b479beb08e83fb1fb', 'text': 'Monica Samille Lewinsky (born July 23, 1973) is an American activist, television personality, fashion designer, and former White House intern. President Bill Clinton admitted to having had what he called an "inappropriate relationship" with Lewinsky while she worked at the White House in 1995–1996. The affair and its repercussions (which included Clinton\'s impeachment) became known later as the Clinton–Lewinsky scandal.'}
>>> 
"""

"""
>>> data[0]
{'metadata': {'title': 'Republican Party (United States)', 'url': 'https://en.wikipedia.org/wiki/Republican_Party_(United_States)', 'id': '35b31d9b4f723f806fd32662ef29edf7', 'path': '35b31d9b4f723f806fd32662ef29edf7.jpg'}, 'image_path': '/data/dataset/multimodal_qa/final_dataset_images/35b31d9b4f723f806fd32662ef29edf7.jpg'}
>>>
"""

import json
import os
from PIL import Image
from tqdm import tqdm

infoseek_data = [
    json.loads(line)
    for line in open(input_data_file, "r")
]

import pickle
image_kb = pickle.load(open("/data/track/mm-RewardRAG/indexs/clip-large/mm_qa/metadata.pkl", "rb"))
image_kb = {item["metadata"]["id"]: item for item in image_kb}

text_kb = pickle.load(open("/data/track/mm-RewardRAG/indexs/bge/mm_qa/metadata.pkl", "rb"))
text_kb = {item["id"]: item for item in text_kb}

import sys
sys.path.append("/data/track/mm-RewardRAG/")
# from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge
from core.answer_generator.generator_w_gt import VLLMAnswerGenerator_w_GT

generator = VLLMAnswerGenerator_w_GT()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/w_gt/results/mmqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

from PIL import Image

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "t",
            "text": item["question"],
            "id": item["qid"],
            "text_documents": [text_kb[doc["doc_id"]]["text"] for doc in item["supporting_context"] if doc["doc_part"] == "text"],
            "image_documents": [
                {
                    "image": Image.open(image_kb[doc["doc_id"]]["image_path"]).convert("RGB").resize((512, 512)),
                    "caption": image_kb[doc["doc_id"]]["metadata"]["title"],
                }
                for doc in item["supporting_context"] if doc["doc_part"] == "image"
            ]
        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
