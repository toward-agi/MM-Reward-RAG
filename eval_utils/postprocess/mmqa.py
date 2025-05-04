import json

prediction_file = "/data/track/mm-RewardRAG/experiments/5-answer/w_all_hn/results/3b/mmqa/answers.jsonl"
prediction_data = [
    json.loads(line) for line in open(prediction_file, "r")
]

gt_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl"
gt_data = [
    json.loads(line) for line in open(gt_file, "r")
]
id_to_data = {
    data["qid"]: data for data in gt_data
}

triples = [
    {
        "question": id_to_data[data["data_id"]]["question"],
        "ground_truth": id_to_data[data["data_id"]]["answers"][0]["answer"],
        "prediction": data["answer"],
        "data_id": data["data_id"],
    }
    for data in prediction_data
]

import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.answer_checker.base_checker import AnswerChecker
checker = AnswerChecker()

results = checker.generate_answer(triples)

new_prediction_data = [
    {
        "data_id": data["data_id"],
        "answer": data["answer"] if results[data["data_id"]] == "False" else id_to_data[data["data_id"]]["answers"][0]["answer"]
    }
    for data in prediction_data
]

new_prediction_file = prediction_file.replace("answers.jsonl", "answers_with_gt.jsonl")
with open(new_prediction_file, "w") as f:
    for data in new_prediction_data:
        f.write(json.dumps(data) + "\n")
