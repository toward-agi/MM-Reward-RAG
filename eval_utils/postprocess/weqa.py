import json

prediction_file = "/data/track/mm-RewardRAG/experiments/5-answer/w_all_hn/results/3b/webqa/answers.jsonl"
prediction_data = [
    json.loads(line) for line in open(prediction_file, "r")
]

gt_file = "/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json"

id_to_data = json.load(open(gt_file, "r"))

triples = [
    {
        "question": id_to_data[data["data_id"]]["Q"],
        "ground_truth": id_to_data[data["data_id"]]["A"][0],
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
        "answer": data["answer"] if results[data["data_id"]] == "False" else id_to_data[data["data_id"]]["A"][0]
    }
    for data in prediction_data
]

new_prediction_file = prediction_file.replace("answers.jsonl", "answers_with_gt.jsonl")
with open(new_prediction_file, "w") as f:
    for data in new_prediction_data:
        f.write(json.dumps(data) + "\n")
