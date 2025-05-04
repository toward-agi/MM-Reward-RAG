import json
from datasets import load_dataset

def check_one(item:dict):
    response = item["response"]
    answers = item["answers"]
    for ans in answers:
        if ans in response:
            return True
    return False

def check_all(results:list):
    acc = 0.0
    acc_list = []
    for result in results:
        if check_one(result):
            acc_list.append(1.0)
            acc += 1.0
        else:
            acc_list.append(0.0)
            acc += 0.0
    return acc,acc_list

def eval_okvqa(responses_path: str):
    # load a jsonl file
    data = [
        json.loads(line)
        for line in open(responses_path, 'r')
    ]
    id_responses = {
        item["data_id"]: item["answer"]
        for item in data
    }

    val_dataset = list(load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR","OKVQA_data")["test"])
    id_to_answer = {
        item["question_id"]: item["answers"]
        for item in val_dataset
    }

    answer_responses = []

    for id in id_to_answer.keys():
        if id not in id_responses:
            print(f"Missing response for {id}")
            continue
        answer_responses.append({
            "data_id": id,
            "answers": id_to_answer[id],
            "response": id_responses[id],
        })
    
    acc, acc_list = check_all(answer_responses)
    avg_acc = acc / len(acc_list)
    print(f"Average accuracy: {avg_acc}")

def test():
    eval_okvqa("/data/track/mm-RewardRAG/experiments/5-answer/preflmr/results/3b/with_reranker_7b/okvqa/3/answers.jsonl")

if __name__ == "__main__":
    test()
