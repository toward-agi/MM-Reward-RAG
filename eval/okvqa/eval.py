import json

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



if __name__=="__main__":
    #load data
    with open("/data/track/mm-RewardRAG/results/okvqa/text_kb_whole/final_qwen-2-vl.json") as f:
        result = json.load(f)
    
    acc,acc_list = check_all(result)

    with open(f"/data/track/mm-RewardRAG/results/okvqa/text_kb_whole/acc_list.json","w") as f:
        json.dump(acc_list,f,indent=4)

    avg_acc = acc / len(acc_list)
    with open("/data/track/mm-RewardRAG/results/okvqa/text_kb_whole/avg_acc.json","w") as f:
        json.dump({"avg_acc":avg_acc},f,indent=4)