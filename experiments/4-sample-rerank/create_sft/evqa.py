file_path = "/data/track/mm-RewardRAG/experiments/4-sample-rerank/results/3b/evqa_sampled_rankings.json"
import json
from tqdm import tqdm
import random
evqa_file = json.load(open(file_path, "r"))
# evqa和infoseek应该都不需要特殊处理吧，都是11个，前1个是pos，其他的都是neg；
output_dir = "/data/track/mm-RewardRAG/experiments/4-sample-rerank/results/pair"
import os
os.makedirs(output_dir, exist_ok=True)
results = []
for single_question in evqa_file:
    single_result = {i: [] for i in range(len(single_question))}
    neg_start_index = 1
    for i in range(len(single_question)):
        if i < neg_start_index:
            if single_question[i][0]["useful"] != [] and single_question[i][0]["useless"] != []:
                pos = random.choice(single_question[i][0]["useful"])
                neg = random.choice(single_question[i][0]["useless"])
                single_result[i].append({
                    "useful->useless":{"pos": pos, "neg": neg}
                    }
                )
            if single_question[i][0]["useful"] != [] and single_question[i][0]["neutral"] != []:
                pos = random.choice(single_question[i][0]["useful"])
                neg = random.choice(single_question[i][0]["neutral"])
                single_result[i].append({
                    "useful->netural":{"pos": pos, "neg": neg}
                    }
                )
        else:
            if single_question[i][0]["useless"] != [] and single_question[i][0]["useful"] != []:
                pos = random.choice(single_question[i][0]["useless"])
                neg = random.choice(single_question[i][0]["useful"])
                single_result[i].append({
                    "useless->useful":{"pos": pos, "neg": neg}
                    }
                )
            if single_question[i][0]["useless"] != [] and single_question[i][0]["neutral"] != []:
                pos = random.choice(single_question[i][0]["useless"])
                neg = random.choice(single_question[i][0]["neutral"])
                single_result[i].append({
                    "useless->netural":{"pos": pos, "neg": neg}
                    }
                )

    results.append(single_result)
with open(os.path.join(output_dir, "evqa_sampled_rankings.json"), "w") as f:
    json.dump(results, f, indent=4)