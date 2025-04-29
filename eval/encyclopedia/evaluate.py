"""
这个文件应该是可以直接使用的；
问题是：
这个evaluation真的合理吗？有没有更好的处理方法？比如llm-as-a-judge；
"""
import evaluation_utils
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义计算分数的函数
def compute_score(item):
    return evaluation_utils.evaluate_example(
        question=item['question'],
        reference_list=item['reference_list'],
        candidate=item['response'],
        question_type=item['question_type']
    )

# 处理 Qwen 和 Llava 的分数，使用并行化
def parallel_evaluate(results, desc="Evaluating Results", max_workers=4):
    scores = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_item = {executor.submit(compute_score, item): item for item in results}
        for future in tqdm(as_completed(future_to_item), total=len(results), desc=desc):
            try:
                scores.append(future.result())
            except Exception as e:
                print(f"Error processing item: {e}")
    return scores

if __name__ == "__main__":
    # 加载结果
   
    with open("/data/track/mm-RewardRAG/results/encyclopedia/text_kb_part/final_qwen-2-vl.json") as f:
        result = json.load(f)

    
    score_list = parallel_evaluate(result, desc="Evaluating Results", max_workers=8)

    # 计算平均分
    # qwen_avg_score = sum(qwen_score_list) / len(qwen_score_list) if qwen_score_list else 0
    avg_score = sum(score_list) / len(score_list) if score_list else 0

    # print(f"Qwen Average Score: {qwen_avg_score}")
    print(f"Average Score: {avg_score}")

    

    # 保存结果到磁盘
    output_dir = "/data/track/mm-RewardRAG/results/encyclopedia/text_kb_part/"

    

    with open(f"{output_dir}scores.json", "w") as file:
        json.dump(score_list, file, indent=4)

    with open(f"{output_dir}avg_score.json","w") as f:
        json.dump({"avg_score":avg_score},f,indent=4)

    print("Scores saved to disk.")

