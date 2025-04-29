import jsonlines

input_data_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_dev.jsonl"
# output_data_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_dev_textQ_ImageQ.jsonl"
output_data_file = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl"
cnt = 0
output_writer = jsonlines.open(output_data_file, mode='w')
with jsonlines.open(input_data_file) as reader:
    for item in reader:
        # if item['metadata']['type'] != 'TextQ' and item['metadata']['type'] != 'ImageQ':
        if 'TableQ' not in item['metadata']['type']:
            if 'TextQ' in item['metadata']['type'] or 'ImageQ' in item['metadata']['type']:
                output_writer.write(item)
                cnt += 1
        #         print(item['metadata']['type'])
        #         continue
        # print(item['metadata']['type'])
        # cnt += 1
        # output_writer.write(item)
print(cnt)