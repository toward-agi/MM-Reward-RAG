import json 
from pathlib import Path

label='hard_neg'
in_path = Path(f'/data/track/mm-RewardRAG/results/webqa/image_kb/qwen-2-vl-{label}.json')
out_path = Path(f'/data/track/mm-RewardRAG/results/webqa/image_kb/qwen-2-vl-{label}.tsv')

header = ['Guid', 'Qcate', 'Q', 'A', 'Output_conf', 'Output']
with open(in_path, 'r') as f:
    data = json.load(f)

with open(out_path, 'w') as f:
    f.write('\t'.join(header) + '\n')
    for item in data:
        # use json.dumps(my_list) to convert list to string
        values = [str(item['Guid']), str(item['Qcate']), str(item['Q']), json.dumps(item['A']), str(item['Output_conf']), json.dumps(item['Output'])]
        f.write('\t'.join(values) + '\n')
