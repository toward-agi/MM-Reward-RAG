import json
from tqdm import tqdm

# 加载数据
webqa_test = "/data/dataset/webqa/WebQA_data_first_release/WebQA_test.json"
webqa_train_val = "/data/dataset/webqa/WebQA_data_first_release/WebQA_train_val.json"

# 处理测试集
test_data = json.load(open(webqa_test, "r"))
imgs = []
texts = []
imgs_ids = set()  # 使用set而不是list来提高查找效率
texts_ids = set()

# 从测试集提取并去重
print("Processing test data...")
for item in tqdm(test_data):
    if "img_Facts" in test_data[item]:
        for img_fact in test_data[item]["img_Facts"]:
            if img_fact["image_id"] not in imgs_ids:
                imgs.append(img_fact)
                imgs_ids.add(img_fact["image_id"])
    if "txt_Facts" in test_data[item]:
        for txt_fact in test_data[item]["txt_Facts"]:
            if txt_fact["snippet_id"] not in texts_ids:
                texts.append(txt_fact)
                texts_ids.add(txt_fact["snippet_id"])

# 保存测试集结果
with open("/data/dataset/webqa/webqa_test_img_facts.json", "w") as f:
    json.dump(imgs, f, indent=2)
with open("/data/dataset/webqa/webqa_test_txt_facts.json", "w") as f:
    json.dump(texts, f, indent=2)

# 处理训练验证集
train_val_data = json.load(open(webqa_train_val, "r"))
imgs = []
texts = []
imgs_ids = set()
texts_ids = set()

# 从训练验证集提取并去重
print("Processing train/val data...")
for item in tqdm(train_val_data):
    data = train_val_data[item]
    # 处理正样本和负样本
    for key in ['img_posFacts', 'img_negFacts']:
        if key in data:
            for img_fact in data[key]:
                if img_fact["image_id"] not in imgs_ids:
                    imgs.append(img_fact)
                    imgs_ids.add(img_fact["image_id"])
    
    for key in ['txt_posFacts', 'txt_negFacts']:
        if key in data:
            for txt_fact in data[key]:
                if txt_fact["snippet_id"] not in texts_ids:
                    texts.append(txt_fact)
                    texts_ids.add(txt_fact["snippet_id"])

# 保存训练验证集结果
with open("/data/dataset/webqa/webqa_train_val_img_facts.json", "w") as f:
    json.dump(imgs, f, indent=2)
with open("/data/dataset/webqa/webqa_train_val_txt_facts.json", "w") as f:
    json.dump(texts, f, indent=2)

print(f"Done! Saved all files.")