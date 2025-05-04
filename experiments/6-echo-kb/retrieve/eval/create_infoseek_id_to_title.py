import pickle
data = pickle.load(open("/data/track/mm-RewardRAG/indexs/bge/infoseek/metadata.pkl","rb"))

id_to_title = {
    item["wikidata_id"]: item["wikipedia_title"] for item in data
}

with open("/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/eval/infoseek_id_to_title.pkl", "wb") as f:
    pickle.dump(id_to_title, f)
print("ID to title mapping saved.")
