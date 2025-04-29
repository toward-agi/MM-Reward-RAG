import sys
sys.path.append("/data/track/mm-RewardRAG")

from core.embeddings.e5_embedding import E5TextEmbedder

import json
import os
from tqdm import tqdm
import numpy as np

embedder = E5TextEmbedder()

"""
{"id": "0", "contents": "\"Evan Morris\"\nEvan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington, D.C. He began his career in Washington as an intern in the Clinton White House at age 18. He began his lobbying work at Patton Boggs before moving on to Roche in 2005. His early work at Roche involved government policy on Medicare and Medicaid, flu preparedness, and hepatitis C. His work at Genentech is being investigated by the Wall Street Journal, which states that \u201cshaping up to be one of the biggest U.S. investigations into"}
{"id": "1", "contents": "\"Horatio Hale\"\nconsisted of an Algonkin vocabulary, which he gathered from a band of Indians who had camped on the college grounds. Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate, for the post of ethnologist and philologist, and obtained the appointment. From 1838 to 1842, he was employed in the work of the expedition, visiting South America, Australasia, Polynesia, and North-western America, then known as Oregon. From this point he returned overland. The Hale Passages of Puget Sound were named in recognition of"}
{"id": "2", "contents": "\"Dibba Al-Hisn\"\nis believed to be the site where the Portuguese during the Habsburg Dynasty built a fort and a wall around the city. In August 1648, the Arabs besieged Muscat, Oman, and on October 31, 1648, a treaty was signed between the two opponents. The terms required the Portuguese to build the fortress of Kuriyat, Dibba Al-Hisn and Matrah (Oman). Note: There have been land disputes between Dibba Al-Hisn and Dibba Al-Baya, but these were resolved in the 1990s. </div> A branch of the Community College of the University of Sharjah is available in Dibba Al-Hisn. Alkhalidya Secondary School Dibba Al-Hisn"}
{"id": "3", "contents": "\"Ao Oni (film)\"\nthe door as a big blue hand grabs his head. The group hears Takeshi's scream and come back to check on him. As they reach the room, they see the door open and Takeshi, bleeding, is thrown out. He asks for help before being dragged back and the door closes. As the group enters the room, they finds Takeshi's dismembered body. Anna and Hiroshi pressure Mika about Takuro's whereabouts and about Naoki, but Mika refuses to tell them and runs away, chased by Hiroshi. Anna confronts Shun who describes that this event is the same as the game he created."}
...
"""

input_data_file = "/data/wiki-corpus/wiki-18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
text_feature_dir = "/data/wiki-corpus/wiki-18/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5-m-large"

import os
os.makedirs(text_feature_dir, exist_ok=True)

with open(input_data_file, "r") as f:
    text_data = [json.loads(line) for line in f]

print(f"Total number of text data: {len(text_data)}")

# Embed text
batch_size = 1024

os.makedirs(text_feature_dir, exist_ok=True)

for i in tqdm(range(0, len(text_data), batch_size)):
    batch = text_data[i:i + batch_size]
    texts = [item["contents"] for item in batch]
    text_features = embedder.embed_corpus(texts)

    batch_file = os.path.join(text_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, text_features)
