from generators.base import BaseGenerator

from typing import List, Dict, Any, Optional, Tuple

class BaselineLLMGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("baseline_llm_generator")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "id" in kwargs and kwargs["id"] is not None
        assert "user_prompt" in kwargs and kwargs["user_prompt"] is not None

        prompts = self.prompts
        messages = [
            # {
            #     "role": "system",
            #     "content": prompts["system"]
            # }, for baseline, system prompt is not necessary
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": kwargs["user_prompt"]
                    },
                ]
            }
        ]
        

        # print(messages)
        
        return messages, kwargs["id"]
    
    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        id = kwargs["id"]
        user_prompt = kwargs["user_prompt"]
        messages,id = self._prepare_messages(id=id,user_prompt=user_prompt)
        return self._generate_single(messages,id)

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        assert "id_prompts_list" in kwargs and kwargs["id_prompts_list"] is not None
        id_prompts_list = kwargs["id_prompts_list"]
        messages_list = []
        for id, user_prompt in id_prompts_list:
            messages, id = self._prepare_messages(id=id, user_prompt=user_prompt)
            messages_list.append((messages, id))
        return messages_list

    def generate_description_batch(self, id_list: List[Tuple[int, str]]) -> List[Tuple[str, int]]:
        """
        Input format:
        [
            (id, user_prompt),
            ...
        ]
        Output format:
        [
            (response, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(id_prompts_list=id_list)
        return self.generate_batch(message_list=messages)

def test():
    llm_generator = BaselineLLMGenerator()
    # image = Image.open("test/images/hk.png")
    # print(description_generator.generate(image=image, id=0))
    # image_list = [(image, i) for i in range(2)]
    # messages_list = description_generator._prepare_messages_for_batch(image_id_list=image_list)
    # # print(messages_list[0][0][0])
    # print(len(messages_list))
    # descriptions = description_generator.generate_batch(message_list=messages_list)
    # print(descriptions)
    input_data_file = "/data/dataset/webqa/WebQA_data_first_release/WebQA_train_val.json"
    """
    [
     "d5bbc6d80dba11ecb1e81171463288e9": {
        "Q": "\"Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color?\"",
        "A": [
            "\"Yes, both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida are beige.\""
        ],
        "topic": "strange architecture",
        "split": "train",
        "Qcate": "YesNo",
        "Guid": "d5bbc6d80dba11ecb1e81171463288e9" }
    ]
    """

    output_file_path = "results/baseline_llm/llava-16-7b.jsonl"
    import os
    os.makedirs("results/baseline_llm", exist_ok=True)

    import json
    from tqdm import tqdm
    input_data = json.load(open(input_data_file))
    # filter the valid data
    valid_data = []
    for key in input_data:
        data = input_data[key]
        if data["split"] != "train":
            valid_data.append({
                "data_id": data["Guid"],
                "question": data["Q"],
            })
        
    print(f"valid data size: {len(valid_data)}")
    exit()

    buffer_size = 1000
    for i in tqdm(range(0, len(valid_data), buffer_size)):
        buffer = valid_data[i:i + buffer_size]
        id_prompts_list = []
        for data in buffer:
            id_prompts_list.append((data["data_id"], data["question"]))
        descriptions = llm_generator.generate_description_batch(id_list=id_prompts_list)
        with open(output_file_path, "a") as f:
            for description, id in descriptions:
                f.write(json.dumps({"data_id": id, "response": description}) + "\n")
if __name__ == "__main__":
    test()
