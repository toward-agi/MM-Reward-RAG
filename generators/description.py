from .base import BaseGenerator
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

class DescriptionGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("description_generator")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "image" in kwargs and kwargs["image"] is not None
        assert "id" in kwargs and kwargs["id"] is not None

        prompts = self.prompts
        messages = [
            {
                "role": "system",
                "content": prompts["system"]
            }
        ]
        
        user_content = []

        user_prompt = prompts["user"]
        
        if "text" in kwargs:
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
            
        # Image content is required
       
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{self._encode_image(kwargs['image'])}"
            }
        })
            
        messages.append({
            "role": "user",
            "content": user_content
        })

        # print(messages)
        
        return messages, kwargs["id"]
    
    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        """Generate a detailed description of the image."""
        image = kwargs["image"]
        id = kwargs["id"]
        messages,id = self._prepare_messages(image=image,id=id)
        return self._generate_single(messages,id)

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        assert "image_id_list" in kwargs and kwargs["image_id_list"] is not None
        image_id_list = kwargs["image_id_list"]
        messages_list = []
        for image, id in image_id_list:
            messages, id = self._prepare_messages(image=image, id=id)
            messages_list.append((messages, id))
        return messages_list

    def generate_description_batch(self, image_id_list: List[Tuple[Image.Image, int]]):
        """
        Input format:
        [
            (image: PIL.Image, id),
            ...
        ]
        Output format:
        [
            (description, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(image_id_list=image_id_list)
        return self.generate_batch(message_list=messages)

def test():
    description_generator = DescriptionGenerator()
    image = Image.open("test/images/hk.png")
    print(description_generator.generate(image=image, id=0))
    image_list = [(image, i) for i in range(2)]
    messages_list = description_generator._prepare_messages_for_batch(image_id_list=image_list)
    # print(messages_list[0][0][0])
    print(len(messages_list))
    descriptions = description_generator.generate_batch(message_list=messages_list)
    print(descriptions)

if __name__ == "__main__":
    test()
