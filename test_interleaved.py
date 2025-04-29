import base64
import os
from openai import OpenAI

# Set up client with your vLLM endpoint
client = OpenAI(
    api_key="EMPTY",  # Can be any string when using vLLM
    base_url="http://0.0.0.0:8000/v1"  # Point to your vLLM server
)

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_qwen_vl_model(contents):
    """
    Query the vLLM-deployed Qwen2.5-VL model with interleaved text and images
    
    Args:
        contents: List of content objects (text and images interleaved)
    """
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {"role": "user", "content": contents}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# Example of interleaved text and images
if __name__ == "__main__":
    # Create an interleaved content list
    interleaved_content = [
        {
            "type": "text",
            "text": "Here's the first image of an animal:"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_to_base64('cat.jpeg')}"
            }
        },
        {
            "type": "text",
            "text": "Can you describe both animals and compare them?"
        }
    ]
    
    # Query the model with interleaved content
    response = query_qwen_vl_model(interleaved_content)
    print(response)
