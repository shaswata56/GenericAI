from langchain_core.tools import ToolException
from pillow_heif import register_heif_opener
from abc import ABC, abstractmethod
from helpers.utils import get_hash
from PIL import Image
import requests
import base64
import json
import os
import io

register_heif_opener()

cache = {}

class ImageDescriptionProvider(ABC):
    @abstractmethod
    def get_image_description(self, image_file_path: str) -> str:
        pass

    @abstractmethod
    def get_image_description_from_encoded(self, encoded_image: str) -> str:
        pass

    def encode_image(self, file_path):
        with Image.open(file_path) as image:
            with io.BytesIO() as img_bytes:
                jpeg_img = image.convert('RGB')
                jpeg_img.save(img_bytes, format='JPEG')
                return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

class TogetherAIImageDescriptionProvider(ImageDescriptionProvider):
    def get_image_description(self, image_file_path: str) -> str:
        return self.get_image_description_from_encoded(self.encode_image(image_file_path))
    
    def get_image_description_from_encoded(self, encoded_image: str) -> str:
        response = requests.post(
            url="https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": os.getenv('VLM_MODEL'),
                "messages": [
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": "Describe the image in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }]
                }]
            })
        )
        resp_obj = json.loads(response.text)
        print(resp_obj)
        return resp_obj['choices'][0]['message']['content']


class OpenAIImageDescriptionProvider(ImageDescriptionProvider):
    def get_image_description(self, image_file_path: str) -> str:
        return self.get_image_description_from_encoded(self.encode_image(image_file_path))
        
    def get_image_description_from_encoded(self, encoded_image: str) -> str:
        # Implement OpenAI's image description API here
        pass

class OllamaImageDescriptionProvider(ImageDescriptionProvider):
    def get_image_description(self, image_file_path: str) -> str:
        return self.get_image_description_from_encoded(self.encode_image(image_file_path))
        
    def get_image_description_from_encoded(self, encoded_image: str) -> str:
        url = 'http://localhost:11434/api/generate'
        payload = json.dumps({
            "model": os.getenv('VLM_MODEL'),
            "prompt": "Describe the image in detail",
            "stream": False,
            "images": [encoded_image]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        resp_obj = json.loads(response.text)
        print(resp_obj)
        return resp_obj['response']

class ImageDescriptionProviderFactory:
    @staticmethod
    def get_provider(provider_name: str) -> ImageDescriptionProvider:
        if provider_name.lower() == "together":
            return TogetherAIImageDescriptionProvider()
        elif provider_name.lower() == "openai":
            return OpenAIImageDescriptionProvider()
        elif provider_name.lower() == "ollama":
            return OllamaImageDescriptionProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

def encode_image_from_bytes(img_bytes):
    with Image.open(img_bytes) as image:
        jpeg_img = image.convert('RGB')
        output_bytes = io.BytesIO()
        jpeg_img.save(output_bytes, format='JPEG')
        jpeg_img.close()
        image.close()
        return base64.b64encode(output_bytes.getvalue()).decode('utf-8')

def get_image_description(image_file_path: str) -> str:
    image_file_path = image_file_path.strip()
    image_file_path = os.path.normpath(image_file_path)
    
    file_hash = get_hash(image_file_path)
    if file_hash in cache:
        return cache[file_hash]
    else:
        provider_name = os.getenv("VLM_PROVIDER", "together")
        provider = ImageDescriptionProviderFactory.get_provider(provider_name)
        content = provider.get_image_description(image_file_path)
        cache[file_hash] = content
    return content

def get_image_description_url(image_url: str) -> str:
    image_url = image_url.strip()

    if image_url in cache:
        return cache[image_url]
    else:
        provider_name = os.getenv("VLM_PROVIDER", "together")
        provider = ImageDescriptionProviderFactory.get_provider(provider_name)
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ToolException("Can not fetch image from this url")
        img_bytes = io.BytesIO(response.content)
        encoded_image = encode_image_from_bytes(img_bytes)
        content = provider.get_image_description_from_encoded(encoded_image)
        cache[image_url] = content
    return content
