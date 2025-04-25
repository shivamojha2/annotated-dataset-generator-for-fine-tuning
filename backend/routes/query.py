from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import torch
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import numpy as np
from utils.redis_connection import redis_client
import json
import json

# npm run dev
# uvicorn main:app  --reload
# brew install redis
# redis-server
# redis-cli

user_id = "unique_user_id"

clip = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        torch_dtype=torch.float32,
        device=-1
    )

feat_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.IndexFlatIP(512)

api_key = os.getenv('API_KEY', 'vd7rF1WgLJD1')
api_base = "https://dg-kube.nix.cccis.com/llm-inference-dev/api"
model = "Phi-4-multimodal-instruct"

router = APIRouter()

# Request model
class QueryRequest(BaseModel):
    query: Optional[str] = None
    selected_images: Optional[List[str]] = None

# Response model
class QueryResponse(BaseModel):
    images: Optional[List[str]] = None
    message: Optional[str] = None

def query_vlm_with_text(query) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given this text query, can you simplify it so that I can straightwawy use it to search web, I want it to be as simple as possible for example - if query is 'I want photos of animals with four legs' we can simplify it to 'animals with four heads'",
                    },
                    {
                        "type": "text",
                        "text": f"HERE is the query :- '{query}'"
                    }
                ]
            }
        ],
        "max_completion_tokens": 6400
    }
    
    # Set up headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Make the API request
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=payload
    )
    
    # Process the response
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    return None

def get_bing_images(query):
    url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch Bing images.")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img", class_="mimg")
    img_urls = [urljoin(url, img["src"]) for img in img_tags if "src" in img.attrs]
    return img_urls

def get_image_embedding_from_url(image_url: str):
    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw).convert("RGB")  # ensure it's RGB

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = feat_model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.squeeze(0)  # shape: (512,)

def add_image_to_faiss(image_url: str):
    embedding = get_image_embedding_from_url(image_url)
    embedding_np = embedding.numpy().reshape(1, -1)  # reshape for FAISS (1, 512)
    
    index.add(embedding_np)

# POST route
@router.post("/", response_model=QueryResponse)
def test_query(payload: QueryRequest):
    if payload.query:
        print("Received query:", payload.query)
        pre_query = query_vlm_with_text(payload.query)
        print("Pre processed query:", pre_query)

        images = get_bing_images(pre_query)
        final_images = []
        for _, img in enumerate(images):
            labels = [pre_query, "not related"]
            output = clip(img, candidate_labels=labels)
            if output[0]['label'] == pre_query and output[0]["score"] > 0.9:
                final_images.append(img)
                if len(final_images) == 6:
                    break
        redis_client.set(user_id, json.dumps({"query": pre_query, "images": images}))
        return QueryResponse(images=final_images)

    if payload.selected_images:
        for img in payload.selected_images:
            print("Received selected image:", img)
            add_image_to_faiss(img)
        faiss.write_index(index, f"image_embeddings_{user_id}.index")
        return QueryResponse(message="Selection submitted successfully!")
    return QueryResponse(message="No valid input provided.")
        

    
