from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Literal
from typing import AsyncGenerator
from pydantic import BaseModel
from utils.redis_connection import redis_client
from utils.models import  DetectBBOxesAndMasks
import json
import requests
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import base64
from urllib.parse import urlparse
import ast
import numpy as np
from PIL import Image, ImageDraw
import re
import uuid
from io import BytesIO
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel


# Initialize the CLIP model and processor
feat_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the object detection and segmentation models
detection_models = DetectBBOxesAndMasks(
    detector_id="google/owlv2-base-patch16-ensemble",
    segmenter_id="facebook/sam-vit-base",
    threshold=0.5
)

# Create a directory for mask images if it doesn't exist
MASK_DIR = Path("static/masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Load LLAMA Model Endpoint
api_key = os.getenv('API_KEY', "vd7rF1WgLJQ1")
api_base = "https://dg-kube.nix.cccis.com/llm-inference-prod2/api"
model = "Llama-4-Scout-17B-16E-Instruct"

api_key_text = os.getenv('API_KEY', 'vd7rF1WgLJD1')
api_base_text = "https://dg-kube.nix.cccis.com/llm-inference-dev/api"
model_text = "Phi-4-multimodal-instruct"

bbox_pattern = r"^\(\s*(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?),\s*(-?\d+(\.\d+)?)\s*\)$"

router = APIRouter()

# Define the request body schema
class AnnotationRequest(BaseModel):
    annotationTypes: List[str]


def query_vlm_with_text(query) -> None:
    payload = {
        "model": model_text,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given this text query, return the object of interest in the image. For example if the query is 'people on the beach', the object of interest is 'a person'. Your task is to simply return the object of interest in the image but should be followed by 'a'. Do not return any other information.. no explaination, no reasoning, just the object of interest.",
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
        "Authorization": f"Bearer {api_key_text}"
    }
    
    # Make the API request
    response = requests.post(
        f"{api_base_text}/chat/completions",
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


def get_image_embedding_from_url(image_url: str):
    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw).convert("RGB")  # ensure it's RGB

    inputs = processor_clip(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = feat_model.get_image_features(**inputs)
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
    embedding = image_features.squeeze(0)
    return  embedding.numpy().reshape(1, -1)

def query_faiss(index, image_path: str, k: int = 1):
    """
    Query the FAISS index with the given image and return the top-k results.
    """
    # Get the embedding for the input image
    query_vector = get_image_embedding_from_url(image_path)

    # Perform the FAISS search
    distances, indices = index.search(query_vector, k)

    # Return the results
    return distances[0], indices[0]


# Update the save_image_with_bbox function to also save a mask image
def save_image_with_bbox_and_mask(np_image: np.ndarray, bbox: tuple, masks: np.ndarray, output_path: str) -> tuple:
    """
    Draw a bounding box on an image and save both the bbox image and mask.

    Args:
        np_image (np.ndarray): The input image as a NumPy array (H x W x C).
        bbox (tuple): Bounding box in the format (x, y, width, height).
        masks (np.ndarray): Binary mask from SAM model.
        output_path (str): Path to save the output image with the bounding box.

    Returns:
        tuple: (bbox_image_path, mask_image_path)
    """
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(np_image)

    # Draw the bounding box
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # Save the image with the bounding box
    image.save(output_path)
    print(f"Image with bounding box saved to: {output_path}")
    
    # Generate a unique filename for the mask
    mask_filename = f"mask_{uuid.uuid4().hex}.png"
    mask_path = str(MASK_DIR / mask_filename)
    
    if masks is not None and masks.shape[0] > 0:
        # Convert the first binary mask to a PIL image (255 for visibility)
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        # Save the mask as a PNG
        mask_image.save(mask_path)
        print(f"Mask image saved to: {mask_path}")
        
        return output_path, mask_path
    
    return output_path, None


def encode_local_image(image_path: str) -> str:
    """Encode an image (from a local path or URL) to base64 format."""
    parsed = urlparse(image_path)
    
    if parsed.scheme in ('http', 'https'):
        # Handle URL
        response = requests.get(image_path)
        response.raise_for_status()
        image_data = response.content
        img = Image.open(BytesIO(image_data))
        width, height = img.size
    else:
        # Handle local file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No such file: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        img = Image.open(image_path)
        width, height = img.size
    
    return base64.b64encode(image_data).decode('utf-8'), width, height


def query_vlm_with_image_llama(
    image_path: str, 
    prompt: str = "Based on the shared images, what panels are damaged in this vehicle?", 
) -> None:
    """Query VLM with one or more local image files using direct HTTP request."""
    
    # Choose endpoint based on API type
    endpoint = f"{api_base}/chat/completions"
    
    # Chat completions format
    image_base64, _, _ = encode_local_image(image_path)
    content = []
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        }
    })
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_completion_tokens": 256
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload
    )
    # Process the response
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    print(f"Error: {response.status_code}")
    print(response.text)
    return 0


def query_vlm_with_local_image(image_path: str, query: str) -> None:
    """Query VLM with a local image file using direct HTTP request."""
    image_base64, _, _ = encode_local_image(image_path)
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Check weather the image matches this query - {query}. If yes give me bounding box around the object in (x, y, w, h) format. The output should be strictly a tuple of 4 integers. If the object is not found, return empty tuple (). Only RETURN tuple nothing else, no text or explaination needed"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_completion_tokens": 64
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
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
    return ()


# Update the stream_annotations function
async def stream_annotations(index, images: List[str], query: str) -> AsyncGenerator:
    for image, (bboxes, mask_urls, scores, shape) in zip(images, get_annotations(index, images, query)):
        if not bboxes:
            continue  # Skip if no bounding box is found
        try:
            # Create a response with multiple bounding boxes and masks
            chunk = json.dumps({
                "URL": image,
                "bboxes": bboxes,  # List of bounding boxes
                "mask_urls": mask_urls,  # List of mask URLs
                "scores": scores.tolist() if scores is not None else None,
                "shape": shape if shape is not None else None,
            }) + "\n"
            yield chunk
        except Exception as e:
            print(f"Error serializing chunk: {e}")
            yield json.dumps({
                "URL": image,
                "bboxes": bboxes,
                "mask_urls": mask_urls,
                "scores": None,
                "shape": None,
                "error": str(e)
            }) + "\n"


def save_image_with_bbox(np_image: np.ndarray, bbox: tuple, output_path: str) -> None:
    """
    Draw a bounding box on a NumPy image and save it to a file.

    Args:
        np_image (np.ndarray): The input image as a NumPy array (H x W x C).
        bbox (tuple): Bounding box in the format (x, y, width, height).
        output_path (str): Path to save the output image with the bounding box.
    """
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(np_image)

    # Draw the bounding box
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # Save the image with the bounding box
    image.save(output_path)
    print(f"Image with bounding box saved to: {output_path}")

def extract_boxes_sorted_by_confidence(response):
    # Step 1: Extract content inside <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answer_match:
        return []

    answer_content = answer_match.group(1).strip()

    # Step 2: Handle 'No Objects' case
    if answer_content == "No Objects":
        return []

    try:
        # Step 3: Safely evaluate the list of dicts from string
        boxes = ast.literal_eval(answer_content)

        # Step 4: Sort by 'Confidence' in descending order
        sorted_boxes = sorted(boxes, key=lambda x: x['Confidence'], reverse=True)
        return sorted_boxes

    except Exception as e:
        print("Error while parsing answer content:", e)
        return []

def rescale_boxes_to_xyxy(boxes, img_width, img_height, scale=1000):
    """
    Rescale boxes from 0-1000 range to image dimensions,
    keeping the [x1, y1, x2, y2] format that SAM expects.
    """
    rescaled = []
    for box in boxes:
        x1, y1, x2, y2 = box['Position']
        # Rescale to image size
        x1 = int(x1 / scale * img_width)
        y1 = int(y1 / scale * img_height)
        x2 = int(x2 / scale * img_width)
        y2 = int(y2 / scale * img_height)

        # Keep the format as [x1, y1, x2, y2]
        rescaled_box = {
            'Position': [x1, y1, x2, y2],
            'Confidence': box['Confidence']
        }
        rescaled.append(rescaled_box)

    return rescaled

def get_annotations(index, images: List[str], query: str, sim_threshold=0.85):
    simple_prompt = query_vlm_with_text(query)
    for i, img in enumerate(images):
        try:
            distances, _ = query_faiss(index, img, k=1)
            if distances[0] < sim_threshold:
                print(f"Image {img} is not similar enough to the query.")
                yield [], [], None, None
                continue
            
            prompt = f"How many objects are there in the image matching the query '{query}', Return the number of objects in the image. The output should be strictly a number. If no object is found, return 0. Only RETURN number nothing else, no text or explaination needed"
            output = query_vlm_with_image_llama(img, prompt)      
            num_objects = int(output.strip())

            image = detection_models.load_image(img)
            width, height = image.size
            results = detection_models.detect(image, [simple_prompt])
            bboxes = detection_models.get_boxes(results)[0]
            if not bboxes:
                print(f"No bbox found")
                yield [], [], None, None
                continue
            
            results = detection_models.segment(image, results)
            masks, bboxes = detection_models.get_masks(results, num_objects)
            
            mask_urls = []
            # Process each mask and save it
            for mask_idx, mask in enumerate(masks):
                mask_filename = f"mask_{i}_{mask_idx}_{uuid.uuid4().hex}.png"
                mask_path = MASK_DIR / mask_filename
                
                if mask.shape != (height, width):
                    print(f"Warning: Mask shape {mask.shape} doesn't match image shape ({height}, {width})")
                    # Resize the mask if needed
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize((width, height))
                    mask = np.array(mask_img) > 0  # Convert back to binary mask
                
                # Save the mask as a PNG
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(mask_path)
                # print(f"Mask {mask_idx} saved to: {mask_path}")
                # Add the URL to the list
                mask_urls.append(f"/static/masks/{mask_filename}")
            
            yield bboxes, mask_urls, None, [height, width]
                
        except Exception as e:
            print(f"Error processing image {img}: {e}")
            import traceback
            traceback.print_exc()
            yield [], [], None, None


@router.post("/")
def generate_annotations(request: AnnotationRequest, user_id: str):
    # Retrieve data from Redis
    user_data = redis_client.get(user_id)
    index = faiss.read_index(f"image_embeddings_{user_id}.index")
    if not user_data:
        raise HTTPException(status_code=404, detail="User data not found")

    data = json.loads(user_data)
    query = data["query"]
    images = data["images"]
    annotation_types = request.annotationTypes
    print("Query:", query)
    print("Images:", images)
    print("Selected Annotations:", annotation_types)

    # Return a StreamingResponse that streams the annotations
    return StreamingResponse(
        stream_annotations(index, images[:10], query),
        media_type="application/json"
    )
