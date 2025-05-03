from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import cv2
from dataclasses import dataclass
import requests
from typing import Union, Tuple


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
    
class DetectBBOxesAndMasks:
    def __init__(self, detector_id: Optional[str] = None, segmenter_id: Optional[str] = None, threshold: float = 0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # setup object detector
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
        self.object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=self.device)
        self.threshold = threshold

        # setup segmenter
        segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(segmenter_id)
    
    def get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]

    def get_masks(self, results: DetectionResult, num_objects: int) -> List[np.ndarray]:
        masks = []
        bboxes = []
        sorted_results = sorted(results, key=lambda result: result.score, reverse=True)[:num_objects]
        for result in sorted_results:
            mask = result.mask
            bbox = result.box.xyxy
            masks.append(mask)
            bboxes.append(bbox)
        return masks, bboxes
    
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    def refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                mask = self.polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def load_image(self, image_str: str) -> Image.Image:
        if image_str.startswith("http"):
            image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_str).convert("RGB")

        return image

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO / OWL to detect a set of labels in an image in a zero-shot fashion.
        """
        # labels = [label if label.endswith(".") else label+"." for label in labels]
        results = self.object_detector(image, candidate_labels=labels)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """

        boxes = self.get_boxes(detection_results)
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self.refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results
     

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
        polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = self.load_image(image)

        detections = self.detect(image, labels)
        if len(detections) == 0:
            return np.array(image), []
        detections = self.segment(image, detections, polygon_refinement)
        return np.array(image), detections