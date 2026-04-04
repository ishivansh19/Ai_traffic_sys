"""
yolo_acquisition.py
Simple YOLO-based acquisition module that:
- Loads a YOLOv8 pretrained model
- Runs inference on an input image
- Converts detections to the format:
    {
      "id": vid,
      "class": v_type,
      "confidence": 0.98,
      "lane": lane,
      "location": (x_center, y_center)
    }
- Determines lane by splitting image width into NUM_LANES vertical regions.
"""

import os
import uuid
import json
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple

# ============ CONFIGURATION =============
MODEL_NAME = "yolov8n.pt"   # small & fast pretrained model
IMAGE_PATH = "images/image.png"   # path to your test image (relative)
CONF_THRESH = 0.35   # minimum confidence to keep detection
NUM_LANES = 3        # how many vertical lanes the image is divided into
OUTPUT_JSON = "detections.json"  # optional save
# ========================================

# Optional mapping: YOLO class name -> desired label used in "class" field.
# If you want ambulance as a class, add appropriate mapping after custom training.
CLASS_MAPPING = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "person",
    # Add more if needed
    # Example: "ambulance": "ambulance"  # only works if model recognizes 'ambulance'
}

def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """Read image from disk and return image array, width, height."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return img, w, h

def map_to_lane(x_center: float, img_width: int, num_lanes: int) -> int:
    """
    Map an x coordinate (pixel) to a lane index (1..num_lanes).
    Left-most lane = 1, Right-most lane = num_lanes.
    """
    lane_width = img_width / num_lanes
    lane_idx = int(x_center // lane_width) + 1
    # clamp
    lane_idx = max(1, min(num_lanes, lane_idx))
    return lane_idx

def box_center_from_bbox(bbox: List[float]) -> Tuple[float, float]:
    """
    bbox format from ultralytics: [x1, y1, x2, y2]
    return center (x_center, y_center)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def human_readable_class(yolo_class_name: str) -> str:
    """Map YOLO class name to desired label, fallback to raw name."""
    return CLASS_MAPPING.get(yolo_class_name, yolo_class_name)

def run_inference(image_path: str,
                  model_name: str = MODEL_NAME,
                  conf_thresh: float = CONF_THRESH,
                  num_lanes: int = NUM_LANES) -> List[Dict]:
    """
    Run YOLO inference on single image and return detection list in requested format.
    """
    # Load model (downloads model if not present)
    model = YOLO(model_name)

    # Read image once (we need width/height for lane mapping)
    img, img_w, img_h = load_image(image_path)

    # Run inference (returns a list of Results; single image -> one result)
    results = model(image_path, conf=conf_thresh, verbose=False)

    # Build output detections
    detections = []
    # ultralytics result handling:
    # each result has .boxes (if detections present)
    for res in results:  # one element for single image
        boxes = res.boxes
        for box in boxes:
            # box.xyxy[0] returns a tensor with [x1,y1,x2,y2]
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else float(box.conf.cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())
            # get class name from model
            # model.names is mapping idx -> class name
            cls_name = model.names.get(cls_idx, str(cls_idx))

            # center point
            cx, cy = box_center_from_bbox(xyxy)

            lane = map_to_lane(cx, img_w, num_lanes)

            detection = {
                "id": str(uuid.uuid4()),              # unique id
                "class": human_readable_class(cls_name),
                "confidence": round(conf, 3),
                "lane": lane,
                "location": (round(cx, 1), round(cy, 1))   # pixel coords (x,y)
            }
            detections.append(detection)

    return detections

def main():
    print("YOLO Acquisition Module — starting")
    print(f"Image: {IMAGE_PATH}\nModel: {MODEL_NAME}\nConfidence threshold: {CONF_THRESH}\nNum lanes: {NUM_LANES}\n")
    # Run
    dets = run_inference(IMAGE_PATH)
    if not dets:
        print("No detections above confidence threshold.")
    else:
        print("Detections:")
        print(json.dumps(dets, indent=2))
        # Save to JSON optionally
        with open(OUTPUT_JSON, "w") as f:
            json.dump(dets, f, indent=2)
        print(f"\nSaved detections to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
