import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'

categories = [
    "person", "man", "woman", "face", "hair", "shirt", "pants", "shoes", 
    "chair", "table", "desk", "computer", "monitor", "screen", "keyboard", 
    "mouse", "phone", "book", "cup", "bottle", "plate", "food", 
    "wall", "floor", "ceiling", "window", "door", "light", "lamp", 
    "painting", "poster", "picture", "frame", "sofa", "couch", "rug", 
    "carpet", "bed", "blanket", "pillow", "cabinet", "shelf", "drawer", 
    "box", "bag", "backpack", "plant", "flower", "tree", "car", "bicycle", 
    "dog", "cat"
]

print("Scanning all possible categories on frame 0 in a single session...")
session_response = tracker.predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = session_response["session_id"]

for text_prompt in categories:
    res = tracker.predictor.add_prompt(
        session_id=session_id,
        frame_idx=0,
        text=text_prompt,
        output_prob_thresh=0.1
    )
    outputs = res["outputs"]
    
    num_objects = len(outputs.get("out_obj_ids", []))
    if num_objects > 0:
        print(f"Prompt: '{text_prompt}' -> Detected {num_objects} objects!")
        print(f"  Object IDs: {outputs['out_obj_ids']}")
        print(f"  Probs: {outputs['out_probs']}")
        print(f"  Boxes (xywh): {outputs['out_boxes_xywh']}")
        
tracker.predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
