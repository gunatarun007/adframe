import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'

categories = [
    "background", "wall", "grey wall", "dark wall", "blank wall", 
    "backdrop", "laptop", "laptop screen", "microphone", "sign", "ON sign", "box"
]

print("Scanning backdrop categories on frame 0 in a single session...")
session_response = tracker.predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = session_response["session_id"]

for text_prompt in categories:
    for thresh in [0.1, 0.05, 0.01]:
        res = tracker.predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=text_prompt,
            output_prob_thresh=thresh
        )
        outputs = res["outputs"]
        
        num_objects = len(outputs.get("out_obj_ids", []))
        if num_objects > 0:
            print(f"Prompt: '{text_prompt}' at thresh {thresh} -> Detected {num_objects} objects!")
            print(f"  Object IDs: {outputs['out_obj_ids']}")
            print(f"  Probs: {outputs['out_probs']}")
            print(f"  Boxes (xywh): {outputs['out_boxes_xywh']}")
            # break thresh loop if detected to keep log clean
            break
        
tracker.predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
