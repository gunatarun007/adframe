import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'

session_response = tracker.predictor.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)
session_id = session_response["session_id"]

# We define a bounding box in normalized [xmin, ymin, width, height] format
# corresponding to the blank wall region: x_min=0.68, y_min=0.15, width=0.20, height=0.35
box = [0.68, 0.15, 0.20, 0.35]

print("Adding bounding box prompt...")
res = tracker.predictor.add_prompt(
    session_id=session_id,
    frame_idx=0,
    bounding_boxes=[box],
    bounding_box_labels=[0]
)
outputs = res["outputs"]

print("Keys in outputs:", list(outputs.keys()))
if "out_binary_masks" in outputs:
    mask = outputs["out_binary_masks"]
    print("out_binary_masks shape:", mask.shape)
    if mask.shape[0] > 0:
        print("Mask sum:", mask[0].sum())

tracker.predictor.handle_request(
    request=dict(type="close_session", session_id=session_id)
)
