import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'
text_prompt = 'wall poster'

print("Starting session...")
session_response = tracker.predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = session_response["session_id"]
print(f"Session ID: {session_id}")

print("Adding prompt...")
tracker.predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=text_prompt,
    )
)

print("Propagating first few frames...")
count = 0
for response in tracker.predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    frame_idx = response["frame_index"]
    outputs = response["outputs"]
    print(f"\nFrame: {frame_idx}")
    print(f"Type of outputs: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"Keys: {list(outputs.keys())}")
        for k, v in outputs.items():
            if hasattr(v, 'shape'):
                print(f"  {k} shape: {v.shape}, dtype: {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print(f"Outputs attributes: {dir(outputs)}")
        
    count += 1
    if count >= 3:
        break

tracker.predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
