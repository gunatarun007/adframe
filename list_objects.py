import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'

print("Scanning prompts...")
for text_prompt in ["poster", "wall", "blank wall", "wall poster", "picture frame", "board"]:
    session_response = tracker.predictor.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = session_response["session_id"]
    
    # We run add_prompt
    tracker.predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )
    
    # We propagate for 1 frame to see outputs
    for prop_response in tracker.predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = prop_response["frame_index"]
        if frame_idx > 0:
            # just stop after first frame to print results
            break
        outputs = prop_response["outputs"]
        num_objects = len(outputs.get("out_obj_ids", []))
        print(f"Prompt: '{text_prompt}', Frame: {frame_idx}, Num Objects: {num_objects}")
        if num_objects > 0:
            print(f"  Object IDs: {outputs['out_obj_ids']}")
            print(f"  Probs: {outputs['out_probs']}")
            
    tracker.predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
