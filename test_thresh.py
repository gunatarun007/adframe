import os
import torch
from tracker import SAM3Tracker

tracker = SAM3Tracker()
video_path = '/workspace/demo.mp4'

for text_prompt in ["poster", "wall poster", "wall", "picture frame"]:
    for thresh in [0.5, 0.3, 0.1]:
        session_response = tracker.predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = session_response["session_id"]
        
        # We run add_prompt
        res = tracker.predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=text_prompt,
            output_prob_thresh=thresh
        )
        outputs = res["outputs"]
        
        num_objects = len(outputs.get("out_obj_ids", []))
        print(f"Prompt: '{text_prompt}', Thresh: {thresh}, Num Objects: {num_objects}")
        if num_objects > 0:
            print(f"  Object IDs: {outputs['out_obj_ids']}")
            print(f"  Probs: {outputs['out_probs']}")
            
        tracker.predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
