import os
import torch
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

class SAM3Tracker:
    """
    Wrapper for Meta's SAM 3 / 3.1 Video Predictor.
    Performs promptable tracking of target surfaces (e.g., posters, walls) across frames.
    """
    def __init__(self, checkpoint_path="./models/sam3.1_multiplex.pt", bpe_path=None, device="cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # If bpe_path is not specified, try to find it within the installed sam3 package
        if bpe_path is None:
            try:
                import sam3
                if hasattr(sam3, "__file__") and sam3.__file__ is not None:
                    sam3_dir = os.path.dirname(sam3.__file__)
                    bpe_path = os.path.join(sam3_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")
                else:
                    bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            except Exception:
                bpe_path = "./sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM 3 checkpoint not found at: {checkpoint_path}. Please run setup.sh first.")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE vocab asset not found at: {bpe_path}.")
            
        print(f"[SAM3Tracker] Initializing predictor with checkpoint: {checkpoint_path} and BPE path: {bpe_path}...")
        
        # Optimize for L40S VRAM (48GB): enable offloads and evaluation flags
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

    def track_video(self, video_path, text_prompt):
        """
        Tracks target surface specified by text_prompt across all video frames.
        Returns a dictionary mapping frame_index (int) -> binary mask (np.ndarray of shape [H, W]).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
            
        print(f"[SAM3Tracker] Starting tracking session for: {video_path}")
        session_response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = session_response["session_id"]
        print(f"[SAM3Tracker] Session ID: {session_id}")
        
        # Prompt the target surface on the first frame (frame 0)
        print(f"[SAM3Tracker] Adding text prompt: '{text_prompt}' on frame 0...")
        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt,
            )
        )
        
        # Propagate the mask forward across the video sequence
        print("[SAM3Tracker] Propagating tracking state across video frames...")
        masks_per_frame = {}
        
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            
            mask = self._extract_mask_from_outputs(outputs)
            if mask is not None:
                masks_per_frame[frame_idx] = mask
                
        # Close the session to free memory
        print("[SAM3Tracker] Closing tracking session...")
        self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        
        print(f"[SAM3Tracker] Tracking completed. Mask extracted for {len(masks_per_frame)} frames.")
        return masks_per_frame

    def _extract_mask_from_outputs(self, outputs):
        """
        Defensively extracts and converts the mask from outputs dictionary.
        Returns a binary numpy array of shape (H, W) where 1 indicates mask and 0 background.
        """
        mask_tensor = None
        if isinstance(outputs, dict):
            for key in ["masks", "mask_logits", "out_mask_logits", "logits"]:
                if key in outputs:
                    mask_tensor = outputs[key]
                    break
        elif hasattr(outputs, "masks"):
            mask_tensor = outputs.masks
        elif hasattr(outputs, "logits"):
            mask_tensor = outputs.logits
            
        if mask_tensor is None:
            return None
            
        # Convert PyTorch tensor to NumPy
        if hasattr(mask_tensor, "cpu"):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = np.array(mask_tensor)
            
        # If float logits, threshold it
        if mask_np.dtype in [np.float32, np.float64]:
            mask_np = mask_np > 0.0
            
        # Squeeze dimensions to get a single (H, W) array
        if mask_np.ndim == 4:
            # Shape is likely (num_objects, num_channels, H, W)
            mask_np = mask_np[0, 0]
        elif mask_np.ndim == 3:
            # Shape is likely (num_objects, H, W)
            mask_np = mask_np[0]
            
        return mask_np.astype(np.uint8)
