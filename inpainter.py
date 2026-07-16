import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.makedirs("/workspace/tmp", exist_ok=True)
import gc
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import WanVACEPipeline

class WanInpainter:
    """
    Wrapper for quantized Wan 2.1 Video Inpainting/VACE pipeline.
    Optimized for NVIDIA L40S GPU (48GB VRAM) running in bfloat16/INT8.
    """
    def __init__(self, model_id="Wan-AI/Wan2.1-VACE-14B-diffusers", load_in_8bit=False, device="cuda"):
        self.device = device
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        
        print(f"[WanInpainter] Loading Wan 2.1 VACE pipeline: {model_id}...")
        
        # Default datatype is bfloat16
        torch_dtype = torch.bfloat16
        
        if self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_skip_modules=["proj_out", "patch_embed"]
                )
                print("[WanInpainter] Initializing pipeline with 8-bit quantization to reduce memory footprint...")
                self.pipe = WanVACEPipeline.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    device_map="auto"
                )
            except Exception as e:
                print(f"[WanInpainter] bitsandbytes/INT8 load failed: {e}. Falling back to standard bfloat16 loading.")
                self.pipe = WanVACEPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype
                )
        else:
            self.pipe = WanVACEPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
            
        # Forced cast to torch.float32 for VAE submodule right after loading to avoid precision errors
        print("[WanInpainter] Casting VAE submodule explicitly to float32...")
        self.pipe.vae.to(dtype=torch.float32)
            
        # GPU memory optimizations for L40S (48GB VRAM)
        print("[WanInpainter] Enforcing memory optimizations (Model CPU Offload & VAE Slicing)...")
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()
        
    def generate_patch_sequence(self, video_patches, mask_patches, prompt, num_frames=16, height=480, width=480):
        """
        Generates/Inpaints a sequence of video frames based on input video patches and mask patches.
        Returns:
            List[PIL.Image.Image]: List of generated video frames (patches).
        """
        print(f"[WanInpainter] Generating patch sequence ({num_frames} frames, {width}x{height}) with prompt: '{prompt}'")
        
        # Convert video_patches to List[PIL.Image] if they are numpy BGR arrays
        processed_video = []
        for i, frame in enumerate(video_patches):
            if isinstance(frame, np.ndarray):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_video.append(Image.fromarray(rgb_frame))
            else:
                processed_video.append(frame)
                
        # Convert mask_patches to List[PIL.Image] if they are numpy arrays
        processed_masks = []
        for i, mask in enumerate(mask_patches):
            if isinstance(mask, np.ndarray):
                # Ensure it's single channel 8-bit image with 0 or 255 values
                if mask.max() <= 1:
                    mask = mask * 255
                processed_masks.append(Image.fromarray(mask.astype(np.uint8)))
            else:
                processed_masks.append(mask)
        
        # Slice inputs to the target generation length
        processed_video = processed_video[:num_frames]
        processed_masks = processed_masks[:num_frames]
        
        # Flush PyTorch cache to optimize VRAM before run
        torch.cuda.empty_cache()
        gc.collect()
        
        # Run inference
        # Wan models require dimensions divisible by 16
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width = (width // 16) * 16
            print(f"[WanInpainter] Resized inference dimensions to divisible-by-16: {width}x{height}")
            
        output = self.pipe(
            video=processed_video,
            mask=processed_masks,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=30,
            guidance_scale=5.0
        )
        
        # Flush PyTorch cache to clean up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Decode output.frames — can be 5D/4D torch.Tensor or np.ndarray (batch,frames,C,H,W)
        frames = output.frames
        pil_frames = []

        def _frame_to_pil(f):
            """Convert a single (C,H,W) tensor or numpy array to a PIL RGB image."""
            import torch as _torch
            if isinstance(f, _torch.Tensor):
                arr = f.float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            elif isinstance(f, np.ndarray):
                if f.ndim == 3 and f.shape[0] in (1, 3, 4):
                    # (C,H,W) numpy — transpose to (H,W,C)
                    arr = np.transpose(f, (1, 2, 0))
                else:
                    arr = f  # already (H,W,C)
                arr = arr.astype(np.float32)
                # if values outside [0,1], assume [0,255]
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arr = np.clip(arr, 0, 1)
            else:
                return Image.fromarray(np.array(f))
            return Image.fromarray((arr * 255).astype(np.uint8))

        if hasattr(frames, "ndim"):
            if frames.ndim == 5:
                frames = frames[0]  # (num_frames, C, H, W)
            if frames.ndim == 4:
                for i in range(frames.shape[0]):
                    pil_frames.append(_frame_to_pil(frames[i]))
            else:
                pil_frames.append(_frame_to_pil(frames))
        elif isinstance(frames, list):
            if len(frames) > 0 and isinstance(frames[0], list):
                frames = frames[0]  # nested [batch][frame]
            for f in frames:
                if isinstance(f, Image.Image):
                    pil_frames.append(f)
                else:
                    pil_frames.append(_frame_to_pil(f))
        else:
            pil_frames = list(frames)

        print(f"[WanInpainter] Generation complete. Generated {len(pil_frames)} frames.")
        return pil_frames

