"""
AdFrame 2.0 — Wan VACE Video Generator
=========================================
Thin wrapper around Wan 2.1 VACE for full-video generation.
Reuses battle-tested patterns from adframe 1.0 inpainter.py:
  - VAE float32 cast
  - enable_model_cpu_offload
  - vae.enable_slicing fallback
  - _frame_to_pil robust decoder (tensor/ndarray/list)

Key changes vs 1.0:
  - Driven by frozen Qwen-approved prompt + placement_bbox
  - Soft mask generated from bbox (no SAM3 required)
  - Configurable num_frames / resolution
  - Returns List[PIL.Image] for downstream judge
"""

import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("adframe2.wan_generator")


def _frame_to_pil(f) -> Image.Image:
    """Convert (C,H,W) tensor or numpy array to PIL RGB image."""
    import torch
    if isinstance(f, torch.Tensor):
        arr = f.float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    elif isinstance(f, np.ndarray):
        if f.ndim == 3 and f.shape[0] in (1, 3, 4):
            arr = np.transpose(f, (1, 2, 0))
        else:
            arr = f
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0, 1)
    else:
        return Image.fromarray(np.array(f))
    return Image.fromarray((arr * 255).astype(np.uint8))


def _decode_output_frames(output) -> List[Image.Image]:
    """Robust decoder for WanVACEPipeline output.frames."""
    frames = output.frames
    pil_frames = []

    if hasattr(frames, "ndim"):
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim == 4:
            for i in range(frames.shape[0]):
                pil_frames.append(_frame_to_pil(frames[i]))
        else:
            pil_frames.append(_frame_to_pil(frames))
    elif isinstance(frames, list):
        if len(frames) > 0 and isinstance(frames[0], list):
            frames = frames[0]
        for f in frames:
            if isinstance(f, Image.Image):
                pil_frames.append(f)
            else:
                pil_frames.append(_frame_to_pil(f))
    else:
        pil_frames = list(frames)

    return pil_frames


def build_soft_mask_from_bbox(
    width: int,
    height: int,
    bbox_normalized: list,
    feather_radius: int = 20,
) -> Image.Image:
    """
    Build a soft (Gaussian-feathered) inpainting mask from a normalized bbox.
    Returns single-channel PIL Image (white = inpaint, black = preserve).

    Args:
        width, height: frame dimensions
        bbox_normalized: [x1, y1, x2, y2] in range [0, 1]
        feather_radius: Gaussian blur radius for mask edge softening
    """
    x1, y1, x2, y2 = bbox_normalized
    px1 = int(x1 * width)
    py1 = int(y1 * height)
    px2 = int(x2 * width)
    py2 = int(y2 * height)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[py1:py2, px1:px2] = 255

    if feather_radius > 0:
        k = feather_radius * 2 + 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), feather_radius)
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    return Image.fromarray(mask, mode="L")


class WanVideoGenerator:
    """
    Wan 2.1 VACE video generator for AdFrame 2.0.
    Accepts Qwen-approved prompts and generates full video output.
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import WanVACEPipeline

        logger.info(f"[WanVideoGenerator] Loading {self.model_id} ...")
        self._pipe = WanVACEPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )

        # VAE float32 for stable decoding
        logger.info("[WanVideoGenerator] Casting VAE to float32...")
        self._pipe.vae.to(dtype=torch.float32)

        # Memory optimizations
        logger.info("[WanVideoGenerator] Enabling CPU offload and VAE slicing...")
        if hasattr(self._pipe, "enable_model_cpu_offload"):
            self._pipe.enable_model_cpu_offload()
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        elif hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "enable_slicing"):
            self._pipe.vae.enable_slicing()

        logger.info("[WanVideoGenerator] Model ready.")

    def generate_video(
        self,
        video_frames: List[Image.Image],       # all frames as PIL RGB
        brand_img: Image.Image,
        placement_bbox: list,                   # [x1, y1, x2, y2] normalized
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 17,                   # must satisfy (n-1) % 4 == 0
        height: int = 480,
        width: int = 480,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate a video sequence with the brand product placed into the scene.

        Steps:
          1. Sample `num_frames` evenly from video_frames.
          2. Composite brand image into anchor frame 0 as VACE conditioning.
          3. Build soft mask from placement_bbox for all frames.
          4. Run VACE pipeline.
          5. Decode and return List[PIL.Image].
        """
        import torch

        self._ensure_loaded()

        # Ensure num_frames satisfies (n-1) % 4 == 0 for VACE
        while (num_frames - 1) % 4 != 0:
            num_frames -= 1
        num_frames = max(num_frames, 5)

        # Sample frames
        total = len(video_frames)
        if total >= num_frames:
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            sampled = [video_frames[i] for i in indices]
        else:
            sampled = video_frames + [video_frames[-1]] * (num_frames - total)

        # Resize all frames to target resolution
        sampled_resized = [
            f.resize((width, height), Image.LANCZOS) for f in sampled
        ]

        # Composite brand into frame 0 for VACE conditioning
        w, h = width, height
        x1, y1, x2, y2 = placement_bbox
        px1, py1, px2, py2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        brand_resized = brand_img.resize((max(1, px2-px1), max(1, py2-py1)), Image.LANCZOS)
        conditioned_frame0 = sampled_resized[0].copy()
        conditioned_frame0.paste(brand_resized, (px1, py1))
        sampled_resized[0] = conditioned_frame0

        # Build masks for each frame
        masks = [
            build_soft_mask_from_bbox(width, height, placement_bbox, feather_radius=15)
            for _ in range(num_frames)
        ]

        # Enforce resolution divisibility
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width = (width // 16) * 16
            logger.warning(f"[WanVideoGenerator] Rounded dims to {width}x{height}")

        # Clear cache before heavy inference
        torch.cuda.empty_cache()
        gc.collect()

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        logger.info(
            f"[WanVideoGenerator] Generating {num_frames} frames @ {width}x{height}, "
            f"{num_inference_steps} steps ..."
        )

        output = self._pipe(
            video=sampled_resized,
            mask=masks,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        torch.cuda.empty_cache()
        gc.collect()

        pil_frames = _decode_output_frames(output)
        logger.info(f"[WanVideoGenerator] Generated {len(pil_frames)} frames.")
        return pil_frames

    def unload(self):
        """Release GPU memory."""
        import torch
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[WanVideoGenerator] Model unloaded.")
