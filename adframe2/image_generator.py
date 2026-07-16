"""
AdFrame 2.0 — Image Generator Stage
======================================
Wraps a single-frame image generator (FLUX.1-dev or SDXL) for rapid
prompt validation before committing to full Wan VACE video generation.

Strategy:
  1. Extract the anchor frame (Qwen's best_frame_index) from the video.
  2. Crop the placement region using Qwen's placement_bbox.
  3. Run img2img inpainting on just that crop.
  4. Return the edited frame for Qwen judge evaluation.

This stage should complete in < 30 seconds.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("adframe2.image_generator")


class ImageGenerator:
    """
    Single-frame image generator for prompt validation.
    Supports FLUX.1-dev (default) and SDXL as backends.
    """

    SUPPORTED_BACKENDS = ["flux", "sdxl"]

    def __init__(
        self,
        backend: str = "flux",
        model_id: Optional[str] = None,
        device: str = "cuda",
    ):
        self.backend = backend.lower()
        assert self.backend in self.SUPPORTED_BACKENDS, (
            f"Unsupported backend '{backend}'. Choose from {self.SUPPORTED_BACKENDS}"
        )
        self.device = device

        # Default model IDs
        if model_id is None:
            model_id = {
                "flux": "black-forest-labs/FLUX.1-dev",
                "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            }[self.backend]
        self.model_id = model_id

        logger.info(f"[ImageGenerator] Loading {self.backend} backend: {model_id}")
        self._load_pipeline()
        logger.info("[ImageGenerator] Pipeline loaded.")

    def _load_pipeline(self):
        import torch

        if self.backend == "flux":
            from diffusers import FluxInpaintPipeline
            self.pipe = FluxInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
            self.pipe.enable_model_cpu_offload()

        elif self.backend == "sdxl":
            from diffusers import AutoPipelineForInpainting
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self.pipe.enable_model_cpu_offload()

    def generate_frame(
        self,
        anchor_frame: Image.Image,
        brand_img: Image.Image,
        placement_bbox: list,         # [x1, y1, x2, y2] normalized 0-1
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.85,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single edited frame with the brand product placed.

        Args:
            anchor_frame: The best video frame as PIL.Image.
            brand_img:    The brand product image.
            placement_bbox: Normalized [x1, y1, x2, y2] from Qwen.
            prompt:       The inpainting prompt from Qwen reasoning.
            negative_prompt: The negative prompt from Qwen reasoning.

        Returns:
            PIL.Image — the generated frame with product placed.
        """
        import torch

        w, h = anchor_frame.size
        x1, y1, x2, y2 = placement_bbox
        # Convert normalized to pixel coords
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        # Build inpainting mask: white in placement region, black elsewhere
        mask = Image.new("L", (w, h), 0)
        mask_arr = np.array(mask)
        mask_arr[py1:py2, px1:px2] = 255
        mask = Image.fromarray(mask_arr)

        # Composite brand image into anchor frame as rough initialization
        brand_resized = brand_img.resize((px2 - px1, py2 - py1), Image.LANCZOS)
        init_frame = anchor_frame.copy()
        init_frame.paste(brand_resized, (px1, py1))

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        logger.info(
            f"[ImageGenerator] Generating frame with {self.backend} "
            f"({num_inference_steps} steps, strength={strength})"
        )

        kwargs = dict(
            image=init_frame,
            mask_image=mask,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        if self.backend == "sdxl":
            kwargs["negative_prompt"] = negative_prompt
            kwargs["guidance_scale"] = guidance_scale
            kwargs["strength"] = strength
        elif self.backend == "flux":
            kwargs["guidance_scale"] = guidance_scale
            kwargs["strength"] = strength
            kwargs["height"] = h
            kwargs["width"] = w

        result = self.pipe(**kwargs)
        generated = result.images[0]

        logger.info("[ImageGenerator] Single-frame generation complete.")
        return generated

    def generate_frame_crop(
        self,
        anchor_frame: Image.Image,
        brand_img: Image.Image,
        placement_bbox: list,
        prompt: str,
        negative_prompt: str = "",
        crop_size: int = 512,
        **kwargs,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Alternate strategy: crop just the placement region, generate at higher
        resolution, then paste back.

        Returns:
            (full_frame_with_edit, crop_pixel_bbox)
        """
        w, h = anchor_frame.size
        x1, y1, x2, y2 = placement_bbox
        px1, py1, px2, py2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)

        # Expand crop with context padding (50% on each side)
        pad_x = int((px2 - px1) * 0.5)
        pad_y = int((py2 - py1) * 0.5)
        cpx1 = max(0, px1 - pad_x)
        cpy1 = max(0, py1 - pad_y)
        cpx2 = min(w, px2 + pad_x)
        cpy2 = min(h, py2 + pad_y)

        crop = anchor_frame.crop((cpx1, cpy1, cpx2, cpy2))
        orig_crop_size = crop.size
        crop = crop.resize((crop_size, crop_size), Image.LANCZOS)

        # Recalculate bbox within cropped coords
        scale_x = crop_size / (cpx2 - cpx1)
        scale_y = crop_size / (cpy2 - cpy1)
        local_bbox = [
            (px1 - cpx1) / (cpx2 - cpx1),
            (py1 - cpy1) / (cpy2 - cpy1),
            (px2 - cpx1) / (cpx2 - cpx1),
            (py2 - cpy1) / (cpy2 - cpy1),
        ]

        generated_crop = self.generate_frame(
            anchor_frame=crop,
            brand_img=brand_img,
            placement_bbox=local_bbox,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )

        # Paste back at original resolution
        generated_crop = generated_crop.resize(orig_crop_size, Image.LANCZOS)
        result_frame = anchor_frame.copy()
        result_frame.paste(generated_crop, (cpx1, cpy1))

        return result_frame, (cpx1, cpy1, cpx2, cpy2)
