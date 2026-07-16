"""
AdFrame 2.0 — Qwen Reasoning Engine
=====================================
Uses Qwen2.5-VL to reason about a video and a brand asset image.
Returns structured JSON with:
  - best frame index for placement
  - placement bounding box [x1, y1, x2, y2] (normalized 0-1)
  - camera angle description
  - lighting conditions
  - recommended product orientation
  - generated inpainting prompt
  - generated negative prompt
  - list of constraints for the image generator
  - initial placement score rationale

No free-form text is returned — always structured JSON.
"""

import json
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("adframe2.qwen_reasoner")


# ---------------------------------------------------------------------------
# Schema (for documentation / validation reference)
# ---------------------------------------------------------------------------
REASONING_SCHEMA = {
    "best_frame_index": "int — index of the best anchor frame for placement",
    "placement_bbox": "[x1, y1, x2, y2] — normalized (0-1) bounding box of the placement region",
    "surface_type": "str — detected surface type (e.g. 'wall', 'table', 'screen')",
    "camera_angle": "str — e.g. 'slight left angle, eye-level'",
    "lighting": {
        "direction": "str — e.g. 'top-right key light'",
        "color_temperature": "str — e.g. 'warm 3200K'",
        "intensity": "str — e.g. 'soft diffuse'"
    },
    "product_orientation": "str — e.g. 'label facing camera, slight 15 degree tilt'",
    "occlusion_risk": "str — e.g. 'low, surface is unobstructed for 90% of frames'",
    "visibility_score": "float 0-1 — how visible the surface is across the video",
    "reflections": "str — e.g. 'minimal, matte wall surface'",
    "scale_guidance": "str — e.g. 'product should occupy 15-25% of frame width'",
    "inpainting_prompt": "str — detailed positive inpainting prompt for the image/video generator",
    "negative_prompt": "str — negative prompt to avoid artifacts",
    "constraints": ["list of str — specific constraints for generation"],
    "reasoning_summary": "str — 1-2 sentence summary of why this placement was chosen",
}

JUDGE_SCHEMA = {
    "score": "float 0-10 — overall realism score",
    "realism": "float 0-10",
    "lighting_match": "float 0-10",
    "perspective": "float 0-10",
    "contact_shadow": "float 0-10",
    "reflections": "float 0-10",
    "product_scale": "float 0-10",
    "visual_integration": "float 0-10",
    "artifact_detection": "float 0-10 (10 = no artifacts)",
    "issues": ["list of str — specific detected problems"],
    "updated_prompt": "str — refined inpainting prompt based on issues found",
    "updated_negative_prompt": "str — updated negative prompt",
    "accept": "bool — True if score >= threshold",
}

VIDEO_JUDGE_SCHEMA = {
    "overall_score": "float 0-10",
    "temporal_consistency": "float 0-10",
    "placement_stability": "float 0-10",
    "flicker": "float 0-10 (10 = no flicker)",
    "realism": "float 0-10",
    "lighting_consistency": "float 0-10",
    "artifacts": "float 0-10 (10 = no artifacts)",
    "issues": ["list of str"],
    "correction_notes": "str — guidance for retry",
    "accept": "bool — True if overall_score >= threshold",
}


# ---------------------------------------------------------------------------
# Helper: encode image to base64 data-URI for Qwen multimodal input
# ---------------------------------------------------------------------------
def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _load_pil(path_or_array) -> Image.Image:
    if isinstance(path_or_array, (str, Path)):
        return Image.open(path_or_array).convert("RGB")
    elif isinstance(path_or_array, np.ndarray):
        if path_or_array.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB))
        return Image.fromarray(path_or_array)
    elif isinstance(path_or_array, Image.Image):
        return path_or_array.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(path_or_array)}")


def _parse_json_from_response(text: str) -> dict:
    """Extract the first JSON block from a model response."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find ```json ... ``` block
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find bare { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# QwenReasoner — core class
# ---------------------------------------------------------------------------
class QwenReasoner:
    """
    Loads Qwen2.5-VL and exposes three reasoning calls:
      1. reason_about_placement(frames, brand_img) -> reasoning JSON
      2. judge_frame(generated_frame, brand_img, reasoning) -> judge JSON
      3. judge_video(keyframes, brand_img, reasoning) -> video_judge JSON
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        import torch
        self.torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        logger.info(f"[QwenReasoner] Loading {model_id} ...")
        self._load_model()
        logger.info("[QwenReasoner] Model loaded.")

    def _load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def _infer(self, messages: list) -> str:
        """Run Qwen inference and return the generated text."""
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 1. Placement Reasoning
    # ------------------------------------------------------------------
    def reason_about_placement(
        self,
        sampled_frames: list,      # List of PIL.Image (sampled from video)
        brand_img: Image.Image,
        video_fps: float = 25.0,
        num_sample_frames: int = 5,
        return_raw: bool = False,
    ):
        """
        Analyze sampled video frames + brand image.
        Returns structured reasoning JSON (see REASONING_SCHEMA).
        """
        logger.info("[QwenReasoner] Running placement reasoning ...")

        # Sample uniformly if too many frames provided
        if len(sampled_frames) > num_sample_frames:
            indices = np.linspace(0, len(sampled_frames) - 1, num_sample_frames, dtype=int)
            sampled_frames = [sampled_frames[i] for i in indices]

        # Build multimodal message
        content = []

        # Add sampled video frames
        for idx, frame in enumerate(sampled_frames):
            content.append({
                "type": "image",
                "image": _pil_to_base64(frame),
            })

        # Add brand image
        content.append({
            "type": "image",
            "image": _pil_to_base64(brand_img),
        })

        content.append({
            "type": "text",
            "text": (
                f"You are an expert virtual product placement director.\n"
                f"The first {len(sampled_frames)} images are uniformly sampled frames from a video "
                f"(sampled at ~{video_fps:.1f} FPS spacing).\n"
                f"The LAST image is the brand product asset to be placed into the video.\n\n"
                "Your task: identify the BEST surface in the video to place this product, "
                "and return a detailed placement specification.\n\n"
                "Consider:\n"
                "- Which surface (wall, table, shelf, screen) is most suitable\n"
                "- Camera angle and perspective (foreshortening, vanishing points)\n"
                "- Lighting direction, color temperature, and intensity\n"
                "- Product orientation for realism\n"
                "- Occlusion risk across the video\n"
                "- Visibility consistency across frames\n"
                "- Ideal product scale relative to the scene\n"
                "- Contact shadows and reflections needed\n\n"
                "CRITICAL: Return ONLY valid JSON. No explanation text. No markdown. Just raw JSON.\n"
                "Use EXACTLY this structure:\n"
                "{\n"
                '  "best_frame_index": <int>,\n'
                '  "placement_bbox": [x1, y1, x2, y2],\n'
                '  "surface_type": "<str>",\n'
                '  "camera_angle": "<str>",\n'
                '  "lighting": {"direction": "<str>", "color_temperature": "<str>", "intensity": "<str>"},\n'
                '  "product_orientation": "<str>",\n'
                '  "occlusion_risk": "<str>",\n'
                '  "visibility_score": <float 0-1>,\n'
                '  "reflections": "<str>",\n'
                '  "scale_guidance": "<str>",\n'
                '  "inpainting_prompt": "<str>",\n'
                '  "negative_prompt": "<str>",\n'
                '  "constraints": ["<str>", ...],\n'
                '  "reasoning_summary": "<str>"\n'
                "}"
            ),
        })

        messages = [{"role": "user", "content": content}]
        raw = self._infer(messages)
        logger.debug(f"[QwenReasoner] Raw placement response:\n{raw[:800]}")
        result = _parse_json_from_response(raw)
        if return_raw:
            return result, raw
        return result

    # ------------------------------------------------------------------
    # 2. Single-Frame Judge
    # ------------------------------------------------------------------
    def judge_frame(
        self,
        generated_frame: Image.Image,
        brand_img: Image.Image,
        reasoning: dict,
        score_threshold: float = 7.0,
        return_raw: bool = False,
    ):
        """
        Evaluate a single generated frame for realism.
        Returns structured judge JSON (see JUDGE_SCHEMA).
        """
        logger.info("[QwenReasoner] Running single-frame judge ...")

        prior_prompt = reasoning.get("inpainting_prompt", "")
        surface_type = reasoning.get("surface_type", "surface")

        content = [
            {"type": "image", "image": _pil_to_base64(generated_frame)},
            {"type": "image", "image": _pil_to_base64(brand_img)},
            {
                "type": "text",
                "text": (
                    "You are a photorealism quality judge for virtual product placement.\n"
                    "Image 1: The generated frame with the product composited into the scene.\n"
                    "Image 2: The original brand product asset.\n\n"
                    f"The product was intended to be placed on: '{surface_type}'.\n"
                    f"The prompt used was: \"{prior_prompt}\"\n\n"
                    "Score each dimension from 0 to 10 (10 = perfect):\n"
                    "- realism: overall photorealistic appearance\n"
                    "- lighting_match: does the product lighting match the scene?\n"
                    "- perspective: correct foreshortening and vanishing point alignment?\n"
                    "- contact_shadow: appropriate shadow at the product base?\n"
                    "- reflections: correct surface reflections on/around product?\n"
                    "- product_scale: is the product a believable size in the scene?\n"
                    "- visual_integration: does it feel like the product was always there?\n"
                    "- artifact_detection: 10=no artifacts, 0=severe artifacts\n\n"
                    "Also provide:\n"
                    "- issues: list of specific problems detected\n"
                    "- updated_prompt: improved inpainting prompt to fix these issues\n"
                    "- updated_negative_prompt: refined negative prompt\n"
                    f"- accept: true if overall score >= {score_threshold}, else false\n\n"
                    "CRITICAL: Return ONLY valid JSON. No markdown. No explanation text.\n"
                    "{\n"
                    '  "score": <float>,\n'
                    '  "realism": <float>,\n'
                    '  "lighting_match": <float>,\n'
                    '  "perspective": <float>,\n'
                    '  "contact_shadow": <float>,\n'
                    '  "reflections": <float>,\n'
                    '  "product_scale": <float>,\n'
                    '  "visual_integration": <float>,\n'
                    '  "artifact_detection": <float>,\n'
                    '  "issues": ["<str>", ...],\n'
                    '  "updated_prompt": "<str>",\n'
                    '  "updated_negative_prompt": "<str>",\n'
                    '  "accept": <bool>\n'
                    "}"
                ),
            },
        ]

        messages = [{"role": "user", "content": content}]
        raw = self._infer(messages)
        logger.debug(f"[QwenReasoner] Raw judge response:\n{raw[:800]}")
        result = _parse_json_from_response(raw)

        # Ensure accept field is set based on score
        if "score" in result:
            result["accept"] = float(result["score"]) >= score_threshold
        if return_raw:
            return result, raw
        return result

    # ------------------------------------------------------------------
    # 3. Video Judge
    # ------------------------------------------------------------------
    def judge_video(
        self,
        keyframes: list,           # List of PIL.Image sampled from output video
        brand_img: Image.Image,
        reasoning: dict,
        score_threshold: float = 7.0,
        return_raw: bool = False,
    ):
        """
        Evaluate keyframes from the generated video for temporal quality.
        Returns structured video judge JSON (see VIDEO_JUDGE_SCHEMA).
        """
        logger.info("[QwenReasoner] Running video judge ...")

        content = []
        for frame in keyframes[:6]:  # cap at 6 keyframes for context window
            content.append({"type": "image", "image": _pil_to_base64(frame)})
        content.append({"type": "image", "image": _pil_to_base64(brand_img)})

        surface_type = reasoning.get("surface_type", "surface")

        content.append({
            "type": "text",
            "text": (
                "You are evaluating temporal quality of virtual product placement in a video.\n"
                f"The first images are uniformly sampled keyframes from the output video.\n"
                f"The last image is the brand product asset.\n"
                f"Product was placed on: '{surface_type}'.\n\n"
                "Score each dimension from 0 to 10:\n"
                "- temporal_consistency: does the placement look the same across frames?\n"
                "- placement_stability: is the product position stable (no jitter/drift)?\n"
                "- flicker: 10=no flicker, 0=severe flicker\n"
                "- realism: overall photorealistic appearance across frames\n"
                "- lighting_consistency: consistent lighting across frames?\n"
                "- artifacts: 10=no artifacts, 0=severe artifacts\n\n"
                "Also provide:\n"
                "- issues: specific problems detected\n"
                "- correction_notes: guidance for retry\n"
                f"- accept: true if overall_score >= {score_threshold}, else false\n\n"
                "CRITICAL: Return ONLY valid JSON. No markdown. No explanation text.\n"
                "{\n"
                '  "overall_score": <float>,\n'
                '  "temporal_consistency": <float>,\n'
                '  "placement_stability": <float>,\n'
                '  "flicker": <float>,\n'
                '  "realism": <float>,\n'
                '  "lighting_consistency": <float>,\n'
                '  "artifacts": <float>,\n'
                '  "issues": ["<str>", ...],\n'
                '  "correction_notes": "<str>",\n'
                '  "accept": <bool>\n'
                "}"
            ),
        })

        messages = [{"role": "user", "content": content}]
        raw = self._infer(messages)
        logger.debug(f"[QwenReasoner] Raw video judge response:\n{raw[:800]}")
        result = _parse_json_from_response(raw)

        if "overall_score" in result:
            result["accept"] = float(result["overall_score"]) >= score_threshold
        if return_raw:
            return result, raw
        return result
