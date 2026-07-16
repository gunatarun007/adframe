"""
AdFrame 2.0 — Main Orchestration Pipeline
============================================
Self-improving VPP pipeline driven entirely by foundation models.

Pipeline:
  Video
    -> [Qwen] Placement Reasoning -> reasoning.json
    -> [Image Gen] Single-frame edit + [Qwen] Judge loop
          (repeat up to max_image_iterations, stop when score >= image_threshold)
    -> [Wan VACE] Full video generation
    -> [Qwen] Video judge
          (repeat up to max_video_iterations, stop when score >= video_threshold)
    -> Export final video

Usage:
  python adframe2.0/run.py \\
    --video /workspace/demo.mp4 \\
    --brand /workspace/demo_brand.jpg \\
    --output_dir adframe2.0/experiments \\
    --image_threshold 7.0 \\
    --video_threshold 7.0 \\
    --max_image_iterations 3 \\
    --max_video_iterations 2 \\
    --image_backend flux \\
    --wan_model Wan-AI/Wan2.1-VACE-1.3B-diffusers \\
    --qwen_model Qwen/Qwen2.5-VL-7B-Instruct \\
    --num_wan_frames 17 \\
    --seed 42
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Environment setup (RunPod disk quota workaround) ────────────────────────
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/huggingface/hub")
os.environ.setdefault("TMPDIR", "/tmp")

# ── Allow importing sibling modules ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))  # adframe root
sys.path.insert(0, str(Path(__file__).parent))          # adframe2.0 root

from experiment_logger import ExperimentLogger
from qwen_reasoner import QwenReasoner
from image_generator import ImageGenerator
from wan_generator import WanVideoGenerator
from video_utils import (
    get_video_metadata,
    sample_frames_from_video,
    get_frame_at_index,
    load_frames_from_video,
    paste_generated_frames_into_video,
    write_video,
    extract_keyframes,
)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adframe2")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class AdFrame2Pipeline:
    """
    Full self-improving VPP pipeline for AdFrame 2.0.
    """

    def __init__(self, args):
        self.args = args
        self.exp = ExperimentLogger()
        self.exp.log(f"Pipeline started. Video: {args.video}, Brand: {args.brand}")
        self.exp.log(f"Config: {vars(args)}")

    def run(self) -> str:
        """
        Execute the full pipeline end-to-end.
        Returns the path to the final output video.
        """
        args = self.args
        exp = self.exp

        # ── Stage 0: Load metadata ────────────────────────────────────────
        exp.log("=== Stage 0: Loading video metadata ===")
        meta = get_video_metadata(args.video)
        exp.log(f"Video: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} FPS, "
                f"{meta['total_frames']} frames ({meta['duration_seconds']:.1f}s)")

        # ── Stage 1: Qwen Placement Reasoning ────────────────────────────
        exp.log("=== Stage 1: Qwen Placement Reasoning ===")
        qwen = QwenReasoner(
            model_id=args.qwen_model,
            device="cuda",
        )

        from PIL import Image
        brand_img = Image.open(args.brand).convert("RGB")

        sampled_frames, frame_indices, fps = sample_frames_from_video(
            args.video, num_samples=args.num_sample_frames
        )
        exp.log(f"Sampled {len(sampled_frames)} frames at indices {frame_indices}")

        reasoning = qwen.reason_about_placement(
            sampled_frames=sampled_frames,
            brand_img=brand_img,
            video_fps=fps,
            num_sample_frames=args.num_sample_frames,
        )
        exp.log(f"Reasoning complete. Surface: {reasoning.get('surface_type')}, "
                f"BestFrame: {reasoning.get('best_frame_index')}, "
                f"BBox: {reasoning.get('placement_bbox')}")
        exp.save_reasoning(reasoning)

        prompt = reasoning.get("inpainting_prompt", "product placement, photorealistic")
        negative_prompt = reasoning.get("negative_prompt", "blurry, distorted, unrealistic")
        placement_bbox = reasoning.get("placement_bbox", [0.3, 0.1, 0.7, 0.5])
        best_frame_index = reasoning.get("best_frame_index", 0)

        # ── Stage 2: Image Generator + Qwen Judge Loop ───────────────────
        exp.log("=== Stage 2: Image Generation + Judge Loop ===")
        img_gen = ImageGenerator(backend=args.image_backend, device="cuda")

        anchor_frame = get_frame_at_index(args.video, best_frame_index)
        if anchor_frame is None:
            anchor_frame = sampled_frames[0]

        judge_history = []
        best_frame_gen = None
        best_frame_score = 0.0

        for iteration in range(args.max_image_iterations):
            exp.log(f"--- Image iteration {iteration + 1}/{args.max_image_iterations} ---")
            exp.log(f"Prompt: {prompt[:120]}")

            seed = args.seed + iteration if args.seed is not None else None
            generated_frame = img_gen.generate_frame(
                anchor_frame=anchor_frame,
                brand_img=brand_img,
                placement_bbox=placement_bbox,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )

            exp.save_output_frame(generated_frame, iteration=iteration)

            # Judge the generated frame
            exp.log("Running Qwen frame judge ...")
            judge_result = qwen.judge_frame(
                generated_frame=generated_frame,
                brand_img=brand_img,
                reasoning=reasoning,
                score_threshold=args.image_threshold,
            )
            judge_result["iteration"] = iteration + 1
            judge_result["prompt_used"] = prompt
            judge_history.append(judge_result)

            score = float(judge_result.get("score", 0))
            accept = judge_result.get("accept", False)
            issues = judge_result.get("issues", [])
            exp.log(f"Judge score: {score:.1f}/10, Accept: {accept}, Issues: {issues}")

            if score > best_frame_score:
                best_frame_score = score
                best_frame_gen = generated_frame

            if accept:
                exp.log(f"Frame accepted at iteration {iteration + 1} with score {score:.1f}")
                prompt = judge_result.get("updated_prompt", prompt)
                negative_prompt = judge_result.get("updated_negative_prompt", negative_prompt)
                break
            else:
                # Refine prompt for next iteration
                prompt = judge_result.get("updated_prompt", prompt)
                negative_prompt = judge_result.get("updated_negative_prompt", negative_prompt)
                exp.log(f"Frame rejected. Refined prompt: {prompt[:100]}")

        exp.save_judge_history(judge_history)
        exp.save_prompt(prompt, negative_prompt)
        exp.log(f"Image judge loop complete. Best score: {best_frame_score:.1f}")

        # Free image generator memory before loading Wan
        del img_gen
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

        # ── Stage 3: Wan VACE Full Video Generation ───────────────────────
        exp.log("=== Stage 3: Wan VACE Video Generation ===")
        wan = WanVideoGenerator(model_id=args.wan_model, device="cuda")

        # Load all frames as PIL RGB for Wan input
        all_frames_bgr, _ = load_frames_from_video(args.video)
        import numpy as np
        from PIL import Image as PILImage
        all_frames_pil = [
            PILImage.fromarray(cv2_frame_to_rgb(f)) for f in all_frames_bgr
        ]

        # Video retry loop
        video_judge_result = None
        final_video_path = None

        for video_iter in range(args.max_video_iterations):
            exp.log(f"--- Video iteration {video_iter + 1}/{args.max_video_iterations} ---")
            seed = (args.seed + 100 + video_iter) if args.seed is not None else None

            generated_video_frames = wan.generate_video(
                video_frames=all_frames_pil,
                brand_img=brand_img,
                placement_bbox=placement_bbox,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=args.num_wan_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                seed=seed,
            )
            exp.log(f"Wan generated {len(generated_video_frames)} frames.")

            # Composite back into full-resolution video
            output_frames_bgr = paste_generated_frames_into_video(
                original_frames_bgr=all_frames_bgr,
                generated_pil_frames=generated_video_frames,
                placement_bbox=placement_bbox,
                blend_feather=args.blend_feather,
            )

            tmp_video_path = str(exp.experiment_dir / f"output_video_iter{video_iter:02d}.mp4")
            write_video(output_frames_bgr, tmp_video_path, fps=fps)
            exp.log(f"Video written: {tmp_video_path}")

            # Video judge
            exp.log("Running Qwen video judge ...")
            keyframes = extract_keyframes(tmp_video_path, num_keyframes=6)
            video_judge_result = qwen.judge_video(
                keyframes=keyframes,
                brand_img=brand_img,
                reasoning=reasoning,
                score_threshold=args.video_threshold,
            )
            video_judge_result["iteration"] = video_iter + 1
            exp.log(f"Video judge score: {video_judge_result.get('overall_score', '?'):.1f}/10, "
                    f"Accept: {video_judge_result.get('accept', False)}")

            final_video_path = tmp_video_path

            if video_judge_result.get("accept", False):
                exp.log(f"Video accepted at iteration {video_iter + 1}.")
                break
            else:
                correction_notes = video_judge_result.get("correction_notes", "")
                exp.log(f"Video rejected. Correction notes: {correction_notes}")
                # Could refine prompt here in future iterations

        exp.save_video_judge(video_judge_result or {})
        if final_video_path:
            exp.save_output_video(final_video_path)

        wan.unload()

        # ── Stage 4: Save metrics and summary ─────────────────────────────
        metrics = {
            "video_metadata": meta,
            "surface_type": reasoning.get("surface_type"),
            "placement_bbox": placement_bbox,
            "best_frame_index": best_frame_index,
            "image_judge_iterations": len(judge_history),
            "best_image_score": best_frame_score,
            "video_judge_score": (
                float(video_judge_result.get("overall_score", 0))
                if video_judge_result else None
            ),
            "video_accepted": (
                video_judge_result.get("accept", False)
                if video_judge_result else False
            ),
            "final_prompt": prompt,
            "final_negative_prompt": negative_prompt,
        }
        exp.save_metrics(metrics)
        exp.print_summary()

        return final_video_path or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cv2_frame_to_rgb(frame_bgr):
    import cv2
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="AdFrame 2.0 — Foundation Model VPP Pipeline"
    )
    # Inputs
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--brand", required=True, help="Path to brand product image")

    # Models
    parser.add_argument(
        "--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL model ID"
    )
    parser.add_argument(
        "--image_backend", default="flux", choices=["flux", "sdxl"],
        help="Image generator backend for frame judge loop"
    )
    parser.add_argument(
        "--wan_model", default="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        help="Wan 2.1 VACE model ID"
    )

    # Judge loop config
    parser.add_argument("--image_threshold", type=float, default=7.0,
                        help="Minimum Qwen score to accept generated frame (0-10)")
    parser.add_argument("--video_threshold", type=float, default=7.0,
                        help="Minimum Qwen score to accept generated video (0-10)")
    parser.add_argument("--max_image_iterations", type=int, default=3,
                        help="Max image generation retries per experiment")
    parser.add_argument("--max_video_iterations", type=int, default=2,
                        help="Max video generation retries per experiment")

    # Sampling
    parser.add_argument("--num_sample_frames", type=int, default=6,
                        help="Number of frames sampled for Qwen placement reasoning")

    # Wan VACE generation
    parser.add_argument("--num_wan_frames", type=int, default=17,
                        help="Number of frames to generate with Wan VACE (must satisfy (n-1)%4==0)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)

    # Compositing
    parser.add_argument("--blend_feather", type=int, default=30,
                        help="Gaussian feather radius for mask edge blending")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = AdFrame2Pipeline(args)
    output_path = pipeline.run()
    print(f"\n[AdFrame 2.0] Done. Output: {output_path}")


if __name__ == "__main__":
    main()
