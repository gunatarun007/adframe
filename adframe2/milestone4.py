"""
AdFrame 2.0 — Milestone 4 (Reproducibility & Experiment Package)
==================================================================
This script runs the full self-improving pipeline while enforcing
the strict 17-point directory structure required for scientific
reproducibility.

It leverages the new `ExperimentPackage` and `system_info` modules.
"""

import os
import sys
import time
import logging
from pathlib import Path
from PIL import Image

# Enforce RunPod disk quota cache locations
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["TMPDIR"] = "/tmp"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("milestone4")

from adframe2.experiment_package import ExperimentPackage
from adframe2.system_info import collect_system_info, collect_model_registry_entry
from adframe2.qwen_reasoner import QwenReasoner
from adframe2.image_generator import ImageGenerator

def main():
    logger.info("=== Starting Milestone 4 (Strict Reproducibility) ===")
    
    # 1. Initialize Experiment Package
    exp = ExperimentPackage()
    exp.record_vram_snapshot("boot")

    try:
        # Save System Info
        sys_info = collect_system_info(git_root="/workspace/adframe")
        exp.save_system_info(sys_info)

        # Paths setup
        workspace_root = Path("/workspace")
        video_path = workspace_root / "demo.mp4"
        brand_path = workspace_root / "demo_brand.jpg"

        # 2. Input Setup
        logger.info("Saving inputs...")
        exp.save_input_video(str(video_path))
        brand_img = Image.open(brand_path).convert("RGB")
        exp.save_brand_reference(brand_img)

        # Extract frame 0
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read frame0 from demo.mp4")
        frame0_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame0 = Image.fromarray(frame0_rgb)
        exp.save_frame0(frame0)
        
        # Also extract sampled frames for reasoning
        exp.extract_and_score_frames(str(video_path), None, brand_img, sample_interval=50)
        # Assuming we just use frame0 for reasoning for now as in M3
        exp.save_sampled_frames([frame0], [0])

        # 3. Model Initialization (with registry tracking)
        logger.info("Loading Qwen2.5-VL...")
        t0 = time.time()
        from adframe2.system_info import current_vram_mb
        vram_before = current_vram_mb()
        reasoner = QwenReasoner(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        vram_after = current_vram_mb()
        exp.register_model("judge", collect_model_registry_entry(
            "Qwen/Qwen2.5-VL-7B-Instruct", "judge", "bfloat16", time.time()-t0, vram_before, vram_after
        ))
        exp.record_vram_snapshot("after_qwen_load")

        # 4. Placement Reasoning
        logger.info("Running placement reasoning...")
        t0 = time.time()
        parsed_reasoning, raw_reasoning = reasoner.reason_about_placement(
            [frame0], brand_img, video_fps=25.0, num_sample_frames=1, return_raw=True
        )
        inf_time = time.time() - t0
        
        # System prompt equivalent for logging
        sys_prompt = "You are an expert virtual product placement director..."
        
        # Force prompt for M4 demo
        initial_prompt = "a red Coca-Cola soda can placed realistically on the surface, photorealistic, high detail, volumetric lighting"
        initial_negative = parsed_reasoning.get("negative_prompt", "blurry, low quality, artifacts")
        
        exp.save_reasoning_iteration(raw_reasoning, parsed_reasoning, initial_prompt, sys_prompt, inf_time)
        exp.record_vram_snapshot("after_reasoning")

        # Normalize bounding box
        placement_bbox = parsed_reasoning.get("placement_bbox", [0, 0, 1, 1])
        if any(val > 1.0 for val in placement_bbox):
            y1, x1, y2, x2 = placement_bbox
            normalized_bbox = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
        else:
            normalized_bbox = placement_bbox

        # 5. Image Generator Load
        logger.info("Loading SDXL Generator...")
        t0 = time.time()
        vram_before = current_vram_mb()
        img_gen = ImageGenerator(backend="sdxl")
        vram_after = current_vram_mb()
        exp.register_model("generator", collect_model_registry_entry(
            "stabilityai/stable-diffusion-xl-base-1.0", "generator", "float16", time.time()-t0, vram_before, vram_after
        ))
        exp.record_vram_snapshot("after_sdxl_load")

        # 6. Judge Loop
        max_iterations = 10
        target_score = 9.0
        accepted = False
        
        current_prompt = initial_prompt
        current_negative = initial_negative
        
        candidates = []
        scores = []
        judge_history = []
        
        for i in range(1, max_iterations + 1):
            logger.info(f"\n--- Iteration {i}/{max_iterations} ---")
            
            # Generate
            t0 = time.time()
            gen_config = {"backend": "sdxl", "steps": 30, "seed": 42 + i, "bbox": normalized_bbox}
            candidate = img_gen.generate_frame(
                anchor_frame=frame0,
                brand_img=brand_img,
                placement_bbox=normalized_bbox,
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_inference_steps=30,
                seed=42 + i
            )
            gen_time = time.time() - t0
            exp.record_timing(f"gen_{i}_s", gen_time)
            exp.save_generation(candidate, gen_config, current_prompt, current_negative)
            candidates.append(candidate)
            
            # Judge
            t0 = time.time()
            judge_res, raw_judge = reasoner.judge_frame(
                generated_frame=candidate,
                brand_img=brand_img,
                reasoning=parsed_reasoning,
                score_threshold=target_score,
                return_raw=True
            )
            judge_time = time.time() - t0
            
            score = float(judge_res.get("score", 0.0))
            scores.append(score)
            judge_history.append(judge_res)
            
            exp.save_judge_iteration(judge_res, raw_judge, judge_time)
            exp.record_prompt(
                prompt=current_prompt, 
                negative=current_negative, 
                iteration=i, 
                changes=f"Feedback applied: {judge_res.get('issues', [])}" if i > 1 else "Initial"
            )
            exp.record_vram_snapshot(f"after_iter_{i}")
            
            if score >= target_score:
                logger.info(f"Target score reached: {score}/10")
                accepted = True
                break
                
            # Update prompts for next iteration if not accepted
            current_prompt = judge_res.get("updated_prompt", current_prompt)
            current_negative = judge_res.get("updated_negative_prompt", current_negative)

        # 7. Comparison Grid
        exp.build_comparison_grid(frame0, candidates, scores)

        # 8. Reports
        final_score = scores[-1] if scores else 0.0
        exp.generate_report(parsed_reasoning, judge_history, final_score, accepted)
        exp.generate_research_summary(final_score, accepted, len(candidates))

    except Exception as e:
        logger.exception("Pipeline failed:")
    finally:
        exp.close()


if __name__ == "__main__":
    main()
