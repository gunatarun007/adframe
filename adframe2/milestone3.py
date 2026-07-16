"""
AdFrame 2.0 — Milestone 3 (Inference Test Runner)
==================================================
Standalone script to execute Milestone 3:
1. Load Qwen2.5-VL and reason about placement on frame0
2. Save reasoning.json
3. Load SDXL/FLUX inpaint pipeline
4. Generate a single realistic edited frame (Coca-Cola can)
5. Save candidate_v1.png
6. Run Qwen frame judge to evaluate candidate_v1.png
7. Loop up to 10 iterations to improve prompt if score < 9
8. Save all iteration details to experiments/experiment_001/
"""

import os
import json
import logging
import time
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
logger = logging.getLogger("milestone3")

from adframe2.qwen_reasoner import QwenReasoner
from adframe2.image_generator import ImageGenerator


def main():
    start_time = time.time()
    
    # Paths setup
    workspace_root = Path("/workspace")
    video_path = workspace_root / "demo.mp4"
    brand_path = workspace_root / "demo_brand.jpg"
    
    exp_dir = Path("adframe2/experiments/experiment_001")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Phase 4: Frame Extraction ===")
    frame0_path = exp_dir / "frame0.png"
    if not frame0_path.exists():
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read frame0 from demo.mp4")
        cv2.imwrite(str(frame0_path), frame)
        logger.info(f"Extracted frame0 to {frame0_path}")
    else:
        logger.info(f"frame0.png already exists at {frame0_path}")
        
    frame0 = Image.open(frame0_path).convert("RGB")
    brand_img = Image.open(brand_path).convert("RGB")

    # Benchmarking metrics
    metrics = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qwen_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "image_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "iterations": []
    }
    
    logger.info("=== Phase 5: Qwen Reasoner Initialization ===")
    reasoner = QwenReasoner(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    
    # 1. Placement Reasoning
    logger.info("Running placement reasoning on frame0...")
    qwen_start = time.time()
    reasoning = reasoner.reason_about_placement([frame0], brand_img, video_fps=25.0, num_sample_frames=1)
    qwen_end = time.time()
    qwen_inference_time = qwen_end - qwen_start
    metrics["qwen_inference_time_seconds"] = round(qwen_inference_time, 2)
    logger.info(f"Placement Reasoning complete in {qwen_inference_time:.1f}s.")
    
    # Save reasoning.json
    reasoning_path = exp_dir / "reasoning.json"
    with open(reasoning_path, "w", encoding="utf-8") as f:
        json.dump(reasoning, f, indent=2)
    logger.info(f"Saved reasoning to {reasoning_path}")
    
    # Extract placement params
    placement_bbox = reasoning.get("placement_bbox")
    # Qwen returned: [139, 418, 327, 486] -> this looks like absolute coords [ymin, xmin, ymax, xmax] or similar.
    # The image generator expects normalized coordinates [x1, y1, x2, y2] (0 to 1).
    # Let's normalize it defensively.
    w_img, h_img = frame0.size
    
    # If any value is > 1.0, assume absolute coordinates
    if any(val > 1.0 for val in placement_bbox):
        # We need to map them back. Let's inspect coordinates: [139, 418, 327, 486]
        # Bounding box coordinates often come in [y_min, x_min, y_max, x_max] format from models.
        # Let's identify the scaling factor. Typical sizes could be 960x540 or similar.
        # Let's sort and match them to coordinate axes.
        # To be safe, we normalize assuming the format is [y_min, x_min, y_max, x_max] on 1000-scale coordinates
        # standard for Qwen2.5-VL detection outputs (which maps all images to 1000x1000 grid).
        y1, x1, y2, x2 = placement_bbox
        normalized_bbox = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
    else:
        normalized_bbox = placement_bbox
        
    logger.info(f"Normalized bounding box: {normalized_bbox}")

    # Initial positive and negative prompts
    # Force a Coca-Cola placement prompt context specifically since the user mission requested it
    prompt = "a red Coca-Cola soda can placed realistically on the surface, photorealistic, high detail, volumetric lighting"
    negative_prompt = reasoning.get("negative_prompt", "blurry, low quality, artifacts")
    
    # Save initial prompts
    with open(exp_dir / "prompt_v1.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
        
    logger.info("=== Phase 6: Image Generator (SDXL) Initialization ===")
    img_gen = ImageGenerator(backend="sdxl")
    
    # 2. Iterative Judge Loop (Phase 8)
    max_iterations = 10
    target_score = 9.0
    accepted = False
    
    for i in range(1, max_iterations + 1):
        logger.info(f"\n--- Iteration {i}/{max_iterations} ---")
        logger.info(f"Current Prompt: {prompt}")
        
        # Save prompt log
        with open(exp_dir / f"prompt_v{i}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            
        # Generate candidate frame
        gen_start = time.time()
        candidate = img_gen.generate_frame(
            anchor_frame=frame0,
            brand_img=brand_img,
            placement_bbox=normalized_bbox,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            seed=42 + i
        )
        gen_end = time.time()
        gen_time = gen_end - gen_start
        logger.info(f"Candidate v{i} generated in {gen_time:.1f}s.")
        
        # Save candidate image
        candidate_path = exp_dir / f"candidate_v{i}.png"
        candidate.save(candidate_path)
        # Save as candidate_v1.png for Phase 6 requirement specifically
        if i == 1:
            candidate.save(exp_dir / "candidate_v1.png")
            
        # Judge candidate
        logger.info("Running Qwen judge on candidate image...")
        judge_start = time.time()
        judge_res = reasoner.judge_frame(
            generated_frame=candidate,
            brand_img=brand_img,
            reasoning=reasoning,
            score_threshold=target_score
        )
        judge_end = time.time()
        judge_time = judge_end - judge_start
        
        score = float(judge_res.get("score", 0))
        issues = judge_res.get("issues", [])
        updated_prompt = judge_res.get("updated_prompt", prompt)
        
        logger.info(f"Judge Score: {score}/10")
        logger.info(f"Issues detected: {issues}")
        
        # Save judge result
        judge_path = exp_dir / f"judge_v{i}.json"
        with open(judge_path, "w", encoding="utf-8") as f:
            json.dump(judge_res, f, indent=2)
            
        iteration_metrics = {
            "iteration": i,
            "prompt": prompt,
            "generation_time_seconds": round(gen_time, 2),
            "judge_time_seconds": round(judge_time, 2),
            "score": score,
            "issues": issues
        }
        metrics["iterations"].append(iteration_metrics)
        
        if score >= target_score:
            logger.info(f"Target score reached! Freezing prompt at iteration {i}.")
            accepted = True
            with open(exp_dir / "final_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            # Save the final frame
            candidate.save(exp_dir / "output_frame.png")
            break
        else:
            logger.info("Score below threshold. Updating prompt for next iteration...")
            prompt = updated_prompt
            
    if not accepted:
        logger.warning(f"Failed to reach target score {target_score} after {max_iterations} iterations.")
        with open(exp_dir / "final_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
            
    # Compile final metrics
    end_time = time.time()
    metrics["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["total_duration_seconds"] = round(end_time - start_time, 2)
    metrics["final_score"] = metrics["iterations"][-1]["score"]
    metrics["accepted"] = accepted
    
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info("=== Milestone 3 Run Completed Successfully ===")
    logger.info(f"Total time elapsed: {metrics['total_duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()
