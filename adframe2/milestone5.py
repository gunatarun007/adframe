"""
AdFrame 2.0 — Milestone 5 (Wan 2.1 VACE Full Video Integration)
==================================================================
This script orchestrates the end-to-end video pipeline:
1. Qwen Placement Reasoning
2. SDXL Prompt Refinement Loop
3. SAM 3 Video Tracking
4. Wan 2.1 VACE Video Generation
5. Qwen Video Judge & Retry Loop
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

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
logger = logging.getLogger("milestone5")

# Ensure sys.path knows about root imports for tracker/inpainter/adframe2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adframe2.experiment_package import ExperimentPackage
from adframe2.system_info import collect_system_info, collect_model_registry_entry, current_vram_mb
from adframe2.qwen_reasoner import QwenReasoner
from adframe2.image_generator import ImageGenerator

from tracker import SAM3Tracker
from inpainter import WanInpainter

def load_video_frames(video_path, max_frames=30):
    """Load video frames up to a maximum (e.g. for a 1-second clip at 30fps or full video)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR, WanInpainter expects either PIL or BGR numpy array
        # We will keep them as BGR numpy arrays and let WanInpainter handle conversion
        frames.append(frame)
    cap.release()
    return frames

def save_video_from_pil(frames, output_path, fps=25):
    """Save a list of PIL Images as an MP4 video."""
    if not frames:
        return
    width, height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for pil_frame in frames:
        # Convert PIL RGB to OpenCV BGR
        cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        out.write(cv_frame)
    out.release()
    logger.info(f"Saved video to {output_path} ({len(frames)} frames)")

def get_keyframes(pil_frames, interval=15):
    """Extract keyframes at given interval."""
    return [pil_frames[i] for i in range(0, len(pil_frames), interval)]

def main():
    logger.info("=== Starting Milestone 5 (Wan VACE Video Integration) ===")
    
    # 1. Initialize Experiment Package
    # Auto-generates experiment_NNN
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
        
        if not video_path.exists() or not brand_path.exists():
            raise FileNotFoundError("Missing demo.mp4 or demo_brand.jpg in /workspace")

        # 2. Input Setup
        logger.info("Saving inputs...")
        exp.save_input_video(str(video_path))
        brand_img = Image.open(brand_path).convert("RGB")
        exp.save_brand_reference(brand_img)

        # Load video frames
        video_frames_bgr = load_video_frames(str(video_path), max_frames=30)
        total_frames = len(video_frames_bgr)
        
        # Frame 0 for reasoning
        frame0_rgb = cv2.cvtColor(video_frames_bgr[0], cv2.COLOR_BGR2RGB)
        frame0 = Image.fromarray(frame0_rgb)
        exp.save_frame0(frame0)

        # 3. Model Initialization (Qwen)
        logger.info("Loading Qwen2.5-VL...")
        t0 = time.time()
        vram_before = current_vram_mb()
        reasoner = QwenReasoner(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        exp.register_model("judge", collect_model_registry_entry(
            "Qwen/Qwen2.5-VL-7B-Instruct", "judge", "bfloat16", time.time()-t0, vram_before, current_vram_mb()
        ))
        exp.record_vram_snapshot("after_qwen_load")

        # 4. Placement Reasoning (Qwen)
        logger.info("Running placement reasoning...")
        t0 = time.time()
        parsed_reasoning, raw_reasoning = reasoner.reason_about_placement(
            [frame0], brand_img, video_fps=25.0, num_sample_frames=1, return_raw=True
        )
        inf_time = time.time() - t0
        
        sys_prompt = "You are an expert virtual product placement director..."
        
        # Default starting prompt
        initial_prompt = "a realistic Coca-Cola can placed naturally inside the scene, photorealistic, cinematic lighting"
        initial_negative = parsed_reasoning.get("negative_prompt", "blurry, out of focus, low quality, distortion")
        
        exp.save_reasoning_iteration(raw_reasoning, parsed_reasoning, initial_prompt, sys_prompt, inf_time)
        exp.record_vram_snapshot("after_reasoning")

        # Normalize bounding box
        placement_bbox = parsed_reasoning.get("placement_bbox", [0.1, 0.1, 0.9, 0.9])
        if any(val > 1.0 for val in placement_bbox):
            y1, x1, y2, x2 = placement_bbox
            normalized_bbox = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
        else:
            normalized_bbox = placement_bbox

        # 5. Image Generator Load for Prompt Refinement Loop
        logger.info("Loading SDXL Generator for prompt refinement...")
        t0 = time.time()
        vram_before = current_vram_mb()
        img_gen = ImageGenerator(backend="sdxl")
        exp.register_model("generator", collect_model_registry_entry(
            "stabilityai/stable-diffusion-xl-base-1.0", "generator", "float16", time.time()-t0, vram_before, current_vram_mb()
        ))
        exp.record_vram_snapshot("after_sdxl_load")

        # 6. SDXL Judge Loop for the Approved Prompt
        # Limit to 3 iterations to save time for the video demo
        max_img_iterations = 3
        target_score = 9.0
        
        current_prompt = initial_prompt
        current_negative = initial_negative
        
        logger.info(f"--- SDXL Image Prompt Refinement Loop (max {max_img_iterations} iterations) ---")
        for i in range(1, max_img_iterations + 1):
            logger.info(f"Iteration {i}: Generating anchor frame...")
            candidate = img_gen.generate_frame(
                anchor_frame=frame0,
                brand_img=brand_img,
                placement_bbox=normalized_bbox,
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_inference_steps=30,
                seed=42 + i
            )
            
            logger.info(f"Iteration {i}: Judging anchor frame...")
            judge_res = reasoner.judge_frame(
                generated_frame=candidate,
                brand_img=brand_img,
                reasoning=parsed_reasoning,
                score_threshold=target_score,
            )
            
            score = float(judge_res.get("score", 0.0))
            logger.info(f"Score: {score}/10")
            
            if score >= target_score:
                logger.info("Target score reached! Approved Prompt found.")
                break
                
            current_prompt = judge_res.get("updated_prompt", current_prompt)
            current_negative = judge_res.get("updated_negative_prompt", current_negative)

        approved_prompt = current_prompt
        approved_negative = current_negative
        logger.info(f"Approved Prompt for Video: {approved_prompt}")
        
        # Free SDXL to save VRAM for Wan and SAM3
        del img_gen
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        exp.record_vram_snapshot("after_sdxl_unload")

        # 7. SAM 3 Tracking
        logger.info("Loading SAM 3 Tracker...")
        t0 = time.time()
        vram_before = current_vram_mb()
        sam_tracker = SAM3Tracker(checkpoint_path="/workspace/adframe/models/sam3.1_multiplex.pt")
        exp.register_model("tracker", collect_model_registry_entry(
            "sam3.1_multiplex.pt", "tracker", "float32", time.time()-t0, vram_before, current_vram_mb()
        ))
        
        # Save a temp clip of the max_frames we loaded so SAM3 doesn't track 990 frames and OOM
        clip_path = str(workspace_root / "clip.mp4")
        save_video_from_pil([Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in video_frames_bgr], clip_path, fps=25)
        
        surface_type = parsed_reasoning.get("surface_type", "surface")
        logger.info(f"Tracking surface: '{surface_type}' in video...")
        masks_dict = sam_tracker.track_video(clip_path, surface_type)
        
        mask_patches = []
        for i in range(total_frames):
            if i in masks_dict and masks_dict[i] is not None:
                mask_patches.append(masks_dict[i])
            else:
                # Fallback empty mask if tracking failed for a frame
                h, w = video_frames_bgr[0].shape[:2]
                mask_patches.append(np.zeros((h, w), dtype=np.uint8))
        
        # Unload SAM 3 to save VRAM
        del sam_tracker
        torch.cuda.empty_cache()
        gc.collect()
        exp.record_vram_snapshot("after_sam3_unload")

        # 8. Load Wan 2.1 VACE
        logger.info("Loading Wan 2.1 VACE (8-bit quantized)...")
        t0 = time.time()
        vram_before = current_vram_mb()
        wan_inpainter = WanInpainter(load_in_8bit=True)
        exp.register_model("wan", collect_model_registry_entry(
            "Wan-AI/Wan2.1-VACE-14B-diffusers", "video_generator", "int8/bfloat16", time.time()-t0, vram_before, current_vram_mb()
        ))
        exp.record_vram_snapshot("after_wan_load")

        # 9. Video Generation & Judge Retry Loop
        max_vid_attempts = 5
        vid_target_score = 7.5
        
        vid_prompt = approved_prompt
        vid_negative = approved_negative
        
        for attempt in range(1, max_vid_attempts + 1):
            logger.info(f"\n--- Video Generation Attempt {attempt}/{max_vid_attempts} ---")
            logger.info(f"Prompt: {vid_prompt}")
            
            # Use lower resolution for generation if OOM is a risk, but L40S should handle 480p easily
            h, w = video_frames_bgr[0].shape[:2]
            # Ensure divisible by 16
            gen_h = (h // 16) * 16
            gen_w = (w // 16) * 16
            
            t0 = time.time()
            try:
                logger.info(f"Generating video sequence ({total_frames} frames)...")
                generated_pil_frames = wan_inpainter.generate_patch_sequence(
                    video_patches=video_frames_bgr,
                    mask_patches=mask_patches,
                    prompt=vid_prompt,
                    num_frames=total_frames,
                    height=gen_h,
                    width=gen_w,
                )
            except Exception as e:
                logger.error(f"Generation failed with error: {e}")
                # Generate a fix plan manually for OOM
                fix_plan = {"error": str(e), "correction_notes": "Reduce frame count or resolution."}
                with open(exp.exp_dir / "fix_plan.json", "w") as f:
                    json.dump(fix_plan, f, indent=2)
                continue
                
            gen_time = time.time() - t0
            exp.record_timing(f"wan_gen_attempt_{attempt}_s", gen_time)
            
            # Save output.mp4
            output_vid_path = str(exp.exp_dir / "output.mp4")
            save_video_from_pil(generated_pil_frames, output_vid_path, fps=25)
            
            # Save Wan config
            wan_config = {
                "attempt": attempt,
                "prompt": vid_prompt,
                "negative_prompt": vid_negative,
                "num_frames": total_frames,
                "width": gen_w,
                "height": gen_h,
                "steps": 30,
                "guidance_scale": 5.0
            }
            exp.save_wan_config(wan_config)
            
            # Extract keyframes and Judge Video
            logger.info("Extracting keyframes and judging video...")
            keyframes = get_keyframes(generated_pil_frames, interval=15)
            
            t0 = time.time()
            video_judge_res = reasoner.judge_video(
                keyframes=keyframes,
                brand_img=brand_img,
                reasoning=parsed_reasoning,
                score_threshold=vid_target_score,
            )
            judge_time = time.time() - t0
            exp.record_timing(f"wan_judge_attempt_{attempt}_s", judge_time)
            
            # Save video report
            video_report_path = exp.exp_dir / "video_report.json"
            with open(video_report_path, "w") as f:
                json.dump(video_judge_res, f, indent=2)
                
            score = float(video_judge_res.get("overall_score", 0.0))
            logger.info(f"Video Judge Score: {score}/10")
            
            if score >= vid_target_score or video_judge_res.get("accept", False):
                logger.info("Video target score reached! Saving output and finishing.")
                # We have our successful output.mp4
                exp.save_output_video(output_vid_path) # converts to gif and low res too
                break
            else:
                logger.warning(f"Video failed validation. Issues: {video_judge_res.get('issues', [])}")
                fix_plan = {
                    "attempt": attempt,
                    "score": score,
                    "issues": video_judge_res.get("issues", []),
                    "correction_notes": video_judge_res.get("correction_notes", "")
                }
                with open(exp.exp_dir / "fix_plan.json", "w") as f:
                    json.dump(fix_plan, f, indent=2)
                
                # Perturb the prompt based on feedback
                vid_prompt = vid_prompt + f", {video_judge_res.get('correction_notes', 'more photorealistic')}"
                logger.info(f"Refined Video Prompt for retry: {vid_prompt}")

    except Exception as e:
        logger.exception("Pipeline failed:")
    finally:
        exp.close()


if __name__ == "__main__":
    main()
