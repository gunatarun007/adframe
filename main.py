import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.makedirs("/workspace/tmp", exist_ok=True)
import argparse
import cv2
import numpy as np
from PIL import Image

from tracker import SAM3Tracker
from inpainter import WanInpainter
from compositor import VPPCompositor

def main():
    parser = argparse.ArgumentParser(description="Virtual Product Placement (VPP) SaaS Pipeline")
    parser.add_argument("--video", type=str, default="/workspace/demo.mp4",
                        help="Path to full-res source video")
    parser.add_argument("--brand", type=str, default="/workspace/demo_brand.jpg",
                        help="Path to brand asset image")
    parser.add_argument("--prompt", type=str, default="wall poster",
                        help="Text prompt for the surface to segment and track")
    parser.add_argument("--output", type=str, default="./output.mp4",
                        help="Path to save output video")
    parser.add_argument("--sam_checkpoint", type=str, default="./models/sam3.1_multiplex.pt",
                        help="Path to SAM 3 checkpoint")
    parser.add_argument("--sam_bpe", type=str, default=None,
                        help="Path to SAM 3 BPE text tokenizer vocab file")
    parser.add_argument("--wan_model", type=str, default="Wan-AI/Wan2.1-VACE-14B-diffusers",
                        help="Wan 2.1 Model ID on Hugging Face")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Enable 8-bit quantization for Wan 2.1 model")
    args = parser.parse_args()

    # 1. Load Video and Brand Asset
    print(f"[Main] Loading source video: {args.video}...")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Input video not found: {args.video}")
        
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    total_frames = len(frames)
    print(f"[Main] Loaded {total_frames} frames from video.")
    if total_frames == 0:
        raise ValueError("Video file contains 0 frames.")

    print(f"[Main] Loading brand asset: {args.brand}...")
    if not os.path.exists(args.brand):
        raise FileNotFoundError(f"Brand asset not found: {args.brand}")
    brand_img = cv2.imread(args.brand)

    # 2. Track Surface using SAM 3
    tracker = SAM3Tracker(checkpoint_path=args.sam_checkpoint, bpe_path=args.sam_bpe)
    masks = tracker.track_video(args.video, args.prompt)
    
    if len(masks) == 0:
        print("[Main] Warning: No masks were tracked. Exiting without modifications.")
        return

    # 3. Create Compositor and Crop Stabilized Patches
    # We use 480x480 crop size matching Wan 2.1 I2V 480P expectations
    compositor = VPPCompositor(crop_width=480, crop_height=480, blur_kernel_size=25)
    
    print("[Main] Calculating mask centroids for stabilization...")
    centroids = compositor.calculate_centroids(masks, total_frames)
    
    print("[Main] Extracting stabilized crops and masks...")
    crop_patches, crop_coords = compositor.extract_stabilized_crops(frames, centroids)
    crop_masks = compositor.crop_masks(masks, centroids, total_frames)

    # 4. Blend Brand Asset onto first frame crop
    print("[Main] Generating composite for first frame crop...")
    first_crop = crop_patches[0]
    first_mask = crop_masks[0]
    composite_first_crop = compositor.blend_brand_asset_onto_crop(first_crop, first_mask, brand_img)

    # Construct the input video sequence for the VACE pipeline
    # Frame 0 has the brand asset composite
    vace_input_patches = [composite_first_crop]
    # Subsequent frames have the original crop patches
    for f_patch in crop_patches[1:]:
        vace_input_patches.append(f_patch)

    # 5. Run Wan 2.1 Inpainting / Video Generation
    inpainter = WanInpainter(model_id=args.wan_model, load_in_8bit=args.load_in_8bit)
    
    # Generate the patch sequence starting from the brand-composited frame
    diffusion_prompt = f"a clean corporate branding asset perfectly embedded on a {args.prompt}, photorealistic, matching lights and shadows, ultra-detailed"
    generated_pil_patches = inpainter.generate_patch_sequence(
        video_patches=vace_input_patches,
        mask_patches=crop_masks,
        prompt=diffusion_prompt,
        num_frames=total_frames,
        height=480,
        width=480
    )

    # 6. Blend generated patches back into full-res frames
    print("[Main] Compositing generated patches back into full-resolution frames...")
    output_frames = []
    num_generated = len(generated_pil_patches)
    
    for i in range(total_frames):
        # Fallback in case generation frame count differs from original video frames (looping/padding)
        patch_idx = i % num_generated
        
        gen_patch_pil = generated_pil_patches[patch_idx]
        gen_patch_rgb = np.array(gen_patch_pil)
        gen_patch_bgr = cv2.cvtColor(gen_patch_rgb, cv2.COLOR_RGB2BGR)
        
        # Original mask for blending
        orig_mask = masks.get(i)
        if orig_mask is None:
            orig_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            
        final_frame = compositor.paste_and_blend_frame(
            original_frame=frames[i],
            generated_patch=gen_patch_bgr,
            original_mask=orig_mask,
            crop_coord=crop_coords[i]
        )
        output_frames.append(final_frame)

    # 7. Save output video
    print(f"[Main] Saving final composited video to: {args.output}...")
    h_orig, w_orig = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w_orig, h_orig))
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    print("[Main] Video editing pipeline executed successfully!")

if __name__ == "__main__":
    main()
