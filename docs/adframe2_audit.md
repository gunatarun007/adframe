# AdFrame 1.0 — Codebase Audit
> Generated: 2026-07-16 | Auditor: Antigravity AI
> Purpose: Baseline audit before designing AdFrame 2.0 experimental branch.

---

## 1. Repository Layout

```
adframe/
├── main.py                  # CLI entrypoint & orchestration
├── tracker.py               # SAM 3 video tracking wrapper
├── inpainter.py             # Wan 2.1 VACE diffusion wrapper
├── compositor.py            # Crop/paste/blend geometry logic
├── setup.sh                 # Linux environment bootstrap script
│
├── extract_frame0.py        # One-off util: save frame 0 as JPEG
├── list_hf_files.py         # One-off util: probe HF repo file sizes
├── list_objects.py          # Debug util: test SAM3 text prompts
├── scan_all_objects.py      # Debug util: scan many categories/thresholds
├── test_box_prompt.py       # Debug util: test SAM3 bounding box prompt
├── test_outputs.py          # Debug util: inspect SAM3 propagate outputs
├── test_thresh.py           # Debug util: sweep detection thresholds
├── patch_sam3.py            # Monkey-patch script for SAM3 argsort bug
│
├── demo_brand.jpg           # Sample brand asset (tracked by git-lfs)
├── frame0.jpg               # Extracted first frame of demo video
├── output.mp4               # Pipeline output video (gitignored in CI)
└── .gitignore
```

**No config files**, **no requirements.txt**, **no tests directory**, **no docs** prior to this audit.
All paths are hardcoded or CLI args. No logging framework, no experiment tracking.

---

## 2. Architecture Diagram (AdFrame 1.0)

```
+-----------------------------------------------------------------+
|                        main.py (CLI)                            |
|  argparse: --video --brand --prompt --output                    |
|            --sam_checkpoint --sam_bpe --wan_model --load_in_8bit|
+---------------------------+-------------------------------------+
                            |
            +---------------v---------------+
            |  [1] Video Loader (cv2)        |
            |  cv2.VideoCapture              |
            |  -> List[np.ndarray BGR]       |
            |  -> fps (float)                |
            +---------------+---------------+
                            |
            +---------------v---------------+
            |  [2] SAM3Tracker (tracker.py)  |
            |  build_sam3_multiplex_video_   |
            |    predictor()                 |
            |  start_session                 |
            |  add_prompt (text, frame 0)    |
            |  propagate_in_video            |
            |  -> Dict[frame_idx, mask(H,W)] |
            |  Fallback: synthetic 200x200   |
            |    mask at fixed coords        |
            +---------------+---------------+
                            |
            +---------------v---------------+
            |  [3] VPPCompositor             |
            |  (compositor.py)               |
            |  calculate_centroids()         |
            |    forward/backward fill       |
            |  extract_stabilized_crops()    |
            |    480x480 patch @ centroid    |
            |    cv2 replicate-pad borders   |
            |  crop_masks()                  |
            |  blend_brand_asset_onto_crop() |
            |    resize brand -> bbox        |
            |    alpha blend on mask         |
            +---------------+---------------+
                            |
            +---------------v---------------+
            |  [4] WanInpainter              |
            |  (inpainter.py)                |
            |  WanVACEPipeline.from_pretrained|
            |    bfloat16 transformer        |
            |  pipe.vae.to(float32)          |
            |  enable_model_cpu_offload()    |
            |  vae.enable_slicing()          |
            |  30-step diffusion, 16 frames  |
            |  _frame_to_pil() decoder       |
            |  -> List[PIL.Image]            |
            +---------------+---------------+
                            |
            +---------------v---------------+
            |  [5] Paste-Back Compositor     |
            |  modulo-index 16 patches       |
            |  paste_and_blend_frame()       |
            |    GaussianBlur mask feather   |
            |    alpha blend patch->frame    |
            +---------------+---------------+
                            |
            +---------------v---------------+
            |  [6] Video Writer              |
            |  cv2.VideoWriter (mp4v)        |
            |  -> output.mp4                 |
            +--------------------------------+
```

---

## 3. Execution Order (Step-by-Step)

| Step | Module | What Happens |
|------|--------|-------------|
| 1 | `main.py` | Parse CLI args |
| 2 | `main.py` | `cv2.VideoCapture` load all frames into RAM as BGR numpy arrays |
| 3 | `tracker.py` | Load SAM 3.1 multiplex checkpoint; monkey-patch `init_state` |
| 4 | `tracker.py` | Start session, add text prompt on frame 0, propagate across all frames |
| 5 | `tracker.py` | If 0 masks returned -> generate synthetic fixed-region fallback mask |
| 6 | `compositor.py` | Calculate per-frame centroid from mask; interpolate missing ones |
| 7 | `compositor.py` | Extract 480x480 stabilized crop patches + matching mask patches |
| 8 | `compositor.py` | Alpha-blend brand image onto frame 0's crop using mask bbox |
| 9 | `inpainter.py` | Load Wan 2.1 VACE 1.3B; cast VAE->float32; enable CPU offload + slicing |
| 10 | `inpainter.py` | Run 30-step diffusion on 16 frames @ 480x480 |
| 11 | `inpainter.py` | Decode 5D numpy/tensor output -> List[PIL.Image] |
| 12 | `main.py` | For all 750 frames: modulo-index patches, paste+blend back |
| 13 | `main.py` | `cv2.VideoWriter` -> write `output.mp4` |

---

## 4. Key Dependencies

| Library | Role | Version Constraint |
|---------|------|--------------------|
| `torch` | GPU compute, bfloat16/float32 | CUDA 12.1 (`cu121`) |
| `diffusers` | `WanVACEPipeline` | Latest from source (git) |
| `transformers` | Tokenizer, BitsAndBytes config | Any recent |
| `accelerate` | `enable_model_cpu_offload` | Any recent |
| `opencv-python` | Video I/O, image ops | Any recent |
| `Pillow` | PIL Image format bridging | Any recent |
| `numpy` | Array operations throughout | Any recent |
| `sam3` | Meta SAM 3 tracking | Cloned from `facebookresearch/sam3` |
| `bitsandbytes` | INT8 quantization (optional) | Any recent |
| `huggingface_hub` | Model downloading | Any recent |
| `ffmpeg` (system) | Video codec support | System package |

**No `requirements.txt`.** Dependencies are installed imperatively via `setup.sh`.

---

## 5. Reusable Modules (Keep for AdFrame 2.0)

| Module / Utility | What to Reuse | Notes |
|------------------|---------------|-------|
| `compositor.py::extract_stabilized_crops` | Centroid-based stable crop extraction | Solid geometry |
| `compositor.py::paste_and_blend_frame` | Gaussian-blur mask feathering blend-back | Good boundary smoothing |
| `compositor.py::calculate_centroids` | Forward/backward fill interpolation | Handles missing frames |
| `inpainter.py::WanInpainter` | Wan VACE loading + `_frame_to_pil` | Robust dtype decoder |
| `inpainter.py` env var block | `HF_HOME`, `TMPDIR` redirects | L40S quota workaround |
| `setup.sh` | System deps + PyTorch + diffusers | Reuse and extend |
| cv2 VideoCapture pattern | Frame loading + fps | Simple and portable |
| `_frame_to_pil` helper | Handles tensor/ndarray/list polymorphism | Battle-tested |

---

## 6. Bottlenecks

| Bottleneck | Severity | Root Cause |
|-----------|----------|------------|
| SAM3 text prompt -> 0 detections | CRITICAL | SAM3-Multiplex fails on flat "stuff" surfaces (walls, posters). Text-grounding silently returns nothing. |
| 16-frame modulo tiling | CRITICAL | Only 16 unique AI frames looped across 750 -> visible repetition, no temporal coherence. |
| Full video loaded into RAM | MEDIUM | 750 frames at 960x540 BGR ~1.1 GB. Scales poorly for longer videos. |
| Prompt is hardcoded / guessed | MEDIUM | No intelligence about what object to track or where to place. Human must guess. |
| Fixed 480x480 crop size | MEDIUM | Does not adapt to actual mask size/aspect ratio. |
| No feedback loop | MEDIUM | Single-pass. Bad result -> no retry or refinement. |
| No evaluation metric | LOW | No PSNR/SSIM/VQA score. Quality is subjective. |
| cv2 mp4v container | LOW | Poor browser compatibility. Should use avc1 or ffmpeg post-encode. |
| SAM3 monkey-patch | LOW | Fragile against SAM3 upstream changes. |
| No experiment tracking | LOW | Every run overwrites output.mp4. No history, no reproducibility. |
| No config system | LOW | All parameters inline or ad-hoc CLI args. No YAML/JSON registry. |

---

## 7. Architecture Summary — What to Keep vs Replace

### Keep
- `compositor.py` crop/paste/blend geometry (pure numpy/cv2, no model dependency)
- `inpainter.py` Wan VACE wrapper core (loading + dtype decoder)
- `setup.sh` base structure for environment bootstrapping
- `cv2.VideoCapture` frame loading pattern
- `/tmp` cache redirect strategy for RunPod disk quota
- `.gitignore` patterns (models, checkpoints, outputs)

### Replace / Remove
- **SAM3 tracking** -> Replace with Qwen2.5-VL for surface reasoning and bbox output
- **Static text-prompt grounding** -> Qwen multi-frame analysis for best placement frame selection
- **Single-pass pipeline** -> Iterative judge loop: Image Gen -> Qwen VQA Judge -> Refine -> Wan Video
- **Fixed diffusion prompt** -> Qwen-generated structured prompt with reasoning JSON
- **Hardcoded fallback mask** -> Qwen bounding box -> soft mask (or zero-shot VACE conditioning)
- **Modulo frame tiling** -> Generate full-length video using VACE's full temporal capacity
- **No experiment logging** -> Structured `experiments/experiment_NNN/` directories with all artifacts
- **No evaluation** -> Qwen-as-judge loop with structured score JSON

---

## 8. Open Questions for AdFrame 2.0 Design

1. Does Qwen2.5-VL expose bbox coordinate output reliable enough to replace SAM3 for surface localization?
2. Which image generator for the single-frame judge loop? FLUX.1-dev vs SDXL vs Wan I2I mode?
3. How to pass Qwen bbox output to Wan VACE without classical segmentation? Approximate soft mask from bbox?
4. Temporal coherence: Wan VACE 1.3B generates ~17 usable frames at 480x480. Full 750-frame generation
   is infeasible at 1.3B scale. Strategy options: generate anchor frames then RIFE interpolate remaining?
5. Judge threshold: what scoring rubric and accept threshold does Qwen use?
6. FLUX.1-dev requires ~24 GB VRAM. L40S has 48 GB. Sequential use alongside Wan is feasible.

---

*End of AdFrame 1.0 Audit. Proceed to AdFrame 2.0 skeleton implementation.*
