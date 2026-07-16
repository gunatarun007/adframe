# AdFrame 2.0 — README

## Research Question
> Can modern foundation models alone perform convincing virtual product placement
> without a large classical CV pipeline?

## What's Different from AdFrame 1.0

| Aspect | AdFrame 1.0 | AdFrame 2.0 |
|--------|-------------|-------------|
| Surface detection | SAM3 text prompt (fails ~100% on walls) | Qwen2.5-VL reasoning → bbox |
| Mask generation | SAM3 propagation + synthetic fallback | Soft Gaussian mask from Qwen bbox |
| Prompt | Static hardcoded string | Qwen-generated structured JSON prompt |
| Feedback | None — single pass | Judge loop: Image Gen → Qwen → Refine |
| Video generation | 16 frames, modulo tiled | Wan VACE with Qwen-approved prompt |
| Video evaluation | None | Qwen video judge on keyframes |
| Experiment logging | None — overwrites output.mp4 | Auto-numbered experiments/ dirs |

## Models Used

| Model | Purpose | VRAM |
|-------|---------|------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | Placement reasoning + frame/video judge | ~18 GB |
| `black-forest-labs/FLUX.1-dev` | Single-frame prompt validation | ~24 GB |
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | Full video inpainting | ~10 GB |

Total peak VRAM: ~24 GB (sequential loading, not concurrent)
L40S available: 48 GB ✅

## Directory Structure

```
adframe2.0/
├── run.py                  # Main CLI entrypoint
├── qwen_reasoner.py        # Qwen2.5-VL placement reasoner + judge
├── image_generator.py      # FLUX.1-dev / SDXL single-frame generator
├── wan_generator.py        # Wan 2.1 VACE video generator
├── experiment_logger.py    # Experiment directory management
├── video_utils.py          # Video I/O, sampling, compositing, export
├── setup.sh                # Environment bootstrap (extends v1.0 setup)
├── README.md               # This file
└── experiments/
    └── experiment_001/
        ├── prompt.txt
        ├── reasoning.json
        ├── judge.json
        ├── video_judge.json
        ├── output_frame.png
        ├── output_video.mp4
        ├── metrics.json
        └── logs.txt
```

## Quick Start (RunPod L40S)

```bash
cd /workspace/adframe
bash adframe2.0/setup.sh

export HF_HOME=/tmp/huggingface
export HF_HUB_CACHE=/tmp/huggingface/hub
export TMPDIR=/tmp

python adframe2.0/run.py \
    --video /workspace/demo.mp4 \
    --brand /workspace/demo_brand.jpg \
    --image_backend flux \
    --wan_model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
    --image_threshold 7.0 \
    --video_threshold 7.0 \
    --max_image_iterations 3 \
    --max_video_iterations 2 \
    --seed 42
```

## Milestones

- [x] **M1** — Codebase audit (`docs/adframe2_audit.md`)
- [x] **M2** — adframe2.0 skeleton created (this directory)
- [ ] **M3** — Qwen reasoning pipeline working (test on demo.mp4)
- [ ] **M4** — Image generation feedback loop working
- [ ] **M5** — Wan VACE integration working
- [ ] **M6** — Automatic self-improving pipeline (end-to-end)
- [ ] **M7** — Demo: Coca-Cola can placement in 36-second podcast video
