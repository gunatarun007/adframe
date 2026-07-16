"""
AdFrame 2.0 — Video I/O Utilities
====================================
Shared video loading, frame sampling, and video export utilities.
Reuses patterns from adframe 1.0 main.py video loader,
but adds streaming mode to avoid loading all frames into RAM.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("adframe2.video_utils")


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_frames_from_video(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Load all (or up to max_frames) frames from a video into RAM.
    Returns:
        (frames_bgr, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    logger.info(f"[VideoUtils] Loaded {len(frames)} frames from {video_path} @ {fps:.2f} FPS")
    return frames, fps


def sample_frames_from_video(
    video_path: str,
    num_samples: int = 8,
) -> Tuple[List[Image.Image], List[int], float]:
    """
    Uniformly sample num_samples frames from a video.
    Returns:
        (pil_frames_rgb, frame_indices, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    indices = np.linspace(0, max(0, total - 1), num_samples, dtype=int).tolist()
    frames = []
    frame_indices = []

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            frame_indices.append(target_idx)

    cap.release()
    logger.info(f"[VideoUtils] Sampled {len(frames)} frames from {video_path}")
    return frames, frame_indices, fps


def get_frame_at_index(
    video_path: str,
    frame_index: int,
) -> Optional[Image.Image]:
    """
    Extract a single frame at a specific index.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error(f"[VideoUtils] Failed to read frame {frame_index} from {video_path}")
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def stream_frames(
    video_path: str,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields (frame_idx, bgr_frame) for streaming processing.
    Avoids loading all frames into RAM.
    """
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1
    cap.release()


def get_video_metadata(video_path: str) -> dict:
    """Return basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": None,
    }
    meta["duration_seconds"] = (
        meta["total_frames"] / meta["fps"] if meta["fps"] > 0 else None
    )
    cap.release()
    logger.info(f"[VideoUtils] Metadata for {Path(video_path).name}: {meta}")
    return meta


# ---------------------------------------------------------------------------
# Frame compositing utilities
# ---------------------------------------------------------------------------
def paste_generated_frames_into_video(
    original_frames_bgr: List[np.ndarray],
    generated_pil_frames: List[Image.Image],
    placement_bbox: list,            # [x1, y1, x2, y2] normalized
    blend_feather: int = 30,
) -> List[np.ndarray]:
    """
    Paste generated frames back into full-resolution original frames.
    Uses Gaussian-blurred bbox mask for seamless edge blending.
    Loops generated frames modulo if shorter than original.
    """
    from adframe2_0.wan_generator import build_soft_mask_from_bbox

    if not original_frames_bgr:
        return []

    h_orig, w_orig = original_frames_bgr[0].shape[:2]
    num_gen = len(generated_pil_frames)

    # Build full-resolution blend mask from bbox
    x1, y1, x2, y2 = placement_bbox
    # Convert soft PIL mask to 3-channel float
    soft_mask_pil = build_soft_mask_from_bbox(w_orig, h_orig, placement_bbox, feather_radius=blend_feather)
    soft_mask = np.array(soft_mask_pil, dtype=np.float32) / 255.0
    soft_mask_3d = soft_mask[:, :, np.newaxis]  # (H, W, 1)

    output_frames = []
    for i, orig_bgr in enumerate(original_frames_bgr):
        gen_pil = generated_pil_frames[i % num_gen]
        gen_rgb = gen_pil.resize((w_orig, h_orig), Image.LANCZOS)
        gen_bgr = cv2.cvtColor(np.array(gen_rgb), cv2.COLOR_RGB2BGR)

        # Blend: use generated only in the placement region
        blended = orig_bgr.astype(np.float32) * (1.0 - soft_mask_3d) + gen_bgr.astype(np.float32) * soft_mask_3d
        output_frames.append(blended.astype(np.uint8))

    return output_frames


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------
def write_video(
    frames_bgr: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
    use_ffmpeg: bool = True,
) -> str:
    """
    Write frames to a video file.
    Prefers ffmpeg pipe for H.264 output (better browser compatibility).
    Falls back to cv2.VideoWriter with mp4v.
    """
    output_path = str(output_path)
    if not frames_bgr:
        raise ValueError("No frames to write.")

    h, w = frames_bgr[0].shape[:2]

    if use_ffmpeg and _ffmpeg_available():
        _write_with_ffmpeg(frames_bgr, output_path, fps, w, h)
    else:
        _write_with_cv2(frames_bgr, output_path, fps, w, h)

    logger.info(f"[VideoUtils] Video saved: {output_path} ({len(frames_bgr)} frames @ {fps:.2f} FPS)")
    return output_path


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _write_with_ffmpeg(frames_bgr, output_path, fps, w, h):
    """Pipe raw BGR frames to ffmpeg for H.264 encoding."""
    import io
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames_bgr:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def _write_with_cv2(frames_bgr, output_path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames_bgr:
        out.write(frame)
    out.release()


def extract_keyframes(
    video_path: str,
    num_keyframes: int = 6,
) -> List[Image.Image]:
    """Extract uniformly spaced keyframes for video judge evaluation."""
    frames, _, _ = sample_frames_from_video(video_path, num_samples=num_keyframes)
    return frames
