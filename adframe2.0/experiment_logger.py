"""
AdFrame 2.0 — Experiment Logger
==================================
Manages the experiments/ directory structure.
Every experiment is immutable and reproducible.

Directory structure:
  adframe2.0/experiments/
    experiment_001/
      prompt.txt          <- final approved inpainting prompt
      reasoning.json      <- Qwen placement reasoning output
      judge.json          <- Qwen frame judge history (all iterations)
      video_judge.json    <- Qwen video judge output
      output_frame.png    <- best generated single frame
      output_video.mp4    <- final composited output video
      metrics.json        <- consolidated scores + metadata
      logs.txt            <- full text log of the run

Experiments are NEVER overwritten. Each run gets a new auto-numbered directory.
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger("adframe2.experiment_logger")


class ExperimentLogger:
    """Manages creation and writing of a single experiment directory."""

    EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

    def __init__(self, experiment_id: Optional[str] = None):
        """
        Creates a new numbered experiment directory.
        experiment_id: optional override (e.g. 'experiment_001').
                       If None, auto-increments from existing dirs.
        """
        self.EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

        if experiment_id is None:
            experiment_id = self._next_experiment_id()

        self.experiment_id = experiment_id
        self.experiment_dir = self.EXPERIMENTS_ROOT / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._log_lines = []
        self._start_time = time.time()

        self.log(f"=== Experiment {experiment_id} started at {datetime.now(timezone.utc).isoformat()} ===")
        logger.info(f"[ExperimentLogger] Created experiment: {self.experiment_dir}")

    def _next_experiment_id(self) -> str:
        existing = sorted(
            d for d in self.EXPERIMENTS_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("experiment_")
        )
        if not existing:
            return "experiment_001"
        last = existing[-1].name
        try:
            num = int(last.split("_")[-1]) + 1
        except ValueError:
            num = len(existing) + 1
        return f"experiment_{num:03d}"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, message: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {message}"
        self._log_lines.append(line)
        logger.info(message)

    def flush_logs(self):
        log_path = self.experiment_dir / "logs.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._log_lines))

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    def save_reasoning(self, reasoning: dict):
        self._save_json(reasoning, "reasoning.json")
        self.log(f"Reasoning saved. Surface: {reasoning.get('surface_type', '?')}, "
                 f"Frame: {reasoning.get('best_frame_index', '?')}")

    def save_judge_history(self, judge_history: list):
        self._save_json(judge_history, "judge.json")
        self.log(f"Judge history saved. Iterations: {len(judge_history)}")

    def save_video_judge(self, video_judge: dict):
        self._save_json(video_judge, "video_judge.json")
        self.log(f"Video judge saved. Score: {video_judge.get('overall_score', '?')}, "
                 f"Accept: {video_judge.get('accept', '?')}")

    def save_prompt(self, prompt: str, negative_prompt: str = ""):
        path = self.experiment_dir / "prompt.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== POSITIVE PROMPT ===\n")
            f.write(prompt + "\n\n")
            if negative_prompt:
                f.write("=== NEGATIVE PROMPT ===\n")
                f.write(negative_prompt + "\n")
        self.log(f"Prompt saved: {prompt[:80]}...")

    def save_output_frame(self, frame: Image.Image, iteration: int = 0):
        path = self.experiment_dir / f"output_frame_iter{iteration:02d}.png"
        frame.save(path)
        # Always symlink/copy the best to output_frame.png
        best_path = self.experiment_dir / "output_frame.png"
        frame.save(best_path)
        self.log(f"Output frame saved: {path.name}")
        return path

    def save_output_video(self, src_path: str):
        dst = self.experiment_dir / "output_video.mp4"
        shutil.copy2(src_path, dst)
        self.log(f"Output video saved: {dst}")
        return dst

    def save_metrics(self, metrics: dict):
        elapsed = time.time() - self._start_time
        metrics["elapsed_seconds"] = round(elapsed, 1)
        metrics["experiment_id"] = self.experiment_id
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._save_json(metrics, "metrics.json")
        self.log(f"Metrics saved. Elapsed: {elapsed:.1f}s")

    def _save_json(self, data, filename: str):
        path = self.experiment_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def print_summary(self):
        elapsed = time.time() - self._start_time
        lines = [
            f"\n{'='*60}",
            f"  Experiment: {self.experiment_id}",
            f"  Directory:  {self.experiment_dir}",
            f"  Elapsed:    {elapsed:.1f}s",
            f"  Files:",
        ]
        for f in sorted(self.experiment_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            lines.append(f"    {f.name:30s} {size_kb:8.1f} KB")
        lines.append("="*60)
        summary = "\n".join(lines)
        self.log(summary)
        print(summary)
        self.flush_logs()
        return summary
