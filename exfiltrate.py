import os
import json
import base64
from pathlib import Path
import subprocess

exp_dir = Path("/workspace/adframe/adframe2/experiments/experiment_004")
out = {}

def cmd(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

out["tree"] = cmd(f"find {exp_dir} -maxdepth 4 | sort")
out["ls"] = cmd(f"ls -lhR {exp_dir}")

def read_txt(p, lines=None):
    try:
        with open(exp_dir / p, "r", encoding="utf-8") as f:
            if lines:
                return "".join([f.readline() for _ in range(lines)])
            return f.read()
    except Exception as e:
        return str(e)

def read_tail(p, n=200):
    try:
        with open(exp_dir / p, "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-n:])
    except Exception as e:
        return str(e)

out["report"] = read_txt("REPORT.md", 100)
out["research_summary"] = read_txt("research_summary.md", 100)
out["reasoning_001"] = read_txt("reasoning/iteration_001.json")
out["reasoning_010"] = read_txt("reasoning/iteration_010.json")
out["judge_001"] = read_txt("judge/iteration_001.json")
out["judge_010"] = read_txt("judge/iteration_010.json")
out["prompt_history"] = read_txt("prompt_history.md")
out["performance"] = read_txt("performance.json")
out["system"] = read_txt("system.json")
out["models"] = read_txt("models.json")
out["video_metrics"] = read_txt("video_metrics.json")
out["terminal_log"] = read_tail("terminal.log", 200)

def b64(p):
    try:
        with open(exp_dir / p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return ""

out["candidate_001"] = b64("generation/candidate_001.png")
out["candidate_005"] = b64("generation/candidate_005.png")
out["candidate_010"] = b64("generation/candidate_010.png")
out["grid"] = b64("comparison/frame_grid.png")

out["output_video"] = cmd(f"ls -lh {exp_dir}/output.mp4")

with open("/workspace/adframe/verification.json", "w") as f:
    json.dump(out, f)
