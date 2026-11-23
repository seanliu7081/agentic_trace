#!/usr/bin/env python
"""Head (agentview) camera of a run's scene, so videos can draw world points (planned paths) on the head view.

    python scripts/head_camera.py RUN_DIR [--gpu N]   -> RUN_DIR/rec/head_camera.json {K, T_wc, size, task, episode}

Evaluator-side: loads the episode's simulator scene exactly as run_episode.py does and reads the fixed head camera
(SimRobot.capture: K with pixel centres at i+0.5, T_wc camera->world with OpenCV axes, image row 0 = top, the same
upright orientation as rec/cam_agent.mp4). Nothing is stepped; the run's own data is not touched.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_MINOR = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}   # nvidia-smi index -> /dev/nvidia minor (EGL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run_dir)
    a = json.loads((run / "args.json").read_text())
    minor = GPU_MINOR[args.gpu]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(minor)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu},{minor}" if args.gpu != minor else str(args.gpu)
    os.environ.setdefault("LIBERO_ROOT", str(ROOT / "third_party" / "LIBERO"))
    sys.path.insert(0, "/workspace/Libero10-Collision")
    sys.path.insert(0, str(ROOT / "src"))
    from libero10_collision import load_episode
    from atrace.sim_iface import SimRobot, HEAD_CAMERA

    with load_episode(a["task"], a["episode"]) as (env, obs, manifest):
        fr = SimRobot(env).capture(HEAD_CAMERA)
    out = {"task": a["task"], "episode": a["episode"], "camera": HEAD_CAMERA, "size": int(fr.rgb.shape[0]),
           "K": fr.K.tolist(), "T_wc": fr.T_wc.tolist()}
    (run / "rec").mkdir(exist_ok=True)
    (run / "rec" / "head_camera.json").write_text(json.dumps(out, indent=1))
    print("head_camera:", run / "rec" / "head_camera.json")


if __name__ == "__main__":
    main()
