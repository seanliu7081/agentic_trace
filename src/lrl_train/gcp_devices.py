"""Physical GPU selection for the contact learner and render-only processes."""
import os
import subprocess


def parse_gpus(value):
    indices = tuple(int(i) for i in (value.split(",") if isinstance(value, str) else value))
    if not indices or len(set(indices)) != len(indices) or any(i not in (0, 1, 2) for i in indices):
        raise ValueError("Select distinct physical GPUs from 0,1,2")
    return indices


def configure_egl_worker(gpu):
    (physical,) = parse_gpus([gpu])
    # A render worker needs EGL only. In particular an EGL minor of 3 on
    # this host denotes physical GPU 0, and must never enter CUDA's allowlist.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    from .render import egl_device
    egl_device(physical)


def configure_learner(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None
    if device != "cuda:0":
        raise ValueError("The GCP learner must use physical GPU 0 (cuda:0)")
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True, timeout=10).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Could not identify physical GPU 0")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    # The parent does no EGL rendering. robosuite validates inherited EGL
    # ordinals against CUDA visibility at import, before workers are forked.
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    return uuid
