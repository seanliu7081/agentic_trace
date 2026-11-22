"""SAM 3.1 text-prompt instance segmentation (in-process, CUDA) with per-image caching."""
from __future__ import annotations

import hashlib
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "sam3.1_multiplex.pt"


def _construct_detector():
    from importlib.resources import files
    from sam3 import model_builder as builder
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
    neck = builder._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=False)
    backbone = SAM3VLBackboneTri(scalp=0, visual=neck, text=builder._create_text_encoder(bpe_path))
    return Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone,
        transformer=builder._create_sam3_transformer(use_fa3=False),
        segmentation_head=builder._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=builder._create_geometry_encoder(),
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=builder._create_dot_product_scoring(),
        supervise_joint_box_scores=True, is_multiplex=True,
    )


class Sam3Segmenter:
    def __init__(self, checkpoint=DEFAULT_CKPT, device="cuda:0", threshold=0.2):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.threshold = threshold
        self._proc = None
        self._torch = None
        self._state_key = None
        self._state = None
        self._results = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.seconds = 0.0

    def load(self):
        if self._proc is not None:
            return
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        det = _construct_detector()
        ck = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        state = ck.get("model", ck)
        sub = {}
        for k, v in state.items():
            for prefix in ("detector.", "sam3_model."):
                if k.startswith(prefix):
                    sub[k[len(prefix):]] = v
                    break
        det.load_state_dict(sub, strict=True)
        det = det.to(device=self.device).eval()
        self._proc = Sam3Processor(det, device=self.device, confidence_threshold=self.threshold)
        self._torch = torch

    def detect(self, rgb, prompt):
        """-> list of (mask (H,W) bool, score) above threshold, sorted by score."""
        with self._lock:
            self.load()
            from PIL import Image
            t0 = time.time()
            key = hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            rk = (key, prompt)
            if rk in self._results:
                return self._results[rk]
            torch = self._torch
            ctx = torch.inference_mode()
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in self.device else nullcontext()
            with ctx, amp:
                if key != self._state_key:
                    self._state = self._proc.set_image(Image.fromarray(rgb))
                    self._state_key = key
                self._proc.reset_all_prompts(self._state)
                out = self._proc.set_text_prompt(state=self._state, prompt=prompt)
                scores = out["scores"]
                if hasattr(scores, "float"):
                    scores = scores.float()
                masks = out["masks"]
                masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
                scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
            if masks.ndim == 4:
                masks = masks[:, 0]
            res = [(masks[i].astype(bool), float(scores[i])) for i in range(len(scores))]
            res.sort(key=lambda t: -t[1])
            if len(self._results) > 400:
                self._results.clear()
            self._results[rk] = res
            self.calls += 1
            self.seconds += time.time() - t0
            return res
