"""Evaluator-side recording for the post-episode videos (task dashboard + 3D obstacle map).

Nothing here is read by the policy: the harness only *calls* the recorder (guarded hooks) and the runner
feeds it every simulator step. Everything lands in <run>/rec/:

  meta.json          task, episode, instruction, operator note, build, stride, fps, map geometry, legend data
  cam_agent.mp4      agent (head) view, upright RGB 512x512, one frame per recorded step   (mp4v intermediate)
  cam_wrist.mp4      wrist view, same indexing
  telemetry.jsonl    one record per recorded frame:
                       frame, step, t (s since harness start, same clock as events.jsonl), eef [m], yaw [rad],
                       finger [m pad gap], force [N, wrist], grip (command), q (9 joint positions), held (oid|null)
  robot_points.npz   per recorded frame k: robot_k (N_r,3) float16 world collision samples, payload_k (N_p,3) or
                     absent; plus 'frames' (int array of frame indices that have payload)
  maps/map_NNNN.npz  S2 scene-map snapshots: state int8 (-1 unknown / 0 free / 1 occupied), label int16 (object id,
                     0 = unlabelled), lo (3,), res, table_z, visual_table_z, step, t, reason
  maps/index.jsonl   one line per snapshot: file, step, t, reason, objects {oid: {name, role, held, part_of}}
  plans.jsonl        every planned move: step, t, tag, how, eef ([[x,y,z], ...] m, densified), n_waypoints

Recorded frame k is taken after simulator step `step` when step % stride == 0 (stride 2 = 10 Hz sim time).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Recorder:
    def __init__(self, run_dir, stride=2, size=512, meta=None):
        self.dir = Path(run_dir) / "rec"
        (self.dir / "maps").mkdir(parents=True, exist_ok=True)
        self.stride = int(stride)
        self.size = int(size)
        self.meta = dict(meta or {})
        self.h = None
        self.t0 = time.time()
        self.k = 0
        self._tel = open(self.dir / "telemetry.jsonl", "w")
        self._plans = open(self.dir / "plans.jsonl", "w")
        self._maps = open(self.dir / "maps" / "index.jsonl", "w")
        four = cv2.VideoWriter_fourcc(*"mp4v")
        self._vw = {n: cv2.VideoWriter(str(self.dir / f"cam_{n}.mp4"), four, 10, (self.size, self.size))
                    for n in ("agent", "wrist")}
        self._robot, self._payload = {}, {}
        self._n_maps = 0
        self._lock = threading.Lock()
        self._threads = []
        self.closed = False

    # ------------------------------------------------------------ wiring
    def attach(self, harness):
        """Called once the harness exists: shares its clock and gives access to its (read-only) state."""
        self.h = harness
        self.t0 = harness.t0
        harness.recorder = self

    def _t(self):
        return round(time.time() - self.t0, 3)

    # ------------------------------------------------------------ per simulator step (runner)
    def on_step(self, robot, ob):
        if self.closed or robot.steps % self.stride != 0:
            return
        for n, key in (("agent", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
            img = ob.get(key)
            if img is None:
                img = np.zeros((self.size, self.size, 3), np.uint8)
            img = np.ascontiguousarray(img[::-1])                      # upright
            if img.shape[0] != self.size:
                img = cv2.resize(img, (self.size, self.size))
            self._vw[n].write(img[:, :, ::-1])
        R, p = robot.eef_pose()
        F, _ = robot.wrench()
        q = robot.q()
        rec = {"frame": self.k, "step": int(robot.steps), "t": self._t(), "eef": np.round(p, 4).tolist(),
               "yaw": round(float(np.arctan2(R[1, 0], R[0, 0])), 4), "finger": round(float(robot.finger_opening()), 5),
               "force": np.round(F, 2).tolist(), "grip": float(getattr(robot, "grip_cmd", 0.0) or 0.0),
               "q": np.round(q[:9], 4).tolist(), "held": None}
        h = self.h
        if h is not None and hasattr(h, "checker"):
            try:
                W, att = h.checker.world_points(np.asarray(q)[None, :7], finger=max(0.0, float(q[7] - q[8]) / 2))
                self._robot[self.k] = W[0].astype(np.float16)
                if att is not None:
                    pts = att[0]
                    if len(pts) > 900:
                        pts = pts[np.linspace(0, len(pts) - 1, 900).astype(int)]
                    self._payload[self.k] = pts.astype(np.float16)
                rec["held"] = h.held
            except Exception:
                pass
        self._tel.write(json.dumps(rec) + "\n")
        self.k += 1

    # ------------------------------------------------------------ harness hooks
    def on_map(self, harness, reason):
        """Snapshot of S2's voxel map (copied now, compressed to disk in the background)."""
        if self.closed:
            return
        sc = harness.scene
        objs = {int(oid): {"name": m.name, "role": m.role, "held": bool(m.held), "part_of": m.part_of,
                           "virtual": bool(getattr(m, "virtual", False))}
                for oid, m in harness.objects.items()}
        idx = self._n_maps
        self._n_maps += 1
        fname = f"map_{idx:04d}.npz"
        line = {"file": fname, "step": int(harness.rb.steps), "t": self._t(), "reason": reason, "objects": objs}
        data = dict(state=sc.state.copy(), label=sc.label.copy(), lo=np.asarray(sc.lo, float), res=float(sc.res),
                    table_z=float(sc.table_z if sc.table_z is not None else np.nan),
                    visual_table_z=float(getattr(sc, "visual_table_z", None) or np.nan),
                    step=int(harness.rb.steps), t=line["t"])
        path = self.dir / "maps" / fname

        def save():
            np.savez_compressed(path, **data)
        th = threading.Thread(target=save, daemon=True)
        th.start()
        self._threads.append(th)
        with self._lock:
            self._maps.write(json.dumps(line) + "\n")
            self._maps.flush()

    def on_plan(self, harness, path, finger, tag, how):
        if self.closed:
            return
        try:
            from atrace.motion import densify
            Q = densify(np.asarray(path, float)[:, :7], 0.02)
            _, ps = harness.rm.site_pose(harness.rm.full_q(Q, finger))
            eef = np.round(ps, 4).tolist()
        except Exception:
            eef = []
        self._plans.write(json.dumps({"step": int(harness.rb.steps), "t": self._t(), "tag": tag, "how": how,
                                      "n_waypoints": int(len(path)), "eef": eef}) + "\n")
        self._plans.flush()

    # ------------------------------------------------------------ end
    def close(self):
        if self.closed:
            return
        self.closed = True
        for vw in self._vw.values():
            vw.release()
        for f in (self._tel, self._plans, self._maps):
            f.close()
        for th in self._threads:
            th.join(timeout=60)
        arrays = {}
        for k, v in self._robot.items():
            arrays[f"robot_{k}"] = v
        for k, v in self._payload.items():
            arrays[f"payload_{k}"] = v
        arrays["payload_frames"] = np.array(sorted(self._payload), int)
        arrays["n_frames"] = np.array(self.k)
        np.savez_compressed(self.dir / "robot_points.npz", **arrays)
        meta = dict(self.meta)
        meta.update(stride=self.stride, fps=10, n_frames=self.k, n_maps=self._n_maps, image_size=self.size)
        h = self.h
        if h is not None and hasattr(h, "scene"):
            sc = h.scene
            meta.update(map_lo=np.asarray(sc.lo, float).tolist(), map_shape=list(sc.shape), map_res=float(sc.res),
                        table_z=float(h.table_z), visual_table_z=float(getattr(h, "visual_table_z", h.table_z)),
                        robot_base=np.asarray(h.rm.root_p, float).tolist())
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))
