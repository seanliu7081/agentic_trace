"""The contact stage as data (CONTACT_STAGE_PLAN.md 4.1-4.5): one guarded leg that carries its expected contact, one
reaction table, the release gate, the hand-empty test, re-localisation of a known shape, and the put-down ("unified"
placement) built from them. Nothing here knows what the object, the destination or the task is.

A leg is judged against its expectation:
  window (lo, hi)  contact is expected after lo..hi metres of travel (a set-down: the object meets its support);
  window None      contact is an error (an approach to a hand-off, a drop's way down, a departure).
Verdicts: as_expected | early (contact before the window, or where none was expected) | none (no contact by the end of
the window) | no_line (no arm posture along the line) | dropped | tracking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from traceplan.contract import EndCandidate

from . import manip
from .geometry import rot_z
from .perception import mask_points, voxel_down
from .sim_iface import HEAD_CAMERA, WRIST_CAMERA
from .snapshot import entity_names

DOWN = np.array([0.0, 0.0, -1.0])
LOOK = {"top_k": 5, "side_k": 16}  # look-ahead: grasps judged per family (the hand fitted at the pick first)
REST = {
    "cell": 0.005,          # m: top-height grid of the observed surfaces
    "radius": (0.06, 0.10),  # m: search radius around the pointed spot (the second only when the first has none)
    "spread": 0.015,         # m: a rest whose support varies more than this under the footprint stands on something
    "observed": 0.6,         # share of footprint cells with an observed surface
    "on_share": 0.6,         # "on": share of the object's footprint over the destination's own surface
    "above_table": 0.004,    # m: a surface this far above the table stands out from it (a plate, a stove top)
    "flat_dest": 0.008,      # m: a destination whose top is less than this above the table: its labels decide "on"
    "support_close": 0.015,  # m: "on": the destination's support region is closed this much (gaps in its cloud)
    "support_min_pts": 30,   # "on": fewer support points than this: its labels decide
    "beside_tol": 0.012,     # m: a rest beside a reference is on the table
    "raise": 0.004,          # m: the object is checked this far above its rest
    "palm": 0.004,           # m: palm clearance at release / on the way down
    "finger": 0.001,         # m: finger clearance at release
    "sweep": (0.02, 0.26, 0.02),  # m: vertical entry checked from the release pose up to the hand-off region
    "fall": 0.12,           # m: a drop lets go at most this high above the rest
    "in_drop": 0.045,       # m: "in", set-down stage: a bounded drop from at most this high when only the hand blocks
    "drop_wall_gap": 0.01,  # m: ... and the footprint is at least this far inside every wall of the container
    "tol": (0.025, 0.015),   # m: expected-contact window around the support: before / after the planned rest
    "max_3d": 36,           # costly 3D checks per call (spread over the hand's turns and over the area)
    "wall": 0.015,          # m: a surface this far above the object's bottom, next to it, is a wall ("in")
    "walls": 6,             # of 8 directions around the footprint that must have a wall ("in")
    "ring": (0.003, 0.04),  # m: walls are looked for from this far outside the footprint ...
    "ray": 0.30,            # m: ... out to this far from its middle
    "dest_pad": 0.015,      # m: the destination's footprint hull is widened this much
    "cap_pad": 0.03,        # m: surfaces higher than the pointed level + the object's height + this are not supports
}
ENTRY = {
    "lift": (0.02, 0.035),        # m: a slide moves in this far above the rest, then sets the object down
    "out_max": 0.24,              # m: longest slide from the rest out to where hand and object can rise
    "step": 0.02,                 # m: sweep step along a slide
    "rise": (0.03, 0.07, 0.11),   # m: from the slide's outer point hand and object must rise this far clear
    "oblique": np.deg2rad(45.0),  # elevation of a straight oblique entry
    "n_dirs": 8,                  # horizontal directions considered (System 1's hint and the arm's side first) ...
    "max_dirs": 4,                # ... of which at most this many are swept
    "arm": 0.006,                 # m: arm clearance along the entry legs (observed geometry)
}


@dataclass
class Outcome:
    cause: str              # done | contact | force | tracking | dropped | no_line
    travel: float           # m moved along the leg
    short: float            # m left to the leg's goal
    force: list | None      # N, world, arm on hand, change at the stop
    opening: float          # finger opening after the leg (m)
    nearest: dict | None    # nearest mapped thing to the touching bodies at the stop
    steps: int = 0
    verdict: str | None = None

    @property
    def touched(self):
        return self.cause in ("contact", "force")


def judge(out, window):
    """The leg's verdict against its expectation (module docstring)."""
    if out.cause in ("no_line", "dropped", "tracking"):
        return out.cause
    if window is None:
        return "early" if out.touched else "as_expected"
    lo, hi = window
    if out.touched:
        return "early" if out.travel < lo else "as_expected"
    return "none"


class ContactMixin:
    """Methods of the Harness (self.rb, self.checker, self.scene, self.tm, ...) for the shared contact stage."""

    # ------------------------------------------------------------------------------------------- one guarded leg
    def guarded_leg(self, R, p, speed, window=None, grip=None, finger=None, f_lim=8.0, tag="leg"):
        """A guarded straight leg to (R, p): the contact detector always on, plus the force along the motion above
        f_lim (one convention for every direction: arm on hand, projected on the motion). Returns an Outcome judged
        against `window` (module docstring)."""
        finger = self._finger() if finger is None else finger
        R0, p0 = self.rb.eef_pose()
        d = np.asarray(p, float) - p0
        L = float(np.linalg.norm(d))
        u = d / L if L > 1e-9 else np.zeros(3)

        def rule(robot, F, F0, k):
            return "force" if float((F - F0) @ u) > f_lim and k > 2 else None
        res = self.move_line(np.asarray(R, float), np.asarray(p, float), grip=grip, speed=speed, finger=finger,
                             guard=self._guard("contact", rule), force_limit=1e9)
        _, p1 = self.rb.eef_pose()
        travel = float((p1 - p0) @ u) if L > 1e-9 else 0.0
        out = Outcome("no_line" if res is None else res.status, travel, max(0.0, L - travel),
                      None if res is None else res.info.get("force"), float(self.rb.finger_opening()), None,
                      0 if res is None else res.steps)
        if out.touched:
            out.nearest = self.nearest_obstacle(self._touch_points(), ())
        out.verdict = judge(out, window)
        self.log("leg", tag=tag, cause=out.cause, travel=round(travel, 4), short=round(out.short, 4),
                 window=None if window is None else [round(window[0], 4), round(window[1], 4)], verdict=out.verdict,
                 force=out.force, nearest=out.nearest)
        return out

    def _touch_points(self):
        """Hand and held-object points at the current configuration (what can touch)."""
        W, att = self.checker.world_points(self.rb.q()[None, :7], self._finger())
        pts = W[0][self.checker.pg >= 1]
        return pts if att is None else np.concatenate([pts, att[0]])

    def mark_contact(self, u):
        """An unexpected contact becomes geometry: a small observed patch at the leading points of hand and held object
        along the motion u, so the next plan sees what was touched (no model call)."""
        P = self._touch_points()
        s = P @ np.asarray(u, float)
        lead = P[s >= np.percentile(s, 95)]
        c = lead.mean(0) + 0.003 * np.asarray(u, float)
        a = np.cross(u, [1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.cross(u, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(u, a)
        g = np.stack(np.meshgrid(np.linspace(-0.015, 0.015, 7), np.linspace(-0.015, 0.015, 7)), -1).reshape(-1, 2)
        patch = c + g[:, :1] * a + g[:, 1:] * b
        self.scene.add_surface(patch)
        self._forget_free(patch)
        self.log("contact_marked", at=c)
        return c

    # ------------------------------------------------------------------------------------------- release
    def release_width(self, R, p):
        """Finger joint to open to at a release: the widest opening whose finger spheres stay clear of the observed
        scene (the held object excluded), down to a little more than the held width."""
        held = (self.held,) if self.held is not None else ()
        w0 = float(self.held_width or 0.0)
        for f in (manip.FULL_JOINT, 0.032, 0.026, max(0.004, w0 / 2 + 0.008)):
            L, grp, rad = self._hand_local(f)
            Wf = np.asarray(p, float) + L[grp == 2] @ np.asarray(R, float).T
            c = self.checker.point_clearance(Wf, held, unknown=False) - rad[grp == 2]
            if float(c.min()) >= REST["finger"]:
                return f
        return max(0.004, w0 / 2 + 0.004)

    def hand_empty(self, model_site, rest_pts):
        """After a release and the departure: did the object stay where it was let go, or does it hang on the hand?
        Two hypotheses for the same known shape: still held (model_site posed with the hand now) and left behind
        (rest_pts, where it was released). Depth points (head and wrist views) that only the first explains mean it
        hangs on; a departure shorter than the object leaves the two overlapping, and then only the parts that differ
        count."""
        Rs, ps = self.rb.eef_pose()
        A = ps + np.asarray(model_site, float) @ Rs.T
        B = np.asarray(rest_pts, float)
        ta, tb = cKDTree(voxel_down(A, 0.004)), cKDTree(voxel_down(B, 0.004))
        n_a = n_b = 0
        frames = [self.capture(HEAD_CAMERA), self.capture(WRIST_CAMERA)]
        self.frames += frames
        for fr in frames:
            P, _ = fr.backproject(mask=~ndimage.binary_dilation(fr.robot_mask, iterations=2), stride=2)
            da, _ = ta.query(P, distance_upper_bound=0.03)
            db, _ = tb.query(P, distance_upper_bound=0.03)
            n_a += int(((da < 0.006) & (db > 0.012)).sum())
            n_b += int(((db < 0.006) & (da > 0.012)).sum())
        held = n_a >= 40 and n_a > 2 * n_b
        self.log("hand_empty_check", only_if_held=n_a, only_if_released=n_b, still_held=held)
        return not held, frames

    # ------------------------------------------------------------------------------------------- one object, one model
    def _without_others(self, P, oid):
        """Points that are not on another tracked object's model (a destination never absorbs what stands on it)."""
        P = np.asarray(P, float).reshape(-1, 3)
        others = [m.points for o, m in self.objects.items()
                  if o != oid and not m.held and len(m.points) and not getattr(m, "virtual", False)]
        if not len(P) or not others:
            return P
        d, _ = cKDTree(np.concatenate(others)).query(P, distance_upper_bound=0.007)
        return P[~np.isfinite(d)]

    def relocalize(self, obj, frames, iters=25):
        """The object's known shape registered to fresh depth around its predicted place (rigid ICP), instead of a new
        segmentation: its identity and extent survive a release or a failure. Returns the fit or None."""
        M = voxel_down(obj.points, 0.004)
        if len(M) < 30:
            return None
        lo, hi = M.min(0) - 0.07, M.max(0) + 0.07
        S = []
        for fr in frames:
            P, _ = fr.backproject(mask=~ndimage.binary_dilation(fr.robot_mask, iterations=3), stride=1)
            sel = np.all((P >= lo) & (P <= hi), axis=1) & (P[:, 2] > self.table_z + 0.004)
            S.append(P[sel])
        S = voxel_down(np.concatenate(S), 0.004) if S else np.zeros((0, 3))
        S = self._without_others(S, obj.oid)
        if len(S) < 30:
            return None
        tree = cKDTree(S)
        X, R, t = M.copy(), np.eye(3), np.zeros(3)
        for it in range(iters):
            d, i = tree.query(X, distance_upper_bound=0.04 if it < 4 else 0.02)
            ok = np.isfinite(d)
            if ok.sum() < 20:
                return None
            A, B = X[ok], S[i[ok]]
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
            dR = Vt.T @ D @ U.T
            dt = cb - dR @ ca
            X = X @ dR.T + dt
            R, t = dR @ R, dR @ t + dt
            if np.linalg.norm(dt) < 1e-4 and np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)) < 1e-3:
                break
        d, _ = tree.query(X)
        inl = float(np.mean(d < 0.006))
        turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        fit = {"inliers": round(inl, 3), "rms_mm": round(1e3 * float(np.sqrt(np.mean(np.minimum(d, 0.02) ** 2))), 1),
               "shift_m": round(float(np.linalg.norm(t)), 4), "turn_deg": round(turn, 1)}
        self.log("relocalized", object=obj.name, accepted=inl >= 0.5, **fit)
        if inl < 0.5:
            return None
        new = obj.points @ R.T + t
        obj.set_points(new)
        obj.support_z = float(new[:, 2].min()) - 0.002
        obj._finalize()
        self._forget_free(new)
        return {**fit, "bottom_above_table": round(float(new[:, 2].min()) - self.table_z, 4),
                "center_xy": np.round(obj.footprint_center(), 4).tolist()}

    # ------------------------------------------------------------------------------------------- rest poses
    def _surface_grid(self, lo, hi, exclude, z_cap=None):
        """Per REST cell over an xy box: the top height of the observed surfaces (surface cloud and object models, not
        the excluded ones), its owner, an observed flag, and the same top height and flag counting only surfaces at or
        below z_cap (the level the object is put to: a roof, a shelf or a handle above it is not its support).
        Returns (H, Lb, obs, H_all, obs_all, Lb_all); without z_cap the capped grid is the full one."""
        c = REST["cell"]
        n = np.maximum(np.ceil((hi - lo) / c).astype(int), 1)
        near = self.checker.near
        near._labels(self.scene)
        clouds = [(near.cloud, near.lab)]
        for oid, m in self.objects.items():
            if m.held or getattr(m, "virtual", False) or oid in exclude or not len(m.points):
                continue
            clouds.append((m.points, np.full(len(m.points), oid)))

        def grid(cap):
            H = np.full(n, -np.inf)
            Lb = np.zeros(n, int)
            for P, lab in clouds:
                if not len(P):
                    continue
                keep = ~np.isin(lab, list(exclude)) if exclude else np.ones(len(P), bool)
                if cap is not None:
                    keep &= P[:, 2] <= cap
                ij = np.floor((P[:, :2] - lo) / c).astype(int)
                keep &= (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1])
                P, lb, ij = P[keep], lab[keep], ij[keep]
                order = np.argsort(P[:, 2])                 # highest last: it wins the cell
                H_ = H.copy()
                np.maximum.at(H_, (ij[:, 0], ij[:, 1]), P[:, 2])
                win = P[order, 2] >= H_[ij[order, 0], ij[order, 1]] - 1e-9
                Lb[ij[order][win, 0], ij[order][win, 1]] = lb[order][win]
                H = H_
            obs = np.isfinite(H)
            # the bare table has no surface points (it is the analytic plane): a cell shows the table top when the
            # space just above the table there was seen (free or occupied in the map)
            g = lo + (np.argwhere(~obs) + 0.5) * c
            if len(g):
                idx = self.scene.index(np.c_[g, np.full(len(g), self.table_z + 0.015)])
                ins = self.scene.inside(idx)
                seen = np.zeros(len(g), bool)
                seen[ins] = self.scene.state[idx[ins, 0], idx[ins, 1], idx[ins, 2]] != -1
                ij = np.argwhere(~obs)[seen]
                obs[ij[:, 0], ij[:, 1]] = True
            return np.where(np.isfinite(H), H, self.table_z), Lb, obs
        H_all, Lb_all, obs_all = grid(None)
        if z_cap is None:
            return H_all, Lb_all, obs_all, H_all, obs_all, Lb_all
        H, Lb, obs = grid(z_cap)
        return H, Lb, obs, H_all, obs_all, Lb_all

    def _dest_mask(self, dest, lo, shape, extra=None):
        """The destination's footprint on the REST grid: the filled hull of its model's xy (and of `extra`: its
        fixture's surface), widened by REST["dest_pad"] (what is inside a drawer or a cavity is inside this hull even
        where no label reaches)."""
        c = REST["cell"]
        if getattr(dest, "virtual", False) or len(dest.points) < 4:
            return None
        g = lo + (np.argwhere(np.ones(shape, bool)) + 0.5) * c
        P = dest.points if extra is None or not len(extra) else np.concatenate([dest.points, extra])
        xy = voxel_down(np.c_[P[:, :2], np.zeros(len(P))], 0.005)[:, :2]
        m = manip._hull_mask(xy, g).reshape(shape)
        return ndimage.binary_dilation(m, iterations=max(1, int(round(REST["dest_pad"] / c))))

    def _entry_dirs(self, x_xy, hint=None, open_dirs=None):
        """Horizontal directions in which an entry leaves the rest pose (at most ENTRY["max_dirs"]): System 1's hint,
        the spot's open sides (where the walls test found no wall of the destination: a slide can only leave there),
        then the side the arm comes from (toward the robot base) and its neighbours."""
        b = np.asarray(self.rm.root_p[:2], float) - np.asarray(x_xy, float)
        a0 = float(np.arctan2(b[1], b[0]))
        angs = a0 + np.arange(ENTRY["n_dirs"]) * 2 * np.pi / ENTRY["n_dirs"]
        angs = sorted(angs, key=lambda t: abs((t - a0 + np.pi) % (2 * np.pi) - np.pi))
        dirs = []
        if hint is not None and np.linalg.norm(np.asarray(hint, float)[:2]) > 1e-6:
            dirs.append(np.r_[np.asarray(hint, float)[:2], 0.0])
        if open_dirs is not None:
            dirs += [np.r_[np.asarray(d, float)[:2], 0.0] for d in open_dirs]
            # an enclosed spot leaves only through its open sides
            angs = angs[:1] if len(open_dirs) else angs
        dirs += [np.array([np.cos(t), np.sin(t), 0.0]) for t in angs]
        out = []
        for d in dirs:
            d = d / max(1e-9, np.linalg.norm(d))
            if all(float(d @ e) < 0.95 for e in out):
                out.append(d)
        return out[:ENTRY["max_dirs"]]

    def entry_search(self, p_rel, Rp, Wp, Lc, radc, held, dirs, dz=0.0, up_first=True):
        """How the hand and the held object get to a rest pose: a free sweep of both (observed geometry) along
        straight down; an oblique line; or a slide (down in front, then in at a little above the rest, then the
        set-down). A side opening and a drawer under an overhang are the same case. Returns the entry
        {kind, path (hand-off point .. release point), approach (unit, along the first leg), dir} or (None, blocker):
        the first thing in the way of the vertical entry."""
        up = np.array([0.0, 0.0, 1.0])

        Ws = voxel_down(np.asarray(Wp, float), 0.01)         # the object thinned to 1 cm for the sweeps
        LcR = Lc @ Rp.T

        def clear_at(pos):
            ch = self.checker.point_clearance(pos + LcR, held, unknown=False) - radc
            if ch.min() < REST["finger"]:
                return False
            cp = self.checker.point_clearance(pos + Ws, held, unknown=False) - 0.002
            return bool(cp.min() >= 0.0)
        blocked = None
        if up_first:
            for z in np.arange(REST["sweep"][0], REST["sweep"][1] - dz + 1e-9, REST["sweep"][2]):
                if not clear_at(p_rel + up * z):
                    q0 = p_rel + up * z
                    blocked = (self.nearest_obstacle(np.concatenate([q0 + Lc @ Rp.T, q0 + Wp]), held) or {})
                    break
            if blocked is None:
                return {"kind": "vertical", "path": [p_rel], "approach": DOWN, "dir": None}, None
        for h in dirs:
            # oblique: one straight line, from above-outside down to the release
            e = np.cos(ENTRY["oblique"]) * h + np.sin(ENTRY["oblique"]) * up
            if all(clear_at(p_rel + e * s) for s in np.arange(REST["sweep"][0], REST["sweep"][1] - dz + 1e-9,
                                                              REST["sweep"][2])):
                return {"kind": "oblique", "path": [p_rel], "approach": -e, "dir": h}, blocked
            # slide: in at a little above the rest (from where the hand can rise), then down onto it
            for z1 in ENTRY["lift"]:
                p_mid = p_rel + up * z1
                if not clear_at(p_mid):
                    continue
                for s in np.arange(ENTRY["step"], ENTRY["out_max"] + 1e-9, ENTRY["step"]):
                    p = p_mid + h * s
                    if not clear_at(p):
                        break
                    if all(clear_at(p + up * r) for r in ENTRY["rise"]):
                        return {"kind": "slide", "path": [p, p_mid, p_rel], "approach": DOWN, "dir": h,
                                "slide_m": round(float(s), 3), "lift_m": z1}, blocked
                else:
                    continue
        return None, blocked

    def _entry_posture(self, Rp, entry, q_seed, held, finger):
        """An arm posture at the release that can go back out along the whole entry and 8 cm on along the hand-off's
        slide, with the arm clear of the observed scene; None when there is none."""
        path, a = entry["path"], np.asarray(entry["approach"], float)
        p_rel = path[-1]
        sols = self.rm.solve_ik(Rp, p_rel, q_seed, n_random=8)
        if not len(sols):
            sols = self.rm.solve_ik(Rp, p_rel, q_seed, n_random=32)
        lim = self.rm.limits[:7]
        for q in sols[:6]:
            if np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q)) < 0.05:
                continue
            qq, Q, ok = q, [q], True
            for wp in list(reversed(path[:-1])) + [path[0] - a * 0.08]:
                line = self.line_configs(Rp, wp, qq, finger=finger)
                if line is None:
                    ok = False
                    break
                Q.extend(line[1::2])
                qq = line[-1]
            if not ok:
                continue
            good, c = self.checker.ok(np.array(Q), {"default": -1.0, "arm": ENTRY["arm"]}, exclude=held,
                                      finger=finger, unknown=False)
            if good.all():
                return q
            if getattr(self, "_posture_blocker", None) is None:
                # what the arm would meet there (a door, a fixture): a reason System 1 and D4 can act on
                k = int(np.argmin(c["arm"]))
                W, _ = self.checker.world_points(np.array(Q)[k:k + 1], finger)
                self._posture_blocker = (self.nearest_obstacle(W[0][self.checker.pg == 0], held) or {}).get("object")
        return None

    @staticmethod
    def _footprint_share(lb, inside, family, flat=False):
        """"On": the share of the object's footprint cells (all of them) over the destination's own support region
        (`inside`: the closed support mask at the footprint cells) and not showing another object's surface. Its labels
        decide only where geometry cannot: no support region (inside None: too sparse) or a destination too flat to
        stand out from the table. lb: owner of the top surface per footprint cell."""
        lb = np.asarray(lb)
        if not len(lb):
            return 0.0
        own_lab = np.isin(lb, list(family))
        if inside is None or flat:
            return float(np.mean(own_lab))
        other = (lb != 0) & ~own_lab
        return float(np.mean(np.asarray(inside, bool) & ~other))

    @staticmethod
    def _rays(offs):
        """For the 8 directions around a footprint (cell offsets): the cells of a ray from just outside the footprint
        out to REST["ray"], nearest first (offsets from the footprint's reference cell), each with its distance beyond
        the footprint's edge (m)."""
        c = REST["cell"]
        ctr = offs.mean(0)
        rays = []
        for k in range(8):
            a = -np.pi + (k + 0.5) * np.pi / 4
            d = np.array([np.cos(a), np.sin(a)])
            edge = max(float(((offs - ctr) @ d).max()) * c, 0.0)
            r = np.arange(edge + REST["ring"][0], REST["ray"] + 1e-9, c / 2)
            cells = np.round(ctr + r[:, None] * d / c).astype(int)
            _, i = np.unique(cells, axis=0, return_index=True)
            i = np.sort(i)
            rays.append((cells[i], r[i] - edge))
        return rays

    def _support_mask(self, dest, lo, shape, extra=None):
        """"On": the destination's own support region on the REST grid: its points (and the unlabelled surface
        connected to it) that stand out from the table, closed morphologically (REST["support_close"]) with holes
        filled, so gaps in its cloud (a patch hidden by the object, points of another model removed, a concave plate's
        inner surface) do not count as off it. Returns (mask, n points); mask None when too sparse to decide."""
        c = REST["cell"]
        if getattr(dest, "virtual", False) or not len(dest.points):
            return None, 0
        P = dest.points if extra is None or not len(extra) else np.concatenate([dest.points, extra])
        P = P[P[:, 2] > self.table_z + REST["above_table"]]
        if len(P) < REST["support_min_pts"]:
            return None, int(len(P))
        k = max(1, int(round(REST["support_close"] / c)))
        ij = np.floor((P[:, :2] - lo) / c).astype(int) + k
        m = np.zeros((shape[0] + 2 * k, shape[1] + 2 * k), bool)          # padded: closing is not cut at the edge
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < m.shape[0]) & (ij[:, 1] < m.shape[1])
        m[ij[ok, 0], ij[ok, 1]] = True
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        m = ndimage.binary_closing(m, structure=(xx ** 2 + yy ** 2) <= k * k)
        m = ndimage.binary_fill_holes(m)
        return m[k:-k, k:-k], int(len(P))

    def _rest_ctx(self, held, rel, drop, hint, f_carry, w_held, why, dbg_3d):
        """What the 3D checks of a rest pose need (hand models, the release-width ladders, the arm's start)."""
        f_rel = manip.FULL_JOINT
        L, grp, rad = self._hand_local(f_rel)
        Lc, _, radc = self._hand_local(f_carry)
        tight = min(f_rel, max(0.004, w_held / 2 + 0.008))   # the narrowest opening that still lets the object go
        ladder = [f for f in (f_rel, 0.032, 0.026) if f > tight] + [tight]
        hands = {f: self._hand_local(f) for f in ladder}
        return {"held": held, "rel": rel, "drop": drop, "hint": hint, "f_rel": f_rel, "L": L, "grp": grp, "rad": rad,
                "Lc": Lc, "radc": radc, "wide": [(f, *hands[f]) for f in ladder],
                "tight": [(f, *hands[f]) for f in sorted(set(ladder))], "q_now": self.rb.q()[:7],
                "why": why, "dbg": dbg_3d}

    def _bad(self, ctx, k):
        ctx["why"][k] = ctx["why"].get(k, 0) + 1

    def _release_hand(self, ctx, Rp, site_rest, heights, ladder):
        """The lowest release height (from `heights`) and the width (from `ladder`, in order) at which the open hand
        is clear of the observed scene (palm and fingers); (None, None) when none is."""
        up = np.array([0.0, 0.0, 1.0])
        for z in heights:
            for f, Lf, grpf, radf in ladder:
                ch = self.checker.point_clearance(site_rest + up * z + Lf @ Rp.T, ctx["held"], unknown=False) - radf
                if ch[grpf == 1].min() >= REST["palm"] and ch[grpf == 2].min() >= REST["finger"]:
                    return float(z), float(f)
        return None, None

    def _blocked_below(self, ctx, Rp, site_rest, rise=0.015):
        """At the floor, is what the (tightest-open) hand meets below it (a wall top, a rim: raising the hand clears
        it), not beside or above it (a cabinet face, a shelf, a handle overhead)? Raising the hand by `rise` must
        improve its worst clearance."""
        f, Lf, grpf, radf = ctx["tight"][0]
        up = np.array([0.0, 0.0, 1.0])
        marg = np.where(grpf == 1, REST["palm"], REST["finger"])
        worst = lambda z: float(np.min(self.checker.point_clearance(site_rest + up * z + Lf @ Rp.T, ctx["held"],
                                                                    unknown=False) - radf - marg))
        c0 = worst(0.0)
        return c0 < 0.0 and worst(rise) > c0 + 0.005

    def _rest_3d(self, ctx, Rp, Wp, site_rest, spot):
        """The hand's part of a rest pose: the open hand at the release, the way in (entry search) and an arm posture
        along it. Rp, site_rest: the hand at the rest; Wp: payload offsets from it (world axes); spot: the 2D result
        (xy, yaw, rest_z, rim, open_dirs, wall_gap, score). Returns the candidate or None (the reason counted).
        Release: at the rest (a set-down); a "drop" stage up to REST["fall"] above it; an "in" set-down stage whose
        payload is clear at the floor and well inside the container's walls, when only the hand blocks at the floor,
        a bounded drop from at most REST["in_drop"] with the tightest release width that is clear."""
        up = np.array([0.0, 0.0, 1.0])
        held, rel, drop = ctx["held"], ctx["rel"], ctx["drop"]
        heights = np.arange(0.0, REST["fall"] + 1e-9, 0.01) if drop else np.array([0.0])
        dz, f_open = self._release_hand(ctx, Rp, site_rest, heights, ctx["wide"])
        bounded = False
        if dz is None and rel == "in" and not drop:
            bd = ctx.setdefault("bounded_drop", {"blocked": 0, "wall_too_close": 0, "not_below": 0, "still_blocked": 0,
                                                 "dropped": 0})
            bd["blocked"] += 1
            if spot.get("wall_gap") is None or spot["wall_gap"] < REST["drop_wall_gap"]:
                bd["wall_too_close"] += 1
            elif not self._blocked_below(ctx, Rp, site_rest):
                bd["not_below"] += 1
            else:
                dz, f_open = self._release_hand(ctx, Rp, site_rest, np.arange(0.005, REST["in_drop"] + 1e-9, 0.005),
                                                ctx["tight"])
                bounded = dz is not None
                bd["dropped" if bounded else "still_blocked"] += 1
        x, yaw = spot["xy"], spot["yaw"]
        if dz is None:
            L, grp, rad = ctx["L"], ctx["grp"], ctx["rad"]
            self._bad(ctx, "hand_meets " + str((self.nearest_obstacle(site_rest + L @ Rp.T, held) or {}).get("object")))
            if len(ctx["dbg"]) < 4:
                Wd = site_rest + L @ Rp.T
                c0 = self.checker.point_clearance(Wd, held, unknown=False)
                ch = c0 - rad
                i0 = int(np.argmin(ch))
                ctx["dbg"].append({"xy": np.round(x, 3), "yaw": round(float(yaw), 2), "why": "hand",
                                   "palm_mm": round(1e3 * float(ch[grp == 1].min()), 1),
                                   "finger_mm": round(1e3 * float(ch[grp == 2].min()), 1),
                                   "site": np.round(site_rest, 3), "pt": np.round(Wd[i0], 3),
                                   "near": self.nearest_obstacle(Wd[i0:i0 + 1], held),
                                   "wall_gap": spot.get("wall_gap")})
            return None
        # the way in: a free sweep of hand and object (vertical, oblique, or a slide in under what is above)
        ctx["n_entry"] = ctx.get("n_entry", 0) + 1
        p_rel = site_rest + up * dz
        entry, blocked = self.entry_search(p_rel, Rp, Wp, ctx["Lc"], ctx["radc"], held,
                                           self._entry_dirs(x, ctx["hint"], spot.get("open_dirs")), dz)
        if entry is None:
            self._bad(ctx, "overhang " + str((blocked or {}).get("object")))
            if len(ctx["dbg"]) < 4:
                ctx["dbg"].append({"xy": np.round(x, 3), "yaw": round(float(yaw), 2), "why": "entry"})
            return None
        self._posture_blocker = None
        q = self._entry_posture(Rp, entry, ctx["q_now"], held, ctx["f_rel"])
        if q is None:
            # none along the entry, or none the arm can leave by (and what the arm would meet, if that)
            self._bad(ctx, "no_arm_posture" + (f" {self._posture_blocker}" if self._posture_blocker else ""))
            return None
        return {"xy": np.asarray(x, float), "yaw": float(yaw), "R": Rp, "p": p_rel, "p_rest": site_rest,
                "rest_z": spot["rest_z"], "release_dz": dz, "q": q, "score": spot["score"],
                "rim_z": spot.get("rim") if entry["kind"] == "vertical" else None,
                "bottom_off": float(Wp[:, 2].min()), "entry": entry["kind"], "path": entry["path"],
                "approach": np.asarray(entry["approach"], float), "entry_dir": entry.get("dir"),
                "slide_m": entry.get("slide_m"), "lift_m": entry.get("lift_m"),
                "bounded_drop": bounded, "release_finger": f_open if bounded else None}

    def rest_poses(self, obj, dest, rel, aim_xy, yaws, failed, drop, n_max=4, aim_z=None, payload=None, hint=None,
                   max_3d=None, spots_only=False):
        """Where the held object can come to rest near the pointed spot, from all observed geometry: on what is under
        its footprint at the pointed level (the destination's floor or top: "in" = the destination's walls around it,
        "on" = most of its footprint over the destination's own support region; the table beside a reference),
        nothing standing inside the footprint, the open hand clear at the release, a free entry for hand and object
        (searched: vertical, oblique, slide-in), and an arm posture along it. System 1's spot ranks the candidates;
        geometry decides.
        payload: (points relative to a reference point, rotation of that reference) for a grasp not made yet or for
        the object alone; default: what is held now (points in the hand frame, the hand's rotation).
        spots_only: the object alone (no hand): the 2D stage and the object's own clearance; returns spots (xy, yaw,
        ref (the reference point at the rest), rest_z, rim, open_dirs, wall_gap, score) for _grasp_puts.
        Returns (ranked candidates, rejection counts)."""
        max_3d = (40 if spots_only else REST["max_3d"]) if max_3d is None else int(max_3d)
        if payload is None:
            P = self.checker.attached.pts_site
            Rs, _ = self.rb.eef_pose()
            held = (self.held,) if self.held is not None else ()
            f_carry = self._finger()
            w_held = float(self.held_width or 0.0)
        else:
            P, Rs = np.asarray(payload[0], float), np.asarray(payload[1], float)
            held = (obj.oid,)
            f_carry = manip.FULL_JOINT
            w_held = 0.0
        if len(P) > 500:
            P = P[np.random.default_rng(0).choice(len(P), 500, replace=False)]
        why, dbg_3d = {}, []
        ctx = None if spots_only else self._rest_ctx(held, rel, drop, hint, f_carry, w_held, why, dbg_3d)
        c = REST["cell"]
        bad = lambda k: why.__setitem__(k, why.get(k, 0) + 1)
        inside_oids = [m.oid for m in self._contents_of(dest)]      # what already lies in the destination
        # the destination's own surfaces: it, its parts, the fixture it belongs to (a drawer's cabinet), and the
        # unlabeled surface connected to them (a segment that covers only part of it: a roof, a back wall)
        parts = [m.oid for m in self.objects.values() if m.part_of == dest.oid]
        family = [dest.oid] + parts + ([dest.part_of] if dest.part_of is not None else [])
        fx_pts = None
        if rel in ("in", "on") and not getattr(dest, "virtual", False) and dest.oid in self.objects:
            fx_oids, fx_pts = self._fixture(dest, obj)
            family += list(fx_oids)
        flat_dest = bool(len(dest.points)) and not getattr(dest, "virtual", False) and \
            dest.top_z - self.table_z < REST["flat_dest"]
        h_pay = float(np.ptp((P @ Rs.T)[:, 2]))
        z_cap = None if aim_z is None else float(aim_z) + h_pay + REST["cap_pad"]
        self._rest_info = {"z_cap": z_cap, "roofed": False, "n3d": 0, "open_dir": None}
        dbg_in, dbg_on = [], []
        for radius in REST["radius"]:
            out, n3d = [], 0
            r_pay = float(np.linalg.norm(P[:, :2], axis=1).max())
            lo = np.asarray(aim_xy, float) - radius - r_pay - 0.03
            hi = np.asarray(aim_xy, float) + radius + r_pay + 0.03
            H, Lb, obs, H_all, obs_all, Lb_all = self._surface_grid(lo, hi, held, z_cap)
            ia = np.clip(np.floor((np.asarray(aim_xy, float) - lo) / c).astype(int), 0, np.array(H.shape) - 1)
            self._rest_info["roofed"] = bool(z_cap is not None and H_all[ia[0], ia[1]] > z_cap)
            dmask = self._dest_mask(dest, lo, H.shape, fx_pts) if rel == "in" else None
            smask, n_sup = self._support_mask(dest, lo, H.shape, fx_pts) if rel == "on" else (None, 0)
            self._rest_info.update(support_cells=None if smask is None else int(smask.sum()), support_pts=n_sup)
            seen_share = round(float(obs.mean()), 3)
            g = np.arange(-radius, radius + 1e-9, 0.01)
            C = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
            C = C[np.linalg.norm(C, axis=1) <= radius + 1e-9] + aim_xy
            C = C[np.argsort(np.linalg.norm(C - aim_xy, axis=1))]
            C = np.array([x for x in C if all(np.linalg.norm(x - f) > r for f, r in failed)])
            if not len(C):
                continue
            pre = []
            for yaw in yaws:
                Rp = rot_z(yaw) @ Rs
                W = P @ Rp.T                               # payload offsets from the reference, world axes
                c_off = 0.5 * (W[:, :2].min(0) + W[:, :2].max(0))
                bottom = float(W[:, 2].min())
                offs = np.unique(np.floor((W[:, :2] - c_off) / c).astype(int), axis=0)
                rays = self._rays(offs)
                for x in C:
                    ij = np.floor((x - lo) / c).astype(int) + offs
                    if (ij < 0).any() or (ij[:, 0] >= H.shape[0]).any() or (ij[:, 1] >= H.shape[1]).any():
                        bad("outside_view")
                        continue
                    h, lb, ob = H[ij[:, 0], ij[:, 1]], Lb[ij[:, 0], ij[:, 1]], obs[ij[:, 0], ij[:, 1]]
                    if ob.mean() < REST["observed"]:
                        bad("unobserved")
                        continue
                    rest_z = float(h.max())
                    # "on" / beside: the object rests upright on a flat support, nothing else standing under it. "in":
                    # any pose counts inside (leaning on a sloped liner, a wall or what is already in it), so the floor
                    # need not be flat: inside (walls around), no overlap and a bounded fall decide
                    if rest_z - float(np.median(h)) > (REST["spread"] if rel != "in" else max(REST["spread"],
                                                                                             0.5 * h_pay)):
                        # ("in": a sloped floor or what lies in it is fine; more than half the object's height across
                        # its footprint is a wall top or a rim edge under part of it)
                        bad("would_rest_on_something_else")
                        continue
                    xc = np.floor((x - lo) / c).astype(int)
                    rim_z, open_dirs, wall_gap = None, None, None
                    if getattr(dest, "virtual", False):
                        # the table itself as the support (a set-down somewhere neutral): rest on the table plane
                        if abs(rest_z - self.table_z) > REST["beside_tol"]:
                            bad("not_on_the_table")
                            continue
                    elif rel == "on":
                        # most of the object's footprint over the destination's own support region
                        inside = None if smask is None else smask[ij[:, 0], ij[:, 1]]
                        share = self._footprint_share(lb, inside, family, flat_dest)
                        if share < REST["on_share"]:
                            bad("not_on_the_destination")
                            if len(dbg_on) < 3:
                                own = np.isin(lb, family)
                                dbg_on.append({"xy": np.round(x, 3), "yaw": round(float(yaw), 2),
                                               "share": round(share, 3), "cells": int(len(lb)),
                                               "in_support": None if inside is None else int(np.sum(inside)),
                                               "other": int(np.sum((lb != 0) & ~own)), "own_label": int(np.sum(own)),
                                               "labels": {str(k): int(v) for k, v in
                                                          zip(*np.unique(lb, return_counts=True))}})
                            continue
                    elif rel == "in":
                        # inside the destination means its walls around the spot: in most directions the first
                        # surface that rises above the object's bottom (at any height and distance: a roof over a
                        # cavity, the far wall of a wide drawer) is the destination's own (its label, a part of it,
                        # its fixture, or inside its footprint hull); a spot on a rim or beside it has open sides or
                        # other things around
                        walls, rims, gaps, dbg_rays, open_k = 0, [], [], [], []
                        for k_ray, (ray, rdist) in enumerate(rays):
                            rij = xc + ray
                            okr = (rij[:, 0] >= 0) & (rij[:, 1] >= 0) & (rij[:, 0] < H.shape[0]) & \
                                (rij[:, 1] < H.shape[1])
                            keep = np.cumprod(okr).astype(bool)
                            rij, rdist_k = rij[keep], rdist[keep]
                            if not len(rij):
                                dbg_rays.append([0])
                                open_k.append(k_ray)
                                continue
                            hr = np.where(obs_all[rij[:, 0], rij[:, 1]], H_all[rij[:, 0], rij[:, 1]], -np.inf)
                            hit = np.flatnonzero(hr >= rest_z + REST["wall"])
                            # the gap for a bounded drop: to the walls at the object's own height (the capped grid),
                            # not to what overhangs it (a handle, a roof edge above)
                            hc = np.flatnonzero(np.where(obs[rij[:, 0], rij[:, 1]], H[rij[:, 0], rij[:, 1]], -np.inf)
                                                >= rest_z + REST["wall"])
                            if len(hc):
                                gaps.append(float(rdist_k[hc[0]]))
                            if not len(hit):
                                dbg_rays.append([len(rij), None])
                                open_k.append(k_ray)
                                continue
                            cell = rij[hit[0]]
                            lab = int(Lb_all[cell[0], cell[1]])
                            # the destination's own body (and its fixture, what lies in it) encloses; a part of it
                            # that moves (an open door, a lid) does not
                            mine = (lab in family and lab not in parts) or (lab in inside_oids) or \
                                (lab not in parts and dmask is not None and bool(dmask[cell[0], cell[1]]))
                            dbg_rays.append([len(rij), int(hit[0]), round(float(hr[hit[0]]), 3), lab, mine])
                            if mine:
                                walls += 1
                                rims.append(float(hr[hit[0]]))
                        if self._rest_info.get("open_dir") is None and 0 < len(open_k) < 8:
                            # the side(s) of the spot nearest the aim without a wall: where it is open (a cavity's
                            # front, a drawer's front): a look into it comes from there
                            a_open = -np.pi + (np.array(open_k) + 0.5) * np.pi / 4
                            v_open = np.array([np.cos(a_open).sum(), np.sin(a_open).sum(), 0.0])
                            if np.linalg.norm(v_open) > 1e-6:
                                self._rest_info["open_dir"] = v_open / np.linalg.norm(v_open)
                        open_dirs = [np.array([np.cos(-np.pi + (k + 0.5) * np.pi / 4),
                                               np.sin(-np.pi + (k + 0.5) * np.pi / 4), 0.0]) for k in open_k]
                        if walls < REST["walls"]:
                            bad("not_inside")
                            if len(dbg_in) < 2:
                                dbg_in.append({"xy": np.round(x, 3), "yaw": round(float(yaw), 2), "walls": walls,
                                               "rest_z": round(rest_z, 4), "rays": dbg_rays})
                            continue
                        rim_z = float(min(rims))      # the lowest wall around: the opening's rim for a drop
                        wall_gap = float(min(gaps)) if gaps else REST["ray"]   # inside the nearest side wall by
                    elif abs(rest_z - self.table_z) > REST["beside_tol"]:
                        bad("not_on_the_table")
                        continue
                    ref_rest = np.r_[x - c_off, rest_z - bottom]
                    pre.append({"score": float(np.linalg.norm(x - aim_xy)) + 0.01 * abs(yaw), "yaw": float(yaw),
                                "Rp": Rp, "ref": ref_rest, "rest_z": rest_z, "xy": np.asarray(x, float),
                                "rim": rim_z if rel == "in" and not getattr(dest, "virtual", False) else None,
                                "open_dirs": open_dirs, "wall_gap": wall_gap})
            pre.sort(key=lambda t: t["score"])
            # the costly 3D checks are spread over the turns (nearest spots first within each): a turn that puts the
            # hand against something fails the same way at every nearby spot
            per_yaw = int(np.ceil(max_3d / max(1, len(yaws))))
            n_yaw, seen_yaw = {}, {}
            for spot in pre:
                if len(out) >= n_max or n3d >= max_3d:
                    break
                x, yaw, Rp, site_rest = spot["xy"], spot["yaw"], spot["Rp"], spot["ref"]
                if any(np.linalg.norm(x - o["xy"]) < 0.02 and abs(yaw - o["yaw"]) < 0.3 for o in out):
                    continue
                ky = round(float(yaw), 3)
                if n_yaw.get(ky, 0) >= per_yaw:
                    continue
                # ... and over the area: a spot 1.5 cm from one tried at the same turn fails the same way
                if any(np.linalg.norm(x - t) < 0.015 for t in seen_yaw.get(ky, [])):
                    continue
                seen_yaw.setdefault(ky, []).append(np.asarray(x, float))
                n_yaw[ky] = n_yaw.get(ky, 0) + 1
                n3d += 1
                self._rest_info["n3d"] += 1
                Wp = P @ Rp.T
                up = np.array([0.0, 0.0, 1.0])
                # the object itself just above its rest: clear of everything (its support is below it)
                cp = self.checker.point_clearance(site_rest + up * REST["raise"] + Wp, held, unknown=False) - 0.002
                if cp.min() < 0.0:
                    no = self.nearest_obstacle(site_rest + Wp, held) or {}
                    bad("object_overlaps " + str(no.get("object")))
                    if len(dbg_3d) < 4:
                        dbg_3d.append({"xy": np.round(x, 3), "yaw": round(float(yaw), 2), "why": "object",
                                       "depth_mm": round(-1e3 * float(cp.min()), 1),
                                       "at": np.round((site_rest + Wp)[int(np.argmin(cp))] - site_rest, 3)})
                    continue
                if spots_only:
                    out.append({k: spot[k] for k in ("xy", "yaw", "ref", "rest_z", "rim", "open_dirs", "wall_gap",
                                                      "score")})
                    continue
                cand = self._rest_3d(ctx, Rp, Wp, site_rest, spot)
                if cand is not None:
                    out.append(cand)
            if out:
                self.log("rest_poses", n=len(out), radius=radius, z_cap=z_cap, spots_only=spots_only,
                         best=[{"xy": o["xy"], "yaw": round(o["yaw"], 2), "rest_z": o["rest_z"],
                                "dz": o.get("release_dz"), "entry": o.get("entry"), "dir": o.get("entry_dir"),
                                "bounded_drop": o.get("bounded_drop")} for o in out[:4]], rejected=why)
                return out, why
        self.log("rest_poses", n=0, rejected=why, seen_share=locals().get("seen_share"), z_cap=z_cap,
                 spots_only=spots_only, roofed=self._rest_info["roofed"], n3d=self._rest_info["n3d"],
                 support_cells=self._rest_info.get("support_cells"), support_pts=self._rest_info.get("support_pts"),
                 bounded_drop=(ctx or {}).get("bounded_drop"), not_inside=dbg_in, not_on=dbg_on, rejected_3d=dbg_3d)
        return [], why

    # ------------------------------------------------------------------------------------------- the put-down
    def put_down(self, obj, dest, cands, drop):
        """Hand-off at the outer end of one of the rest poses' entries (TRACE slides back from each and takes the first
        it certifies), the guarded legs of the entry (no contact expected) and the last one with its expected contact,
        the release gate, the departure back along the hand's approach and the entry, the hand-empty test and the
        measured rest. Returns a typed result."""
        ent = entity_names(self).get(dest.oid)          # None: a spot on the table (staging)
        finger = self._finger()
        ends = []
        for i, c in enumerate(cands):
            path = c.get("path") or [c["p"]]
            a = np.asarray(c.get("approach", DOWN), float)
            e = EndCandidate(c["R"], np.asarray(path[0], float), q=None, approach=a,
                             standoff=(0.03, 0.30) if len(path) == 1 else (0.0, 0.25), entity=ent)
            e.k = i
            ends.append(e)
        res = self.tm.move(ends, "to_place", grip=0.0, finger=finger, entity=ent, approach=ends[0].approach,
                           keep_ends=len(ends))
        if res is not None and res.status == "dropped":
            return {"status": "dropped_in_transit"}
        if res is None or res.status != "done" or self.tm.last_end is None:
            lm = getattr(self, "last_move", None) or {}
            return {"status": "could_not_reach", "why": lm.get("why") or {"kind": lm.get("status")}}
        c = cands[int(self.tm.last_end.k)]
        path = [np.asarray(w, float) for w in (c.get("path") or [c["p"]])]
        Rc, pc = self.rb.eef_pose()
        base = {"xy": c["xy"], "yaw_deg": round(float(np.degrees(c["yaw"])), 1),
                "planned_rest_above_table": round(c["rest_z"] - self.table_z, 4),
                "release_height_above_rest": round(c["release_dz"], 3), "entry": c.get("entry", "vertical"),
                "bounded_drop": bool(c.get("bounded_drop"))}
        visited = [pc.copy()]
        for i, wp in enumerate(path[:-1]):
            # the entry's first legs (down in front, in under what is above): any contact is an error
            out = self.guarded_leg(Rc, wp, speed=0.04, window=None, grip=0.0, finger=finger, tag=f"place_entry{i}")
            if out.verdict != "as_expected":
                at = self.mark_contact((wp - visited[-1]) / max(1e-9, np.linalg.norm(wp - visited[-1]))) \
                    if out.touched else None
                self._back_out(visited, finger)
                return {**base, "status": {"early": "early_contact", "no_line": "no_line"}.get(out.verdict,
                                                                                             out.verdict),
                        "descent": out.verdict, "leg": i, "touched": out.nearest,
                        "touched_height_above_table": None if at is None else round(float(at[2]) - self.table_z, 4)}
            visited.append(self.rb.eef_pose()[1])
        Rc, pc = self.rb.eef_pose()
        d_rel = float(np.linalg.norm(c["p"] - pc))
        u = (c["p"] - pc) / max(d_rel, 1e-9)
        into = drop and c.get("rim_z") is not None
        if into:
            # a drop into a container: it opens once the object is down inside the opening (below the rim), on
            # contact with the container or at the planned pose, whichever comes first (any pose counts inside)
            goal, window = c["p"] + u * REST["tol"][1], None
        elif c["release_dz"] <= 0.0005:
            # set-down: the object is expected to meet its support where planned (perception error both ways)
            t_lo, t_hi = REST["tol"]
            goal, window = c["p"] + u * t_hi, (max(0.0, d_rel - t_lo), d_rel + t_hi)
        else:
            goal, window = c["p"], None                   # a bounded drop: nothing to touch before the release height
        out = self.guarded_leg(Rc, goal, speed=0.05 if len(path) == 1 else 0.03, window=window, grip=0.0,
                               finger=finger, tag="place_descent")
        if into:
            Rn, pn = self.rb.eef_pose()
            low = float(pn[2] + c["bottom_off"])          # the object's bottom now
            inside = low < float(c["rim_z"]) - 0.01
            out.verdict = ("as_expected" if (inside or not out.touched) else "early") \
                if out.verdict in ("as_expected", "early", "none") else out.verdict
            self.log("drop_gate", bottom_above_table=round(low - self.table_z, 4),
                     rim_above_table=round(float(c["rim_z"]) - self.table_z, 4), touched=out.touched,
                     verdict=out.verdict)
        base.update(descent=out.verdict, travel_m=round(out.travel, 4))
        if out.verdict == "dropped":
            return {**base, "status": "dropped_in_transit"}
        if out.verdict != "as_expected":
            # the release gate is closed: keep the object, mark what was touched, back off along the entry
            at = self.mark_contact(u) if out.touched else None
            self._back_out(visited, finger)
            status = {"early": "early_contact", "none": "no_contact", "no_line": "no_line"}.get(out.verdict,
                                                                                               out.verdict)
            return {**base, "status": status, "touched": out.nearest,
                    "touched_height_above_table": None if at is None else round(float(at[2]) - self.table_z, 4)}
        # release: as wide as the surroundings allow, then leave along the hand's approach and the entry
        Rr, pr = self.rb.eef_pose()
        self._v2_release_p = pr.copy()
        # a bounded drop ("in", only the hand blocked at the floor) opens the tightest width that was clear
        f = float(c["release_finger"]) if c.get("bounded_drop") and c.get("release_finger") else \
            self.release_width(Rr, pr)
        att = self.checker.attached
        model_site = att.model_site if att.model_site is not None else att.pts_site
        self.ex.gripper_to(2 * (f + manip.PAD_OFFSET))
        bottom = float(self._payload_world(Rr, pr)[:, 2].min())
        self._finish_release(obj, bottom if out.touched else c["rest_z"], touched=out.touched)
        rest_pts = obj.points.copy()
        self._depart_after_release(Rr, pr, path, visited, f)
        self.ex.gripper(-1.0, max_steps=12)
        empty, frames = self.hand_empty(model_site, rest_pts)
        if not empty:
            # it hangs on the hand: it is still held (no release counted); the caller puts it down again
            self.checker.attached = att
            self.held, obj.held = obj.oid, True
            self.held_width = self.rb.finger_opening()
            self.scene.clear_label(obj.oid)
            self.log("not_released", object=obj.name)
            return {**base, "status": "not_released"}
        fit = self.relocalize(obj, frames)
        self.rebuild_map(self.frames[-8:])
        return {**base, "status": "released", "ended_on_contact": bool(out.touched),
                "release_finger_joint": round(f, 4), "measured_rest": fit}

    def _back_out(self, visited, finger, tag="place_back"):
        """Leave an entry that ended on an unexpected contact (or had no posture): back through the points it went
        through, then up; the object stays held."""
        Rs, ps = self.rb.eef_pose()
        if len(visited) <= 1:
            self.retreat(Rs, ps + np.array([0.0, 0.0, 0.10]), grip=0.0, speed=0.08, finger=finger, tag=tag)
            return
        for wp in reversed(visited[1:]):
            self.retreat(Rs, wp, grip=0.0, speed=0.06, finger=finger, tag=tag, until_clear=False)
        self.retreat(Rs, visited[0] + np.array([0.0, 0.0, 0.06]), grip=0.0, speed=0.08, finger=finger, tag=tag)

    def _depart_after_release(self, Rr, pr, path, visited, f):
        """After a release: out along the reverse of the hand's approach while the hand came in from the side (its
        fingers are around a part of the object: going up first would lift it by that part), then back along the
        entry (up from a slide's set-down, out under what is above, up), then up until clear."""
        a = Rr[:, 2]
        if abs(float(a[2])) < 0.7:
            self.retreat(Rr, pr - a * 0.06, grip=0.0, speed=0.06, finger=f, tag="release_back", until_clear=False)
        if len(path) > 1:
            Rs, ps = self.rb.eef_pose()
            shift = ps - pr
            for wp in reversed(path[:-1]):
                self.retreat(Rs, np.asarray(wp, float) + shift, grip=0.0, speed=0.06, finger=f, tag="release_out",
                             until_clear=False)
        Rs, ps = self.rb.eef_pose()
        self.retreat(Rs, ps + np.array([0.0, 0.0, 0.10]), grip=0.0, speed=0.08, finger=f, tag="release_lift")

    # ------------------------------------------------------------------------------------------- regrasp (D3)
    @staticmethod
    def object_clearance(W_obj, pts, side=None, cache=None):
        """Signed clearance of points to an object model (its points W_obj), as a 2.5D solid seen along `side`: the
        direction a side grasp comes from (the generator's own frame), else straight down. Seen from above, everything
        under a handle's top bar is solid; seen from the handle's side, the fingers beside its bar are outside."""
        from .perception import ObjectModel
        W_obj, pts = np.asarray(W_obj, float), np.asarray(pts, float)
        key = None if side is None else (len(W_obj), round(float(W_obj[:, 0].sum()), 5), tuple(np.round(side, 3)))
        if cache is not None and key is not None and key in cache:
            Q, c, tmp = cache[key]
        else:
            Q = np.eye(3) if side is None else manip.up_frame(np.asarray(side, float))
            c = W_obj.mean(0)
            loc = (W_obj - c) @ Q.T
            tmp = ObjectModel(-6, "o", "o", "target", float(loc[:, 2].min()) - 0.002)
            tmp.set_points(loc)
            if cache is not None and key is not None:
                cache[key] = (Q, c, tmp)
        return tmp.signed_clearance((pts - c) @ Q.T)

    def _hand_fits_pick(self, obj, R, p, W_obj, width, all_widths=False, side=None):
        """The pick without the arm (cheap): the open (or pre-shaped) hand at the grasp and 4 cm back along its
        approach clear of the scene and the table, and of the object model standing at W_obj (seen along `side` for a
        side grasp). Returns the first finger opening that fits (all of them, in order, with all_widths), or None / []."""
        from .harness import GRASP_NEAR, NEAR
        cache = self.__dict__.setdefault("_side_model_cache", {})
        excl = (obj.oid,)
        a = R[:, 2]
        fits = []
        # unsegmented surface hugging the object (a handle, a rim edge outside its mask) is the object itself
        fn, self.checker.free_near = self.checker.free_near, [(np.asarray(W_obj, float), 0.01, True)]
        try:
            return self._hand_fits_pick_inner(obj, R, p, W_obj, width, all_widths, side, cache, excl, a, fits)
        finally:
            self.checker.free_near = fn

    def _hand_fits_pick_inner(self, obj, R, p, W_obj, width, all_widths, side, cache, excl, a, fits):
        from .harness import GRASP_NEAR, NEAR
        for f in (manip.FULL_JOINT,) + tuple(manip.preshape_finger(width, gap) for gap in manip.APPROACH_GAPS):
            L, grp, rad = self._hand_local(f)
            ok = True
            for back in (0.0, 0.04):
                Wh = p - a * back + L @ R.T
                # observed geometry (user decision D2: unseen space is looked at, then crossed slowly)
                c = self.checker.point_clearance(Wh, excl, unknown=False) - rad
                zc = Wh[:, 2] - self.table_z - 0.5 * rad
                if c[grp == 1].min() < NEAR["palm"] or c[grp == 2].min() < NEAR["finger"]:
                    k = int(np.argmin(c - np.where(grp == 1, NEAR["palm"], NEAR["finger"])))
                    ok, self._pick_why = False, "scene %s at %s (%s %.1f mm, %d cm back)" % (
                        (self.nearest_obstacle(Wh[k:k + 1], excl) or {}).get("object"), np.round(Wh[k], 3).tolist(),
                        "palm" if grp[k] == 1 else "finger", 1e3 * float(c[k]), round(100 * back))
                    break
                if zc[grp == 1].min() < GRASP_NEAR["plane_palm"] or zc[grp == 2].min() < GRASP_NEAR["plane_finger"]:
                    ok, self._pick_why = False, "table"
                    break
            if not ok:
                continue
            dd = self.object_clearance(W_obj, p + L @ R.T, side, cache) - rad
            if dd[grp == 1].min() < 0.006 or dd[grp == 2].min() < 0.0015:
                self._pick_why = "object (%s %.1f mm)" % ("palm" if dd[grp == 1].min() < 0.006 else "finger",
                                                          1e3 * float(min(dd[grp == 1].min(), dd[grp == 2].min())))
                continue
            if not all_widths:
                return f
            fits.append(f)
        return fits if all_widths else None

    def object_spots(self, obj, dest, rel, aim, P_obj, drop):
        """Where the object itself (no hand) can rest near the pointed spot, at 12 turns: the 2D stage and its own
        clearance, once for every grasp to be judged (CONTACT_STAGE_PLAN 4.5 look-ahead). P_obj: its points relative
        to its reference (footprint middle at its bottom), world axes. Returns (spots, rejection counts)."""
        yaws = [k * np.pi / 6 for k in range(12)]
        return self.rest_poses(obj, dest, rel, np.asarray(aim.get("rest_xy", aim["xy"]), float), yaws, [], drop,
                               n_max=12, aim_z=aim.get("z_top"), payload=(P_obj, np.eye(3)), spots_only=True)

    def _grasp_puts(self, obj, g, o0, P_obj, spots, rel, drop, hint, max_entry=3):
        """Can grasp g (world, the object at its reference o0) put the object down at one of the object's rest spots:
        the open hand at the release first (cheap), then the way in and the arm posture for at most max_entry spots.
        Returns (candidate or None, rejection counts)."""
        why, dbg = {}, []
        ctx = self._rest_ctx((obj.oid,), rel, drop, hint, max(0.002, g.width / 2 - manip.PAD_OFFSET), g.width, why,
                             dbg)
        for spot in spots:
            Rz = rot_z(spot["yaw"])
            R_h = Rz @ g.R
            p_h = spot["ref"] + Rz @ (np.asarray(g.p, float) - o0)
            Wp = P_obj @ Rz.T + spot["ref"] - p_h                 # payload offsets from the hand, world axes
            cand = self._rest_3d(ctx, R_h, Wp, p_h, spot)
            if cand is not None:
                return cand, why
            if ctx.get("n_entry", 0) >= max_entry:
                break
        return None, why

    def _side_cands(self, obj):
        """Side grasp candidates on the object's model (cached per model)."""
        key = (obj.oid, len(obj.points), round(float(obj.points[:, 0].sum()), 4) if len(obj.points) else 0.0)
        cache = getattr(self, "_side_cache", {})
        if key not in cache:
            R0, _ = self.rb.eef_pose()
            cache = {key: manip.side_grasp_candidates(obj.points, obj.name, manip.side_directions(), cur_R=R0)}
            self._side_cache = cache
        return cache[key]

    def _grasp_executable(self, obj, R, p, W_obj, width, side=None):
        """Can the hand take the object (standing at W_obj) with the grasp (R, p): the open (or pre-shaped) hand clear
        of the scene, the table and the object model, an arm posture away from the joint limits that can come in
        along the approach and lift 10 cm. Returns the arm posture or None. (The object's points in the scene are
        excluded: it is somewhere else now.)"""
        from .harness import GRASP_JOINT_SLACK, GRASP_NEAR
        excl = (obj.oid,)
        a = R[:, 2]
        for f in self._hand_fits_pick(obj, R, p, W_obj, width, all_widths=True, side=side):
            att, self.checker.attached = self.checker.attached, None      # the phantom payload must not count
            fn, self.checker.free_near = self.checker.free_near, [(np.asarray(W_obj, float), 0.01, True)]
            try:
                lim = self.rm.limits[:7]
                for q in self.ik_clear(R, p, self.rb.q()[:7], exclude=excl, finger=f, n_random=48)[:3]:
                    if float(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q))) < GRASP_JOINT_SLACK:
                        continue
                    good, _ = self.checker.ok(q[None], GRASP_NEAR, exclude=excl, finger=f)
                    if not good[0]:
                        continue
                    if self.line_configs(R, p - a * 0.10, q, finger=f) is None or \
                            self.line_configs(R, p + np.array([0.0, 0.0, 0.10]), q, finger=f) is None:
                        continue
                    return q, f
            finally:
                self.checker.attached = att
                self.checker.free_near = fn
        return None

    def plan_regrasp(self, obj, dest, rel, aim, drop, hint, budget_s=45.0):
        """D3: the grasp held now has no put-down here. Where the object itself can rest there (object_spots, once);
        grasps of it in any direction that can put it down at one of those spots (_grasp_puts: the open hand at the
        release, then the way in and the arm), and a staging spot on the table where the object, set down with the
        grasp held now (turned as needed), can be taken with one of them. Returns {table, staging, want, ...} or None
        (with self._regrasp_why)."""
        import time as _time
        from .perception import ObjectModel
        t0 = _time.time()
        att = self.checker.attached
        Rs, ps = self.rb.eef_pose()
        P_site = att.pts_site
        M_site = att.model_site if att.model_site is not None else P_site
        W_now = ps + M_site @ Rs.T
        full = ps + P_site @ Rs.T
        o0 = np.r_[0.5 * (full[:, :2].min(0) + full[:, :2].max(0)), float(full[:, 2].min())]
        P_obj = full - o0
        spots, why = self.object_spots(obj, dest, rel, aim, P_obj, drop)
        if not spots:
            self._regrasp_why = "the object itself has no place to rest there (" + self._rest_words(why) + ")"
            self.log("regrasp_lookahead", spots=0, rejected=why, s=round(_time.time() - t0, 1))
            return None
        tmp = ObjectModel(-4, obj.name, obj.category, "target", float(W_now[:, 2].min()) - 0.002)
        tmp.set_points(W_now)
        alts = manip.side_grasp_candidates(W_now, obj.name, manip.side_directions(), cur_R=Rs) + \
            manip.grasp_candidates(tmp, tmp.support_z)
        same = lambda g: np.degrees(np.arccos(np.clip(g.R[:, 2] @ Rs[:, 2], -1, 1))) < 25 and \
            np.linalg.norm(g.p - ps) < 0.02
        # the grasp held is blocked by the hand's own room or the way in: the grasps whose hand stays lowest first;
        # never from below (the hand would end up between the object and its support)
        alts = [g for g in alts if not same(g) and float(g.R[2, 2]) <= 0.25]
        if alts:
            hh = self.hand_height(alts, float(W_now[:, 2].max()))
            alts = [alts[i] for i in np.lexsort((-np.array([g.score for g in alts]), np.round(hh, 2)))]
        t_eval = _time.time()                   # the evaluation budget starts after generating the alternatives
        compat, tried, kinds = [], 0, {}
        for g in alts:
            if len(compat) >= 3 or _time.time() - t_eval > 0.6 * budget_s or tried >= 16:
                break
            tried += 1
            cand, why_g = self._grasp_puts(obj, g, o0, P_obj, spots, rel, drop, hint)
            for k, v in why_g.items():
                kinds[k.split(" ")[0]] = kinds.get(k.split(" ")[0], 0) + v
            if cand is not None:
                compat.append((g, Rs.T @ g.R, Rs.T @ (g.p - ps), cand))
        self.log("regrasp_lookahead", spots=len(spots), alternatives=len(alts), tried=tried, compatible=len(compat),
                 entries=[c[3]["entry"] for c in compat], approach=[np.round(c[0].R[:, 2], 2) for c in compat],
                 rejected=kinds, s=round(_time.time() - t0, 1))
        if not compat:
            self._regrasp_why = "no grasp of the object has a put-down there (" + self._rest_words(kinds) + ")"
            return None
        # staging: where it was taken from (free and reachable), then halfway to the destination
        aim_xy = np.asarray(aim.get("rest_xy", aim["xy"]), float)
        origin = getattr(self, "_grasp_origin", {}).get(obj.oid)
        aims = [a for a in (origin, None if origin is None else 0.5 * (origin + aim_xy)) if a is not None]
        aims.append(np.asarray(ps[:2], float))
        for sxy in aims:
            if _time.time() - t0 > budget_s:
                break
            table = self._table_patch(sxy)
            stc, _ = self.rest_poses(obj, table, "on", np.asarray(sxy, float), [k * np.pi / 4 for k in range(8)], [],
                                     False, n_max=6, aim_z=self.table_z)
            for sc in stc:
                Ws = sc["p_rest"] + M_site @ sc["R"].T
                for g, R_rel, p_rel, cd in compat:
                    Rg, pg = sc["R"] @ R_rel, sc["p_rest"] + sc["R"] @ p_rel
                    sd = g.info.get("side")
                    sd = None if sd is None else sc["R"] @ Rs.T @ np.asarray(sd, float)   # turned with the object
                    ex = self._grasp_executable(obj, Rg, pg, Ws, g.width, side=sd)
                    if ex is not None:
                        self.log("regrasp_plan", staging_xy=sc["xy"], staging_yaw_deg=round(np.degrees(sc["yaw"]), 1),
                                 grasp_approach=np.round(Rg[:, 2], 2), dest_entry=cd["entry"],
                                 s=round(_time.time() - t0, 1))
                        return {"table": table, "staging": sc, "want": (Rg, pg), "dest_entry": cd["entry"]}
        self._regrasp_why = "no staging spot where the object can be set down and taken again that way"
        self.log("regrasp_plan", found=False, why=self._regrasp_why, s=round(_time.time() - t0, 1))
        return None

    def _look_at_rest(self, aim, dest, hint=None):
        """A wrist look at the pointed spot when it is not seen well enough: from above, or, when something is above
        it (a roof, a shelf), level with it from its open side (where the walls test found no wall; else System 1's
        entry side; else the arm's side). The view is added to the destination's model. True when a frame came."""
        rest_xy = np.asarray(aim.get("rest_xy", aim["xy"]), float)
        if self._rest_info.get("roofed") and aim.get("z_top") is not None:
            at = np.r_[rest_xy, float(aim["z_top"]) + 0.03]
            od = self._rest_info.get("open_dir")
            if od is None and hint is not None:
                od = np.asarray(hint, float)
            if od is None:
                od = np.r_[self.rm.root_p[:2] - rest_xy, 0.0]
            h = -np.r_[np.asarray(od, float)[:2], 0.0]
            h = h / max(1e-9, np.linalg.norm(h))
            fr, tried = None, 0
            for down, turn, dist in ((0.35, 0.0, 0.30), (0.7, 0.0, 0.28), (0.5, 0.35, 0.28), (0.5, -0.35, 0.28)):
                v = rot_z(turn) @ h + np.array([0.0, 0.0, -down])
                tried += 1
                fr = self.look_toward(at, v, dist=dist, allow_rrt=tried == 1)
                if fr is not None:
                    break
            self.log("look_at_rest", kind="level", view_dir=np.round(v, 2), got=fr is not None, tries=tried)
        else:
            fr = self.look_from_above(np.r_[rest_xy, dest.top_z], height=0.30)
            self.log("look_at_rest", kind="above", got=fr is not None)
        if fr is None:
            return False
        mt = self.mask_matching(fr, [dest.category, dest.name], dest)
        if mt is not None:
            dest.add_points(self._without_others(self.filter_points(mask_points(fr, mt[0]), dest), dest.oid))
        self.rebuild_map(self.frames[-8:])
        return True

    def lookahead_grasp(self, obj, dest, st, place_pt):
        """CONTACT_STAGE_PLAN 4.5 look-ahead, before the grasp of a transfer. System 1 points at the destination now
        (the call the put-down would make, made earlier; its answer is the put-down's first aim). Where the object
        itself can rest there is computed once (object_spots); none: early exit (no grasp changes that). Then the
        usual (top-down) grasps, the hand fitted at the pick without the arm, are judged at those spots: the open hand
        at the release first (cheap), the way in and the arm posture only after it (_grasp_puts); when none of the
        first LOOK["top_k"] can put it down, grasps from the side, the hand lowest first, never from below. Returns
        the first grasp that can put it down (R, p) as the grasp to take, or None (grasp as usual; the put-down may
        still regrasp at a staging spot, D3)."""
        import time as _time
        t0 = _time.time()
        rel = st["relation"]
        release = st.get("release") or ("drop" if rel == "in" else "place")
        drop = release == "drop"
        aim = self._v2_pose(obj, dest, st, release, [], place_pt, before_grasp=True)
        self._pending_aim = (st["id"], aim)
        t1 = _time.time()
        hint = self._entry_hint(aim.get("entry"))
        full = np.asarray(self.payload_model(obj)[0], float)
        o0 = np.r_[obj.footprint_center(), float(full[:, 2].min())]
        P_obj = full - o0
        spots, why = self.object_spots(obj, dest, rel, aim, P_obj, drop)
        looked = False
        if not spots and why.get("unobserved", 0) and self._look_at_rest(aim, dest, hint):
            looked = True                     # the hand is free now: look at the spot, then judge again
            spots, why = self.object_spots(obj, dest, rel, aim, P_obj, drop)
        timing = lambda: {"s_model": round(t1 - t0, 1), "s_geometry": round(_time.time() - t1, 1)}
        if not spots:
            self.log("lookahead", grasp=None, spots=0, looked=looked, rejected=why, **timing())
            return None
        R0, _ = self.rb.eef_pose()
        cur_yaw = float(np.arctan2(R0[1, 0], R0[0, 0]))
        tried, skipped = [], []
        # a coarse preview of the usual grasps (12 turns, 2 depths, 3 tilts: under a second, against ~10 s for the
        # full set); the grasp taken is then chosen near the one judged here
        preview = manip.grasp_candidates(obj, obj.support_z, cur_yaw=cur_yaw, n_yaw=12, depth_options=(0.008, 0.028),
                                         tilts=(0.0, np.deg2rad(20), -np.deg2rad(20)), xy_step=0.015, voxel=0.005)
        for family, cands in (("top", preview), ("side", None)):
            if family == "side":
                cands = [g for g in self._side_cands(obj) if float(g.R[2, 2]) <= 0.25]
                if cands:
                    hh = self.hand_height(cands, obj.top_z)
                    cands = [cands[i] for i in np.lexsort((-np.array([g.score for g in cands]), np.round(hh, 2)))]
            n = 0
            for g in cands:
                if n >= LOOK[family + "_k"]:
                    break
                self._pick_why = None
                if self._hand_fits_pick(obj, g.R, g.p, obj.points, g.width, side=g.info.get("side")) is None:
                    if family == "side" and len(skipped) < 12:
                        skipped.append({"approach": np.round(g.R[:, 2], 2).tolist(), "p": np.round(g.p, 3).tolist(),
                                        "pick": self._pick_why})
                    continue
                n += 1
                cand, why_g = self._grasp_puts(obj, g, o0, P_obj, spots, rel, drop, hint)
                tried.append({"family": family, "approach": np.round(g.R[:, 2], 2).tolist(),
                              "p": np.round(g.p, 3).tolist(),
                              "result": cand["entry"] if cand is not None else max(why_g, key=why_g.get, default=None)})
                if cand is not None:
                    self.log("lookahead", grasp=family, approach=np.round(g.R[:, 2], 2), entry=cand["entry"],
                             bounded_drop=cand.get("bounded_drop"), spots=len(spots), tried=tried, looked=looked,
                             **timing())
                    return (g.R, g.p)
        self.log("lookahead", grasp=None, spots=len(spots), tried=tried, skipped_at_pick=skipped, looked=looked,
                 n_side=len(self._side_cands(obj)), **timing())
        return None

    def _table_patch(self, xy):
        """The table top around xy as a destination (a staging spot): a virtual 30 x 30 cm patch."""
        from .perception import ObjectModel
        g = np.stack(np.meshgrid(np.arange(-0.15, 0.1501, 0.01), np.arange(-0.15, 0.1501, 0.01)), -1).reshape(-1, 2)
        table = ObjectModel(-1, "table", "table", "context", self.table_z - 0.01)
        table.set_points(np.c_[g + np.asarray(xy, float)[:2], np.full(len(g), self.table_z)])
        table.virtual = True
        return table

    GEOM_3D = ("overhang", "hand_meets", "no_arm_posture", "object_overlaps")

    def _geometry_blocked(self, why):
        """The rest-pose search failed in its 3D stage (hand, entry, arm), not on where the object would rest: at least
        8 spots reached that stage and the hand/entry/arm reasons are most of its rejections (another grasp can
        change those, D3)."""
        n3 = sum(v for k, v in why.items() if k.split(" ")[0] in self.GEOM_3D)
        hand = sum(v for k, v in why.items() if k.split(" ")[0] in ("overhang", "hand_meets", "no_arm_posture"))
        return getattr(self, "_rest_info", {}).get("n3d", 0) >= 8 and hand >= 0.5 * max(n3, 1)

    def _movable_blockers(self, why, dest=None):
        """Parts of fixtures (a door, a drawer, a lid) that the rest-pose search named as being in the way of most of
        the spots that reached its 3D stage; never the destination itself (moving it moves the spot along)."""
        n3 = sum(v for k, v in why.items() if k.split(" ")[0] in self.GEOM_3D)
        names = {k.split(" ", 1)[1] for k, v in why.items() if " " in k and k.split(" ", 1)[0] in self.GEOM_3D
                 and v >= 0.5 * max(n3, 1)}
        return [m for m in self.objects.values() if m.name in names and m.part_of is not None and
                m.part_of in self.objects and not m.held and (dest is None or m.oid != dest.oid)]

    def make_room(self, obj, st, part, result):
        """D4: a part that moves (a door, a drawer, a lid) is in the way of the put-down: set the held object down where
        it was taken from, move the part to its open end stop (the general actuation, goal "open"), take the object
        again. Once per stage. Returns True when the object is held again."""
        from .harness import StageFailure
        sid = entity_names(self).get(part.oid)
        origin = getattr(self, "_grasp_origin", {}).get(obj.oid)
        if sid is None or sid not in self.id_map or origin is None:
            return False
        table = self._table_patch(origin)
        stc, _ = self.rest_poses(obj, table, "on", np.asarray(origin, float), [k * np.pi / 4 for k in range(8)], [],
                                 False, n_max=4, aim_z=self.table_z)
        if not stc:
            return False
        put = self.put_down(obj, table, stc, False)
        self.log("make_room", part=part.name, staging=put["status"])
        if put["status"] == "dropped_in_transit":
            raise StageFailure("dropped_in_transit")
        if put["status"] != "released":
            return False
        st2 = {"id": f"{st['id']}_room", "op": "actuate", "object": sid, "motion": "pull", "goal": "open",
               "relation": None, "reference": None, "turn_direction": None, "terms": []}
        try:
            self.do_actuate(st2)
            status = "moved"
        except StageFailure as f:
            status = f.reason
        result["attempts"].append({"phase": "make_room", "part": part.name, "status": status})
        self.grasp_object(obj, st, result)
        return True

    def _entry_hint(self, word):
        """System 1's entry hint as a horizontal direction from the rest pose toward where the object comes from (in
        the head camera's frame), or None ("from_above", "any")."""
        if word in (None, "any", "from_above") or self.head0 is None:
            return None
        R = self.head0.T_wc[:3, :3]
        v = {"from_image_left": -R[:, 0], "from_image_right": R[:, 0], "from_camera_side": -R[:, 2],
             "from_far_side": R[:, 2]}.get(word)
        if v is None or np.linalg.norm(v[:2]) < 1e-6:
            return None
        return np.r_[v[:2] / np.linalg.norm(v[:2]), 0.0]

    def _unified_yaws(self, fp, long_side):
        """Turns of the hand about the vertical: as v2, and every 30 degrees (twelve) when the long side does not
        matter or the hand holds the object from the side (its approach must face the way in)."""
        yaws = self._v2_yaws(fp, long_side)
        Rs, _ = self.rb.eef_pose()
        extra = (np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3, 2 * np.pi / 3, -2 * np.pi / 3, 5 * np.pi / 6,
                 -5 * np.pi / 6) if abs(float(Rs[2, 2])) < 0.9 or long_side in (None, "any") else ()
        return yaws + [y for y in extra if all(abs(y - k) > 0.05 for k in yaws)]

    def place_object_unified(self, obj, dest, st, place_pt, result):
        """Placement "unified" (CONTACT_STAGE_PLAN M2): System 1 points; geometry finds the rest poses near the spot
        and the hand-off; the way down carries its expected contact and the gripper opens only when the object meets
        its support there (or, for a drop, when it hangs over the opening within the fall bound); System 1 judges the
        result and points again when it is not right. Contacts before the window are handled locally (back off, the
        contact becomes geometry, next rest pose) without a model call."""
        from .harness import V2_TRIES, StageFailure
        rel = st["relation"]
        release = st.get("release") or ("drop" if rel == "in" else "place")
        drop = release == "drop"
        notes, aim, failed, local, looked, cleared, regrasped, roomed = [], None, [], 0, {}, False, False, False
        pend = getattr(self, "_pending_aim", None)
        if pend is not None and pend[0] == st["id"]:
            aim, self._pending_aim = pend[1], None          # pointed at before the grasp (look-ahead)
        attempt = 0
        while attempt < V2_TRIES:
            self._v2_half = self._v2_held_half()
            if aim is None:
                aim = self._v2_pose(obj, dest, st, release, notes, place_pt)
            hint = self._entry_hint(aim.get("entry"))
            fp = self._payload_world(*self.rb.eef_pose())[:, :2]
            yaws = self._unified_yaws(fp - 0.5 * (fp.min(0) + fp.max(0)), aim.get("long_side"))
            rest_xy = np.asarray(aim.get("rest_xy", aim["xy"]), float)
            cands, why = self.rest_poses(obj, dest, rel, rest_xy, yaws, failed, drop, aim_z=aim.get("z_top"),
                                         hint=hint)
            if cands and hint is not None:
                d0 = cands[0].get("entry_dir")
                self.log("entry_hint", hint=aim.get("entry"), used=cands[0]["entry"],
                         agrees=bool(d0 is not None and float(np.asarray(d0) @ hint) > 0.7))
            if not cands and why.get("unobserved", 0) and looked.get(str(aim.get("pixel")), 0) < 2:
                # the spot is not seen well enough: look (at most twice per pointed spot), then again
                looked[str(aim.get("pixel"))] = looked.get(str(aim.get("pixel")), 0) + 1
                if self._look_at_rest(aim, dest, hint):
                    looked[str(aim.get("pixel"))] = 2
                    continue
            if not cands and not regrasped and self.motion != "cp5" and self._geometry_blocked(why):
                # D3: the grasp held now has no way in here; another grasp of the object might (look-ahead), taken
                # after setting it down at a staging spot
                regrasped = True
                plan = self.plan_regrasp(obj, dest, rel, aim, drop, hint)
                if plan is not None:
                    put = self.put_down(obj, plan["table"], [plan["staging"]], False)
                    self.log("regrasp_staging", status=put["status"], xy=put.get("xy"))
                    result["attempts"].append({"phase": "regrasp", "status": put["status"],
                                               "dest_entry": plan["dest_entry"]})
                    if put["status"] == "released":
                        self.grasp_object(obj, st, result, want=plan["want"])
                        failed, local = [], 0
                        continue
                    if put["status"] == "dropped_in_transit":
                        raise StageFailure("dropped_in_transit")
                why = {**why, "regrasp: " + getattr(self, "_regrasp_why", "staging failed"): 1}
            if not cands and not roomed and self.motion != "cp5":
                parts = self._movable_blockers(why, dest)
                if parts:
                    roomed = True                     # D4: move the part out of the way, then the same spot again
                    if self.make_room(obj, st, parts[0], result):
                        failed, local = [], 0
                        continue
            if not cands:
                put = {"status": "no_rest_pose", "why": self._rest_words(why)}
            else:
                put = self.put_down(obj, dest, cands, drop)
                if put["status"] == "could_not_reach" and not cleared and self._start_side(put.get("why")):
                    cleared = True                     # blocked where the move starts: lift clear, same poses again
                    self._lift_clear()
                    put = self.put_down(obj, dest, cands, drop)
                thing = (put.get("why") or {}).get("thing") if put["status"] == "could_not_reach" else None
                parts = self._movable_blockers({f"overhang {thing}": 1}, dest) if thing and not roomed else []
                if parts and self.motion != "cp5":
                    roomed = True                      # D4: the way to the hand-off is blocked by a part that moves
                    if self.make_room(obj, st, parts[0], result):
                        failed, local = [], 0
                        continue
            note = {"attempt": attempt + 1, "aimed_at_pixel": aim.get("pixel"), "long_side": aim.get("long_side"),
                    "result": put}
            self.log("unified_put", **{k: v for k, v in put.items() if k != "measured_rest"},
                     measured_rest=put.get("measured_rest"))
            if put["status"] in ("early_contact", "no_contact", "no_line") and local < 3:
                # S2 acts first: the touched place is geometry now; the next rest pose near the same spot, no model call
                local += 1
                failed.append((np.asarray(put["xy"], float), 0.02))
                if put["status"] == "no_contact":
                    # the support was not where the model put it: look at the spot again before the next pose
                    fr = self.look_from_above(np.r_[put["xy"], self.table_z + put["planned_rest_above_table"]],
                                              height=0.28)
                    if fr is not None:
                        self.rebuild_map(self.frames[-8:])
                result["attempts"].append({"phase": "place", "status": put["status"], "xy": put["xy"]})
                notes.append(note)
                continue
            if put["status"] == "dropped_in_transit":
                raise StageFailure("dropped_in_transit")
            if put["status"] == "not_released":
                result["attempts"].append({"phase": "place", "status": "not_released"})
                notes.append(note)
                attempt += 1
                continue
            if put["status"] != "released":
                result["attempts"].append({"phase": "place", "status": put["status"], "xy": aim["xy"],
                                           "why": put.get("why")})
                notes.append(note)
                aim, local = None, 0
                attempt += 1
                continue
            look, nxt = self._v2_look(obj, dest, st, release, put, notes, relocalized=bool(put.get("measured_rest")))
            check = {"satisfied": bool(look.get("done")), "by": "system1_look", "evidence": look.get("evidence"),
                     "ended_on_contact": put.get("ended_on_contact"), "measured_rest": put.get("measured_rest")}
            if look.get("done") is None:
                check["unverified"] = True
            result["attempts"].append({"phase": "place", "status": "released", "check": check, "xy": put["xy"]})
            result["relation_check"] = check
            self.log("placed", object=obj.name, check=check)
            if check["satisfied"] or check.get("unverified"):
                return True
            notes.append({**note, "seen_after_release": look.get("evidence")})
            attempt += 1
            if attempt >= V2_TRIES:
                break
            if not put.get("measured_rest") and not look.get("refreshed") and \
                    not self.relocate_object(obj, self._v2_release_p if hasattr(self, "_v2_release_p") else
                                             np.r_[put["xy"], self.table_z]):
                raise StageFailure("object_lost_after_release", check=check, notes=notes[-2:])
            self.grasp_object(obj, st, result)
            aim, local, failed = nxt, 0, []
        raise StageFailure("placement_failed", attempts=result["attempts"][-3:], notes=notes[-3:])

    @staticmethod
    def _rest_words(why):
        """Rejection counts of the rest-pose search in words for System 1 (the most frequent first)."""
        words = {"unobserved": "the spot is not seen well enough", "outside_view": "too close to the edge of what is seen",
                 "not_inside": "the object would not be down inside the destination (on its rim or edge)",
                 "would_rest_on_something_else": "the object would rest on something else standing there, or across "
                                                 "a rim or wall",
                 "not_on_the_destination": "too little of the object would be on the destination (it would stand beside it or "
                                           "hang over its edge, or rest on something else on it)", "not_on_the_table":
                 "the spot beside the reference is not on the table", "no_arm_posture": "the arm cannot reach it"}
        out = []
        for k, n in sorted(why.items(), key=lambda kv: -kv[1]):
            head, _, thing = k.partition(" ")
            if head == "no_arm_posture" and thing:
                txt = f"the arm would meet {thing}"
            else:
                txt = words.get(head) or {"object_overlaps": f"the object would overlap {thing}",
                                          "hand_meets": f"the open hand would meet {thing}",
                                          "overhang": f"{thing} is in the way from above, and no slanted or side "
                                                      "way in is free"}.get(head, k)
            out.append(f"{txt} ({n})")
        return "no place near the pointed spot where the object can rest: " + "; ".join(out[:4])

