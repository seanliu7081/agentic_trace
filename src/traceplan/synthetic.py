"""Analytic scenes for tests and the bench (TRACE_method.md §8.3): boxes and spheres with an exact SDF, the
Panda-like default hand, the standard scenes S1-S6 and a trivial arm check.

Frames: world z up, table plane z = table_z. The grip-site frame has +z = approach, +x = finger-closing axis; the
"pointing down" pose R_DOWN maps eef z to world -z.
"""
from __future__ import annotations

import numpy as np

from .contract import GROUPS, ArmResult, EndCandidate, Entity, SceneSnapshot, Spheres, Term, TraceRequest

R_DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
MARGINS = {"guard": {"arm": 0.010, "palm": 0.008, "finger": 0.005, "link": 0.010, "payload": 0.005},
           "guard_plane": {"arm": 0.015, "palm": 0.008, "finger": 0.004, "link": 0.015, "payload": 0.003},
           "perception": 0.005, "tube0": 0.004, "tube_k": 0.022, "v_min": 0.05}
LIMITS = {"v_max": 0.40, "a_max": 0.5, "w_max": 1.2, "dt": 0.05, "v_slow": 0.05}
SAFETY_SIGMA = 0.002          # clearance tolerance; margins add it so v <= 1 still keeps the ladder distance (§5.1)
SHOULDER = (-0.1, 0.0, 0.333)  # synthetic robot base at (-0.1, 0, table) plus the Panda shoulder height
REACH = 0.855


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0], [0, 0, 1.0]])


# --------------------------------------------------------------------------------------------- shapes
def _sdf_box(x, b):
    q = (x - np.asarray(b["center"], float)) @ rotz(b.get("yaw", 0.0))      # world -> box frame (row vectors)
    d = np.abs(q) - np.asarray(b["half"], float)
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


def _sdf_sphere(x, s):
    return np.linalg.norm(x - np.asarray(s["center"], float), axis=-1) - s["radius"]


def _box_surface(b, step=0.01):
    h = np.asarray(b["half"], float)
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        g = [np.linspace(-h[i], h[i], max(int(np.ceil(2 * h[i] / step)) + 1, 2)) for i in o]
        uv = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 2)
        for sgn in (-1, 1):
            P = np.zeros((len(uv), 3))
            P[:, o] = uv
            P[:, ax] = sgn * h[ax]
            pts.append(P)
    P = np.concatenate(pts) @ rotz(b.get("yaw", 0.0)).T + np.asarray(b["center"], float)
    return P[np.linspace(0, len(P) - 1, min(600, len(P))).astype(int)]


def _sphere_surface(s, n=400):
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    u = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
    return np.asarray(s["center"], float) + s["radius"] * u


def box_scene(boxes=(), spheres=(), table_z=0.0, bounds=((0.05, -0.45, -0.02), (0.85, 0.45, 0.62)), res=0.01,
              cov=0.003, unknown=(), version=1):
    """Analytic scene. boxes: [{id, center, half, yaw?}], spheres: [{id, center, radius}], unknown: boxes of
    unobserved space [{center, half, yaw?, shadow_of?}]. cov: isotropic std (m) for every entity, or {id: (3,3)}.
    `exact` is the analytic SDF of the solids (the table plane is a separate primitive)."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    shape = tuple(int(n) for n in np.ceil((hi - lo) / res - 1e-9))
    C = lo + (np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) + 0.5) * res
    shapes = [(b, _sdf_box, _box_surface, b["half"][2]) for b in boxes] + \
             [(s, _sdf_sphere, _sphere_surface, s["radius"]) for s in spheres]
    occ, labels = np.zeros(shape, bool), np.zeros(shape, np.int16)
    entities = {}
    for i, (d, sdf, surf, hz) in enumerate(shapes):
        inside = sdf(C, d) <= 0
        occ |= inside
        labels[inside] = i + 1
        c = np.asarray(d["center"], float)
        S = np.asarray(cov[d["id"]], float) if isinstance(cov, dict) else np.eye(3) * cov ** 2
        entities[d["id"]] = Entity(d["id"], i + 1, surf(d), S, {"top_center": c + [0, 0, hz], "base": c - [0, 0, hz],
                                                                "centroid": c})
    unk, shadow = np.zeros(shape, bool), np.zeros(shape, np.int16)
    unknown_boxes = list(unknown)
    for u in unknown:
        m = _sdf_box(C, u) <= 0
        unk |= m & ~occ
        if u.get("shadow_of"):
            shadow[m & ~occ] = entities[u["shadow_of"]].label

    def exact(pts, exclude=(), unknown=False):
        pts = np.asarray(pts, float).reshape(-1, 3)
        ds = [sdf(pts, d) for d, sdf, _, _ in shapes if d["id"] not in exclude]
        if unknown:
            ds += [_sdf_box(pts, u) for u in unknown_boxes if u.get("shadow_of") not in exclude]
        return np.min(ds, axis=0) if ds else np.full(len(pts), 1.0)

    return SceneSnapshot(version, lo, float(res), occ, unk, labels, shadow, table_z, entities, exact, {})


# --------------------------------------------------------------------------------------------- bodies
def _box_spheres(lo, hi, cells, group):
    """Cover a box by the circumscribed spheres of a cells[0] x cells[1] x cells[2] partition."""
    lo, hi, n = np.asarray(lo, float), np.asarray(hi, float), np.asarray(cells)
    h = (hi - lo) / n
    c = np.stack(np.meshgrid(*[lo[i] + (np.arange(n[i]) + 0.5) * h[i] for i in range(3)], indexing="ij"),
                 -1).reshape(-1, 3)
    return Spheres(c, np.full(len(c), 0.5 * np.linalg.norm(h)), np.full(len(c), GROUPS.index(group)))


def default_hand(opening=0.08):
    """Panda-like hand in the grip-site frame: palm box, two finger boxes, link-7 cylinder."""
    palm = _box_spheres((-0.10, -0.03, -0.11), (0.10, 0.03, -0.031), (8, 2, 3), "palm")
    xf = opening / 2 + 0.01
    fingers = [_box_spheres((s * xf - 0.01, -0.01, -0.045), (s * xf + 0.01, 0.01, 0.009), (2, 2, 4), "finger")
               for s in (-1, 1)]
    z = np.linspace(-0.11, -0.25, 9)[1::2]                      # 4 spheres, spacing 0.035
    link = Spheres(np.stack([np.zeros(4), np.zeros(4), z], -1), np.full(4, np.hypot(0.045, 0.0175)),
                   np.full(4, GROUPS.index("link")))
    return palm + fingers[0] + fingers[1] + link


def payload_column(radius, z_lo, z_hi, step=0.025):
    """An upright cylinder held between the fingers (axis = eef z), as spheres of group payload."""
    z = np.arange(z_lo + radius * 0.5, z_hi - radius * 0.5 + 1e-9, step)
    return Spheres(np.stack([np.zeros_like(z), np.zeros_like(z), z], -1), np.full(len(z), radius),
                   np.full(len(z), GROUPS.index("payload")))


# --------------------------------------------------------------------------------------------- terms
def tube_vmin(m):
    return m["tube0"] + m["tube_k"] * m["v_min"]


def safety_terms(bodies, holding, table_z, margins=MARGINS, zone=0.05):
    """What the harness writes (levels.safety_terms): clearance and plane clearance per group, reach limit and the
    smoothness regularizers."""
    groups = [g for g in GROUPS if (bodies.groups == GROUPS.index(g)).any()]
    tv = tube_vmin(margins)
    excl = [holding] if holding else []
    out = []
    for g in groups:
        out.append(Term(f"clear_{g}", "clearance", {"body": g, "field": {"exclude": excl}, "zone": zone,
                                                     "margin": margins["guard"][g] + tv + margins["perception"]
                                                     + SAFETY_SIGMA}, "ineq", "safety", sigma=SAFETY_SIGMA,
                        src="perception", param="margin", unit="m"))
        if table_z is not None:
            out.append(Term(f"plane_{g}", "plane_clearance", {"body": g, "z": table_z, "zone": zone,
                                                              "margin": margins["guard_plane"][g] + tv + SAFETY_SIGMA},
                            "ineq", "safety", sigma=SAFETY_SIGMA, src="perception", param="margin", unit="m"))
    out.append(Term("reach", "reach_limit", {"center": list(SHOULDER), "r_max": REACH}, "ineq", "safety",
                    sigma=0.01, src="robot", param="r_max", unit="m"))
    out.append(Term("smooth_pos", "smooth_pos", {}, "eq", "prefer", sigma=1.0, src="regularizer"))
    out.append(Term("smooth_rot", "smooth_rot", {}, "eq", "prefer", sigma=4.0, src="regularizer"))
    return out


def tilt_term(level="upright", role="require", relax_to="tilted", max_deg=None, sigma_deg=None, tid="upright"):
    lv = {"upright": (10, 3), "tilted": (25, 5), "loose": (45, 8)}[level]
    mx, sg = (lv[0] if max_deg is None else max_deg), (lv[1] if sigma_deg is None else sigma_deg)
    return Term(tid, "tilt_limit", {"u": "payload.up", "v": "world.up", "max": np.deg2rad(mx), "u0": [0.0, 0, 1]},
                "ineq", role, sigma=np.deg2rad(sg), src="language", level=level, relax_to=relax_to, param="max",
                unit="rad")


def trivial_arm_check(R, p, s, q_ref):
    """Always reachable; the rolled-out joints are the reference (linear from start_q to the end q)."""
    return ArmResult(True, np.array(q_ref, float))


# --------------------------------------------------------------------------------------------- scenes
def _scene(name, snap, p0, pe, R0=R_DOWN, Re=R_DOWN, holding=None, payload=None, task_terms=(), opening=0.08):
    bodies = default_hand(opening) + payload if payload is not None else default_hand(opening)
    return {"name": name, "snap": snap, "start": {"R": R0, "p": np.asarray(p0, float), "q": np.zeros(7)},
            "ends": [EndCandidate(Re, np.asarray(pe, float), q=np.zeros(7))], "bodies": bodies, "holding": holding,
            "finger": opening, "terms": safety_terms(bodies, holding, snap.table_z) + list(task_terms)}


def standard_scenes():
    mug = payload_column(0.035, -0.02, 0.09)
    S = {}
    S["S1"] = _scene("S1", box_scene(boxes=[{"id": "block", "center": [0.6, 0.3, 0.05], "half": [0.04, 0.04, 0.05]}]),
                     [0.30, -0.20, 0.30], [0.45, 0.15, 0.22])
    S["S2"] = _scene("S2", box_scene(boxes=[{"id": "box", "center": [0.40, 0.0, 0.17], "half": [0.06, 0.05, 0.17]}]),
                     [0.40, -0.25, 0.22], [0.40, 0.25, 0.22])
    # S3: a shelf board at z = 0.24..0.26 with a hole a little wider than the hand (palm x +-0.12, link r 0.048)
    hole_x, hole_y, zb = (0.29, 0.61), (-0.09, 0.09), (0.24, 0.26)
    board = [((0.15, hole_x[0]), (-0.35, 0.35)), ((hole_x[1], 0.75), (-0.35, 0.35)),
             (hole_x, (-0.35, hole_y[0])), (hole_x, (hole_y[1], 0.35))]
    boxes = [{"id": f"shelf_{i}", "center": [np.mean(x), np.mean(y), np.mean(zb)],
              "half": [np.ptp(x) / 2, np.ptp(y) / 2, np.ptp(zb) / 2]} for i, (x, y) in enumerate(board)]
    S["S3"] = _scene("S3", box_scene(boxes=boxes), [0.30, 0.22, 0.44], [0.45, 0.0, 0.12])
    # S4: upright carry over clutter that the payload would hit on the straight line
    clutter = [{"id": "c1", "center": [0.36, -0.06, 0.08], "half": [0.05, 0.04, 0.08]},
               {"id": "c2", "center": [0.30, 0.08, 0.06], "half": [0.04, 0.05, 0.06], "yaw": 0.4},
               {"id": "c3", "center": [0.47, 0.02, 0.07], "half": [0.04, 0.04, 0.07]}]
    S["S4"] = _scene("S4", box_scene(boxes=clutter), [0.32, -0.26, 0.20], [0.40, 0.26, 0.20], holding="mug",
                     payload=mug, task_terms=[tilt_term("upright")], opening=0.07)
    # S5: carry with an attached payload: the hand clears the can, the payload does not (recorded mug/can case)
    S["S5"] = _scene("S5", box_scene(boxes=[{"id": "can", "center": [0.38, 0.0, 0.06], "half": [0.035, 0.035, 0.06]}]),
                     [0.38, -0.22, 0.20], [0.38, 0.22, 0.20], holding="mug", payload=mug,
                     task_terms=[tilt_term("tilted", role="prefer", relax_to=None)], opening=0.07)
    # S6: contradictory: tilt <= 5 deg while the pinned end is tilted 40 deg from the start
    S["S6"] = _scene("S6", box_scene(), [0.35, -0.15, 0.30], [0.40, 0.15, 0.30], Re=rotx(np.deg2rad(40)) @ R_DOWN,
                     holding="cup", payload=mug, task_terms=[tilt_term(max_deg=5, sigma_deg=2)], opening=0.07)
    return S


def make_request(sc, request_id=None, seed=0, budget_ms=2000.0, mode="trace", ends=None, terms=None, **kw):
    return TraceRequest(request_id or f"{sc['name']}_{seed}", sc["snap"].version, seed, budget_ms,
                        sc["start"]["R"], sc["start"]["p"], sc["start"]["q"], sc["finger"], sc["holding"],
                        sc["ends"] if ends is None else ends, sc["terms"] if terms is None else terms, sc["bodies"],
                        MARGINS, LIMITS, mode=mode, **kw)
