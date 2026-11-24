"""The TRACE contract: the only types the harness and TRACE share.

The harness depends on this module, not on TRACE's internals, so a simpler solver can stand behind the same
contract (TRACE integration.md, "The contract"). Everything here is plain data: numpy arrays in SI units
(metres, radians, seconds), world frame unless stated otherwise. The end-effector frame is the grip site
(+z = approach axis, +x = finger-closing axis).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import numpy as np

RESULTS = ("certified", "conflict", "not_certifiable", "no_path", "program_error")
GROUPS = ("palm", "finger", "link", "payload")          # body groups of the collision spheres
ROLES = ("safety", "require", "prefer")


# ----------------------------------------------------------------------------------------------- bodies
@dataclass
class Spheres:
    """Collision spheres rigid in the end-effector (grip-site) frame."""
    centers: np.ndarray            # (S,3) eef frame
    radii: np.ndarray              # (S,)
    groups: np.ndarray             # (S,) int index into GROUPS

    def select(self, group: str) -> "Spheres":
        m = self.groups == GROUPS.index(group)
        return Spheres(self.centers[m], self.radii[m], self.groups[m])

    def __add__(self, other: "Spheres") -> "Spheres":
        return Spheres(np.concatenate([self.centers, other.centers]), np.concatenate([self.radii, other.radii]),
                       np.concatenate([self.groups, other.groups]))


# ------------------------------------------------------------------------------------------ scene snapshot
@dataclass
class Entity:
    """A perceived object in one snapshot. `cov` is the 3x3 covariance (m^2) of its position estimate."""
    id: str                        # stable id (the plan's object id when there is one)
    label: int                     # its label value in SceneSnapshot.labels / .shadow (0 = none)
    points: np.ndarray             # (M,3) surface samples (subsampled, <= ~600)
    cov: np.ndarray                # (3,3)
    keypoints: dict = field(default_factory=dict)   # name -> (3,), e.g. "top_center", "base", "centroid"

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(0)

    @property
    def top_z(self) -> float:
        return float(np.percentile(self.points[:, 2], 99.5))

    def keypoint(self, name: str) -> np.ndarray:
        if name in self.keypoints:
            return np.asarray(self.keypoints[name], float)
        if name == "centroid":
            return self.centroid
        lo, hi = self.points.min(0), self.points.max(0)
        c = 0.5 * (lo + hi)
        if name == "top_center":
            return np.array([c[0], c[1], self.top_z])
        if name == "base":
            return np.array([c[0], c[1], lo[2]])
        raise KeyError(name)


@dataclass
class SceneSnapshot:
    """One versioned picture of the scene, built by the harness from perception.

    Voxel grids share origin/res/shape; voxel (i,j,k) has its centre at origin + (ijk + 0.5) * res.
    occ: observed or inferred solid (the held object is never in it). unknown: never observed free or solid.
    labels: entity label per occupied voxel (0 = unlabelled). shadow: entity label of unknown voxels hidden
    behind that entity (0 = none). The table top is the analytic plane z = table_z (not in the grids).
    exact(points (N,3), exclude: tuple[str]) -> (N,) distance (m) from points to the nearest solid or unknown
    surface, the same geometry the harness's guard uses; None for synthetic or replayed snapshots (TRACE then
    uses the grids). It never includes the table plane.
    """
    version: int
    origin: np.ndarray
    res: float
    occ: np.ndarray
    unknown: np.ndarray
    labels: np.ndarray
    shadow: np.ndarray
    table_z: float | None
    entities: dict
    exact: object = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.occ.shape

    def label_of(self, entity_ids) -> tuple:
        return tuple(self.entities[e].label for e in entity_ids if e in self.entities and self.entities[e].label)

    def save(self, path) -> None:
        ents = {k: {"label": int(e.label), "points": e.points, "cov": e.cov,
                    "keypoints": {n: np.asarray(v, float).tolist() for n, v in e.keypoints.items()}}
                for k, e in self.entities.items()}
        np.savez_compressed(path, version=self.version, origin=self.origin, res=self.res, occ=self.occ,
                            unknown=self.unknown, labels=self.labels, shadow=self.shadow,
                            table_z=np.nan if self.table_z is None else self.table_z,
                            entities=np.array(json.dumps({k: {"label": v["label"], "cov": np.asarray(v["cov"]).tolist(),
                                                              "keypoints": v["keypoints"]} for k, v in ents.items()})),
                            meta=np.array(json.dumps(to_jsonable(self.meta))),
                            **{f"pts__{k}": v["points"] for k, v in ents.items()})

    @staticmethod
    def load(path) -> "SceneSnapshot":
        z = np.load(path, allow_pickle=False)
        ents = json.loads(str(z["entities"]))
        entities = {k: Entity(k, int(v["label"]), z[f"pts__{k}"], np.asarray(v["cov"]),
                              {n: np.asarray(p) for n, p in v["keypoints"].items()}) for k, v in ents.items()}
        tz = float(z["table_z"])
        return SceneSnapshot(int(z["version"]), z["origin"], float(z["res"]), z["occ"], z["unknown"], z["labels"],
                             z["shadow"], None if np.isnan(tz) else tz, entities, None, json.loads(str(z["meta"])))


# ------------------------------------------------------------------------------------------------- terms
@dataclass
class Term:
    """One requirement: residual (from the primitive library) + tolerance + role (TRACE_method.md §5.1).

    args: primitive arguments; entity references are Entity ids, numbers are SI. sigma is in the residual's unit.
    Fields after `src` are integration metadata: `entity` + `effect` drive uncertainty fusion (§5.5: an
    "attract" term is loosened, a "repel" term tightened, by that entity's covariance); `level`/`relax_to` are the
    named level the term came from and the looser level the plan allows the sequencer to switch to; `param`/`unit`
    name the args entry a conflict quote is expressed in.
    """
    id: str
    prim: str
    args: dict
    shape: str                     # "eq" | "ineq"
    role: str                      # "safety" | "require" | "prefer"
    scope: object = "path"         # "path" | "terminal" | [a, b] (fraction of the segment's parameter)
    sigma: float = 0.01
    kappa: float = 1.0
    src: str = "goal"              # "goal" | "perception" | "language" | "regularizer" | "robot"
    entity: str | None = None
    effect: str | None = None      # "attract" | "repel" | None
    level: str | None = None
    relax_to: str | None = None
    param: str | None = None
    unit: str | None = None


# ---------------------------------------------------------------------------------------------- request
@dataclass
class EndCandidate:
    """One acceptable end of the segment, ranked by the harness (first = preferred).

    (R, p) is the contact pose when `approach` is given (grasp / place / press pose) and the segment then ends at
    a standoff before it: p_end = p - standoff * approach. standoff "auto" = computed by TRACE from the entity's
    uncertainty along `approach`, the tracking tube and the guard distance (TRACE integration.md, "Where the
    boundary sits"); a number = fixed; 0 = the segment ends at (R, p) itself; (s_min, s_max) = slide: the hand-off
    sits where certification ends, i.e. the smallest standoff from max(auto, s_min) up to s_max (1 cm steps) whose
    end pose passes the end test (search.end_test) and has an arm posture; the guarded move covers the rest. `window` (m) is the lateral
    tolerance of the contact pose (perpendicular to approach): when perception plus tracking error exceed it the
    end is "not certifiable with this perception" (look closer). q: arm configuration at (R, p) (posture target of
    the arm rollout). q_path/p_path/R_path: an optional seed path (e.g. from an existing joint-space planner),
    resampled by arc length.
    """
    R: np.ndarray
    p: np.ndarray
    q: np.ndarray | None = None
    approach: np.ndarray | None = None
    standoff: object = 0.0
    window: float | None = None
    entity: str | None = None
    q_path: np.ndarray | None = None
    p_path: np.ndarray | None = None
    R_path: np.ndarray | None = None


@dataclass
class TraceRequest:
    """One contact-free segment. `terms` are fully resolved (the harness's level table has already turned named
    levels into tolerances) and include the harness-written safety terms. margins = the margin ladder:
    {"guard": {group: m}, "guard_plane": {group: m}, "perception": m, "tube0": m, "tube_k": s, "v_min": m/s}
    with tracking tube(v) = tube0 + tube_k * v. limits = {"v_max", "a_max", "w_max", "dt", "v_slow"}.
    free_end (optional, used when `ends` is empty): {"entity", "relation": "above" | "at", "height", "approach"}
    for TRACE's own end-pose solve (§6.1). mode: "trace" (fast path, then search) | "v0" (fast path, then the
    provided seed paths only) | "straight" (fast path only).
    """
    request_id: str
    scene_version: int
    seed: int
    budget_ms: float
    start_R: np.ndarray
    start_p: np.ndarray
    start_q: np.ndarray
    finger: float
    holding: str | None
    ends: list
    terms: list
    bodies: Spheres
    margins: dict
    limits: dict
    free_end: dict | None = None
    mode: str = "trace"
    zone: float = 0.05             # "do not get closer" zone at both ends of the path (m)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------- callbacks
@dataclass
class ArmResult:
    """Answer of the harness's arm check (TRACE_method.md §7.2 as changed: the guard's controller model rolled
    along the path, plus whole-arm clearance and joint limits, all on the harness's geometry)."""
    ok: bool
    q: np.ndarray | None           # (N,7) rolled-out joints (the posture reference used in execution)
    reason: str | None = None      # "joint_limit" | "unreachable" | "arm_clearance" | "discontinuous"
    at_t: float | None = None      # path parameter of the first failure
    min_clearance: float | None = None
    clearance: np.ndarray | None = None   # (N,) per-sample arm clearance minus margin (for retiming), optional
    detail: dict | None = None            # typed reason: {"joint", "q"} | {"at", "clearance"} | {"err_p", "err_r"}

# arm_check(R (N,3,3), p (N,3), s (N,) path parameter in [0,1], q_ref (N,7)) -> ArmResult
# ik_check(R (3,3), p (3,), q_seed (7,) | None) -> (ok: bool, q: (7,) | None)          (free ends only)


# --------------------------------------------------------------------------------------------- response
@dataclass
class Timed:
    """The executable trajectory: samples on the control grid (t[0] = 0, step limits["dt"])."""
    t: np.ndarray                  # (K,)
    p: np.ndarray                  # (K,3)
    R: np.ndarray                  # (K,3,3)
    q: np.ndarray                  # (K,7) posture reference (arm rollout)
    tube: np.ndarray               # (K,) allowed tracking error (m) at each sample
    speed: np.ndarray              # (K,) planned eef speed (m/s)


@dataclass
class TraceResponse:
    request_id: str
    result: str                    # one of RESULTS
    timed: Timed | None = None
    spline: dict | None = None     # SE3Spline.to_dict() of the certified path (logging / replay)
    end_index: int | None = None   # which EndCandidate the path ends at
    end_R: np.ndarray | None = None
    end_p: np.ndarray | None = None
    standoff: float | None = None
    valid_while: dict | None = None    # {"scene_version", "tube_mm", "holding", "start_q"}
    report: list = field(default_factory=list)   # [{term, prim, role, src, level, max_violation, at_t, kappa, certifiable}]
    quote: list | None = None      # conflict: [{term, prim, level, relax_to, param, value, needed, unit}]
    look_at: dict | None = None    # not_certifiable: {entity, term, uncertainty, tolerance, axis}
    findings: list | None = None   # program_error: [{test, term, detail}]
    stats: dict = field(default_factory=dict)    # {stage: fast_path|search|seed, solve_ms, certify_ms, evals, ...,
    #   ends_dropped: [{end, why: "blocked"|"no_posture", group, clearance, at}], arm: last arm-check refusal,
    #   end_unseen: {group, depth, at} when the chosen end overlaps unseen space}


# ----------------------------------------------------------------------------------------------- utils
def to_jsonable(o):
    """JSON-friendly copy (arrays -> rounded lists, dataclasses -> dicts, callables dropped)."""
    if is_dataclass(o) and not isinstance(o, type):
        return {k: to_jsonable(v) for k, v in asdict(o).items() if not callable(v)}
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if not callable(v)}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return np.round(o.astype(float), 5).tolist() if o.dtype.kind == "f" else o.tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o
