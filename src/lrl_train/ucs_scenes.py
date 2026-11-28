"""Procedural MJCF scene builders for Unified Contact Stage (UCS) training and unseen Gate 1 testing.

Covers the contact regimes missing from pure 1-DoF handle generators:
  1. `ucs_lever_knob`: centered knobs AND offset rotary levers (teaches `r >= 0` rotary kinematics without seeing
     `faucet` or `basin_faucet`).
  2. `ucs_cavity_box`: side-opening roofed cavities with optional partial side door + free 6-DoF rigid payload
     (`ucs_payload`) for constrained horizontal slide-in & set-down.
  3. `ucs_overhang_drawer`: open lower drawer/container beneath a protruding upper overhang/handle + free 6-DoF
     rigid payload (`ucs_payload`) for oblique/down-then-in placement.
  4. Unseen test-only variants (`unseen_arch=True`, `unseen_bin=True`) and `payload_xml` for `short_fridge` /
     `short_cabinet` unseen benchmark evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import scenes


@dataclass
class ContainerTarget:
    """Ground-truth target volume for evaluator scoring & critic/reward computation (never seen by actor)."""
    kind: str                    # "cavity_insert" | "overhang_place"
    fixture_body: str
    floor_body: str
    payload_body: str
    payload_half_r: float        # horizontal half-extent (m)
    payload_half_h: float        # vertical half-height (m)
    # Local coordinates in `fixture_body` frame (transformed to world once model is built):
    floor_center_local: np.ndarray
    floor_half_xy: tuple[float, float]
    open_normal_local: np.ndarray    # unit outward normal of cavity mouth (or outward front of overhang drawer)
    ceiling_gap: float               # vertical clearance from floor to ceiling/overhang (m)
    # Populated after StageSim loads the XML:
    floor_center_world: np.ndarray | None = None
    R_world: np.ndarray | None = None
    open_normal_world: np.ndarray | None = None
    sill_height: float = 0.


def _v(arr) -> str:
    return " ".join(f"{float(x):.4f}" for x in arr)


def _box_geom(name: str, lo, hi, rgba: str = "0.55 0.52 0.48 1", density: float = 600.0) -> str:
    lo_a, hi_a = np.asarray(lo, float), np.asarray(hi, float)
    mid = 0.5 * (lo_a + hi_a)
    half = np.maximum(0.5 * (hi_a - lo_a), 0.002)
    return (
        f'<geom name="{name}" type="box" pos="{_v(mid)}" size="{_v(half)}" '
        f'rgba="{rgba}" density="{density:.0f}" friction="0.9 0.02 0.001"/>'
    )


def _flat_support_contacts(geoms: str) -> str:
    """Add interior witnesses tangent to existing coplanar support faces.

    MuJoCo's default convex collision can generate a single contact for a
    broad base resting on a plane. Four coplanar contacts support both tilt
    axes. The witnesses are invisible, massless and inside the original solid.
    """
    import mujoco
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{geoms}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rotation = data.geom_xmat.reshape(-1, 3, 3)
    centers = data.geom_xpos + np.einsum("nij,nj->ni", rotation, model.geom_aabb[:, :3])
    extents = np.einsum("nij,nj->ni", np.abs(rotation), model.geom_aabb[:, 3:])
    floor = np.min(centers[:, 2] - extents[:, 2])
    up = np.array([0., 0., 1.])
    radius = .002
    xml = []
    for g in np.flatnonzero(centers[:, 2] - extents[:, 2] <= floor + 1e-7):
        for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            xy = centers[g, :2] + .60 * extents[g, :2] * signs
            # Inscribe a box enclosing the sphere. Its entire lower face lies
            # on the support plane; each upper corner must still be in the
            # same convex component. This also excludes rounded feet.
            flat = True
            for corner in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                pos = np.r_[xy + radius*np.array(corner), floor - radius]
                args = data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g]
                low = mujoco.mju_rayGeom(*args, pos, up, model.geom_type[g])
                top = mujoco.mju_rayGeom(*args, pos + .5*up, -up, model.geom_type[g])
                if abs(low-radius) > 1.e-7 or top < 0. or .5 - top < 3*radius:
                    flat = False
                    break
            if flat:
                center = np.r_[xy, floor + radius]
                position = " ".join(format(value, ".17g") for value in center)
                friction = " ".join(format(value, ".17g") for value in model.geom_friction[g])
                xml.append(f'<geom name="ucs_support_{len(xml)}" type="sphere" size="{radius}" '
                           f'pos="{position}" friction="{friction}" '
                           'rgba="0 0 0 0" mass="0"/>')
    return "".join(xml)


def make_payload_xml(
    rng: np.random.Generator,
    kind: str = "mug",
    spawn_xyz=(0.0, 0.32, 0.96),
    yaw: float | None = None,
) -> tuple[str, float, float]:
    """Build a free 6-DoF rigid object (`ucs_payload`) with realistic graspable geometry & mass (0.06 - 0.55 kg).
    Supports `mug`, `bowl`, `plate`, `thin_book`, and `box`. Returns `(mjcf_xml, half_r, half_h)`."""
    rho = float(rng.uniform(250.0, 750.0))
    if kind == "mug":
        r = float(rng.uniform(0.026, 0.036))
        hh = float(rng.uniform(0.032, 0.046))
        # Cylinder body + side handle block so side grasps and rim grasps both work physically
        geoms = (
            f'<geom name="ucs_payload_cyl" type="cylinder" pos="0 0 0" size="{r:.4f} {hh:.4f}" '
            f'rgba="0.90 0.80 0.25 1" density="{rho:.0f}" friction="1.2 0.03 0.001"/>'
            f'<geom name="ucs_payload_handle" type="box" pos="{r + 0.012:.4f} 0 0" size="0.012 0.006 {0.65 * hh:.4f}" '
            f'rgba="0.92 0.92 0.92 1" density="{rho:.0f}" friction="1.4 0.03 0.001"/>'
        )
        half_r = r + 0.012
    elif kind == "bowl":
        # Approximated with flat bottom + octagonal/box rim walls so grasps on rim or sides hold firmly
        r = float(rng.uniform(0.038, 0.056))
        hh = float(rng.uniform(0.022, 0.034))
        wt = 0.008
        geoms = (
            _box_geom("ucs_payload_base", [-r, -r, -hh], [r, r, -hh + wt], "0.18 0.18 0.20 1", rho)
            + _box_geom("ucs_payload_wx0", [-r, -r, -hh + wt], [-r + wt, r, hh], "0.18 0.18 0.20 1", rho)
            + _box_geom("ucs_payload_wx1", [r - wt, -r, -hh + wt], [r, r, hh], "0.18 0.18 0.20 1", rho)
            + _box_geom("ucs_payload_wy0", [-r + wt, -r, -hh + wt], [r - wt, -r + wt, hh], "0.18 0.18 0.20 1", rho)
            + _box_geom("ucs_payload_wy1", [-r + wt, r - wt, -hh + wt], [r - wt, r, hh], "0.18 0.18 0.20 1", rho)
        )
        half_r = r
    elif kind == "plate":
        # Wide flat plate/dish for non-prehensile planar pushing (matches LIBERO flat plate geometry)
        r = float(rng.uniform(0.062, 0.086))
        hh = float(rng.uniform(0.008, 0.013))
        geoms = (
            f'<geom name="ucs_payload_plate" type="cylinder" pos="0 0 0" size="{r:.4f} {hh:.4f}" '
            f'rgba="0.88 0.90 0.92 1" density="{rho:.0f}" friction="0.45 0.01 0.001"/>'
        )
        half_r = r
    elif kind == "thin_book":
        # Narrow standing book / pack slab (2.2 - 4.2 cm thickness, matches caddy compartment tasks)
        rx = float(rng.uniform(0.011, 0.021))
        ry = float(rng.uniform(0.044, 0.062))
        hh = float(rng.uniform(0.044, 0.060))
        geoms = _box_geom("ucs_payload_book", [-rx, -ry, -hh], [rx, ry, hh], "0.68 0.24 0.22 1", rho)
        half_r = max(rx, ry)
    else:
        rx = float(rng.uniform(0.022, 0.035))
        ry = float(rng.uniform(0.028, 0.048))
        hh = float(rng.uniform(0.030, 0.048))
        geoms = _box_geom("ucs_payload_box", [-rx, -ry, -hh], [rx, ry, hh], "0.25 0.50 0.78 1", rho)
        half_r = max(rx, ry)

    geoms += _flat_support_contacts(geoms)
    euler_attr = "" if yaw is None else f' euler="0 0 {float(yaw):.5f}"'
    xml = f'<body name="ucs_payload" pos="{_v(spawn_xyz)}"{euler_attr}><freejoint name="ucs_payload_free"/>{geoms}</body>'
    return xml, half_r, hh


def _mount_pose(rng: np.random.Generator, x_back: float, y_half: float, top_h: float, yaw_deg: float = 55.0) -> tuple[np.ndarray, float, str]:
    """Sample a reachable tabletop pose facing roughly toward the Panda base `(-0.66, 0.0)`."""
    base = np.array(scenes.CFG["base_xy"], dtype=float)
    for _ in range(100):
        r = float(rng.uniform(0.46, 0.66))
        bearing = float(rng.uniform(-1.0, 1.0) * np.deg2rad(32.0))
        xy = base + r * np.array([np.cos(bearing), np.sin(bearing)])
        yaw = bearing + np.pi + float(rng.uniform(-1.0, 1.0) * np.deg2rad(yaw_deg))
        c, s = np.cos(yaw), np.sin(yaw)
        corners = xy + np.array([[x * c - y * s, x * s + y * c] for x in (x_back, 0.04) for y in (-y_half, y_half)])
        if np.linalg.norm(corners - base, axis=1).min() >= 0.28:
            attr = f'pos="{xy[0]:.4f} {xy[1]:.4f} {scenes.CFG["table_z"]:.4f}" euler="0 0 {yaw:.5f}"'
            return xy, yaw, attr
    xy = np.array([-0.10, 0.0])
    yaw = np.pi
    return xy, yaw, f'pos="{xy[0]:.4f} {xy[1]:.4f} {scenes.CFG["table_z"]:.4f}" euler="0 0 {yaw:.5f}"'


def ucs_lever_knob(rng: np.random.Generator, name: str = "ucs_valve"):
    """Procedural rotary knob OR offset lever valve (teaches both r=0 knobs and r>0 faucet-like levers)."""
    is_lever = bool(rng.random() < 0.55)
    th = float(np.round(rng.uniform(np.deg2rad(65.0), np.deg2rad(165.0)), 4))
    rho = float(rng.uniform(400.0, 900.0))
    damp = float(rng.uniform(0.05, 2.5))
    fric = float(rng.uniform(0.0, 0.35))

    if rng.random() < 0.65:
        # Horizontal mount (vertical rotation axis +z, like stove / sink / basin faucet)
        W, D, H_base = float(rng.uniform(0.16, 0.28)), float(rng.uniform(0.16, 0.26)), float(rng.uniform(0.03, 0.11))
        n = np.array([0.0, 0.0, 1.0])
        u = np.array([0.0, 1.0, 0.0])
        pivot = np.array([-0.04, float(rng.uniform(-0.03, 0.03)), H_base + 0.004])
        base_geom = _box_geom(f"{name}_mount", [-D, -W / 2, 0.0], [0.0, W / 2, H_base], "0.35 0.35 0.38 1", rho)
    else:
        # Front vertical panel mount (horizontal rotation axis +x)
        W, D, H_base = float(rng.uniform(0.18, 0.28)), float(rng.uniform(0.10, 0.18)), float(rng.uniform(0.14, 0.24))
        z0 = float(rng.uniform(0.0, 0.08))
        n = np.array([1.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0])
        pivot = np.array([0.004, 0.0, z0 + 0.5 * H_base])
        base_geom = _box_geom(f"{name}_mount", [-D, -W / 2, z0], [0.0, W / 2, z0 + H_base], "0.35 0.35 0.38 1", rho)

    v = np.cross(n, u)
    if is_lever:
        lever_len = float(rng.uniform(0.055, 0.105))
        bar_w = float(rng.uniform(0.010, 0.017))
        stem_h = float(rng.uniform(0.025, 0.048))
        # Stem along n, lever arm extending along u!
        geoms = (
            f'<geom name="{name}_stem" type="cylinder" fromto="0 0 0 {_v(n * stem_h)}" size="{bar_w:.4f}" '
            f'rgba="0.75 0.75 0.78 1" density="{rho:.0f}"/>'
            + _box_geom(
                f"{name}_lever",
                n * (stem_h - 0.008) - bar_w * v - 0.01 * u,
                n * (stem_h + 0.008) + bar_w * v + lever_len * u,
                "0.82 0.82 0.85 1",
                rho,
            )
        )
        hp_local = n * stem_h + (0.65 * lever_len) * u
        h_axis = u
        h_half = (0.45 * lever_len, bar_w)
    else:
        r = float(rng.uniform(0.015, 0.028))
        h = float(rng.uniform(0.020, 0.042))
        geoms = (
            f'<geom name="{name}_cap" type="cylinder" fromto="0 0 0 {_v(n * h)}" size="{r:.4f}" '
            f'rgba="0.25 0.25 0.28 1" density="{rho:.0f}" friction="1.4 0.03 0.001"/>'
            + _box_geom(f"{name}_fin", n * (0.4 * h) - r * u - 0.006 * v, n * (h + 0.008) + r * u + 0.006 * v, "0.85 0.85 0.85 1", rho)
        )
        hp_local = n * (0.65 * h)
        h_axis = u
        h_half = (r, 0.009 if rng.random() < 0.5 else r)

    _, _, attr = _mount_pose(rng, -D, W / 2, H_base + 0.06, yaw_deg=60.0)
    s_init = float(rng.choice([-1.0, 1.0]))
    j_lo, j_hi = (-th, 0.0) if s_init < 0 else (0.0, th)
    closed_v, open_v = (0.0, -th) if s_init < 0 else (0.0, th)

    xml = (
        f'<body name="{name}" {attr}>{base_geom}'
        f'<body name="{name}_p0" pos="{_v(pivot)}">'
        f'<joint name="{name}_j0" type="hinge" axis="{_v(n)}" limited="true" range="{j_lo:.4f} {j_hi:.4f}" '
        f'damping="{damp:.3f}" frictionloss="{fric:.3f}"/>'
        f"{geoms}</body></body>"
    )
    part = scenes.Part(
        name=f"{name}_p0",
        joint=f"{name}_j0",
        kind="hinge",
        body=f"{name}_p0",
        fixture_bodies=(name,),
        range=(j_lo, j_hi),
        closed=closed_v,
        open=open_v,
        half_extents=h_half,
        protrusion=0.03,
        face_point_b=hp_local,
        face_normal_b=n,
        face_u_b=u,
        handle_point_b=hp_local,
        handle_axis_b=h_axis,
        handle_half=h_half,
        handle_kind="rotary",
    )
    return xml, [part]


def ucs_cavity_box(rng: np.random.Generator, unseen_arch: bool = False) -> tuple[list[str], ContainerTarget]:
    """Side-opening roofed cavity (+x outward opening in local frame) + free rigid payload."""
    name = "ucs_cavity"
    wt = 0.014
    z0 = float(rng.uniform(0.0, 0.09))
    if not unseen_arch:
        W = float(rng.uniform(0.24, 0.36))     # interior y width
        H = float(rng.uniform(0.15, 0.23))     # interior z height
        D = float(rng.uniform(0.18, 0.26))     # interior x depth
        sill = 0.0
    else:
        # Unseen Gate 1 variant: narrower arched/baffled mouth with raised lip sill
        W = float(rng.uniform(0.21, 0.30))
        H = float(rng.uniform(0.145, 0.20))
        D = float(rng.uniform(0.19, 0.25))
        sill = 0.012

    rho = 700.0
    # Interior x spans [-D, 0], y spans [-W/2, W/2], z spans [z0 + wt, z0 + wt + H]
    z_floor = z0 + wt
    geoms = ""
    if z0 > 0.002:
        geoms += _box_geom(f"{name}_plinth", [-D - wt, -W / 2 - wt, 0.0], [0.0, W / 2 + wt, z0], "0.30 0.30 0.32 1", rho)
    geoms += _box_geom(f"{name}_floor", [-D - wt, -W / 2 - wt, z0], [0.0, W / 2 + wt, z_floor], "0.75 0.75 0.78 1", rho)
    geoms += _box_geom(f"{name}_roof", [-D - wt, -W / 2 - wt, z_floor + H], [0.0, W / 2 + wt, z_floor + H + wt], "0.45 0.45 0.48 1", rho)
    geoms += _box_geom(f"{name}_back", [-D - wt, -W / 2 - wt, z_floor], [-D, W / 2 + wt, z_floor + H], "0.45 0.45 0.48 1", rho)
    geoms += _box_geom(f"{name}_left", [-D, -W / 2 - wt, z_floor], [0.0, -W / 2, z_floor + H], "0.45 0.45 0.48 1", rho)
    geoms += _box_geom(f"{name}_right", [-D, W / 2, z_floor], [0.0, W / 2 + wt, z_floor + H], "0.45 0.45 0.48 1", rho)

    if unseen_arch:
        # Raised front lip + side pillars narrowing the opening slightly (tests compliant clearance, not memorization)
        geoms += _box_geom(f"{name}_sill", [-0.016, -W / 2, z_floor], [0.0, W / 2, z_floor + sill], "0.50 0.40 0.35 1", rho)
        geoms += _box_geom(f"{name}_arch_top", [-0.02, -W / 2, z_floor + H - 0.018], [0.0, W / 2, z_floor + H], "0.50 0.40 0.35 1", rho)

    # Optional side door standing open (70-105 deg) on one side, mimicking microwave/cabinet door obstacle!
    if rng.random() < 0.65:
        side_s = float(rng.choice([-1.0, 1.0]))
        ang = float(rng.uniform(np.deg2rad(75.0), np.deg2rad(105.0))) * side_s
        hinge_xy = [0.0, side_s * (W / 2 + 0.005), z_floor + 0.5 * H]
        door_len = 0.85 * W
        geoms_door = _box_geom(
            f"{name}_open_door",
            [0.0, -side_s * door_len if side_s > 0 else 0.0, -0.48 * H],
            [0.014, 0.0 if side_s > 0 else -side_s * door_len, 0.48 * H],
            "0.35 0.38 0.42 1",
            rho,
        )
        door_body = f'<body name="{name}_door" pos="{_v(hinge_xy)}" euler="0 0 {ang:.4f}">{geoms_door}</body>'
    else:
        door_body = ""

    _, _, attr = _mount_pose(rng, -D - wt, W / 2 + wt + 0.12, z_floor + H + wt, yaw_deg=42.0)
    cav_xml = f'<body name="{name}" {attr}>{geoms}{door_body}</body>'

    pay_kind = str(rng.choice(["mug", "box"]))
    pay_xml, half_r, half_h = make_payload_xml(rng, kind=pay_kind, spawn_xyz=(-0.25, 0.32, 0.96))

    target = ContainerTarget(
        kind="cavity_insert",
        fixture_body=name,
        floor_body=name,
        payload_body="ucs_payload",
        payload_half_r=half_r,
        payload_half_h=half_h,
        floor_center_local=np.array([-0.52 * D, 0.0, z_floor]),
        floor_half_xy=(0.36 * D, 0.38 * W),
        open_normal_local=np.array([1.0, 0.0, 0.0]),
        ceiling_gap=H - sill,
    )
    return [cav_xml, pay_xml], target


def ucs_overhang_drawer(rng: np.random.Generator, unseen_bin: bool = False, fit_payload: bool = False) -> tuple[list[str], ContainerTarget]:
    """Open lower drawer/bin standing out beneath a protruding upper drawer/shelf overhang (+x outward)."""
    name = "ucs_ohdrawer"
    wt = 0.012
    W = float(rng.uniform(0.24, 0.36))
    D_carc = float(rng.uniform(0.20, 0.26))
    out_ext = float(rng.uniform(0.14, 0.20))      # open drawer extends [0, out_ext] in +x
    wall_h = float(rng.uniform(0.034, 0.048))     # drawer side/front wall height (LIBERO bottom drawer: ~4.2 cm)
    gap_above = float(rng.uniform(0.105, 0.148))  # vertical clearance from drawer floor to upper overhang!
    oh_ext = float(rng.uniform(0.045, 0.105))     # how far the upper drawer/handle overhangs +x over the open drawer!
    z0 = float(rng.uniform(0.0, 0.06))
    rho = 650.0

    pay_kind = str(rng.choice(["bowl", "mug", "box"]))
    pay_xml, half_r, half_h = make_payload_xml(rng, kind=pay_kind, spawn_xyz=(-0.25, -0.32, 0.96))
    # Ensure physical passability: payload height + front drawer wall must fit below the upper overhang!
    gap_above = max(gap_above, wall_h + 2.0 * half_h + 0.036)

    z_floor = z0 + wt
    # Carcass + overhanging upper drawer/handle block
    geoms = _box_geom(f"{name}_carc", [-D_carc, -W / 2 - wt, 0.0], [0.0, W / 2 + wt, z_floor + gap_above + 0.12], "0.85 0.85 0.85 1", rho)
    # Protruding upper drawer + handle bar (the exact t03 kinematic obstacle!)
    geoms += _box_geom(
        f"{name}_upper_oh",
        [0.0, -W / 2, z_floor + gap_above],
        [oh_ext, W / 2, z_floor + gap_above + 0.10],
        "0.80 0.78 0.75 1",
        rho,
    )
    if unseen_bin:
        # Unseen lipped front overhang bar + side divider
        geoms += _box_geom(
            f"{name}_upper_bar",
            [oh_ext, -0.35 * W, z_floor + gap_above + 0.015],
            [oh_ext + 0.024, 0.35 * W, z_floor + gap_above + 0.035],
            "0.55 0.55 0.60 1",
            rho,
        )

    # Open lower drawer box: floor from [-0.04, out_ext], walls around [-0.04..out_ext, -W/2..W/2]
    d_geoms = (
        _box_geom(f"{name}_dfloor", [-0.04, -W / 2, z0], [out_ext, W / 2, z_floor], "0.88 0.86 0.82 1", rho)
        + _box_geom(f"{name}_dfront", [out_ext - wt, -W / 2, z_floor], [out_ext, W / 2, z_floor + wall_h], "0.90 0.88 0.85 1", rho)
        + _box_geom(f"{name}_dleft", [-0.04, -W / 2, z0 + 0.0], [out_ext - wt, -W / 2 + wt, z_floor + wall_h], "0.88 0.86 0.82 1", rho)
        + _box_geom(f"{name}_dright", [-0.04, W / 2 - wt, z_floor], [out_ext - wt, W / 2, z_floor + wall_h], "0.88 0.86 0.82 1", rho)
    )

    _, _, attr = _mount_pose(rng, -D_carc, W / 2 + wt, z_floor + gap_above + 0.12, yaw_deg=45.0)
    box_xml = f'<body name="{name}" {attr}>{geoms}<body name="{name}_open_box" pos="0 0 0">{d_geoms}</body></body>'

    x_mid = 0.5 * (max(0.02, oh_ext - 0.02) + (out_ext - wt - 0.02))
    if fit_payload:
        # The payload footprint must fit between the carcass and the inner face
        # of the front wall after descending from the insertion corridor.
        inner_depth = out_ext - wt
        clearance = min(.5 * inner_depth, np.sqrt(2.) * half_r + .004)
        x_mid = float(np.clip(x_mid, clearance, inner_depth - clearance))
    target = ContainerTarget(
        kind="overhang_place", sill_height=wall_h,
        fixture_body=name,
        floor_body=f"{name}_open_box",
        payload_body="ucs_payload",
        payload_half_r=half_r,
        payload_half_h=half_h,
        floor_center_local=np.array([x_mid, 0.0, z_floor]),
        floor_half_xy=(max(0.03, 0.42 * (out_ext - wt)), 0.36 * W),
        open_normal_local=np.array([1.0, 0.0, 0.0]),
        ceiling_gap=gap_above,
    )
    return [box_xml, pay_xml], target


def ucs_caddy_slot(rng: np.random.Generator) -> tuple[list[str], ContainerTarget]:
    """Open-top narrow compartment / desk caddy with taller rear divider (+z upward opening)."""
    name = "ucs_caddy"
    wt = 0.010
    W_slot = float(rng.uniform(0.112, 0.148))      # compartment y width (clears jaw width while constraining placement)
    L_slot = float(rng.uniform(0.17, 0.24))        # x length [-L_slot, 0]
    wall_h = float(rng.uniform(0.042, 0.062))      # front/side compartment wall height
    back_extra = float(rng.uniform(0.035, 0.065))  # taller rear tier divider
    z0 = float(rng.uniform(0.0, 0.030))
    rho = 650.0
    z_floor = z0 + wt

    geoms = ""
    if z0 > 0.002:
        geoms += _box_geom(f"{name}_plinth", [-L_slot - wt, -W_slot / 2 - wt, 0.0], [wt, W_slot / 2 + wt, z0], "0.52 0.40 0.28 1", rho)
    geoms += (
        _box_geom(f"{name}_floor", [-L_slot - wt, -W_slot / 2 - wt, z0], [wt, W_slot / 2 + wt, z_floor], "0.62 0.48 0.34 1", rho)
        + _box_geom(f"{name}_front", [0.0, -W_slot / 2 - wt, z_floor], [wt, W_slot / 2 + wt, z_floor + wall_h], "0.58 0.44 0.30 1", rho)
        + _box_geom(f"{name}_back", [-L_slot - wt, -W_slot / 2 - wt, z_floor], [-L_slot, W_slot / 2 + wt, z_floor + wall_h + back_extra], "0.58 0.44 0.30 1", rho)
        + _box_geom(f"{name}_left", [-L_slot, -W_slot / 2 - wt, z_floor], [0.0, -W_slot / 2, z_floor + wall_h], "0.58 0.44 0.30 1", rho)
        + _box_geom(f"{name}_right", [-L_slot, W_slot / 2, z_floor], [0.0, W_slot / 2 + wt, z_floor + wall_h], "0.58 0.44 0.30 1", rho)
    )
    _, _, attr = _mount_pose(rng, -L_slot - wt, W_slot / 2 + wt, z_floor + wall_h + back_extra, yaw_deg=55.0)
    slot_xml = f'<body name="{name}" {attr}>{geoms}</body>'

    pay_kind = str(rng.choice(["thin_book", "thin_book", "box", "mug"]))
    pay_xml, half_r, half_h = make_payload_xml(rng, kind=pay_kind, spawn_xyz=(-0.25, -0.32, 0.96))

    target = ContainerTarget(
        kind="slot_place",
        fixture_body=name,
        floor_body=name,
        payload_body="ucs_payload",
        payload_half_r=half_r,
        payload_half_h=half_h,
        floor_center_local=np.array([-0.48 * L_slot, 0.0, z_floor]),
        floor_half_xy=(0.34 * L_slot, 0.34 * W_slot),
        open_normal_local=np.array([0.0, 0.0, 1.0]),
        ceiling_gap=0.26,
    )
    return [slot_xml, pay_xml], target


def bind_container_world(target: ContainerTarget, m, d) -> ContainerTarget:
    """Populate world-frame floor center, rotation, and outward mouth normal after MuJoCo forward."""
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target.fixture_body)
    R = d.xmat[bid].reshape(3, 3).copy()
    xpos = d.xpos[bid].copy()
    target.R_world = R
    target.floor_center_world = xpos + R @ target.floor_center_local
    target.open_normal_world = R @ target.open_normal_local
    return target
