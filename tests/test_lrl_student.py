"""LRL_PLAN M1E, E2: the student's rendering (render.Cameras), observation with the D9 sensor model (student_obs), network
(student) and DAgger parts (dagger), on one generated drawer layout rendered with EGL."""
import os

import mujoco
import numpy as np
import pytest
import torch

from atrace.lrl_e2e import ACT_DIM, PROPRIO_DIM, Belief, frame_a, start_region
from lrl_train import scenes
from lrl_train.env import layout_rng
from lrl_train.render import GPU_MINOR, HEAD, WRIST, Cameras
from lrl_train.student_obs import CFG, FIXTURE, N_POINTS, OBS_DIM, PART, POINT_DIM, ROBOT_C, SensorModel, StudentObs


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """A drawer stack with the arm at a start-region pose of its first drawer, and head + wrist cameras at 96 px on
    the first visible GPU. EGL picks devices by /dev/nvidia minor; robosuite asserts at import that this id is in
    CUDA_VISIBLE_DEVICES, so it is imported before the id is set (changing CUDA_VISIBLE_DEVICES would confuse
    torch's CUDA initialization later in the process)."""
    import robosuite.utils.binding_utils  # noqa: F401
    mp = pytest.MonkeyPatch()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    mp.setenv("MUJOCO_EGL_DEVICE_ID", str(GPU_MINOR[int(cvd.split(",")[0])] if cvd else 0))
    from lrl_train.sim import StageSim
    rng = layout_rng("drawer_stack", 3)
    sim = StageSim(scenes.sample_layout(rng, "drawer_stack", tmp_path_factory.mktemp("bddl")))
    part = sim.parts[0]
    belief = Belief(part.face_point, part.face_normal)
    assert any(sim.entry(R, p, grip=-1.0) for R, p in start_region(belief))
    cams = Cameras(sim.robot.m, sim.robot.d, 96)
    yield sim, part, belief, cams
    cams.close()
    sim.close()
    mp.undo()


def test_cameras_match_robosuite_render(scene):
    """Depth and the body at every pixel as SimRobot.capture would read them (robosuite's render, two passes)."""
    sim, _, _, cams = scene
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    rs = MjRenderContextOffscreen(sim.env.env.sim, device_id=-1)
    m = sim.robot.m
    for v in cams.render():
        rs.gl_ctx.make_current()
        seg, z = sim.env.env.sim.render(width=96, height=96, camera_name=v.name, depth=True, segmentation=True)
        near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
        depth = near / (1.0 - z[::-1].astype(np.float64) * (1.0 - near / far))
        typ, oid = seg[::-1][..., 0], seg[::-1][..., 1]
        body = np.full(typ.shape, -1)
        g = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (oid >= 0)
        s = (typ == int(mujoco.mjtObj.mjOBJ_SITE)) & (oid >= 0)
        body[g], body[s] = m.geom_bodyid[oid[g]], m.site_bodyid[oid[s]]
        assert np.abs(v.depth - depth).max() < 1e-5
        assert (v.body == body).mean() == 1.0
    del rs


def test_part_cloud_without_sensors_is_the_truth_mask(scene):
    """The wrist camera at the start pose looks at the drawer's front (the head camera sees its back: NOTES.md)."""
    sim, part, belief, cams = scene
    so = StudentObs(sensors=False)
    so.episode(sim.robot.m, part, np.random.default_rng(0))
    wrist = cams.render((WRIST,))[0]
    cls = so.cls[wrist.body]
    pts = so.part_cloud([wrist])
    assert 100 < len(pts) <= (cls == PART).sum()
    # every part point lies on the drawer's front or handle: within its half extents of the face centre, in its plane
    r = pts - part.face_point
    assert np.abs(r @ part.face_normal).max() < 0.05
    assert np.abs(r @ part.face_u).max() < part.half_extents[0] + 0.01


def test_observation_layout(scene):
    sim, part, belief, cams = scene
    so = StudentObs()
    so.episode(sim.robot.m, part, np.random.default_rng(1), share=1.0)
    frame = frame_a(sim.robot.eef_pose()[1], belief)
    views = cams.render()
    so.start(frame, so.part_cloud(views))
    obs = so.obs(views, np.zeros(PROPRIO_DIM), frame)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    pts = obs[PROPRIO_DIM:].reshape(N_POINTS, POINT_DIM)
    live, part_rows = pts[:CFG["live_points"]], pts[CFG["live_points"]:]
    assert (part_rows[:, 3] == 1).all() and (live[:, 3] == 0).all()        # "start" mode: the part rows carry it
    lo, hi = 5 * np.array(CFG["crop_lo"]), 5 * np.array(CFG["crop_hi"])
    assert ((live[:, :3] >= lo - 1e-4) & (live[:, :3] <= hi + 1e-4)).all()
    assert set(np.unique(live[:, 4])) <= {0.0, 1.0} and live[:, 4].any()   # wrist points present
    # the part rows sit around A's origin, which is the believed (here: true) face centre
    assert np.linalg.norm(part_rows[:, :3].mean(0)) / 5 < 0.1


def test_robot_pixels_are_dropped(scene):
    sim, part, belief, cams = scene
    so = StudentObs(sensors=False)
    so.episode(sim.robot.m, part, np.random.default_rng(2))
    wrist = cams.render((WRIST,))[0]
    cls = so.cls[wrist.body]
    pts, idx, _ = so._points(wrist)
    assert (cls == ROBOT_C).any()                                            # the fingers are in the wrist view
    assert not (cls.ravel()[idx] == ROBOT_C).any()


@pytest.mark.parametrize("mode", ["part", "handle", "spill", "fixture"])
def test_mask_modes(scene, mode):
    sim, part, _, cams = scene
    so = StudentObs()
    so.episode(sim.robot.m, part, np.random.default_rng(3))
    so.sm.mode = mode
    so.sm.cfg = dict(CFG, mask_sigma=0.0)
    wrist = cams.render((WRIST,))[0]
    cls = so.cls[wrist.body]
    truth, proud = cls == PART, so._proud(wrist, cls)
    mask = so._mask(wrist, cls)
    if mode == "part":
        assert (mask == (truth & ~proud)).all()
    elif mode == "handle":
        assert proud.any() and (mask == proud).all() and not (mask & ~truth).any()
    elif mode == "fixture":
        assert (mask == (truth | (cls == FIXTURE))).all()
    else:
        want = int(truth.sum() * CFG["spill_recall"] / CFG["spill_precision"])
        assert abs(int(mask.sum()) - want) <= 1 and (mask & ~truth).any()


def test_sensor_model_draws():
    rng = np.random.default_rng(4)
    modes = [SensorModel(rng, 96, closed=True).mode for _ in range(400)]
    share = {m: modes.count(m) / 400 for m in CFG["mask_modes"]}
    for m, p in zip(CFG["mask_modes"], CFG["mask_p_closed"]):
        assert abs(share[m] - p) < 0.08
    sm = SensorModel(rng, 96)
    z = np.full(20000, 1.0)
    noisy = sm.noisy(z, HEAD, rng)
    sigma = sm.noise * (CFG["noise_a0"] + CFG["noise_a2"])
    assert abs(np.std(noisy) / sigma - 1) < 0.05


def test_student_fits_and_round_trips(tmp_path):
    from lrl_train.student import Student, load, save
    torch.manual_seed(0)
    model = Student()
    obs = torch.randn(64, OBS_DIM)
    target = torch.cat([torch.rand(64, ACT_DIM - 1) * 2 - 1, (torch.rand(64, 1) > 0.5).float()], 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss, st = model.loss(obs, target)
        first = first or float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.3 * first
    a = model.act(obs, draw=False)
    assert a.shape == (64, ACT_DIM) and set(a[:, -1].tolist()) <= {-1.0, 1.0}
    save(tmp_path / "s.pt", model)
    back, _ = load(tmp_path / "s.pt")
    assert torch.equal(back(obs)[0], model.eval()(obs)[0])


def test_buffer_ring_and_stand_in_teacher():
    from lrl_train.dagger import S_DIM, T_DIM, Buffer, Teacher
    buf = Buffer(10, 3)
    buf.add(np.arange(24, dtype=np.float32).reshape(8, 3), np.zeros((8, ACT_DIM)))
    buf.add(np.arange(24, 36, dtype=np.float32).reshape(4, 3), np.ones((4, ACT_DIM)))
    assert buf.n == 10 and buf.i == 2 and buf.added == 12
    assert buf.obs[0, 0] == 30 and buf.obs[1, 0] == 33 and buf.obs[2, 0] == 6
    other = Buffer(10, 3, buf.shm.name)                     # the trainer process attaches by name
    assert other.obs[1, 0] == 33 and other.n == 10
    other.close()
    buf.close(unlink=True)
    t = Teacher.stand_in(torch.device("cpu"))
    out = t(np.random.randn(5, T_DIM).astype(np.float32))
    assert out.shape == (5, ACT_DIM) and ((out[:, -1] >= 0) & (out[:, -1] <= 1)).all()
    assert S_DIM == OBS_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph of the acting call")
def test_acting_graph_matches_plain_calls():
    from lrl_train.dagger import S_DIM, Acting, Teacher
    from lrl_train.student import Student
    dev = torch.device("cuda:0")
    teacher, student = Teacher.stand_in(dev), Student().to(dev).eval()
    act = Acting(teacher, student, 8, dev)
    obs = np.random.default_rng(5).normal(size=(5, act.inp.shape[1])).astype(np.float32)
    teach = np.array([True, False, True, False, False])
    executed, labels = act(obs, teach)
    np.testing.assert_allclose(labels, teacher(obs[:, S_DIM:]).cpu().numpy(), atol=1e-5)
    with torch.no_grad():
        a, _ = student(torch.as_tensor(obs[:, :S_DIM], device=dev))
    np.testing.assert_allclose(executed[~teach, :ACT_DIM - 1], a.cpu().numpy()[~teach], atol=1e-5)
    np.testing.assert_allclose(executed[teach, :ACT_DIM - 1], labels[teach, :ACT_DIM - 1], atol=1e-6)
    assert set(executed[:, -1].tolist()) <= {-1.0, 1.0}
