"""Reference curves of M0 (lrl_train.eval_stroke): one layout end to end in the real simulator, CPU, no rendering."""
import numpy as np

from lrl_train import eval_stroke as E


def test_press_poses_follow_the_search():
    c, n = np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])
    fam = E.press_poses(c, n)
    assert [len(fam[f]) for f in E.FAMILIES] == [4, 8, 6, 8]
    for mode, a, R, p in fam["tip"]:
        np.testing.assert_allclose(R[:, 2], -n, atol=1e-9)                  # fingertips point into the face
        np.testing.assert_allclose(p + a * (E.manip.TIP_Z + E.CFG["tip_gap"]), c, atol=1e-9)
    for mode, a, R, p in fam["side"]:
        np.testing.assert_allclose(a, [0.0, 0.0, -1.0], atol=1e-9)          # hand pointing down in front of the face
        assert abs((p - c) @ n - E.CFG["side_out"]) < 1e-9
    assert list(E.press_poses(c, np.array([0.0, 0.0, 1.0]))) == ["tip"]     # a level face has the level tip only


def test_one_drawer_layout(tmp_path):
    rows = E.run_layout(("drawer_stack", 0, str(tmp_path)), levels=("none",), families=("tip",), rounds=1)
    assert rows and len(rows) % len(E.METHODS) == 0
    ran = [r for r in rows if r["pose"] is not None]
    assert ran, "no tip pose on any drawer of the layout"
    for r in ran:
        assert r["err_pos"] == 0.0 and r["stop"] in ("stalled", "force_limit", "max_travel", "time", "force")
        assert r["left"] >= 0.0 and r["force_peak"] >= r["block_force"] >= 0.0
        assert r["success"] or E.reason(r) in E.REASONS
    stroke = [r for r in ran if r["method"] == "stroke"]
    # a drawer met square on its front, with no error, closes under today's stroke on at least one level
    assert any(r["success"] and r["touched"] and r["left"] < E.CFG["closed_tol"] for r in stroke)
    s = E.summarize(rows)
    assert s["trials"] == len(rows) and "| drawer_stack | tip |" in E.tables(s)
    assert sum(d["n"] for d in s["success"].values()) == len(rows)
