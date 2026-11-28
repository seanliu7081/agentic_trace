"""Run 50 seeded MuJoCo contact scripts on an isolated parallel-jaw test fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from atrace.gcp_labels import LABELS, classify_contacts, contact_patch_evidence


class PandaAuditFixture:
    """A loaded sliding bar and fixed cabinet lip at the Panda's measured finger pads."""
    def __init__(self, directory):
        from . import scenes
        from .fast_sim import FastSimRobot
        from .sim import StageSim
        xml = """<body name="audit_mount" pos="0 0 3">
            <geom name="cabinet_lip" type="box" pos=".15 0 0" size=".008 .015 .008"/>
            <body name="audit_bar"><joint name="bar_slide" type="slide" axis="0 1 0"
              damping="3" stiffness="18" frictionloss=".12"/>
            <geom name="bar" type="box" size=".009 .018 .007" density="900"
              friction="1.1 .005 .0001"/></body></body>"""
        self.sim = sim = StageSim(scenes.Layout(scenes.base_layout(directory), [xml], [], 234))
        sim.robot.sensorless = True
        sim.set_arm(sim.q_home)
        self.rb = FastSimRobot(sim.robot)
        self.rb.sync_from_robosuite()
        self.start = sim.snapshot()
        self.m, self.d = self.rb.m, self.rb.d
        self.R, self.p = self.rb.eef_pose()
        self.mount = self.m.body("audit_mount").id
        self.joint = self.m.joint("bar_slide")
        self.bar = self.m.geom("bar").id
        self.frame = self.m.geom("cabinet_lip").id
        self.left, self.right = [scenes._subtree(self.m, self.m.body(name).id)[self.m.geom_bodyid]
                                 for name in ("gripper0_leftfinger", "gripper0_rightfinger")]
        self.robot = sim._is_robot[self.m.geom_bodyid]
        self.target = np.arange(self.m.ngeom) == self.bar

    def rollout(self, label, seed):
        rng = np.random.default_rng(seed)
        sim, rb, m, d = self.sim, self.rb, self.m, self.d
        R, p = self.R, self.p
        sim.restore(self.start)
        rb.sync_from_robosuite()
        # Shift fixture geometry for each preassigned script. No contact is
        # fabricated: only the motor commands and an external slip load vary.
        offset = np.array([0., 0., -.004]) + rng.uniform(-.001,.001,3)
        offset[0] += {"cabinet_frame_pinch":-.15, "air_close":.10,
                      "single_finger_graze":.034}.get(label,0.)
        if label == "push_face_contact":
            offset[1] += .034
        m.body_pos[self.mount] = p + R@offset
        mujoco.mju_mat2Quat(m.body_quat[self.mount], np.ascontiguousarray(R.ravel()))
        m.geom_size[self.bar,0] = .045 if label=="push_face_contact" else .009
        m.geom_rbound[self.bar] = np.linalg.norm(m.geom_size[self.bar])
        mujoco.mj_forward(m,d)
        rows=[]
        dist = rng.uniform(.032,.038)
        for step in range(48):
            width = .0
            if label=="single_finger_graze":
                width = .060 if step<24 else .080
            if label=="push_face_contact":
                width = .058
            drive = dist*np.clip((step-12)/18.,0,1) if label in (
                "firm_handle_grasp","mid_pull_slip","push_face_contact") else 0.
            if label=="mid_pull_slip" and step>=33:
                d.qfrc_applied[self.joint.dofadr[0]] = 40.
            command = np.clip((width - 2*rb._pad_off)/.04 - 1.,-1.,1.)
            rb._grip_cur[:] = (command,-command)
            rb.step(R,p+drive*R[:,1],0.)
            facts = contact_patch_evidence(d.contact[:d.ncon],self.left,self.right,self.robot,
                                          self.target,R[:,1])
            facts.update(command_width_m=width, width_m=rb.finger_opening(),
                         target_delta=sim.joint("bar_slide"))
            rows.append(facts)
        return rows


def audit(out=None, seed=390701):
    cases = list(np.resize(LABELS, 50))
    np.random.default_rng(seed).shuffle(cases)
    rows = []
    matrix = {label: {result: 0 for result in (*LABELS, "uncertain")} for label in LABELS}
    with TemporaryDirectory() as tmp:
        fixture = PandaAuditFixture(Path(tmp))
        try:
            for idx, expected in enumerate(cases):
                trace = fixture.rollout(expected, seed + idx)
                predicted = classify_contacts(trace)
                matrix[expected][predicted] += 1
                rows.append(dict(seed=seed + idx, expected=expected, predicted=predicted, trace=trace))
        finally:
            fixture.sim.close()
    report = dict(fixture="Panda FastSimRobot with a movable bar and fixed frame at 20 Hz/500 Hz",
                  count=len(rows), correct=sum(r["expected"] == r["predicted"] for r in rows),
                  confusion=matrix, trials=rows)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out)
    print(f'{result["correct"]}/{result["count"]}', result["confusion"])
