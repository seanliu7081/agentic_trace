"""Additional whole-stroke IK certificate for an axis-aligned rolling pinch.

The caller supplies a contact point and orientation, commanded tangent/axis and a
scene collision callback. An available radius estimate sets the sweep length.
"""
from dataclasses import replace

import numpy as np

from .geometry import frame_from_axes, rotvec_to_mat


def _solve(model, R, p, q, finger):
    return model.ik(R, p, q, q_rest=q, finger=finger, iters=52, tol_p=.0017, tol_r=.014)[:2]


def _certify(chain, limits, collision_free, finger, phase, fractions):
    margin = np.minimum(chain - limits[:, 0], limits[:, 1] - chain).min()
    if margin < .043:
        return -np.inf
    for q0, q1, s0, s1 in zip(chain, chain[1:], fractions, fractions[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / .055)))
        for t in np.linspace(0., 1., count + 1):
            if not collision_free(q0 + t*(q1-q0), finger, phase, s0 + t*(s1-s0)):
                return -np.inf
    return float(margin)


def _extend(model, q, waypoints, finger):
    chain = [q]
    valid = np.ones(len(q), dtype=bool)
    for R, p in waypoints:
        q, solved = _solve(model, R, p, q, finger)
        chain.append(q)
        valid &= solved
        if not valid.any():
            break
    return np.asarray(chain).transpose(1, 0, 2), valid


def _score_existing(sampler, candidate, contact, axis, angles, path, finger, fractions, limits):
    approach = [(candidate.R_handoff, candidate.p_handoff + s*(contact - candidate.p_handoff))
                for s in np.linspace(.2, 1., 5)]
    pre, valid = _extend(sampler.model, candidate.q_handoff[None], approach, finger)
    if not valid[0]:
        return -np.inf
    poses = [(rotvec_to_mat(axis*.5*g) @ candidate.R_contact, p)
             for g, p in zip(angles[1:], path[1:])]
    stroke, valid = _extend(sampler.model, pre[0, -1][None], poses, .008)
    if not valid[0]:
        return -np.inf
    return min(_certify(pre[0], limits, sampler.collision_fn, finger, "approach",
                        np.linspace(-1., 0., pre.shape[1])),
               _certify(stroke[0], limits, sampler.collision_fn, .008, "arc", fractions))


def rank_rolling_arc(sampler, contract, base, q_ref, contact, approach, closing, ref, finger, rng,
                     max_found, *, legacy_unsigned_arc=False, radius=.17):
    """Try endpoint-first redundant IK if the nearest entry has low arc margin.

    Roll is relaxed only where the closing/approach frame identifies a grasp
    whose long axis is aligned with the commanded axis. A failed additional
    certificate leaves other validated motion hypotheses eligible.
    """
    from .gcp import EntryCandidate

    if not np.any(contract.goal_axis_ref) or abs(contract.target_rot_rad) < .05:
        return base
    contact = np.asarray(contact, dtype=float)
    approach = np.asarray(approach, dtype=float)
    closing = np.asarray(closing, dtype=float)
    approach = approach / max(np.linalg.norm(approach), 1e-12)
    closing = closing / max(np.linalg.norm(closing), 1e-12)
    axis, tangent = ref @ contract.goal_axis_ref, ref @ contract.goal_dir_ref
    if abs(axis @ np.cross(approach, closing)) < .8 or np.linalg.norm(np.cross(tangent, axis)) < .8:
        return base

    angle = contract.target_rot_rad
    # The contract tangent already includes travel direction. A signed rotation
    # requires the radial vector associated with the positive joint tangent.
    pos_tangent = tangent if legacy_unsigned_arc else np.copysign(1., angle) * tangent
    radial = radius * np.cross(pos_tangent, axis)
    center = contact - radial
    probe = np.copysign(min(abs(angle), .04/radius), angle)
    # Preserve increasing *progress* for either sign of commanded travel.
    angles = np.unique(np.r_[np.linspace(0., angle, 10), probe])
    if angle < 0.:
        angles = angles[::-1]
    fractions = angles / angle
    path = [center + rotvec_to_mat(axis*g) @ radial for g in angles]
    standoff = np.linalg.norm(base[0].p_handoff-contact) if base else .055
    limits = sampler.model.limits[:7]
    best = -np.inf
    if base:
        best = _score_existing(sampler, base[0], contact, axis, angles, path, finger, fractions, limits)
        if np.isfinite(best):
            base = [replace(base[0], arc_joint_margin_rad=best), *base[1:]]
        if best >= .24:
            return base

    # This secondary search must not change sensor noise or later random draws.
    from copy import deepcopy
    proposal_rng = deepcopy(rng)
    chosen = None
    for yaw in (0., -.46, .46, -.82, .82):
        yaw_rot = rotvec_to_mat(axis*yaw)
        ray = yaw_rot @ approach
        hand = contact - standoff*ray
        for sign in (1., -1.):
            R = frame_from_axes(ray, sign*(yaw_rot @ closing))
            if sampler.wrist_visible is not None and not sampler.wrist_visible(R, hand, contact):
                continue
            poses = [(rotvec_to_mat(axis*.5*g) @ R, point) for g, point in zip(angles, path)]
            seeds = sampler.model.solve_ik(*poses[-1], q_ref, n_random=14, rng=proposal_rng,
                                           finger=.008, q_rest=limits.mean(axis=1))
            if not len(seeds):
                continue
            backward, valid = _extend(sampler.model, seeds, reversed(poses[:-1]), .008)
            if not valid.any():
                continue
            approach_poses = [(R, contact-s*standoff*ray) for s in (.2, .4, .6, .8, 1.)]
            withdraw, reachable = _extend(sampler.model, backward[:, -1], approach_poses, finger)
            valid &= reachable
            if not valid.any():
                continue
            stroke = backward[:, ::-1]
            pre = withdraw[:, ::-1]
            margins = np.minimum(stroke-limits[:, 0], limits[:, 1]-stroke).min(axis=(1, 2))
            for i in np.argsort(-margins):
                if not valid[i] or margins[i] <= max(.06, best):
                    continue
                if sampler.wrist_visible is None and not sampler._visible(R, hand, contact, pre[i, 0], finger):
                    continue
                score = _certify(stroke[i], limits, sampler.collision_fn, .008, "arc", fractions)
                if score <= best:
                    continue
                score = min(score, _certify(pre[i], limits, sampler.collision_fn, finger, "approach",
                                             np.linspace(-1., 0., pre.shape[1])))
                if score < max(.06, best + .015):
                    continue
                best = score
                chain = np.concatenate((pre[i], stroke[i, 1:]))
                chosen = EntryCandidate(R, hand, R, contact, pre[i, 0], chain, ray, "axis_arc_roll", score)
            if best > .40:
                break
        if best > .40:
            break
    if chosen is None:
        return base
    return [chosen, *base[:max_found-1]]
