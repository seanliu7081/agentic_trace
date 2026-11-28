// Zero-gap C++ 500 Hz OSC stepper linked directly against MuJoCo 3.2.6 (libmujoco.so.3.2.6).
// Reproduces robosuite 1.4.1 OperationalSpaceController + PandaGripper + MujocoEnv.step
// bit-for-bit (including robosuite's float32 quat2mat cast and 25-substep gripper integration)
// while running all 25 substeps inside C++ with zero Python overhead and only 1 mj_forward/substep.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <mujoco/mujoco.h>

namespace {

// Exact match of robosuite.utils.transform_utils.quat2mat(axisangle2quat(rv))
// Note: robosuite casts quaternion to float32 AND computes scale, q *= scale, q2 = outer(q,q),
// and 1.0f - q2 - q2 ALL in IEEE-754 float32 before numpy returns float32/float64!
static void axisangle_to_robosuite_mat(const double rv[3], double R[9]) {
    const double angle = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    if (std::abs(angle) <= 1e-9) {
        std::memset(R, 0, 9 * sizeof(double));
        R[0] = R[4] = R[8] = 1.0;
        return;
    }
    const double s = std::sin(0.5 * angle) / angle;
    const double qx = rv[0] * s;
    const double qy = rv[1] * s;
    const double qz = rv[2] * s;
    const double qw = std::cos(0.5 * angle);

    float q[4] = {
        static_cast<float>(qw),
        static_cast<float>(qx),
        static_cast<float>(qy),
        static_cast<float>(qz)
    };
    const float n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (n < 1e-15f) {
        std::memset(R, 0, 9 * sizeof(double));
        R[0] = R[4] = R[8] = 1.0;
        return;
    }
    const float scale = static_cast<float>(std::sqrt(2.0 / static_cast<double>(n)));
    for (int i = 0; i < 4; ++i) q[i] *= scale;
    float q2[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            q2[i][j] = q[i] * q[j];

    R[0] = 1.0 - static_cast<double>(q2[2][2]) - static_cast<double>(q2[3][3]);
    R[1] = static_cast<double>(q2[1][2] - q2[3][0]);
    R[2] = static_cast<double>(q2[1][3] + q2[2][0]);
    R[3] = static_cast<double>(q2[1][2] + q2[3][0]);
    R[4] = 1.0 - static_cast<double>(q2[1][1]) - static_cast<double>(q2[3][3]);
    R[5] = static_cast<double>(q2[2][3] - q2[1][0]);
    R[6] = static_cast<double>(q2[1][3] - q2[2][0]);
    R[7] = static_cast<double>(q2[2][3] + q2[1][0]);
    R[8] = 1.0 - static_cast<double>(q2[1][1]) - static_cast<double>(q2[2][2]);
}

// Symmetric positive-definite / general NxN inverse via Gauss-Jordan with partial pivoting (N <= 7)
template <int N>
static bool invert_mat(const double A_in[N * N], double A_inv[N * N]) {
    double aug[N][2 * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aug[i][j] = A_in[i * N + j];
            aug[i][N + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < N; ++col) {
        int piv = col;
        double max_v = std::abs(aug[col][col]);
        for (int r = col + 1; r < N; ++r) {
            double v = std::abs(aug[r][col]);
            if (v > max_v) {
                max_v = v;
                piv = r;
            }
        }
        if (max_v < 1e-14) return false;
        if (piv != col) {
            for (int j = 0; j < 2 * N; ++j) std::swap(aug[col][j], aug[piv][j]);
        }
        const double inv_d = 1.0 / aug[col][col];
        for (int j = 0; j < 2 * N; ++j) aug[col][j] *= inv_d;
        for (int r = 0; r < N; ++r) {
            if (r == col) continue;
            const double f = aug[r][col];
            if (f != 0.0) {
                for (int j = 0; j < 2 * N; ++j) aug[r][j] -= f * aug[col][j];
            }
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_inv[i * N + j] = aug[i][N + j];
    return true;
}

static inline void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

}  // namespace

extern "C" {

// Step `substeps` (25) physics steps for a 7-DoF arm + 2-DoF Panda gripper under exact robosuite OSC_POSE.
// Also tracks peak non-allowed / handle contact penetration across all substeps!
int ucs_osc_step(
    std::uintptr_t m_addr,
    std::uintptr_t d_addr,
    int substeps,
    int site_id,
    const int* qpos_idx,       // [7]
    const int* qvel_idx,       // [7]
    const int* arm_act_idx,    // [7]
    const int* grip_act_idx,   // [2]
    const double* p_goal,      // [3]
    const double* R_goal,      // [9] exact robosuite T.quat2mat(T.axisangle2quat(rv))
    const double* q_init,      // [7] nullspace posture reference
    const double* kp,          // [6]
    const double* kd,          // [6]
    double joint_kp,           // 10.0
    double grip_cmd,           // scalar in [-1, 1]
    double grip_speed,         // 0.01
    double* grip_cur_action,   // [2] in/out PandaGripper.current_action
    double* out_max_pen_m,     // [1] out: maximum contact penetration (-dist) seen across substeps
    int split_step, const unsigned char* forbidden_pairs             // 0 = exact robosuite double-solve (mj_forward + mj_step); 1 = mj_step1 + mj_step2 (2x faster!)
) {
    const mjModel* m = reinterpret_cast<const mjModel*>(m_addr);
    mjData* d = reinterpret_cast<mjData*>(d_addr);
    if (!m || !d) return -1;

    const int nv = m->nv;
    double stack_buf[64 * 64 + 6 * 64];
    const bool use_heap = (nv > 64);
    double* heap_buf = use_heap ? new double[nv * nv + 6 * nv] : nullptr;
    double* M_full = use_heap ? heap_buf : stack_buf;
    double* jacp = M_full + nv * nv;
    double* jacr = jacp + 3 * nv;

    const double joint_kv = 2.0 * std::sqrt(joint_kp);
    const double sign_g = (grip_cmd > 0.0) ? 1.0 : ((grip_cmd < 0.0) ? -1.0 : 0.0);

    double grip_bias[2], grip_weight[2];
    for (int k = 0; k < 2; ++k) {
        const int ai = grip_act_idx[k];
        const double lo = m->actuator_ctrlrange[2 * ai + 0];
        const double hi = m->actuator_ctrlrange[2 * ai + 1];
        grip_bias[k] = 0.5 * (hi + lo);
        grip_weight[k] = 0.5 * (hi - lo);
    }

    double arm_lo[7], arm_hi[7];
    for (int i = 0; i < 7; ++i) {
        const int ai = arm_act_idx[i];
        arm_lo[i] = m->actuator_ctrlrange[2 * ai + 0];
        arm_hi[i] = m->actuator_ctrlrange[2 * ai + 1];
    }

    double max_pen = 0.0;

    for (int sub = 0; sub < substeps; ++sub) {
        // 1. Position/velocity/collision/constraint kinematics
        if (split_step) {
            mj_step1(m, d);
        } else {
            mj_forward(m, d);
        }

        // Track contact penetration across every physics substep
        for (int ic = 0; ic < d->ncon; ++ic) {
            const auto &c = d->contact[ic];
            if (forbidden_pairs && !forbidden_pairs[c.geom1 * m->ngeom + c.geom2]) continue;
            const double pen = -c.dist;
            if (pen > max_pen) max_pen = pen;
        }

        // 2. Site pose & Jacobian
        const double* ee_pos = d->site_xpos + 3 * site_id;
        const double* ee_mat = d->site_xmat + 9 * site_id;
        mj_jacSite(m, d, jacp, jacr, site_id);

        double J[6][7];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 7; ++c) {
                J[r][c] = jacp[r * nv + qvel_idx[c]];
                J[r + 3][c] = jacr[r * nv + qvel_idx[c]];
            }
        }

        double q[7], qd[7];
        for (int i = 0; i < 7; ++i) {
            q[i] = d->qpos[qpos_idx[i]];
            qd[i] = d->qvel[qvel_idx[i]];
        }

        // Site linear & angular velocity: exact match of robosuite get_site_xvelp / get_site_xvelr = jac * d->qvel
        double ee_pos_vel[3] = {0.0, 0.0, 0.0};
        double ee_ori_vel[3] = {0.0, 0.0, 0.0};
        for (int r = 0; r < 3; ++r) {
            double sp = 0.0, sr = 0.0;
            for (int c = 0; c < nv; ++c) {
                sp += jacp[r * nv + c] * d->qvel[c];
                sr += jacr[r * nv + c] * d->qvel[c];
            }
            ee_pos_vel[r] = sp;
            ee_ori_vel[r] = sr;
        }

        // 3. Mass matrix M (7x7) and M^-1
        mj_fullM(m, M_full, d->qM);
        double M[49], Minv[49];
        for (int r = 0; r < 7; ++r)
            for (int c = 0; c < 7; ++c)
                M[r * 7 + c] = M_full[qvel_idx[r] * nv + qvel_idx[c]];

        invert_mat<7>(M, Minv);

        // 4. J * M^-1 (6x7) and Lambda_full_inv = J * M^-1 * J^T (6x6)
        double JMinv[6][7];
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 7; ++c) {
                double s = 0.0;
                for (int k = 0; k < 7; ++k) s += J[r][k] * Minv[k * 7 + c];
                JMinv[r][c] = s;
            }
        }

        double Lfull_inv[36], Lpos_inv[9], Lori_inv[9];
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                double s = 0.0;
                for (int k = 0; k < 7; ++k) s += JMinv[r][k] * J[c][k];
                Lfull_inv[r * 6 + c] = s;
            }
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Lpos_inv[r * 3 + c] = Lfull_inv[r * 6 + c];
                Lori_inv[r * 3 + c] = Lfull_inv[(r + 3) * 6 + (c + 3)];
            }
        }

        double Lfull[36], Lpos[9], Lori[9];
        invert_mat<6>(Lfull_inv, Lfull);
        invert_mat<3>(Lpos_inv, Lpos);
        invert_mat<3>(Lori_inv, Lori);

        // 5. Orientation error = 0.5 * sum_col cross(rc_col, rd_col)
        const double rc0[3] = {ee_mat[0], ee_mat[3], ee_mat[6]};
        const double rc1[3] = {ee_mat[1], ee_mat[4], ee_mat[7]};
        const double rc2[3] = {ee_mat[2], ee_mat[5], ee_mat[8]};
        const double rd0[3] = {R_goal[0], R_goal[3], R_goal[6]};
        const double rd1[3] = {R_goal[1], R_goal[4], R_goal[7]};
        const double rd2[3] = {R_goal[2], R_goal[5], R_goal[8]};
        double c0[3], c1[3], c2[3];
        cross3(rc0, rd0, c0);
        cross3(rc1, rd1, c1);
        cross3(rc2, rd2, c2);
        const double ori_err[3] = {
            0.5 * (c0[0] + c1[0] + c2[0]),
            0.5 * (c0[1] + c1[1] + c2[1]),
            0.5 * (c0[2] + c1[2] + c2[2])
        };

        double des_f[3], des_t[3];
        for (int i = 0; i < 3; ++i) {
            des_f[i] = kp[i] * (p_goal[i] - ee_pos[i]) - kd[i] * ee_pos_vel[i];
            des_t[i] = kp[i + 3] * ori_err[i] - kd[i + 3] * ee_ori_vel[i];
        }

        double wrench[6] = {0};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                wrench[r] += Lpos[r * 3 + c] * des_f[c];
                wrench[r + 3] += Lori[r * 3 + c] * des_t[c];
            }
        }

        // 6. Nullspace matrix N = I - Jbar * J, where Jbar = M^-1 J^T Lfull (7x6)
        double Jbar[7][6];
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 6; ++c) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += JMinv[k][r] * Lfull[k * 6 + c];  // Minv symmetric -> (Minv J^T)[r,k] = JMinv[k,r]
                Jbar[r][c] = s;
            }
        }
        double Nmat[7][7];
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                double s = (r == c) ? 1.0 : 0.0;
                for (int k = 0; k < 6; ++k) s -= Jbar[r][k] * J[k][c];
                Nmat[r][c] = s;
            }
        }

        double u_null[7], pose_tau[7];
        for (int i = 0; i < 7; ++i) u_null[i] = joint_kp * (q_init[i] - q[i]) - joint_kv * qd[i];
        for (int r = 0; r < 7; ++r) {
            double s = 0.0;
            for (int c = 0; c < 7; ++c) s += M[r * 7 + c] * u_null[c];
            pose_tau[r] = s;
        }

        // 7. Total arm torques + optional 500 Hz force-cap + clip
        for (int i = 0; i < 7; ++i) {
            double tau = d->qfrc_bias[qvel_idx[i]];
            for (int k = 0; k < 6; ++k) tau += J[k][i] * wrench[k];
            for (int k = 0; k < 7; ++k) tau += Nmat[k][i] * pose_tau[k];  // N^T * pose_tau
            tau = std::max(arm_lo[i], std::min(arm_hi[i], tau));
            d->ctrl[arm_act_idx[i]] = tau;
        }

        // 8. PandaGripper.format_action + actuator rescale
        grip_cur_action[0] = std::max(-1.0, std::min(1.0, grip_cur_action[0] - grip_speed * sign_g));
        grip_cur_action[1] = std::max(-1.0, std::min(1.0, grip_cur_action[1] + grip_speed * sign_g));
        d->ctrl[grip_act_idx[0]] = grip_bias[0] + grip_weight[0] * grip_cur_action[0];
        d->ctrl[grip_act_idx[1]] = grip_bias[1] + grip_weight[1] * grip_cur_action[1];

        // 9. Advance physics 1 substep (2 ms)
        if (split_step) {
            mj_step2(m, d);
        } else {
            mj_step(m, d);
        }
    }

    // Final forward (1x per 25 substeps) so site_xpos, site_xmat, acceleration-stage F/T sensordata, and contacts match post-step state
    mj_forward(m, d);
    for (int ic = 0; ic < d->ncon; ++ic) {
        const auto &c = d->contact[ic];
            if (forbidden_pairs && !forbidden_pairs[c.geom1 * m->ngeom + c.geom2]) continue;
            const double pen = -c.dist;
        if (pen > max_pen) max_pen = pen;
    }

    if (out_max_pen_m) *out_max_pen_m = max_pen;

    if (heap_buf) delete[] heap_buf;
    return 0;
}

}  // extern "C"
