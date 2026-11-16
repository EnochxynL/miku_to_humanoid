"""Generate the robot joint angles that put a robot into an MMD pose.

This is the only place that needs to know *which* robot segment corresponds to *which* MMD limb, and the
only place where the correspondence can be established without already knowing C:

* the source bone of a link comes from ``mmd_bones.LINKS``;
* its child comes from ``mmd_bones.NEXT_JOINT`` (MMD puts twists and sleeve chains between joints, so
  the tree child is usually not the next joint);
* what stays here is robot-side knowledge - the far end of the segment and the joints that move it.

The comparison is on **limb directions**, which are convention-free: both sides are measured as "the
vector from A to B", so no frame convention has to be known. Orientations cannot be used here, because
their relationship *is* C - using them would be circular.

The output is data: a JSON of joint angles in degrees. ``gmr_calibrate`` consumes it.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/gmr_paired_pose.py --pmx <model.pmx> \
        --g1-xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml \
        --out configs/mmd/paired_pose_pmx.json --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmd_pose                                        # noqa: E402
from mmd_bones import LINKS, NEXT_JOINT                 # noqa: E402

# robot link -> (far end of the measured segment, the joints that move that segment)
LIMBS = {
    'left_shoulder_yaw_link': ('left_elbow_link',
        ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint']),
    'left_elbow_link': ('left_wrist_yaw_link', ['left_elbow_joint']),
    'right_shoulder_yaw_link': ('right_elbow_link',
        ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']),
    'right_elbow_link': ('right_wrist_yaw_link', ['right_elbow_joint']),
    'left_hip_yaw_link': ('left_knee_link',
        ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint']),
    'left_knee_link': ('left_ankle_roll_link', ['left_knee_joint']),
    'right_hip_yaw_link': ('right_knee_link',
        ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']),
    'right_knee_link': ('right_ankle_roll_link', ['right_knee_joint']),
}


def segments():
    """(source bone, source child, robot link, robot child, joints) for every measured limb."""
    out = []
    for link, (robot_child, joints) in LIMBS.items():
        bone = LINKS[link]
        out.append((bone, NEXT_JOINT[bone], link, robot_child, joints))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--g1-xml', required=True)
    parser.add_argument('--out', required=True, help='joint angles to write')
    parser.add_argument('--vpd', default=None,
                        help='MMD pose to match; omitted = the rig\'s bind pose')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    limbs = segments()

    # --- source limb directions ---------------------------------------------------------------
    wanted = sorted({b for b, *_ in limbs} | set(LINKS.values()))
    if args.vpd:
        frames, _ = mmd_pose.load(args.pmx, args.vpd, wanted)
        source = frames[0]
        note = 'vpd: %s' % Path(args.vpd).name
    else:
        source = mmd_pose.bind_pose(args.pmx, wanted)[0]
        note = 'bind pose'

    def source_position(name):
        return np.asarray(source[name][0])

    targets = []
    for src_a, src_b, _ra, _rb, _joints in limbs:
        direction = source_position(src_b) - source_position(src_a)
        targets.append(direction / np.linalg.norm(direction))

    # --- solve the robot pose -----------------------------------------------------------------
    robot = mujoco.MjModel.from_xml_path(args.g1_xml)
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)

    def body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    addresses, order = {}, []
    for _, _, _, _, joints in limbs:
        for name in joints:
            jid = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0 and name not in addresses:
                addresses[name] = (int(robot.jnt_qposadr[jid]), robot.jnt_range[jid])
                order.append(name)

    def place(values):
        data.qpos[:] = robot.qpos0
        for name, value in zip(order, values):
            data.qpos[addresses[name][0]] = value
        mujoco.mj_forward(robot, data)

    def direction_errors():
        out = []
        for target, (_a, _b, robot_a, robot_b, _j) in zip(targets, limbs):
            got = np.array(data.xpos[body(robot_b)]) - np.array(data.xpos[body(robot_a)])
            got /= np.linalg.norm(got)
            out.append(np.degrees(np.arccos(np.clip(got @ target, -1.0, 1.0))))
        return out

    def cost(values):
        # Limb directions leave each limb free to spin about its own axis, so the raw cost is flat
        # along that null space: the solver then stops wherever it happens to be (two runs diverged by
        # 71 deg). A tiny ridge makes the minimum unique and picks the least-twisted pose.
        place(values)
        return float(np.sum(np.square(direction_errors())) + 1e-6 * np.sum(np.square(values)))

    result = minimize(cost, np.zeros(len(order)), method='L-BFGS-B',
                      bounds=[tuple(addresses[n][1]) for n in order],
                      options={'maxiter': 2000, 'ftol': 1e-12})
    place(result.x)
    errors = direction_errors()
    mean_error = float(np.mean(errors))

    payload = {
        'robot_xml': Path(args.g1_xml).name,
        'note': ('joint angles that put this robot into the source pose; consumed by '
                 'gmr_calibrate, which derives C from it'),
        'source_pose': args.vpd or '(bind pose)',
        'joints': {n: round(float(v), 4) for n, v in zip(order, result.x)},
        'limb_direction_error_deg': round(mean_error, 3),
        'per_limb_error_deg': {src_a: round(float(err), 3)
                               for (src_a, *_rest), err in zip(limbs, errors)},
        'solve_cost': round(float(result.fun), 9),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    if args.report:
        print('source: %s   solve cost %.6f   iterations %d' % (note, result.fun, result.nit))
        print('%-8s %10s' % ('limb', 'dir err'))
        for (src_a, _b, _c, _d, _j), err in zip(limbs, errors):
            print('%-8s %9.2f' % (src_a, err))
        print('mean %.3f deg over %d segments' % (mean_error, len(limbs)))
    print('written', out)


if __name__ == '__main__':
    main()
