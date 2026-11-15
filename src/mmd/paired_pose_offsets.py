"""Derive per-link offsets from the MMD bind pose, by posing the robot to match it.

The key property of an MMD rig: **bones carry no rest rotation**, so at the bind pose every bone's
world orientation is the identity. That collapses GMR's offset definition to one reading:

    target(t) = R_src(t) · C     and at bind R_src = I   =>   C = R_link(matched pose)

so C is simply *the robot link's world orientation while the robot is standing like the MMD bind
pose*. No fitting over an animation, no second axis, no roll search.

The robot cannot reproduce the source pose exactly (fewer DOF, different proportions), so the pose is
solved to match the **limb directions** - the part that is directly measurable on both sides - and the
residual mismatch is what C absorbs. Arms in the MMD bind pose are 54 deg off vertical, which keeps
the limbs unambiguous (not a degenerate straight-limb pose).

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/paired_pose_offsets.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --config runtime/gmr_inputs/glb_to_g1.json --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glb_to_gmr import LINKS   # noqa: E402

# limb (source bone, source child, robot link, robot child body) and the robot joints that move it
LIMBS = [
    ('左腕', '左ひじ', 'left_shoulder_yaw_link', 'left_elbow_link',
     ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint']),
    ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link', ['left_elbow_joint']),
    ('右腕', '右ひじ', 'right_shoulder_yaw_link', 'right_elbow_link',
     ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']),
    ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link', ['right_elbow_joint']),
    ('左足', '左ひざ', 'left_hip_yaw_link', 'left_knee_link',
     ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint']),
    ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link', ['left_knee_joint']),
    ('右足', '右ひざ', 'right_hip_yaw_link', 'right_knee_link',
     ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']),
    ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link', ['right_knee_joint']),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--arm-pos-weight', type=float, default=0.0,
                        help='0 keeps GMR\'s own convention: orientation only for the arms')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    # --- source: MMD bind pose (every bone frame is the identity) --------------------------------
    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mujoco.mj_forward(src, src_data)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}

    def source_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    # --- robot ----------------------------------------------------------------------------------
    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    joint_address = {}
    for _, _, _, _, joints in LIMBS:
        for name in joints:
            jid = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                joint_address[name] = (int(robot.jnt_qposadr[jid]), robot.jnt_range[jid])
                robot_data.qpos[int(robot.jnt_qposadr[jid])] = robot.qpos0[int(robot.jnt_qposadr[jid])]
    names = list(joint_address.keys())

    targets = []
    for src_a, src_b, robot_a, robot_b, _joints in LIMBS:
        a = src_data.xpos[source_body(src_b)] - src_data.xpos[source_body(src_a)]
        targets.append(a / np.linalg.norm(a))

    def unpack(values):
        robot_data.qpos[:] = robot.qpos0
        for name, value in zip(names, values):
            robot_data.qpos[joint_address[name][0]] = value
        mujoco.mj_forward(robot, robot_data)

    def cost(values):
        unpack(values)
        total = 0.0
        for target, (_, _, robot_a, robot_b, _joints) in zip(targets, LIMBS):
            got = robot_data.xpos[robot_body(robot_b)] - robot_data.xpos[robot_body(robot_a)]
            got /= np.linalg.norm(got)
            total += np.degrees(np.arccos(np.clip(got @ target, -1.0, 1.0))) ** 2
        return total

    x0 = np.array([0.0] * len(names))
    bounds = [tuple(joint_address[n][1]) for n in names]
    result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12})
    unpack(result.x)

    errors = []
    for target, (_, _, robot_a, robot_b, _joints) in zip(targets, LIMBS):
        got = robot_data.xpos[robot_body(robot_b)] - robot_data.xpos[robot_body(robot_a)]
        got /= np.linalg.norm(got)
        errors.append(np.degrees(np.arccos(np.clip(got @ target, -1.0, 1.0))))

    # --- C = the robot link's orientation in the matched pose -----------------------------------
    bone_for_link = {link: spec[0] for link, spec in LINKS.items()}
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    report = []
    for link in list(config['ik_match_table1'].keys()):
        if link not in LINKS:
            continue
        body = robot_body(link)
        if body < 0:
            continue
        # C = R_src(bind)^-1 * R_link(matched). The identity-bind property that makes this a single
        # reading holds for PMX, but the GLB carries Rx90 on every node, so it has to be divided out
        # or the whole pose comes out 90 deg off (measured: arms 48 deg instead of 2).
        source_bind = R.identity()
        if bone_for_link.get(link) and bone_for_link[link] in index_of:
            source_bind = R.from_matrix(np.array(
                src_data.xmat[source_body(bone_for_link[link])]).reshape(3, 3))
        constant = source_bind.inv() * R.from_matrix(np.array(robot_data.xmat[body]).reshape(3, 3))
        q = constant.as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        report.append((link, constant.magnitude() * 180.0 / np.pi))

    official = {'left_shoulder_yaw_link': 100.0, 'right_shoulder_yaw_link': 100.0}
    for link in ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
                 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link'):
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][1] = float(args.arm_pos_weight)
                config[table][link][2] = (1.0 if args.arm_pos_weight > 0
                                          else official.get(link, 10.0))
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')

    if args.report:
        print('pose solve cost %.3f   iterations %d' % (result.fun, result.nit))
        print('%-8s %10s' % ('limb', 'dir err'))
        for (src_a, _b, _c, _d, _j), err in zip(LIMBS, errors):
            print('%-8s %9.1f' % (src_a, err))
        print()
        print(f"{'link':<26}{'C (deg)':>9}")
        for link, angle in report:
            print(f'{link:<26}{angle:8.1f}')
    print('written', config_path)


if __name__ == '__main__':
    main()
