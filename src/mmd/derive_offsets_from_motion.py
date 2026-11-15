"""Derive GMR offsets as a frame convention: distal axis + the axis of the joint that DRIVES the link.

Correcting the earlier version: the second axis must be the joint that drives the link itself
(``left_elbow_joint`` for the forearm, not the wrist), and that axis has to be expressed in the
link's own frame - the MJCF stores it in world coordinates at rest, so it is rotated in by the
link's rest rotation. The human side measures the same thing from the clip: the axis of
``R_parent(t)^-1 R_bone(t)``, carried into the bone's own frame and averaged.

Acceptance: the last frame rendered with the camera at azimuth 88.1 deg must look right.

    uv run python src/mmd/derive_offsets_from_motion.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --base runtime/gmr_inputs/glb_base.json \
        --out runtime/gmr_inputs/glb_to_g1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))

# link -> (robot joint that drives it, MMD bone it maps to, MMD parent bone, MMD child bone)
CHAIN = {
    'left_shoulder_yaw_link': ('left_shoulder_yaw_joint', '左腕', '左肩', '左ひじ'),
    'left_elbow_link': ('left_elbow_joint', '左ひじ', '左腕', '左手首'),
    'left_wrist_yaw_link': ('left_wrist_yaw_joint', '左手首', '左ひじ', '左中指１'),
    'right_shoulder_yaw_link': ('right_shoulder_yaw_joint', '右腕', '右肩', '右ひじ'),
    'right_elbow_link': ('right_elbow_joint', '右ひじ', '右腕', '右手首'),
    'right_wrist_yaw_link': ('right_wrist_yaw_joint', '右手首', '右ひじ', '右中指１'),
    'left_hip_yaw_link': ('left_hip_yaw_joint', '左足', '下半身', '左ひざ'),
    'left_knee_link': ('left_knee_joint', '左ひざ', '左足', '左足首'),
    'left_ankle_roll_link': ('left_ankle_roll_joint', '左足首', '左ひざ', '左つま先'),
    'right_hip_yaw_link': ('right_hip_yaw_joint', '右足', '下半身', '右ひざ'),
    'right_knee_link': ('right_knee_joint', '右ひざ', '右足', '右足首'),
    'right_ankle_roll_link': ('right_ankle_roll_joint', '右足首', '右ひざ', '右つま先'),
}


def orthonormal(distal, bend):
    d = distal / (np.linalg.norm(distal) or 1.0)
    b = bend - (bend @ d) * d
    n = np.linalg.norm(b)
    if n < 1e-6:
        return None
    b /= n
    return np.column_stack([d, b, np.cross(d, b)])


def mean_bend_axis(frames_rot, parent_rot, own_rot):
    """Axis of parent^-1 * own, carried into the bone's own frame, averaged over frames."""
    axes, weights = [], []
    reference = None
    for rp, ro in zip(parent_rot, own_rot):
        relative = rp.inv() * ro
        if reference is None:
            reference = relative
            continue
        rotvec = (relative * reference.inv()).as_rotvec()
        magnitude = float(np.linalg.norm(rotvec))
        if magnitude > np.radians(3.0):
            world_axis = rp.apply(rotvec / magnitude)
            axes.append(ro.inv().apply(world_axis))
            weights.append(magnitude)
    if not axes:
        return None
    axes = np.array(axes)
    weights = np.array(weights)
    anchor = axes[int(np.argmax(weights))]
    axes = axes * np.sign(axes @ anchor)[:, None]
    mean = (axes * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-6 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--base', required=True, help='config with the right bone names/weights')
    parser.add_argument('--out', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    args = parser.parse_args()

    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}
    anim = np.load(args.anim)['qpos']

    def body_of(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    config = json.loads(Path(args.base).read_text(encoding='utf-8'))
    report = []

    # cache per-bone world rotations across the clip (sampled) for the bend measurement
    sample = list(range(0, len(anim), 4))
    rotations = {}
    for name in {v[1] for v in CHAIN.values()} | {v[2] for v in CHAIN.values()}:
        if name not in index_of:
            continue
        body = body_of(name)
        seq = []
        for i in sample:
            src_data.qpos[:] = anim[i]
            mujoco.mj_forward(src, src_data)
            seq.append(R.from_matrix(np.array(src_data.xmat[body]).reshape(3, 3)))
        rotations[name] = seq

    src_data.qpos[:] = anim[0]
    mujoco.mj_forward(src, src_data)

    for link, (joint_name, bone, parent_bone, child_bone) in CHAIN.items():
        if bone not in index_of or child_bone not in index_of or parent_bone not in index_of:
            continue
        bone_body, child_body = body_of(bone), body_of(child_bone)
        rot_bone = R.from_matrix(np.array(src_data.xmat[bone_body]).reshape(3, 3))
        distal_src = rot_bone.inv().apply(src_data.xpos[child_body] - src_data.xpos[bone_body])
        bend_src = mean_bend_axis(None, rotations[parent_bone], rotations[bone])
        if bend_src is None:
            report.append((link, None, 'no bend'))
            continue

        link_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, link)
        robot_child = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY,
                                        {'left_shoulder_yaw_link': 'left_elbow_link',
                                         'left_elbow_link': 'left_wrist_yaw_link',
                                         'left_wrist_yaw_link': 'left_wrist_roll_link',
                                         'right_shoulder_yaw_link': 'right_elbow_link',
                                         'right_elbow_link': 'right_wrist_yaw_link',
                                         'right_wrist_yaw_link': 'right_wrist_roll_link',
                                         'left_hip_yaw_link': 'left_knee_link',
                                         'left_knee_link': 'left_ankle_roll_link',
                                         'left_ankle_roll_link': 'left_toe_link',
                                         'right_hip_yaw_link': 'right_knee_link',
                                         'right_knee_link': 'right_ankle_roll_link',
                                         'right_ankle_roll_link': 'right_toe_link'}[link])
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        distal_robot = rot_link.T @ (robot_data.xpos[robot_child] - robot_data.xpos[link_body])
        joint_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            report.append((link, None, 'no joint'))
            continue
        # the joint axis lives in the PARENT link's frame; carry it into this link's frame
        parent_body = int(robot.jnt_bodyid[joint_id])
        rot_parent = np.array(robot_data.xmat[parent_body]).reshape(3, 3)
        axis_parent = rot_parent.T @ np.array(robot_data.xaxis[joint_id])
        bend_robot = rot_link.T @ (rot_parent @ axis_parent)

        B_src = orthonormal(distal_src, bend_src)
        B_robot = orthonormal(distal_robot, bend_robot)
        if B_src is None or B_robot is None:
            report.append((link, None, 'degenerate'))
            continue
        constant = R.from_matrix(B_src @ B_robot.T)
        q = constant.as_quat()
        entry_bone = config['ik_match_table1'].get(link, [bone])[0]
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config[table]:
                config[table][link][0] = entry_bone
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        report.append((link, float(np.degrees(constant.magnitude())), 'two-axis'))

    out = Path(args.out)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f"{'link':<26}{'C (deg)':>9}  note")
    for link, angle, note in report:
        print(f'{link:<26}{angle:8.1f}  {note}' if angle is not None else f'{link:<26}{"-":>9}  {note}')
    print('written', out)


if __name__ == '__main__':
    main()
