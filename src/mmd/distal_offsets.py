"""Derive GMR's rotation offsets: one axis for ball-joint links, two for hinge-driven links.

GMR asks for a constant ``C`` per link with ``target(t) = R_src(t) · C``. What ``C`` must get right
depends on how the robot link is driven:

* **ball joint** (upper arm, thigh): the link can take any orientation, so only the limb *direction*
  matters. ``C = rotation_between(a_local, a_src)`` - roll about the limb is free and harmless.
* **hinge** (forearm, shin, wrist, ankle): the joint angle is chosen by *projecting* the target onto
  the hinge plane, so a wrong roll silently picks a wrong angle. ``C`` needs a second axis, the hinge
  axis, measured on the human side **from the clip**: a MMD bone is a free 3-dof joint, so no declared
  axis exists, and the bind pose is straight, which makes any geometry-only second axis degenerate.

    C = B_src · B_robot^T,  B = [distal, hinge, distal x hinge] (orthonormalised)

Ball-joint links keep the single-axis form because the two-axis form measurably hurt them
(upper arm 2 deg -> 14 deg when both axes were forced on every link).

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/distal_offsets.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config runtime/gmr_inputs/glb_to_g1.json --report
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

# link -> (source bone, source child, robot child body)
BALL_LINKS = {
    'left_shoulder_yaw_link': ('左腕', '左ひじ', 'left_elbow_link'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', 'right_elbow_link'),
    'left_hip_yaw_link': ('左足', '左ひざ', 'left_knee_link'),
    'right_hip_yaw_link': ('右足', '右ひざ', 'right_knee_link'),
}

# link -> (source parent bone, source bone, source child, robot child body, robot hinge joint)
HINGE_LINKS = {
    'left_elbow_link': ('左腕', '左ひじ', '左手首', 'left_wrist_yaw_link', 'left_elbow_joint'),
    'right_elbow_link': ('右腕', '右ひじ', '右手首', 'right_wrist_yaw_link', 'right_elbow_joint'),
    'left_wrist_yaw_link': ('左ひじ', '左手首', '左中指１', 'left_wrist_roll_link',
                            'left_wrist_pitch_joint'),
    'right_wrist_yaw_link': ('右ひじ', '右手首', '右中指１', 'right_wrist_roll_link',
                             'right_wrist_pitch_joint'),
    'left_knee_link': ('左足', '左ひざ', '左足首', 'left_ankle_roll_link', 'left_knee_joint'),
    'right_knee_link': ('右足', '右ひざ', '右足首', 'right_ankle_roll_link', 'right_knee_joint'),
    'left_ankle_roll_link': ('左ひざ', '左足首', '左つま先', 'left_toe_link',
                             'left_ankle_pitch_joint'),
    'right_ankle_roll_link': ('右ひざ', '右足首', '右つま先', 'right_toe_link',
                              'right_ankle_pitch_joint'),
}


def rotation_between(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    axis = np.cross(u, v)
    sine = np.linalg.norm(axis)
    cosine = float(u @ v)
    if sine < 1e-9:
        return R.identity()
    return R.from_rotvec(np.arctan2(sine, cosine) * axis / sine)


def orthonormal(distal, hinge):
    d = distal / np.linalg.norm(distal)
    h = hinge - (hinge @ d) * d
    norm = np.linalg.norm(h)
    if norm < 1e-6:
        return None
    h /= norm
    return np.column_stack([d, h, np.cross(d, h)])


def measured_hinge_axis(rotations_parent, rotations_bone):
    """Axis of parent^-1 * bone over the clip, carried into the bone's own frame, sign-aligned."""
    axes, weights = [], []
    reference = None
    for rp, rb in zip(rotations_parent, rotations_bone):
        relative = rp.inv() * rb
        if reference is None:
            reference = relative
            continue
        rotvec = (relative * reference.inv()).as_rotvec()
        magnitude = float(np.linalg.norm(rotvec))
        if magnitude > np.radians(3.0):
            world_axis = rp.apply(rotvec / magnitude)
            axes.append(rb.inv().apply(world_axis))
            weights.append(magnitude)
    if not axes:
        return None
    axes = np.array(axes)
    weights = np.array(weights)
    anchor = axes[int(np.argmax(weights))]
    axes *= np.sign(axes @ anchor)[:, None]
    mean = (axes * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-6 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', default=None,
                        help='needed for the measured hinge axis of hinge-driven links')
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--arm-pos-weight', type=float, default=0.0,
                        help="0 keeps GMR's own convention (orientation for the arms)")
    parser.add_argument('--flip-hinge', action='store_true',
                        help='negate the measured hinge axis; its sign is arbitrary until matched to '
                             'the robot joint\'s positive direction')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mujoco.mj_forward(src, src_data)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}

    def source_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    rotations = {}
    if args.anim:
        anim = np.load(args.anim)['qpos']
        wanted = {name for spec in HINGE_LINKS.values() for name in spec[:3]}
        for name in wanted:
            if name not in index_of:
                continue
            body = source_body(name)
            sequence = []
            for frame in anim[::4]:
                src_data.qpos[:] = frame
                mujoco.mj_forward(src, src_data)
                sequence.append(R.from_matrix(np.array(src_data.xmat[body]).reshape(3, 3)))
            rotations[name] = sequence
        src_data.qpos[:] = src.qpos0
        mujoco.mj_forward(src, src_data)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    report = []

    def store(link, constant):
        q = constant.as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    for link, (bone, child, robot_child) in BALL_LINKS.items():
        if bone not in index_of or child not in index_of:
            continue
        link_body, link_child = robot_body(link), robot_body(robot_child)
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        rot_bone = np.array(src_data.xmat[source_body(bone)]).reshape(3, 3)
        a_local = rot_link.T @ (robot_data.xpos[link_child] - robot_data.xpos[link_body])
        a_src = rot_bone.T @ (src_data.xpos[source_body(child)] - src_data.xpos[source_body(bone)])
        constant = rotation_between(a_local, a_src)
        store(link, constant)
        report.append((link, float(np.degrees(constant.magnitude())), 'distal'))

    for link, (parent, bone, child, robot_child, joint) in HINGE_LINKS.items():
        if not args.anim or bone not in index_of or child not in index_of:
            continue
        if parent not in rotations or bone not in rotations:
            report.append((link, None, 'no clip data'))
            continue
        link_body, link_child = robot_body(link), robot_body(robot_child)
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        rot_bone = np.array(src_data.xmat[source_body(bone)]).reshape(3, 3)
        distal_local = rot_link.T @ (robot_data.xpos[link_child] - robot_data.xpos[link_body])
        distal_src = rot_bone.T @ (src_data.xpos[source_body(child)] -
                                   src_data.xpos[source_body(bone)])

        hinge_src = measured_hinge_axis(rotations[parent], rotations[bone])
        if hinge_src is not None and args.flip_hinge:
            hinge_src = -hinge_src
        joint_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if hinge_src is None or joint_id < 0:
            report.append((link, None, 'no hinge'))
            continue
        hinge_local = rot_link.T @ np.array(robot_data.xaxis[joint_id])

        B_src = orthonormal(distal_src, hinge_src)
        B_robot = orthonormal(distal_local, hinge_local)
        if B_src is None or B_robot is None:
            report.append((link, None, 'degenerate'))
            continue
        constant = R.from_matrix(B_src @ B_robot.T)
        store(link, constant)
        report.append((link, float(np.degrees(constant.magnitude())), 'distal+hinge'))

    official_rot = {'left_shoulder_yaw_link': 100.0, 'right_shoulder_yaw_link': 100.0}
    for link in ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
                 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link'):
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][1] = float(args.arm_pos_weight)
                config[table][link][2] = 1.0 if args.arm_pos_weight > 0 else official_rot.get(link, 10.0)

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    if args.report:
        print(f"{'link':<26}{'C (deg)':>9}  source")
        for link, angle, kind in report:
            print(f'{link:<26}{angle:8.1f}  {kind}' if angle is not None
                  else f'{link:<26}{"-":>9}  {kind}')
    print('written', config_path)


if __name__ == '__main__':
    main()
