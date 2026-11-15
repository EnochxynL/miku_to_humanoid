"""Set GMR's per-link rotation offsets to the minimal distal-axis correspondence.

GMR wants a constant ``C`` per link with ``target(t) = R_src(t) · C``. The only thing ``C`` must get
right for the pose to look correct is the *direction of the limb*, which fixes two of its three
degrees of freedom:

    C · a_local = a_src        a_local: limb axis in the robot link's rest frame
                               a_src:   limb axis in the source bone's local frame

so ``C = rotation_between(a_local, a_src)``. The third degree of freedom (roll about the limb) is
left free, which is harmless - a hinge or a limb's roll does not bend it.

The previous construction (``src_bind^-1 · rest · align``) also folded in a 180-ish yaw and put the
arms 43-65 degrees off, which is what made every limb look twisted to one side.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/distal_offsets.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --config runtime/gmr_inputs/glb_to_g1.json
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

# link -> (source bone, source child) - the limb whose axis the link carries
CHAINS = {
    'left_shoulder_yaw_link': ('左腕', '左ひじ'),
    'left_elbow_link': ('左ひじ', '左手首'),
    'left_wrist_yaw_link': ('左手首', '左中指１'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ'),
    'right_elbow_link': ('右ひじ', '右手首'),
    'right_wrist_yaw_link': ('右手首', '右中指１'),
    'left_hip_yaw_link': ('左足', '左ひざ'),
    'left_knee_link': ('左ひざ', '左足首'),
    'left_ankle_roll_link': ('左足首', '左つま先'),
    'right_hip_yaw_link': ('右足', '右ひざ'),
    'right_knee_link': ('右ひざ', '右足首'),
    'right_ankle_roll_link': ('右足首', '右つま先'),
}

ROBOT_CHILD = {
    'left_shoulder_yaw_link': 'left_elbow_link',
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
    'right_ankle_roll_link': 'right_toe_link',
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--arm-pos-weight', type=float, default=300.0,
                        help='drive the arm chain by position instead of orientation: a hinge elbow '
                             'cannot satisfy a full 3-dof orientation target, and with orientation '
                             'only the forearm stayed 60-90 deg off')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mujoco.mj_forward(src, src_data)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}

    def source_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY,
                                 f'glb_body_{index_of[name]:03d}')

    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    report = []
    for link, (bone, child) in CHAINS.items():
        if bone not in index_of or child not in index_of:
            continue
        bone_body, child_body = source_body(bone), source_body(child)
        link_body, link_child = robot_body(link), robot_body(ROBOT_CHILD[link])
        if link_body < 0 or link_child < 0:
            continue
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        rot_bone = np.array(src_data.xmat[bone_body]).reshape(3, 3)
        a_local = rot_link.T @ (robot_data.xpos[link_child] - robot_data.xpos[link_body])
        a_src = rot_bone.T @ (src_data.xpos[child_body] - src_data.xpos[bone_body])
        if np.linalg.norm(a_local) < 1e-6 or np.linalg.norm(a_src) < 1e-6:
            continue
        constant = rotation_between(a_local, a_src)
        q = constant.as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        report.append((link, float(np.degrees(constant.magnitude()))))

    # arms: position-driven. Measured effect on the forearm direction error: 61/92 -> 11/38 deg.
    if args.arm_pos_weight > 0:
        for link in ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
                     'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link'):
            for table in ('ik_match_table1', 'ik_match_table2'):
                if link in config.get(table, {}):
                    config[table][link][1] = float(args.arm_pos_weight)
                    config[table][link][2] = 1.0
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    if args.report:
        print(f"{'link':<26}{'C (deg)':>9}")
        for link, angle in report:
            print(f'{link:<26}{angle:8.1f}')
    print('written', config_path)


if __name__ == '__main__':
    main()
