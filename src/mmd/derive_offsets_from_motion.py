"""Derive GMR's rotation offsets as a pure frame convention: segment direction + measured bend axis.

The offset ``C`` carries a source bone's local frame onto a robot link's local frame. Both are rigidly
attached to the same physical segment, so ``C`` is a constant that depends on the two *frame
conventions* only - not on any pose. It is fixed by two directions per segment:

  * **distal axis** - bone -> child, in each side's own local frame (geometry);
  * **bend axis**   - the axis the child joint actually rotates about, in the parent's local frame.

The bend axis on the human side has to be *measured from the clip*: a MMD bone is a free 3-dof joint,
so no declared axis exists, and at the bind pose the limb is straight, which makes any
geometry-only second axis degenerate (that is why earlier attempts collapsed to identity). The
robot side gets its axis directly from the MJCF joint.

    uv run python src/mmd/derive_offsets_from_motion.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --out runtime/gmr_inputs/glb_to_g1.json
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
from glb_to_gmr import LINKS, MEASURE   # noqa: E402

# robot link -> (child joint used for the bend axis, source child bone)
BEND = {
    'left_shoulder_yaw_link': ('left_elbow_joint', '左ひじ'),
    'left_elbow_link': ('left_wrist_pitch_joint', '左手首'),
    'left_wrist_yaw_link': ('left_wrist_roll_joint', '左中指１'),
    'right_shoulder_yaw_link': ('right_elbow_joint', '右ひじ'),
    'right_elbow_link': ('right_wrist_pitch_joint', '右手首'),
    'right_wrist_yaw_link': ('right_wrist_roll_joint', '右中指１'),
    'left_hip_yaw_link': ('left_knee_joint', '左ひざ'),
    'left_knee_link': ('left_ankle_pitch_joint', '左足首'),
    'left_ankle_roll_link': ('left_ankle_roll_joint', '左つま先'),
    'right_hip_yaw_link': ('right_knee_joint', '右ひざ'),
    'right_knee_link': ('right_ankle_pitch_joint', '右足首'),
    'right_ankle_roll_link': ('right_ankle_roll_joint', '右つま先'),
}


def orthonormal(distal, bend):
    d = distal / (np.linalg.norm(distal) or 1.0)
    b = bend - (bend @ d) * d
    n = np.linalg.norm(b)
    if n < 1e-6:
        return None
    b /= n
    return np.column_stack([d, b, np.cross(d, b)])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--base', default=None, help='config to keep weights/scales from')
    parser.add_argument('--out', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    args = parser.parse_args()

    gmr_root = Path(args.gmr_root)
    base_path = Path(args.base) if args.base else (
        gmr_root / 'general_motion_retargeting' / 'ik_configs' / 'bvh_lafan1_to_g1.json')

    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}
    anim = np.load(args.anim)['qpos']

    def src_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    # --- source: distal axis and measured bend axis, both in the bone's local frame ------------
    src_data.qpos[:] = anim[0]
    mujoco.mj_forward(src, src_data)
    distal_src, bend_src = {}, {}
    for link, (bone, child, robot_child) in LINKS.items():
        if bone not in index_of or child not in index_of:
            continue
        bone_body, child_body = src_body(bone), src_body(child)
        rot_bone = R.from_matrix(np.array(src_data.xmat[bone_body]).reshape(3, 3))
        world = src_data.xpos[child_body] - src_data.xpos[bone_body]
        distal_src[link] = rot_bone.inv().apply(world)

        # measured bend axis: axis of R_bone(t)^-1 R_child(t), averaged over the clip, sign aligned
        axes, weights = [], []
        reference = None
        for qpos in anim[::2]:
            src_data.qpos[:] = qpos
            mujoco.mj_forward(src, src_data)
            rb_ = R.from_matrix(np.array(src_data.xmat[bone_body]).reshape(3, 3))
            rc_ = R.from_matrix(np.array(src_data.xmat[child_body]).reshape(3, 3))
            relative = rb_.inv() * rc_
            if reference is None:
                reference = relative
                continue
            delta = relative * reference.inv()
            rotvec = delta.as_rotvec()
            magnitude = float(np.linalg.norm(rotvec))
            if magnitude > np.radians(3.0):
                axes.append(rotvec / magnitude)
                weights.append(magnitude)
        if not axes:
            continue
        axes = np.array(axes)
        weights = np.array(weights)
        anchor = axes[int(np.argmax(weights))]
        axes = axes * np.sign(axes @ anchor)[:, None]
        mean = (axes * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-6:
            bend_src[link] = mean / norm

    # --- robot: same two axes, from geometry and the joint axis ---------------------------------
    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)
    distal_robot, bend_robot = {}, {}
    for link, spec in BEND.items():
        if link not in LINKS:
            continue
        bone, child, robot_child = LINKS[link]
        link_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, link)
        child_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_child)
        if link_body < 0 or child_body < 0:
            continue
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        distal_robot[link] = rot_link.T @ (robot_data.xpos[child_body] - robot_data.xpos[link_body])
        joint_name = spec[0]
        joint_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            bend_robot[link] = rot_link.T @ np.array(robot_data.xaxis[joint_id])

    # --- constants ------------------------------------------------------------------------------
    config = json.loads(base_path.read_text(encoding='utf-8'))
    config['human_root_name'] = '腰'
    config['ground_height'] = 0.0
    src_data.qpos[:] = anim[0]
    mujoco.mj_forward(src, src_data)
    config['human_height_assumption'] = float(src_data.xpos[src_body('頭')][2])
    scales = {'腰': 1.0}
    for link, (bone, child, robot_child) in LINKS.items():
        if link == 'pelvis':
            continue
        link_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, link)
        child_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_child)
        robot_length = np.linalg.norm(robot_data.xpos[child_body] - robot_data.xpos[link_body])
        src_length = np.linalg.norm(src_data.xpos[src_body(child)] - src_data.xpos[src_body(bone)])
        if src_length > 1e-6:
            scales.setdefault(bone, float(robot_length / src_length))
    config['human_scale_table'] = scales

    report = []
    for table in ('ik_match_table1', 'ik_match_table2'):
        for link in list(config[table].keys()):
            if link not in LINKS:
                continue
            bone = LINKS[link][0]
            if link in distal_src and link in distal_robot and link in bend_src and link in bend_robot:
                B_src = orthonormal(distal_src[link], bend_src[link])
                B_robot = orthonormal(distal_robot[link], bend_robot[link])
                if B_src is None or B_robot is None:
                    continue
                constant = R.from_matrix(B_src @ B_robot.T)
                report.append((link, float(np.degrees(constant.magnitude())), 'two-axis'))
            elif link in ('pelvis', 'torso_link'):
                # spine: yaw only, from the body lateral axis (its joints have no single hinge)
                src_axis = (src_data.xpos[src_body('右足')] - src_data.xpos[src_body('左足')])
                robot_axis = (robot_data.xpos[mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY,
                                                                'right_hip_yaw_link')] -
                              robot_data.xpos[mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY,
                                                                'left_hip_yaw_link')])
                constant = R.from_rotvec(np.arctan2(np.cross(robot_axis, src_axis)[2],
                                                   robot_axis @ src_axis) * np.array([0, 0, 1.0]))
                report.append((link, float(np.degrees(constant.magnitude())), 'yaw'))
            else:
                continue
            q = constant.as_quat()
            config[table][link][0] = bone
            config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print(f"{'link':<26}{'C (deg)':>9}  source")
    for link, angle, kind in report:
        print(f'{link:<26}{angle:8.1f}  {kind}')
    print('written', out)


if __name__ == '__main__':
    main()
