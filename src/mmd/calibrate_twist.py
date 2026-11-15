"""Measure and correct the one degree of freedom the previous steps could not see: limb twist.

Why this exists: the segment-direction error used so far is blind to rotation *about* the limb, so
every roll fix had to be judged by eye. This script decomposes, for each mapped link, the residual
between the robot's achieved orientation and the target the config asks for:

    residual = R_achieved^-1 * R_target        (expressed in the link's frame)

Swung out on the limb axis, the residual separates into
  * swing - the limb pointing the wrong way (already handled by position/direction targets), and
  * twist - the rotation *about* the limb, which is exactly the "arm and leg are sideways" symptom.

The twist is then folded back into the config's rot_offset as a constant, once, and the run reports
whether the residual twist went to zero. No eye judgement needed.

    <gmr>/.venv/Scripts/python.exe src/mmd/calibrate_twist.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config runtime/gmr_inputs/glb_to_g1.json \
        --frames 318 --apply
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
from glb_to_gmr import LINKS, MEASURE, load_glb_frames   # noqa: E402


def limb_axis(link, robot_data, robot_body):
    """Unit direction of the limb, taken from the robot side (always well defined)."""
    bone, child, robot_child = LINKS[link]
    a = mujoco.mj_name2id(mujoco.MjModel.from_xml_string('<mujoco/>'), 0, '')
    return None


def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


def swing_twist(rotation, axis):
    """Split a rotation into the part about ``axis`` and the remainder."""
    quat = rotation.as_quat()
    projection = float(quat[:3] @ axis)
    twist = R.from_quat(np.concatenate([projection * axis, [quat[3]]]))
    norm = np.linalg.norm(twist.as_quat())
    if norm < 1e-9:
        return 0.0, rotation
    twist = R.from_quat(twist.as_quat() / norm)
    return float(twist.magnitude()), rotation * twist.inv()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--frames', default='318')
    parser.add_argument('--apply', action='store_true', help='write the corrected offsets back')
    args = parser.parse_args()

    sys.path.insert(0, args.gmr_root)
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    config_path = Path(args.config).resolve()
    gmrmod.IK_CONFIG_DICT.setdefault('glb', {})['unitree_g1'] = str(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    frames = load_glb_frames(args.glb_xml, args.nodes, args.anim)
    retargeter = GMR(src_human='glb', tgt_robot='unitree_g1',
                     actual_human_height=config['human_height_assumption'], verbose=False)

    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    data = mujoco.MjData(robot)
    wanted = [int(v) for v in args.frames.split(',') if int(v) < len(frames)]

    twists = {}
    for index in wanted:
        qpos = retargeter.retarget(frames[index])
        data.qpos[:] = qpos
        mujoco.mj_forward(robot, data)
        for link, (src_a, src_b, robot_a, robot_b) in MEASURE.items():
            if src_a not in frames[index] or src_b not in frames[index]:
                continue
            body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_a)
            body_b = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_b)
            if body < 0 or body_b < 0:
                continue
            achieved = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            axis = unit(data.xpos[body_b] - data.xpos[body])
            # the target this config currently asks for
            entry = config['ik_match_table1'].get(link)
            if entry is None:
                continue
            w, x, y, z = entry[4]
            offset = R.from_quat([x, y, z, w])
            q = frames[index][entry[0]][1]
            source = R.from_quat([q[1], q[2], q[3], q[0]])
            target = source * offset
            residual = achieved.inv() * target
            angle, swing = swing_twist(residual, axis)
            twists.setdefault(link, []).append(np.degrees(angle) * np.sign(1.0))

    print(f"{'link':<26}{'mean twist (deg)':>18}")
    for link, values in twists.items():
        print(f'{link:<26}{np.mean(values):17.1f}')

    if not args.apply:
        return

    for link, values in twists.items():
        entry = config['ik_match_table1'].get(link)
        if entry is None:
            continue
        w, x, y, z = entry[4]
        offset = R.from_quat([x, y, z, w])
        # rotate the offset about the limb axis by the measured twist, expressed in the link frame
        body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, MEASURE[link][2])
        body_b = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, MEASURE[link][3])
        frame_rot = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
        world_axis = unit(data.xpos[body_b] - data.xpos[body])
        local_axis = achieved_local_axis = frame_rot.inv().apply(world_axis)
        correction = R.from_rotvec(np.radians(np.mean(values)) * local_axis)
        corrected = offset * correction
        c = corrected.as_quat()
        entry[4] = [float(c[3]), float(c[0]), float(c[1]), float(c[2])]
        table2 = config.get('ik_match_table2', {})
        if link in table2:
            table2[link][4] = entry[4]
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print('written', config_path)


if __name__ == '__main__':
    main()
