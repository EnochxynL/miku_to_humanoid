"""Reproduce the PMX player's rest-pose alignment angles from its own model, read-only.

The player (D:\\MY_DATA\\Downloads\\mmd_mujoco_player) fixes three things our GMR pipeline got wrong:
dividing out each link's rest rotation, rotating the G1 rest limb onto the PMX rest limb (the model
rests in an A-pose, the robot hangs its arms down), and measuring where "straight" actually is. Its
log records the alignment it measured: humerus 49.7 / 50.4 deg, thigh 9.9 / 10.5 deg.

This script recomputes those four numbers from the player's own MJCF and pmx_skeleton.json without
modifying or running the project. If they agree, the alignment is understood well enough to port;
if they do not, stop and ask rather than tuning something else to fit.

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/check_pmx_align.py \
        --player D:\\MY_DATA\\Downloads\\mmd_mujoco_player
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

# copied verbatim from src/mmd_robot_player/robots.py (target_joint_map)
TARGETS = {
    '腰': 'waist_yaw_joint', '下半身': 'waist_roll_joint',
    '上半身': 'waist_pitch_joint', '上半身2': 'waist_pitch_joint',
    '左肩P': 'left_shoulder_pitch_joint', '左肩': 'left_shoulder_pitch_joint',
    '左腕': 'left_shoulder_pitch_joint', '左ひじ': 'left_elbow_joint',
    '右肩P': 'right_shoulder_pitch_joint', '右肩': 'right_shoulder_pitch_joint',
    '右腕': 'right_shoulder_pitch_joint', '右ひじ': 'right_elbow_joint',
    '左足': 'left_hip_pitch_joint', '左ひざ': 'left_knee_joint', '左足首': 'left_ankle_pitch_joint',
    '右足': 'right_hip_pitch_joint', '右ひざ': 'right_knee_joint', '右足首': 'right_ankle_pitch_joint',
}
# copied verbatim from src/mmd_robot_player/robots.py (segment_align)
SEGMENT = {
    'left_shoulder_pitch_joint': {'pmx': ('左腕', '左ひじ'),
                                  'g1_start': 'left_shoulder_yaw_link', 'g1_child': 'left_elbow_link'},
    'right_shoulder_pitch_joint': {'pmx': ('右腕', '右ひじ'),
                                   'g1_start': 'right_shoulder_yaw_link', 'g1_child': 'right_elbow_link'},
    'left_hip_pitch_joint': {'pmx': ('左足', '左ひざ'), 'g1_child': 'left_knee_link'},
    'right_hip_pitch_joint': {'pmx': ('右足', '右ひざ'), 'g1_child': 'right_knee_link'},
}
RECORDED = {'left_shoulder_pitch_joint': 49.7, 'right_shoulder_pitch_joint': 50.4,
            'left_hip_pitch_joint': 9.9, 'right_hip_pitch_joint': 10.5}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', required=True)
    args = parser.parse_args()
    player = Path(args.player)

    model = mujoco.MjModel.from_xml_path(
        str(player / 'models/unitree_g1/xmls/g1_pmx_fk.xml'))
    skeleton = json.loads(
        (player / 'models/unitree_g1/pmx_skeleton.json').read_text(encoding='utf-8'))
    bones = {entry['name']: entry for entry in skeleton['bones']}

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    # probe pose: every driven joint set to its rest inverse, exactly as bind_targets does
    applied = set()
    for target in sorted(set(TARGETS.values())):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        if joint_id < 0:
            print(f'missing joint {target}')
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_BALL:
            continue
        body = int(model.jnt_bodyid[joint_id])
        rest = np.array(model.body_quat[body], dtype=float)
        rest = rest / np.linalg.norm(rest)
        conjugate = np.array([rest[0], -rest[1], -rest[2], -rest[3]])
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address:address + 4] = conjugate
        applied.add(target)
    mujoco.mj_kinematics(model, data)

    def body_position(name):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if body < 0 else np.array(data.xpos[body], dtype=float)

    def unit(vector):
        return vector / (np.linalg.norm(vector) or 1.0)

    print(f"{'joint':<28}{'recomputed':>12}{'recorded':>10}{'diff':>8}")
    for target, spec in SEGMENT.items():
        bone_start, bone_end = spec['pmx']
        start_entry, end_entry = bones.get(bone_start), bones.get(bone_end)
        if start_entry is None or end_entry is None:
            print(f'{target:<28}  missing PMX bone')
            continue
        pmx_start = body_position(f"pmx_body_{start_entry['index']:03d}")
        pmx_end = body_position(f"pmx_body_{end_entry['index']:03d}")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        driven = int(model.jnt_bodyid[joint_id])
        limb_start = body_position(spec['g1_start']) if spec.get('g1_start') else np.array(data.xpos[driven])
        limb_end = body_position(spec['g1_child'])
        if any(v is None for v in (pmx_start, pmx_end, limb_start, limb_end)):
            print(f'{target:<28}  missing body')
            continue
        g1_direction = unit(limb_end - limb_start)
        pmx_direction = unit(pmx_end - pmx_start)
        angle = float(np.degrees(np.arccos(np.clip(g1_direction @ pmx_direction, -1, 1))))
        print(f'{target:<28}{angle:11.1f} {RECORDED[target]:9.1f} {angle - RECORDED[target]:7.1f}')
    print('\nprobe applied to joints:', len(applied))


if __name__ == '__main__':
    main()
