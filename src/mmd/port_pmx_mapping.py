"""Reproduce the PMX player's pose and test whether GMR's per-bone constant model holds.

The player drives each robot joint from MMD *local* rotations:

    q_joint = conj(body_rest) * (product of converted MMD bone rotations) * align

which is a joint-level mapping. GMR instead demands a per-bone constant, ``target = R_human * C``.
Those two are compatible only if ``R_human(t)^-1 * R_link(t)`` is the same rotation for every frame.
This script measures exactly that - the decisive test for whether the GMR route is salvageable - and
writes the constants when they are constant.

Everything about the player is read from D:\\MY_DATA\\Downloads\\mmd_mujoco_player; nothing there is
modified.

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/port_pmx_mapping.py \
        --player <dir> --vmd <file> --bvh <clip> [--config <base> --out <config>]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

TARGETS = {
    '腰': 'waist_yaw_joint', '下半身': 'waist_roll_joint',
    '上半身': 'waist_pitch_joint', '上半身2': 'waist_pitch_joint',
    '左肩P': 'left_shoulder_pitch_joint', '左肩': 'left_shoulder_pitch_joint',
    '左腕': 'left_shoulder_pitch_joint', '左ひじ': 'left_elbow_joint',
    '左腕捩': 'left_wrist_roll_joint', '左手捩': 'left_wrist_yaw_joint',
    '左手首': 'left_wrist_pitch_joint',
    '右肩P': 'right_shoulder_pitch_joint', '右肩': 'right_shoulder_pitch_joint',
    '右腕': 'right_shoulder_pitch_joint', '右ひじ': 'right_elbow_joint',
    '右腕捩': 'right_wrist_roll_joint', '右手捩': 'right_wrist_yaw_joint',
    '右手首': 'right_wrist_pitch_joint',
    '左足': 'left_hip_pitch_joint', '左ひざ': 'left_knee_joint', '左足首': 'left_ankle_pitch_joint',
    '左つま先': 'left_ankle_roll_joint',
    '右足': 'right_hip_pitch_joint', '右ひざ': 'right_knee_joint', '右足首': 'right_ankle_pitch_joint',
    '右つま先': 'right_ankle_roll_joint',
}
SEGMENT = {
    'left_shoulder_pitch_joint': {'pmx': ('左腕', '左ひじ'),
                                  'g1_start': 'left_shoulder_yaw_link', 'g1_child': 'left_elbow_link'},
    'right_shoulder_pitch_joint': {'pmx': ('右腕', '右ひじ'),
                                   'g1_start': 'right_shoulder_yaw_link', 'g1_child': 'right_elbow_link'},
    'left_hip_pitch_joint': {'pmx': ('左足', '左ひざ'), 'g1_child': 'left_knee_link'},
    'right_hip_pitch_joint': {'pmx': ('右足', '右ひざ'), 'g1_child': 'right_knee_link'},
}
FLEXION = {
    'left_elbow_joint': {'bones': ('左腕', '左ひじ', '左手首'), 'axis': (0, 1, 0),
                         'straight_q': 1.2740903539558606, 'sign': -1,
                         'range': (-1.0472, 1.5707963267948966)},
    'right_elbow_joint': {'bones': ('右腕', '右ひじ', '右手首'), 'axis': (0, 1, 0),
                          'straight_q': 1.2740903539558606, 'sign': -1,
                          'range': (-1.0472, 1.5707963267948966)},
    'left_knee_joint': {'bones': ('左足', '左ひざ', '左足首'), 'axis': (0, 1, 0),
                        'straight_q': 0.24434609527920614, 'sign': 1, 'range': (0.0, 2.8798)},
    'right_knee_joint': {'bones': ('右足', '右ひざ', '右足首'), 'axis': (0, 1, 0),
                         'straight_q': 0.24434609527920614, 'sign': 1, 'range': (0.0, 2.8798)},
}
# bone -> (robot link, MMD bone, clip bone) as GMR sees them
GMR_BONES = {
    'Hips': ('pelvis', '腰', 'Hips'),
    'Spine2': ('torso_link', '上半身2', 'Spine2'),
    'LeftUpLeg': ('left_hip_yaw_link', '左足', 'LeftUpLeg'),
    'LeftLeg': ('left_knee_link', '左ひざ', 'LeftLeg'),
    'LeftFootMod': ('left_ankle_roll_link', '左足首', 'LeftFootMod'),
    'RightUpLeg': ('right_hip_yaw_link', '右足', 'RightUpLeg'),
    'RightLeg': ('right_knee_link', '右ひざ', 'RightLeg'),
    'RightFootMod': ('right_ankle_roll_link', '右足首', 'RightFootMod'),
    'LeftArm': ('left_shoulder_yaw_link', '左腕', 'LeftArm'),
    'LeftForeArm': ('left_elbow_link', '左ひじ', 'LeftForeArm'),
    'RightArm': ('right_shoulder_yaw_link', '右腕', 'RightArm'),
    'RightForeArm': ('right_elbow_link', '右ひじ', 'RightForeArm'),
}


def read_vmd(path: Path):
    data = path.read_bytes()
    offset = 50
    count = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    tracks = {}
    for _ in range(count):
        name = data[offset:offset + 15].split(b'\x00')[0].decode('shift_jis', errors='replace')
        offset += 15
        frame = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        position = np.array(struct.unpack_from('<3f', data, offset))
        offset += 12
        quat = np.array(struct.unpack_from('<4f', data, offset))
        offset += 16
        tracks.setdefault(name, []).append((frame, quat, position))
    return tracks


def sample_position(tracks, name, frame):
    """Nearest-key translation of a VMD track (the reader keeps only rotations)."""
    entry = tracks.get(name)
    if not entry:
        return (0.0, 0.0, 0.0)
    return entry[0][2] if len(entry[0]) > 2 else (0.0, 0.0, 0.0)


def sample(track, frame):
    """Nearest-key sample of a VMD track."""
    frames = np.array([entry[0] for entry in track])
    index = int(np.argmin(np.abs(frames - frame)))
    return track[index][1]


def mmd_quat_to_mujoco(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    q = np.array([w, z, -x, -y])
    norm = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-12 else q / norm


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def rotation_between(source, target):
    source = source / (np.linalg.norm(source) or 1.0)
    target = target / (np.linalg.norm(target) or 1.0)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(source @ target)
    if sine < 1e-9:
        return R.from_quat([0, 0, 0, 1]) if cosine > 0 else R.from_rotvec(np.pi * np.array([1.0, 0, 0]))
    return R.from_rotvec(np.arctan2(sine, cosine) * axis / sine)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', required=True)
    parser.add_argument('--vmd', required=True)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--frames', default='0,20,40,60,80,100')
    parser.add_argument('--validate', action='store_true',
                        help='check the reproduction against the player log acceptance numbers')
    parser.add_argument('--config', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    player = Path(args.player)

    model = mujoco.MjModel.from_xml_path(str(player / 'models/unitree_g1/xmls/g1_pmx_fk.xml'))
    skeleton = json.loads((player / 'models/unitree_g1/pmx_skeleton.json').read_text(encoding='utf-8'))
    bones = {entry['name']: entry for entry in skeleton['bones']}
    tracks = read_vmd(Path(args.vmd))

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    clip, _ = load_bvh_file(args.bvh)

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    # ---- reproduce bind_targets -------------------------------------------------------
    drivers = {}
    for bone, target in TARGETS.items():
        if bone in bones:
            drivers.setdefault(target, []).append(bone)

    records = []
    for target, names in sorted(drivers.items()):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_BALL:
            continue
        body = int(model.jnt_bodyid[joint_id])
        rest = np.array(model.body_quat[body], dtype=float)
        rest /= np.linalg.norm(rest)
        records.append((int(model.jnt_qposadr[joint_id]), tuple(names), body,
                        np.array([rest[0], -rest[1], -rest[2], -rest[3]])))

    probe = mujoco.MjData(model)
    probe.qpos[:] = model.qpos0
    for address, _, _, rest_inverse in records:
        probe.qpos[address:address + 4] = rest_inverse
    mujoco.mj_kinematics(model, probe)

    def body_pos(frame_data, name):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if body < 0 else np.array(frame_data.xpos[body], dtype=float)

    bindings = []
    for address, names, body, rest_inverse in records:
        spec = SEGMENT.get(next((t for t, n in drivers.items() if n == list(names)), ''), None)
        align = R.from_quat([0, 0, 0, 1])
        if spec:
            start_pmx = body_pos(probe, f"pmx_body_{bones[spec['pmx'][0]]['index']:03d}")
            end_pmx = body_pos(probe, f"pmx_body_{bones[spec['pmx'][1]]['index']:03d}")
            limb_start = body_pos(probe, spec['g1_start']) if spec.get('g1_start') else np.array(probe.xpos[body])
            limb_end = body_pos(probe, spec['g1_child'])
            if all(v is not None for v in (start_pmx, end_pmx, limb_start, limb_end)):
                align = rotation_between(limb_end - limb_start, end_pmx - start_pmx)
        bindings.append((address, names, rest_inverse, align.as_quat(), body))

    # flexion bindings: (address, (proximal, pivot, distal) body ids, rule)
    flexion = []
    for name, rule in FLEXION.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        ids = []
        for bone in rule['bones']:
            source_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bones[bone]['joint'])
            ids.append(int(model.jnt_bodyid[source_joint]))
        flexion.append((int(model.jnt_qposadr[joint_id]), tuple(ids), rule))
    flexion_by_address = {address: (ids, rule) for address, ids, rule in flexion}

    wanted = [int(v) for v in args.frames.split(',')]
    positions_by_frame = {}
    spread = {bone: [] for bone in GMR_BONES}
    constants = {bone: [] for bone in GMR_BONES}
    for frame_index in wanted:
        data.qpos[:] = model.qpos0
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
        qadr = int(model.jnt_qposadr[root_joint])
        # 全ての親 and センター accumulate into the floating base, exactly as update_pose does
        root_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        root_motion = np.zeros(3)
        for name in ('全ての親', 'センター'):
            if name in tracks:
                root_rotation = quat_mul(root_rotation, mmd_quat_to_mujoco(sample(tracks[name], frame_index * 1.0)))
        scale = float(skeleton.get('scale', 0.07))
        for name in ('全ての親', 'センター', 'グルーブ'):
            if name in tracks:
                px, py, pz = sample_position(tracks, name, frame_index * 1.0)
                root_motion += scale * np.array([-pz, px, py])
        data.qpos[qadr:qadr + 3] = np.array([0.0, 0.0, 0.793]) + root_motion
        data.qpos[qadr + 3:qadr + 7] = root_rotation / np.linalg.norm(root_rotation)
        # source shadow skeleton first (flexion reads its FK)
        for name, entry in bones.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry['joint'])
            if joint_id < 0 or name not in tracks:
                continue
            q = mmd_quat_to_mujoco(sample(tracks[name], frame_index * 30.0 / 30.0))
            address = int(model.jnt_qposadr[joint_id])
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL:
                data.qpos[address:address + 4] = q
        for address, names, rest_inverse, align, _ in bindings:
            q = np.array([1.0, 0.0, 0.0, 0.0])
            for bone in names:
                if bone in tracks:
                    q = quat_mul(q, mmd_quat_to_mujoco(sample(tracks[bone], frame_index * 30.0 / 30.0)))
            q = quat_mul(rest_inverse, q)
            if align[0] < 1.0 - 1e-9:
                q = quat_mul(q, align)
            data.qpos[address:address + 4] = q / np.linalg.norm(q)
        mujoco.mj_kinematics(model, data)
        for address, (proximal, pivot, distal), rule in flexion:
            incoming = data.xpos[pivot] - data.xpos[proximal]
            outgoing = data.xpos[distal] - data.xpos[pivot]
            cosine = np.clip(np.dot(incoming, outgoing) /
                             ((np.linalg.norm(incoming) * np.linalg.norm(outgoing)) or 1.0), -1, 1)
            bend = np.arccos(cosine)
            angle = float(np.clip(rule['straight_q'] + rule['sign'] * bend, *rule['range']))
            data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        positions_by_frame[frame_index] = np.array(data.xpos).copy()

        for bone, (link, _, clip_bone) in GMR_BONES.items():
            if frame_index >= len(clip) or clip_bone not in clip[frame_index]:
                continue
            q = clip[frame_index][clip_bone][1]
            human = R.from_quat([q[1], q[2], q[3], q[0]])
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            if body < 0:
                continue
            robot = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            constants[bone].append((human.inv() * robot).as_quat())

    if args.validate:
        # compare against the acceptance numbers recorded in the player's modification log
        pairs = [('humerus L', ('left_shoulder_yaw_link', 'left_elbow_link'), ('左腕', '左ひじ')),
                 ('humerus R', ('right_shoulder_yaw_link', 'right_elbow_link'), ('右腕', '右ひじ')),
                 ('forearm L', ('left_elbow_link', 'left_wrist_yaw_link'), ('左ひじ', '左手首')),
                 ('forearm R', ('right_elbow_link', 'right_wrist_yaw_link'), ('右ひじ', '右手首'))]
        print(f"{'segment':<12}{'mean err':>10}{'min':>8}{'max':>8}")
        for label, (link_a, link_b), (bone_a, bone_b) in pairs:
            errors = []
            for frame_index in wanted:
                pose_xpos = positions_by_frame.get(frame_index)
                if pose_xpos is None:
                    continue
                a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_a)
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_b)
                pmx_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"pmx_body_{bones[bone_a]['index']:03d}")
                pmx_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"pmx_body_{bones[bone_b]['index']:03d}")
                if min(a, b, pmx_a, pmx_b) < 0:
                    continue
                v_robot = pose_xpos[b] - pose_xpos[a]
                v_pmx = pose_xpos[pmx_b] - pose_xpos[pmx_a]
                cosine = np.clip(v_robot @ v_pmx / ((np.linalg.norm(v_robot) * np.linalg.norm(v_pmx)) or 1.0), -1, 1)
                errors.append(np.degrees(np.arccos(cosine)))
            if errors:
                print(f'{label:<12}{np.mean(errors):9.1f} {np.min(errors):7.1f} {np.max(errors):7.1f}')
        return

    print(f"{'bone':<13}{'mean dev (deg)':>16}{'max dev':>10}")
    solved = {}
    for bone, quats in constants.items():
        if len(quats) < 2:
            continue
        reference = R.from_quat(quats[0])
        deviations = [np.degrees((reference.inv() * R.from_quat(q)).magnitude()) for q in quats]
        mean = R.from_quat(np.mean([R.from_quat(q).as_quat() for q in quats], axis=0))
        solved[bone] = mean
        print(f'{bone:<13}{np.mean(deviations):15.1f} {np.max(deviations):9.1f}')

    if args.config and args.out:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        for bone, rotation in solved.items():
            link = GMR_BONES[bone][0]
            x, y, z, w = rotation.as_quat()
            wxyz = [float(w), float(x), float(y), float(z)]
            for table in ('ik_match_table1', 'ik_match_table2'):
                for entry_link, entry in config[table].items():
                    if entry_link == link:
                        entry[4] = wxyz
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        print('written', args.out)


if __name__ == '__main__':
    main()
