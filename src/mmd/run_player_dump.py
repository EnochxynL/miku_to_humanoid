"""Run the PMX player once, outside its repo, and measure whether GMR's model can express its pose.

The player (D:\\MY_DATA\\Downloads\\mmd_mujoco_player) maps MMD bone rotations onto robot joints in
joint space. GMR instead asks for one constant rotation per link. Those two are compatible only if
``R_human(t)^-1 * R_link(t)`` is the same rotation at every frame; this script borrows the player's
own code to produce that pose - so the measurement is on the real thing, not on a reimplementation -
then reports both the per-bone spread and the player's own acceptance numbers.

Nothing is written inside the player's repository: bytecode writing is disabled and every output
goes to ``--out``.

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/run_player_dump.py \
        --player <dir> --vmd <file> --bvh <clip> --out runtime/player_pose.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.dont_write_bytecode = True   # never litter the borrowed repository with __pycache__

GMR_BONES = {
    'Hips': ('pelvis', 'Hips'),
    'Spine2': ('torso_link', 'Spine2'),
    'LeftUpLeg': ('left_hip_yaw_link', 'LeftUpLeg'),
    'LeftLeg': ('left_knee_link', 'LeftLeg'),
    'LeftFootMod': ('left_ankle_roll_link', 'LeftFootMod'),
    'RightUpLeg': ('right_hip_yaw_link', 'RightUpLeg'),
    'RightLeg': ('right_knee_link', 'RightLeg'),
    'RightFootMod': ('right_ankle_roll_link', 'RightFootMod'),
    'LeftArm': ('left_shoulder_yaw_link', 'LeftArm'),
    'LeftForeArm': ('left_elbow_link', 'LeftForeArm'),
    'RightArm': ('right_shoulder_yaw_link', 'RightArm'),
    'RightForeArm': ('right_elbow_link', 'RightForeArm'),
}
SEGMENT_CHECK = [('humerus L', ('left_shoulder_yaw_link', 'left_elbow_link'), ('左腕', '左ひじ'), 0.0),
                 ('humerus R', ('right_shoulder_yaw_link', 'right_elbow_link'), ('右腕', '右ひじ'), 0.0),
                 ('forearm L', ('left_elbow_link', 'left_wrist_yaw_link'), ('左ひじ', '左手首'), 4.4),
                 ('forearm R', ('right_elbow_link', 'right_wrist_yaw_link'), ('右ひじ', '右手首'), 8.8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', required=True)
    parser.add_argument('--vmd', required=True)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--human-from', choices=('bvh', 'shadow'), default='bvh',
                        help='shadow = the same VMD the player used, so the two sides share a source')
    parser.add_argument('--root-relative', action='store_true',
                        help='compare in the pelvis frame, removing any global rotation mismatch')
    args = parser.parse_args()

    player = Path(args.player).resolve()
    sys.path.insert(0, str(player / 'src'))
    import mujoco
    from mmd_robot_player.player import bind_all, update_pose
    from mmd_robot_player.robots import ROBOTS
    from mmd_robot_player.vmd import read_vmd

    profile = ROBOTS['g1-pmx-fk']
    tracks = read_vmd(Path(args.vmd))
    model = mujoco.MjModel.from_xml_path(str(player / profile['xml']))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    model.opt.timestep = 1 / 240
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mapping = json.loads((player / profile['skeleton']).read_text(encoding='utf-8'))
    bindings = bind_all(model, mapping, profile)

    max_frame = max((key.frame for track in tracks.values() for key in track), default=0)
    frames = list(range(0, int(max_frame) + 1, args.stride))

    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    clip, _ = load_bvh_file(args.bvh)
    shadow_body = {}
    for entry in mapping['bones']:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pmx_body_{entry['index']:03d}")
        if body >= 0:
            shadow_body[entry['name']] = body
    MMD_OF = {'Hips': '腰', 'Spine2': '上半身2', 'LeftUpLeg': '左足', 'LeftLeg': '左ひざ',
              'LeftFootMod': '左足首', 'RightUpLeg': '右足', 'RightLeg': '右ひざ',
              'RightFootMod': '右足首', 'LeftArm': '左腕', 'LeftForeArm': '左ひじ',
              'RightArm': '右腕', 'RightForeArm': '右ひじ'}

    body_ids = {}
    for bone, (link, _) in GMR_BONES.items():
        body_ids[bone] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    segment_ids = []
    for label, (link_a, link_b), (bone_a, bone_b), recorded in SEGMENT_CHECK:
        segment_ids.append((label,
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_a),
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_b),
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                              f"pmx_body_{next(b['index'] for b in mapping['bones'] if b['name'] == bone_a):03d}"),
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                              f"pmx_body_{next(b['index'] for b in mapping['bones'] if b['name'] == bone_b):03d}"),
                            recorded))

    constants = {bone: [] for bone in GMR_BONES}
    segment_errors = {label: [] for label, *_ in segment_ids}
    qpos_log = []
    for frame in frames:
        update_pose(model, data, tracks, float(frame), profile, bindings)
        qpos_log.append(np.array(data.qpos).copy())
        root_h = R.from_quat([clip[min(int(frame), len(clip) - 1)]['Hips'][1][1],
                              clip[min(int(frame), len(clip) - 1)]['Hips'][1][2],
                              clip[min(int(frame), len(clip) - 1)]['Hips'][1][3],
                              clip[min(int(frame), len(clip) - 1)]['Hips'][1][0]])
        pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        root_r = R.from_matrix(np.array(data.xmat[pelvis_body]).reshape(3, 3))
        for bone, (link, clip_bone) in GMR_BONES.items():
            index = int(frame)
            if index >= len(clip) or clip_bone not in clip[index]:
                continue
            body = body_ids[bone]
            if args.human_from == 'shadow':
                shadow = shadow_body.get(MMD_OF.get(bone, ''))
                if shadow is None:
                    continue
                human = R.from_matrix(np.array(data.xmat[shadow]).reshape(3, 3))
            else:
                q = clip[index][clip_bone][1]
                human = R.from_quat([q[1], q[2], q[3], q[0]])
            if body < 0:
                continue
            robot = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            if args.root_relative:
                human = root_h.inv() * human
                robot = root_r.inv() * robot
            constants[bone].append((human.inv() * robot).as_quat())
        for label, a, b, pmx_a, pmx_b, _ in segment_ids:
            if min(a, b, pmx_a, pmx_b) < 0:
                continue
            v_robot = data.xpos[b] - data.xpos[a]
            v_pmx = data.xpos[pmx_b] - data.xpos[pmx_a]
            cosine = np.clip(v_robot @ v_pmx /
                             ((np.linalg.norm(v_robot) * np.linalg.norm(v_pmx)) or 1.0), -1, 1)
            segment_errors[label].append(float(np.degrees(np.arccos(cosine))))

    print('== player acceptance numbers (recomputed with the player itself) ==')
    print(f"{'segment':<12}{'mean':>8}{'min':>8}{'max':>8}{'log says':>10}")
    for label, values in segment_errors.items():
        recorded = next(r for l, *_, r in segment_ids if l == label)
        if values:
            print(f'{label:<12}{np.mean(values):7.1f} {np.min(values):7.1f} {np.max(values):7.1f} {recorded:9.1f}')

    print('\n== is a single constant per link enough for GMR? ==')
    print(f"{'bone':<13}{'mean dev':>10}{'max dev':>10}{'verdict':>10}")
    solved = {}
    for bone, quats in constants.items():
        if len(quats) < 2:
            continue
        reference = R.from_quat(quats[0])
        deviations = [np.degrees((reference.inv() * R.from_quat(q)).magnitude()) for q in quats]
        mean = R.from_quat(np.mean([R.from_quat(q).as_quat() for q in quats], axis=0))
        solved[bone] = mean
        verdict = 'ok' if np.mean(deviations) < 10 else 'NO'
        print(f'{bone:<13}{np.mean(deviations):9.1f} {np.max(deviations):9.1f} {verdict:>10}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.array(frames), qpos=np.array(qpos_log))
    print('\nwrote', out, 'with', len(qpos_log), 'poses')


if __name__ == '__main__':
    main()
