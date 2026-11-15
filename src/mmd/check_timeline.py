"""Check that our BVH and the PMX player's shadow skeleton describe the same frame at the same time.

The shadow skeleton inside the player's MJCF is driven straight from the VMD with only the
documented basis change, so it is the reference for "what the MMD pose is". If our BVH were
misaligned in time (or in frame conversion), the rotation between the two would drift instead of
staying constant - and such a drift would look exactly like the "per-link constant does not hold"
result measured earlier, so this check has to come first.

The rotation between a BVH bone and the matching shadow bone is a fixed basis change, so its
**constancy** is the test, not its size.

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/check_timeline.py \
        --player <dir> --vmd <file> --bvh <clip>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.dont_write_bytecode = True

MMD_BONE = {
    'Hips': '腰', 'Spine': '上半身', 'Spine2': '上半身2', 'Neck': '首', 'Head': '頭',
    'LeftUpLeg': '左足', 'LeftLeg': '左ひざ', 'LeftFoot': '左足首', 'LeftToe': '左つま先',
    'RightUpLeg': '右足', 'RightLeg': '右ひざ', 'RightFoot': '右足首', 'RightToe': '右つま先',
    'LeftShoulder': '左肩', 'LeftArm': '左腕', 'LeftForeArm': '左ひじ', 'LeftHand': '左手首',
    'RightShoulder': '右肩', 'RightArm': '右腕', 'RightForeArm': '右ひじ', 'RightHand': '右手首',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', required=True)
    parser.add_argument('--vmd', required=True)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--step', type=int, default=10)
    args = parser.parse_args()

    player = Path(args.player).resolve()
    sys.path.insert(0, str(player / 'src'))
    import mujoco
    from mmd_robot_player.player import bind_all, update_pose
    from mmd_robot_player.robots import ROBOTS
    from mmd_robot_player.vmd import read_vmd
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    profile = ROBOTS['g1-pmx-fk']
    tracks = read_vmd(Path(args.vmd))
    model = mujoco.MjModel.from_xml_path(str(player / profile['xml']))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    model.opt.timestep = 1 / 240
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mapping = json.loads((player / profile['skeleton']).read_text(encoding='utf-8'))
    bindings = bind_all(model, mapping, profile)
    index_of = {entry['name']: entry['index'] for entry in mapping['bones']}

    clip, _ = load_bvh_file(args.bvh)
    max_frame = max((key.frame for track in tracks.values() for key in track), default=0)
    frames = list(range(0, min(int(max_frame), len(clip) - 1) + 1, args.step))

    body_of = {}
    for bone, mmd in MMD_BONE.items():
        if mmd in index_of:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pmx_body_{index_of[mmd]:03d}")
            if body >= 0:
                body_of[bone] = body

    relative = {bone: [] for bone in body_of}
    root_gap = []
    for frame in frames:
        update_pose(model, data, tracks, float(frame), profile, bindings)
        for bone, body in body_of.items():
            if bone not in clip[frame]:
                continue
            q = clip[frame][bone][1]
            bvh = R.from_quat([q[1], q[2], q[3], q[0]])
            shadow = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            relative[bone].append((bvh.inv() * shadow).as_quat())

    print(f"{'bone':<13}{'mean dev':>10}{'max dev':>10}   (degree; small = timelines agree)")
    worst = 0.0
    for bone, quats in relative.items():
        if len(quats) < 3:
            continue
        reference = R.from_quat(quats[0])
        deviations = [np.degrees((reference.inv() * R.from_quat(q)).magnitude()) for q in quats]
        worst = max(worst, float(np.max(deviations)))
        print(f'{bone:<13}{np.mean(deviations):9.1f} {np.max(deviations):9.1f}')
    print(f'\nworst deviation over all bones: {worst:.1f} deg')
    print('verdict:', 'timelines agree' if worst < 10 else 'MISALIGNED - fix before trusting anything')


if __name__ == '__main__':
    main()
