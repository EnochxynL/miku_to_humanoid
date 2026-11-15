"""Separate two explanations for the BVH-vs-player mismatch: a time offset, or a different pose.

The previous check found the two clips disagree by up to 165 deg. That can be either

* a **time offset** (our BVH frame i is not the player's frame i), or
* a **different pose at the same time** - our BVH came from a GLB whose exporter may have baked MMD
  IK into the bones, while the player's shadow skeleton is the raw VMD with no IK.

Root translation is immune to IK, so comparing its trajectory and searching for the lag that best
aligns the two series separates the two cases: a clean optimum at a non-zero lag means a time
offset; a good match at lag 0 means the timelines agree and the pose difference is IK.

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/check_root_lag.py \
        --player <dir> --vmd <file> --bvh <clip>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', required=True)
    parser.add_argument('--vmd', required=True)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--max-lag', type=int, default=40)
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
    waist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pmx_body_{index_of['腰']:03d}")

    count = min(len(clip), 320)
    ours = np.array([clip[i]['Hips'][0] for i in range(count)], dtype=float)
    theirs = []
    for frame in range(count):
        update_pose(model, data, tracks, float(frame), profile, bindings)
        theirs.append(np.array(data.xpos[waist], dtype=float))
    theirs = np.array(theirs)

    # compare velocities (offsets in the vertical/absolute placement are scale differences)
    ours_v = np.diff(ours, axis=0)
    theirs_v = np.diff(theirs, axis=0)

    def correlation(a, b):
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / denominator) if denominator else 0.0

    best_lag, best_score = 0, -2.0
    scores = {}
    for lag in range(-args.max_lag, args.max_lag + 1):
        if lag >= 0:
            a, b = ours_v[lag:], theirs_v[:len(theirs_v) - lag]
        else:
            a, b = ours_v[:len(ours_v) + lag], theirs_v[-lag:]
        if len(a) < 20:
            continue
        score = correlation(a, b)
        scores[lag] = score
        if score > best_score:
            best_lag, best_score = lag, score

    print(f"frames compared: {count}")
    print(f"root-velocity correlation at lag 0 : {scores.get(0, float('nan')):+.3f}")
    print(f"best lag: {best_lag:+d} frames (score {best_score:+.3f})")
    if best_lag:
        a = ours_v[best_lag:] if best_lag > 0 else ours_v[:len(ours_v) + best_lag]
        b = theirs_v[:len(theirs_v) - best_lag] if best_lag > 0 else theirs_v[-best_lag:]
        print(f"correlation at the best lag        : {correlation(a, b):+.3f}")
    print()
    print('scale check: |ours| mean %.3f   |theirs| mean %.3f' %
          (np.linalg.norm(ours, axis=1).mean(), np.linalg.norm(theirs, axis=1).mean()))


if __name__ == '__main__':
    main()
