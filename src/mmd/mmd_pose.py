"""MMD (PMX + VMD/VPD) -> world poses of the bones GMR drives, with no MuJoCo involved.

The source side only ever needs two things per bone per frame: a world **position** and a world
**orientation**. ``engine.animation.Skeleton`` already computes both (it is the vendored deform
pipeline), so there is no reason to bake local rotations into a qpos vector and then run forward
kinematics to get them back - that round trip is what used to drop the ``センター`` translation.

Conversions to the robot's space, matching the MJCF the viewer uses:

* position: ``(x, y, z) -> (-z, x, y) * scale``   (MMD is left-handed; this is a reflection)
* orientation: ``(x, y, z, w) -> (w, z, -x, -y)``   (the conjugation that reflection implies)

Frames are returned in GMR's ``human_data`` shape: ``{bone name: [pos(3), quat(w,x,y,z)]}``.

    PYTHONPATH=src;src\\mmd uv run python -m mmd_pose --pmx <model.pmx> --pose <motion.vmd> \
        --config configs/generated/mmd_to_g1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.animation import Skeleton, Track        # noqa: E402
from engine.pmx import parse_pmx                    # noqa: E402
from engine.vmd import parse_vmd                    # noqa: E402
from engine.vmd_tracks import (frame_range, ik_schedule, match_tracks,   # noqa: E402
                               vpd_frame)

DEFAULT_SCALE = 0.07


def to_robot_position(position, scale=DEFAULT_SCALE):
    from engine.basis import to_vector
    return to_vector(position, scale)


def to_robot_quaternion(quaternion):
    from engine.basis import to_quaternion
    return to_quaternion(quaternion)


def bind_pose(pmx_path, wanted):
    """The rig's bind pose as one frame (what a calibration that omits the VPD uses)."""
    model = parse_pmx(str(pmx_path))
    bones = model['bones']
    index_of = {}
    for index, bone in enumerate(bones):
        index_of.setdefault(bone['name'], index)
    skeleton = Skeleton(model)
    skeleton.pose({})
    return [_frame(skeleton, bones, index_of, wanted)]


def load(pmx_path, pose_path, wanted, step=1, solve_ik=True, scale=DEFAULT_SCALE):
    """Frames of world poses. ``pose_path`` is a VMD; None means the bind pose."""
    model = parse_pmx(str(pmx_path))
    bones = model['bones']

    index_of = {}
    for index, bone in enumerate(bones):
        index_of.setdefault(bone['name'], index)

    if pose_path is None:
        return bind_pose(pmx_path, wanted), []

    if str(pose_path).lower().endswith('.vpd'):
        # a pose is one frame: no tracks, no IK schedule
        sampled, unmatched = vpd_frame(bones, pose_path)
        skeleton = Skeleton(model)
        skeleton.pose(sampled)
        return [_frame(skeleton, bones, index_of, wanted)], unmatched

    vmd = parse_vmd(str(pose_path))
    tracks, unmatched = match_tracks(bones, vmd)
    enabled_at = ik_schedule(bones, vmd, solve_ik)
    frames = frame_range(vmd, step)

    skeleton = Skeleton(model)
    out = []
    for frame in frames:
        skeleton.pose({i: track.sample(frame) for i, track in tracks.items()},
                      ik_enabled=enabled_at(frame))
        out.append(_frame(skeleton, bones, index_of, wanted))
    return out, unmatched


def _frame(skeleton, bones, index_of, wanted):
    frame = {}
    for name in wanted:
        index = index_of.get(name)
        if index is None:
            continue
        world_q, world_p = skeleton.world(index)
        frame[name] = [to_robot_position(world_p), to_robot_quaternion(world_q)]
    return frame


def wanted_from_config(config_path):
    """The bones the IK config actually drives (its bone-name field), plus the root chain."""
    import json
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    names = set()
    for table in ('ik_match_table1', 'ik_match_table2'):
        for entry in config.get(table, {}).values():
            if entry and entry[0]:
                names.add(entry[0])
    names.add('腰')
    names.add('上半身2')
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--pose', default=None, help='VMD file; omit for the bind pose')
    parser.add_argument('--config', default=None, help='take the bone list from this IK config')
    parser.add_argument('--bones', default='', help='comma separated bone names (overrides --config)')
    parser.add_argument('--out', default=None, help='optional .npz of the frames')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--no-ik', dest='solve_ik', action='store_false')
    args = parser.parse_args()

    wanted = ([b for b in args.bones.split(',') if b] if args.bones
              else wanted_from_config(args.config) if args.config else ['腰'])
    frames, unmatched = load(args.pmx, args.pose, wanted, step=args.step,
                             solve_ik=args.solve_ik, scale=args.scale)
    print('{"frames": %d, "bones": %d, "unmatched_vmd_bones": %d}'
          % (len(frames), len(frames[0]) if frames else 0, len(unmatched)))
    if args.out:
        names = sorted(frames[0]) if frames else []
        positions = np.array([[f[n][0] for n in names] for f in frames])
        quaternions = np.array([[f[n][1] for n in names] for f in frames])
        np.savez_compressed(args.out, names=np.array(names), pos=positions, quat=quaternions)


if __name__ == '__main__':
    main()
