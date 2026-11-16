"""Drive the vendored deform pipeline once, and read it out two ways.

``engine.animation.Skeleton`` turns a PMX + (VMD|VPD) into per-bone transforms. Two consumers want
different things from that same state, and neither is a copy of the other:

* ``world_poses`` - per-bone **world** poses, the shape GMR's ``human_data`` has. This is the full state:
  every bone keeps its own translation, nothing is projected away.
* ``bake`` - **qpos** for one compiled character model. MuJoCo can only animate through qpos, and qpos is
  a lossy projection: rotations only for bones that have a ball joint, and a single free joint carrying
  one bone's translation (MMD moves the character through センター, so that joint rides on it).

Both share one per-frame loop in ``_drive``: one ``pose()``, then each reads what it needs. Deriving qpos
from the world poses is possible (``local = world(parent)^-1 * world``, measured at 0.000e+00 deg over
511 bones), but it needs every bone's world pose plus the parent table, and **no process needs both at
once** - so this module drives once and offers two readouts instead.

Two details of the reference matter, and a hand port misses them; both live in ``engine.vmd_tracks``:

* VMD stores bone names truncated to **15 shift-jis bytes**; the name map has to truncate identically or
  hundreds of tracks silently fail to match;
* only bones with both the IK flag (0x0020) and an ``ik`` block are driven by IK, and the VMD can switch
  them on/off per frame (``ik_frames``).

The MMD -> MuJoCo basis (a reflection, which is why a quaternion does not simply rotate) is in
``engine.basis``.

    # world poses - what GMR consumes
    PYTHONPATH=src;src\\mmd uv run python -m engine.bake --pmx <model.pmx> --vmd <motion.vmd> \
        --poses --out runtime/poses.npz
    # qpos for a compiled character model
    PYTHONPATH=src;src\\mmd uv run python -m engine.bake --pmx <model.pmx> --vmd <motion.vmd> \
        --out runtime/anim.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                          # noqa: E402

from engine.animation import FPS, Skeleton             # noqa: E402
from engine.basis import to_quaternion, to_vector      # noqa: E402
from engine.pmx import parse_pmx                       # noqa: E402
from engine.vmd import parse_vmd                       # noqa: E402
from engine.vmd_tracks import (frame_range, ik_schedule, match_tracks,   # noqa: E402
                               vpd_frame)

DEFAULT_SCALE = 0.07


# ---------------------------------------------------------------------------- preparation
def _prepare(pmx_path, pose_path, step=1, solve_ik=True):
    """(model, bones, frames, tracks, pose_frame, unmatched, enabled_at) for either kind of input."""
    model = parse_pmx(str(pmx_path))
    bones = model['bones']
    if str(pose_path).lower().endswith('.vpd'):
        # a pose is one frame: no tracks, no IK schedule
        pose_frame, unmatched = vpd_frame(bones, pose_path)
        return model, bones, [0], None, pose_frame, unmatched, None

    vmd = parse_vmd(str(pose_path))
    tracks, unmatched = match_tracks(bones, vmd)
    enabled_at = ik_schedule(bones, vmd, solve_ik)
    return model, bones, frame_range(vmd, step), tracks, None, unmatched, enabled_at


def _drive(model, frames, tracks, pose_frame, enabled_at, on_frame):
    """Pose the skeleton once per frame and hand it to ``on_frame`` - the one place that drives it."""
    skeleton = Skeleton(model)
    for frame in frames:
        if pose_frame is not None:
            skeleton.pose(pose_frame)
        else:
            sampled = {i: track.sample(frame) for i, track in tracks.items()}
            skeleton.pose(sampled, ik_enabled=enabled_at(frame))
        on_frame(skeleton, frame)


def _index_of(bones):
    """Exact name -> index, first wins (MMD rigs may repeat a name)."""
    index_of = {}
    for index, bone in enumerate(bones):
        index_of.setdefault(bone['name'], index)
    return index_of


# ---------------------------------------------------------------------------- readout: world poses
def _frame(skeleton, index_of, wanted, scale):
    frame = {}
    for name in wanted:
        index = index_of.get(name)
        if index is None:
            continue
        world_q, world_p = skeleton.world(index)
        frame[name] = [to_vector(world_p, scale), to_quaternion(world_q)]
    return frame


def bind_pose(pmx_path, wanted, scale=DEFAULT_SCALE):
    """The rig's bind pose as one frame (what a calibration that omits the VPD uses)."""
    model = parse_pmx(str(pmx_path))
    index_of = _index_of(model['bones'])
    skeleton = Skeleton(model)
    skeleton.pose({})
    return [_frame(skeleton, index_of, wanted, scale)]


def world_poses(pmx_path, pose_path, wanted, step=1, solve_ik=True, scale=DEFAULT_SCALE):
    """([{bone name: [pos(3), quat(w,x,y,z)]}], unmatched names) - GMR's ``human_data`` shape.

    No MuJoCo and no qpos round trip: the source side needs a world position and orientation per bone,
    which the pipeline already computes. Baking local rotations into qpos and running forward kinematics
    to get them back is the round trip that used to drop the ``センター`` translation.
    """
    if pose_path is None:
        return bind_pose(pmx_path, wanted, scale), []

    model, bones, frames, tracks, pose_frame, unmatched, enabled_at = _prepare(
        pmx_path, pose_path, step, solve_ik)
    index_of = _index_of(bones)
    out = []

    def on_frame(skeleton, _frame_number):
        out.append(_frame(skeleton, index_of, wanted, scale))

    _drive(model, frames, tracks, pose_frame, enabled_at, on_frame)
    return out, unmatched


def wanted_bones(config_path):
    """The bones an IK config actually drives (its bone-name field), plus the root chain."""
    from engine.profile import ROOT_BONE, TORSO_BONE

    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    names = set()
    for table in ('ik_match_table1', 'ik_match_table2'):
        for entry in config.get(table, {}).values():
            if entry and entry[0]:
                names.add(entry[0])
    names.update((ROOT_BONE, TORSO_BONE))
    return sorted(names)


# ---------------------------------------------------------------------------- readout: qpos
def bake(pmx_path, pose_path, robot_model, step=1, solve_ik=True, scale=DEFAULT_SCALE):
    """(qpos frames, unmatched bone names, times, bones without a joint) for one compiled model.

    ``pose_path`` may be a .vpd: a pose is a motion of one frame, so the same call serves both.
    ``robot_model`` only needs the joint names: ``mjcf.build(..., mesh=False)`` is enough.
    """
    model, bones, frames, tracks, pose_frame, unmatched, enabled_at = _prepare(
        pmx_path, pose_path, step, solve_ik)

    # MMD moves the character through センター, which is a *translation*. A qpos built only from ball
    # joints drops it and the character dances in place, so the free joint takes it.
    free_address = None
    for joint in range(robot_model.njnt):
        if robot_model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            free_address = int(robot_model.jnt_qposadr[joint])
            break
    # The bone that carries the whole-body offset is not necessarily a root: MMD models have static
    # 操作中心/全ての親 and translate センター (or a groove above the waist) instead. Pick the shallowest
    # bone whose translation track actually moves - the free joint rides on it.
    depth = [0] * len(bones)
    for index, bone in enumerate(bones):
        parent, d = bone['parent'], 0
        while 0 <= parent < len(bones):
            parent, d = bones[parent]['parent'], d + 1
        depth[index] = d
    if pose_frame is not None:
        moving = [i for i, (translation, _rotation) in pose_frame.items()
                  if abs(translation).max() > 1e-9]
    else:
        moving = [i for i, track in tracks.items()
                  if any(abs(v) > 1e-9 for key in track.keys for v in key['pos'])]
    translate_index = min(moving, key=lambda i: depth[i]) if moving else None
    bind_translate = None
    if translate_index is not None:
        probe = Skeleton(model)
        probe.pose({})
        bind_translate = np.array(probe.world(translate_index)[1], dtype=float)

    addresses, missing = {}, []
    for index, bone in enumerate(bones):
        joint = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, bone['name'])
        if joint >= 0 and robot_model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL:
            addresses[index] = int(robot_model.jnt_qposadr[joint])
        elif joint < 0:
            missing.append(bone['name'])

    out = []

    def on_frame(skeleton, _frame_number):
        qpos = np.array(robot_model.qpos0, dtype=float)
        if free_address is not None and translate_index is not None:
            world = np.array(skeleton.world(translate_index)[1], dtype=float)
            qpos[free_address:free_address + 3] = to_vector(world - bind_translate, scale)
        for index, address in addresses.items():
            qpos[address:address + 4] = to_quaternion(skeleton.local_q[index])
        out.append(qpos)

    _drive(model, frames, tracks, pose_frame, enabled_at, on_frame)
    return np.array(out), unmatched, np.array([f / FPS for f in frames]), missing


# ---------------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vmd', default=None, help='motion or pose; omit for the bind pose')
    parser.add_argument('--out', required=True)
    parser.add_argument('--poses', action='store_true',
                        help='write world poses (GMR human_data) instead of qpos')
    parser.add_argument('--wanted', default='',
                        help='comma separated bones for --poses (default: the whole rig)')
    parser.add_argument('--mesh', action='store_true',
                        help='build the textured model; baking only needs the joints')
    parser.add_argument('--step', type=int, default=1, help='sample every N VMD frames')
    parser.add_argument('--no-ik', dest='solve_ik', action='store_false')
    parser.add_argument('--scale', type=float, default=DEFAULT_SCALE)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.poses:
        wanted = [name for name in args.wanted.split(',') if name]
        if not wanted:
            wanted = [bone['name'] for bone in parse_pmx(args.pmx)['bones']]
        frames, unmatched = world_poses(args.pmx, args.vmd, wanted, step=args.step,
                                        solve_ik=args.solve_ik, scale=args.scale)
        names = sorted(frames[0]) if frames else []
        np.savez_compressed(out,
                            names=np.array(names),
                            pos=np.array([[f[name][0] for name in names] for f in frames]),
                            quat=np.array([[f[name][1] for name in names] for f in frames]))
        print('{"frames": %d, "bones": %d, "unmatched_vmd_bones": %d, "written": "%s"}'
              % (len(frames), len(names), len(unmatched), out))
        return

    from engine import mjcf
    model = mjcf.build(args.pmx, mesh=args.mesh)
    qpos, unmatched, times, missing = bake(args.pmx, args.vmd, model.model,
                                           step=args.step, solve_ik=args.solve_ik,
                                           scale=args.scale)
    np.savez_compressed(out, qpos=qpos, times=times)
    print('{"frames": %d, "unmatched_vmd_bones": %d, "bones_without_joint": %d, "written": "%s"}'
          % (len(qpos), len(unmatched), len(missing), out))


if __name__ == '__main__':
    main()
