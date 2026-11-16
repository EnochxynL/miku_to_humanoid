"""Bake a VMD onto a PMX rig and write MuJoCo qpos.

The deform pipeline is **not** re-implemented here: it is ``engine.animation``, vendored verbatim from
mmd2gltf, which is the bake whose result was accepted visually. Two details of that reference matter,
and a hand port misses them:

* VMD stores bone names truncated to **15 shift-jis bytes**; the name map has to truncate identically
  or hundreds of tracks silently fail to match;
* only bones with both the IK flag (0x0020) and an ``ik`` block are driven by IK, and the VMD can
  switch them on/off per frame (``ik_frames``).

This module only turns the reference's per-bone local rotations into qpos for the MJCF that
``pmx_to_mjcf`` produced: bodies carry the bone names, so joints are found by name, and the
MMD -> MuJoCo basis (a reflection) is undone by ``mmd_quat_to_mujoco``.

    PYTHONPATH=src;src\\mmd uv run python -m engine.bake --pmx <model.pmx> --vmd <motion.vmd> \
        --xml runtime/pmx/skeleton.xml --out runtime/pmx/anim.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                          # noqa: E402

from engine.animation import FPS, Skeleton, Track      # noqa: E402
from engine.pmx import parse_pmx                       # noqa: E402
from engine.vmd import parse_vmd                       # noqa: E402


def mmd_quat_to_mujoco(rotation) -> np.ndarray:
    """MMD quaternion (x, y, z, w) -> MuJoCo (w, x, y, z) under the (-z, x, y) reflection."""
    x, y, z, w = (float(v) for v in rotation)
    return np.array([w, z, -x, -y])


def _name_map(bones):
    mapping = {}
    for index, bone in enumerate(bones):
        try:
            key = bone['name'].encode('shift-jis', errors='replace')[:15]
        except Exception:
            key = bone['name'].encode('utf-8', errors='replace')[:15]
        mapping.setdefault(key, index)
    return mapping


def bake(pmx_path, vmd_path, xml_path, step=1, solve_ik=True):
    """(qpos frames, unmatched vmd bone names, times) for the MJCF derived from ``pmx_path``."""
    model = parse_pmx(str(pmx_path))
    vmd = parse_vmd(str(vmd_path))
    bones = model['bones']
    count = len(bones)

    name_map = _name_map(bones)
    tracks, unmatched = {}, []
    for name, keys in vmd['bones'].items():
        index = name_map.get(name.encode('shift-jis', errors='replace')[:15])
        if index is None:
            unmatched.append(name)
        else:
            tracks[index] = Track(keys)

    all_ik = [i for i, b in enumerate(bones) if b['flags'] & 0x0020 and 'ik' in b]
    base_enabled = set(all_ik) if solve_ik else set()

    ik_states = []
    if vmd.get('ik_frames'):
        ik_name_map = {}
        for index in all_ik:
            key = bones[index]['name'].encode('shift-jis', errors='replace')[:20]
            ik_name_map.setdefault(key, index)
        for frame in vmd['ik_frames']:
            states = {}
            for name, on in frame['ik'].items():
                index = ik_name_map.get(name.encode('shift-jis', errors='replace')[:20])
                if index is not None:
                    states[index] = on
            ik_states.append((frame['frame'], states))

    def enabled_at(frame):
        if not ik_states:
            return base_enabled
        state = {}
        for start, values in ik_states:
            if start > frame:
                break
            state.update(values)
        return {i for i in base_enabled if state.get(i, True)}

    last = vmd['max_frame']
    frames = list(range(0, last + 1, step))
    if frames[-1] != last:
        frames.append(last)

    skeleton = Skeleton(model)

    robot = mujoco.MjModel.from_xml_path(str(xml_path))
    addresses, missing = {}, []
    for index, bone in enumerate(bones):
        joint = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, bone['name'])
        if joint >= 0 and robot.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL:
            addresses[index] = int(robot.jnt_qposadr[joint])
        elif joint < 0:
            missing.append(bone['name'])

    out = []
    for frame in frames:
        sampled = {i: track.sample(frame) for i, track in tracks.items()}
        skeleton.pose(sampled, ik_enabled=enabled_at(frame))
        qpos = np.array(robot.qpos0, dtype=float)
        for index, address in addresses.items():
            qpos[address:address + 4] = mmd_quat_to_mujoco(skeleton.local_q[index])
        out.append(qpos)

    return np.array(out), unmatched, np.array([f / FPS for f in frames]), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vmd', required=True)
    parser.add_argument('--xml', required=True, help='MJCF produced by pmx_to_mjcf for this PMX')
    parser.add_argument('--out', required=True)
    parser.add_argument('--step', type=int, default=1, help='sample every N VMD frames')
    parser.add_argument('--no-ik', dest='solve_ik', action='store_false')
    args = parser.parse_args()

    qpos, unmatched, times, missing = bake(args.pmx, args.vmd, args.xml,
                                           step=args.step, solve_ik=args.solve_ik)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, times=times)
    print('{"frames": %d, "unmatched_vmd_bones": %d, "bones_without_joint": %d, "written": "%s"}'
          % (len(qpos), len(unmatched), len(missing), out))


if __name__ == '__main__':
    main()
