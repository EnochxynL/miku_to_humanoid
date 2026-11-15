"""Bake a VMD onto the PMX-as-MuJoCo model, with MMD's own IK and inheritance, then show it.

The bake follows MMD's deform pipeline, not a generic FK pass, because the motion depends on it:

    world_R(i) = world_R(parent) * local_q[i]              # PMX bones carry no rest rotation
    world_p(i) = world_p(parent) + R(parent) * (rest_i + local_t[i])

1. **付与親 (inherit)**: ``local_q[i] = slerp(I, local_q[src], ratio) * local_q[i]``
2. **IK**: CCD toward ``world(ik_bone).p``; each link may only rotate by
   ``limit * (link_index + 1)`` per pass, then its local rotation is clamped by the link's Euler
   limits; solve in ``(after_physics, layer, index)`` order
3. bone names are matched on the VMD's own key: shift-jis, truncated to 15 bytes

The result is written as MuJoCo qpos: the free joint carries the root motion, and every ball joint
carries ``mmd_quat_to_mujoco(local_q)`` - the same conjugation the player uses, so the two can be
compared frame by frame.

    uv run python src/reference_maker/bake_pmx_vmd.py \
        --pmx <model.pmx> --vmd <motion.vmd> --xml runtime/pmx/herta.xml --out runtime/pmx/herta_qpos.npz
    # add --view to open a MuJoCo window and play the bake back
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from mmd.vendor_pmx import parse_pmx
from mmd.vendor_vmd import read_vmd, sample_track

INHERIT_ROTATION = 0x0100
INHERIT_TRANSLATION = 0x0200
MOVABLE = 0x0004
IK_ENABLED = 0x0020
AFTER_PHYSICS = 0x1000


def mmd_quat_to_mujoco(rotation) -> np.ndarray:
    """(x, y, z, w) in MMD space -> (w, x, y, z) with the handedness flip baked in."""
    x, y, z, w = (float(v) for v in rotation)
    q = np.array([w, z, -x, -y])
    norm = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-12 else q / norm


def mmd_position_to_mujoco(position, scale: float) -> np.ndarray:
    x, y, z = (float(v) for v in position)
    return scale * np.array([-z, x, y])


class Skeleton:
    """MMD deform pipeline on the PMX bone table (port of mmd2gltf's Skeleton)."""

    def __init__(self, bones):
        self.bones = bones
        count = len(bones)
        self.rest = []
        for i, bone in enumerate(bones):
            parent = bone['parent']
            if 0 <= parent < count and parent != i:
                offset = np.array(bone['pos'], float) - np.array(bones[parent]['pos'], float)
            else:
                offset = np.array(bone['pos'], float)
            self.rest.append(offset)
        self.order = sorted(range(count),
                            key=lambda i: (1 if bones[i]['flags'] & AFTER_PHYSICS else 0,
                                           bones[i]['layer'], i))
        self.local_q = [np.array([0.0, 0.0, 0.0, 1.0]) for _ in range(count)]
        self.local_t = [np.zeros(3) for _ in range(count)]
        self._world = {}

    def reset(self):
        count = len(self.bones)
        self.local_q = [np.array([0.0, 0.0, 0.0, 1.0]) for _ in range(count)]
        self.local_t = [np.zeros(3) for _ in range(count)]
        self._world = {}

    def invalidate(self):
        self._world = {}

    def world(self, index):
        cached = self._world.get(index)
        if cached is not None:
            return cached
        parent = self.bones[index]['parent']
        local_t = self.rest[index] + self.local_t[index]
        if 0 <= parent < len(self.bones) and parent != index:
            parent_q, parent_p = self.world(parent)
            rotation = R.from_quat(self.local_q[index])
            world_q = (R.from_quat(parent_q) * rotation).as_quat()
            world_p = parent_p + R.from_quat(parent_q).apply(local_t)
        else:
            world_q = self.local_q[index]
            world_p = local_t
        self._world[index] = (world_q, world_p)
        return world_q, world_p


def bake(pmx_path: Path, vmd_path: Path):
    model = parse_pmx(str(pmx_path))
    bones = model['bones']
    tracks = read_vmd(vmd_path)

    # VMD stores at most 15 bytes of a shift-jis name
    name_map = {}
    for i, bone in enumerate(bones):
        key = bone['name'].encode('shift-jis', errors='replace')[:15]
        name_map.setdefault(key, i)
    index_tracks = {}
    unmatched = []
    for name, keys in tracks.items():
        key = name.encode('shift-jis', errors='replace')[:15]
        index = name_map.get(key)
        if index is None:
            unmatched.append(name)
        else:
            index_tracks[index] = keys

    max_frame = max((key.frame for keys in tracks.values() for key in keys), default=0)
    skeleton = Skeleton(bones)
    frames = []
    for frame in range(int(max_frame) + 1):
        skeleton.reset()
        for index, keys in index_tracks.items():
            position, rotation = sample_track(keys, float(frame))
            if bones[index]['flags'] & MOVABLE:
                skeleton.local_t[index] = np.asarray(position, dtype=float)
            skeleton.local_q[index] = np.asarray(rotation, dtype=float)
        for index in skeleton.order:
            bone = bones[index]
            flags = bone['flags']
            if flags & (INHERIT_ROTATION | INHERIT_TRANSLATION):
                source = bone.get('inherit_parent', -1)
                ratio = bone.get('inherit_ratio', 1.0)
                if 0 <= source < len(bones) and source != index:
                    if flags & INHERIT_ROTATION:
                        # ratio is not limited to [0, 1] in real models, so scale the rotation
                        # vector instead of slerping between identity and the source
                        added = R.from_rotvec(ratio * R.from_quat(skeleton.local_q[source]).as_rotvec())
                        skeleton.local_q[index] = (added * R.from_quat(skeleton.local_q[index])).as_quat()
                    if flags & INHERIT_TRANSLATION:
                        skeleton.local_t[index] = (skeleton.local_t[index] +
                                                   skeleton.local_t[source] * ratio)
                    skeleton.invalidate()
            if flags & IK_ENABLED and 'ik' in bone:
                solve_ik(skeleton, index, bone['ik'])
        frames.append([np.array(q).copy() for q in skeleton.local_q])
    return model, frames, unmatched


def solve_ik(skeleton: Skeleton, ik_index: int, ik):
    target = ik.get('target', -1)
    if not (0 <= target < len(skeleton.bones)):
        return
    goal = skeleton.world(ik_index)[1].copy()
    limit = ik['limit_angle']
    for _ in range(max(1, ik['loop'])):
        settled = False
        for link_index, link in enumerate(ik['links']):
            joint = link['bone']
            if not (0 <= joint < len(skeleton.bones)):
                continue
            joint_q, joint_p = skeleton.world(joint)
            effector = skeleton.world(target)[1]
            if np.linalg.norm(effector - goal) < 1e-4:
                settled = True
                break
            inverse = R.from_quat(joint_q).inv()
            v1 = inverse.apply(effector - joint_p)
            v2 = inverse.apply(goal - joint_p)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            v1, v2 = v1 / n1, v2 / n2
            angle = math.acos(max(-1.0, min(1.0, float(v1 @ v2))))
            if angle < 1e-6:
                continue
            angle = min(angle, limit * (link_index + 1))
            axis = np.cross(v1, v2)
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            delta = R.from_rotvec(axis / norm * angle)
            skeleton.local_q[joint] = (R.from_quat(skeleton.local_q[joint]) * delta).as_quat()
            if 'min' in link:
                euler = R.from_quat(skeleton.local_q[joint]).as_euler('xyz')
                euler = np.clip(euler, np.asarray(link['min'], float), np.asarray(link['max'], float))
                skeleton.local_q[joint] = R.from_euler('xyz', euler).as_quat()
            skeleton.invalidate()
        if settled:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', default=None)
    parser.add_argument('--vmd', default=None)
    parser.add_argument('--play', default=None, help='play an already baked npz instead of baking')
    parser.add_argument('--xml', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(mj_model)

    if args.play:
        qpos_frames = list(np.load(args.play)['qpos'])
        print(json.dumps({'playing': args.play, 'frames': len(qpos_frames)}))
        _view(mj_model, data, qpos_frames, args)
        return

    if not (args.pmx and args.vmd):
        raise SystemExit('either --play, or both --pmx and --vmd')
    model, frames, unmatched = bake(Path(args.pmx), Path(args.vmd))
    tree = json.loads(Path(args.tree).read_text(encoding='utf-8'))
    scale = tree['scale']
    free_address = int(mj_model.jnt_qposadr[
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'pmx_root_free')])

    root_bones = [i for i, bone in enumerate(model['bones']) if bone['parent'] < 0]
    qpos_frames = []
    for frame_values in frames:
        data.qpos[:] = mj_model.qpos0
        # root motion: the MMD roots' own local transform, converted
        root_rotation = np.array([1.0, 0.0, 0.0, 0.0])
        root_offset = np.zeros(3)
        for index in root_bones:
            q = mmd_quat_to_mujoco(frame_values[index])
            root_rotation = np.array(
                (R.from_quat(root_rotation) * R.from_quat(q)).as_quat())
            root_offset = root_offset + mmd_position_to_mujoco(
                model['bones'][index].get('_t', (0, 0, 0)) if False else (0, 0, 0), scale)
        data.qpos[free_address:free_address + 3] = root_offset
        data.qpos[free_address + 3:free_address + 7] = root_rotation
        for index in range(len(model['bones'])):
            address = int(mj_model.jnt_qposadr[
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'pmx_joint_{index:03d}')])
            data.qpos[address:address + 4] = mmd_quat_to_mujoco(frame_values[index])
        qpos_frames.append(np.array(data.qpos).copy())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=np.array(qpos_frames), times=np.arange(len(qpos_frames)) / 30.0)
    print(json.dumps({'frames': len(qpos_frames), 'unmatched_vmd_bones': len(unmatched),
                      'written': str(out)}, ensure_ascii=False))

    if not args.view:
        return
    _view(mj_model, data, qpos_frames, args)


def _view(mj_model, data, qpos_frames, args):
    import time
    import mujoco
    import mujoco.viewer
    end = len(qpos_frames) if args.end < 0 else min(args.end, len(qpos_frames))
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        frame = max(0, args.start)
        while viewer.is_running():
            started = time.perf_counter()
            data.qpos[:] = qpos_frames[frame % end]
            mujoco.mj_forward(mj_model, data)
            viewer.sync()
            frame += 1
            if frame >= end:
                frame = max(0, args.start)
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, 1.0 / 30.0 / args.speed - elapsed))


if __name__ == '__main__':
    main()
