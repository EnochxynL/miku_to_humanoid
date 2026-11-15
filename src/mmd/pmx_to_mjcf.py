"""Build a MuJoCo model from a PMX skeleton, so the PMX rig can act as a rigid-body "robot".

Why this exists: GMR needs *rigid-body poses* as input, not joint angles dressed up as bone
orientations. A PMX skeleton is already a rigid-body tree - one body per bone, parent-relative
offsets, and (unlike MJCF) no rest rotations at all, so a bone's local rotation is exactly the
joint value. Making that tree a MuJoCo model lets us bake the motion onto it and then feed true
link poses to GMR with ``C = identity``.

Layout follows the player's own PMX shadow skeleton (``pmx_body_NNN`` / ``pmx_joint_NNN``), so the
two can be compared directly:

    <body name="pmx_body_042" pos="...">
      <inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />
      <joint name="pmx_joint_042" type="ball" />
      ...

Coordinate change is the documented one: MMD is left-handed (X right, Y up, Z toward the viewer),
MuJoCo is right-handed, so ``(x, y, z) -> (-z, x, y) * scale`` - a reflection, which is why the
joint quaternions must be converted with the matching conjugation ``(x,y,z,w) -> (w, z, -x, -y)``.

    uv run python src/reference_maker/pmx_to_mjcf.py \
        --pmx <model.pmx> --out runtime/pmx/model.xml --tree runtime/pmx/tree.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mmd.vendor_pmx import parse_pmx


# the humanoid chain, so the viewer shows a readable skeleton instead of every hair and skirt bone
MAIN_BONES = {
    'センター', 'グルーブ', '全ての親', '腰', '下半身', '上半身', '上半身2', '首', '頭',
    '左肩P', '左肩', '右肩P', '右肩',
    '左腕', '左ひじ', '左手首', '右腕', '右ひじ', '右手首',
    '左足', '左ひざ', '左足首', '左つま先', '右足', '右ひざ', '右足首', '右つま先',
}


def to_mujoco_position(mmd_position, scale: float) -> np.ndarray:
    """MMD (left-handed, Y up) -> MuJoCo (right-handed, Z up)."""
    x, y, z = (float(v) for v in mmd_position)
    return scale * np.array([-z, x, y])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--scale', type=float, default=0.07,
                        help='MMD units to metres (the player and the glTF exporter both use 0.07)')
    parser.add_argument('--model-name', default='pmx_skeleton')
    parser.add_argument('--all-bones', dest='main_bones_only', action='store_false',
                        help='draw every bone, including hair and skirt chains')
    parser.add_argument('--capsule-radius', type=float, default=0.012,
                        help='draw a capsule per bone towards its tail so the skeleton is visible')
    args = parser.parse_args()

    model = parse_pmx(args.pmx)
    bones = model['bones']
    count = len(bones)
    scale = args.scale

    # parent-relative rest offsets, in MuJoCo coordinates
    offsets = []
    for bone in bones:
        parent = bone['parent']
        position = np.array(bone['pos'], dtype=float)
        if 0 <= parent < count and parent != len(offsets):
            position = position - np.array(bones[parent]['pos'], dtype=float)
        offsets.append(to_mujoco_position(position, scale))

    children = {i: [] for i in range(count)}
    roots = []
    for i, bone in enumerate(bones):
        parent = bone['parent']
        if 0 <= parent < count and parent != i:
            children[parent].append(i)
        else:
            roots.append(i)

    def bone_tail(index: int):
        """Tail vector of a bone in MMD space, or None when it has no length."""
        bone = bones[index]
        if 'tail_bone' in bone and 0 <= bone['tail_bone'] < count:
            return np.array(bones[bone['tail_bone']]['pos'], float) - np.array(bone['pos'], float)
        if 'tail_offset' in bone:
            tail = np.array(bone['tail_offset'], float)
            if np.linalg.norm(tail) > 1e-6:
                return tail
        return None

    def capsule(index: int):
        """Tail vector in the body's local frame (MmJ to MuJoCo basis, scaled)."""
        tail = bone_tail(index)
        if tail is None:
            return None
        vector = np.array([-tail[2], tail[0], tail[1]]) * scale
        length = float(np.linalg.norm(vector))
        if length < 1e-4:
            return None
        return vector, min(args.capsule_radius, length / 2.0)

    lines = ["<?xml version='1.0' encoding='utf-8'?>",
             f'<mujoco model="{args.model_name}">',
             '  <compiler angle="radian" autolimits="true" />',
             '  <visual>',
             '    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1" />',
             '    <map znear="0.01" zfar="20" />',
             '  </visual>',
             '  <statistic center="0 0 0.8" extent="1.1" />',
             '  <worldbody>',
             '    <light pos="1 -1 2" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7" />',
             '    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.28 0.30 0.34 1" contype="0" conaffinity="0" group="0" />',
             '    <body name="pmx_root" pos="0 0 0">',
             '      <freejoint name="pmx_root_free" />',
             '      <inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />']

    def emit(index: int, depth: int):
        pad = '  ' * depth
        x, y, z = offsets[index]
        lines.append(f'{pad}<body name="pmx_body_{index:03d}" pos="{x:.9g} {y:.9g} {z:.9g}">')
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />')
        lines.append(f'{pad}  <joint name="pmx_joint_{index:03d}" type="ball" />')
        cap = capsule(index) if (not args.main_bones_only or bones[index]['name'] in MAIN_BONES) else None
        if cap is not None:
            (vx, vy, vz), radius = cap
            lines.append(
                f'{pad}  <geom type="capsule" fromto="0 0 0 {vx:.6g} {vy:.6g} {vz:.6g}" '
                f'size="{radius:.6g}" rgba="0.72 0.74 0.78 1" '
                f'contype="0" conaffinity="0" group="1"/>')
        for child in children[index]:
            emit(child, depth + 1)
        lines.append(f'{pad}</body>')

    for root in roots:
        emit(root, 3)
    lines += ['    </body>', '  </worldbody>', '</mujoco>']

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    tree = {
        'source': Path(args.pmx).name,
        'scale': scale,
        'coordinate_mapping': 'MMD [x,y,z] -> MuJoCo [-z,x,y] * scale (left-handed to right-handed)',
        'bone_count': count,
        'roots': [bones[i]['name'] for i in roots],
        'bones': [
            {'name': bone['name'], 'index': i,
             'joint': f'pmx_joint_{i:03d}', 'body': f'pmx_body_{i:03d}',
             'parent': bone['parent'],
             'height': float(to_mujoco_position(bone['pos'], scale)[2])}
            for i, bone in enumerate(bones)
        ],
    }
    Path(args.tree).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tree).write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'bones': count, 'roots': tree['roots'][:6],
                      'written': [str(out), str(args.tree)]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
