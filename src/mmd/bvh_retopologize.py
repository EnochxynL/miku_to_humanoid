"""Re-topologise a BVH exported by the MMD evaluator onto the LaFAN1 joint set GMR expects.

Why: our previous path was PMX+VMD -> GLB (a third-party converter) -> BVH. Comparing that GLB's
animation against the VMD shows the converter applied ``(x, y, z, w) -> (-x, -y, z, w)`` to every
local rotation (a 180 deg change of basis about Z) while mirroring positions with ``z -> -z`` -
two different transforms, which is why no per-bone constant could make the retarget agree.

`py_mmd_anim` evaluates PMX+VMD itself and exports BVH with a mirror implementation that has unit
tests, so this script takes that BVH as the motion source and rewrites it with LaFAN1's joint names
and hierarchy, keeping the world orientations untouched.

    uv run python src/reference_maker/bvh_retopologize.py \
        --src runtime/gmr_inputs/herta_from_mmdengine.bvh \
        --out runtime/gmr_inputs/herta_mmd_topology.bvh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# LaFAN1 hierarchy and, for each joint, the MMD bone that carries it
LAFAN1_TREE = [
    ('Hips', None, '腰'),
    ('LeftUpLeg', 'Hips', '左足'), ('LeftLeg', 'LeftUpLeg', '左ひざ'),
    ('LeftFoot', 'LeftLeg', '左足首'), ('LeftToe', 'LeftFoot', '左つま先'),
    ('RightUpLeg', 'Hips', '右足'), ('RightLeg', 'RightUpLeg', '右ひざ'),
    ('RightFoot', 'RightLeg', '右足首'), ('RightToe', 'RightFoot', '右つま先'),
    ('Spine', 'Hips', '上半身'), ('Spine1', 'Spine', None), ('Spine2', 'Spine1', '上半身2'),
    ('Neck', 'Spine2', '首'), ('Head', 'Neck', '頭'),
    ('LeftShoulder', 'Spine2', '左肩'), ('LeftArm', 'LeftShoulder', '左腕'),
    ('LeftForeArm', 'LeftArm', '左ひじ'), ('LeftHand', 'LeftForeArm', '左手首'),
    ('RightShoulder', 'Spine2', '右肩'), ('RightArm', 'RightShoulder', '右腕'),
    ('RightForeArm', 'RightArm', '右ひじ'), ('RightHand', 'RightForeArm', '右手首'),
]

AXES = {'Xrotation': (1.0, 0.0, 0.0), 'Yrotation': (0.0, 1.0, 0.0), 'Zrotation': (0.0, 0.0, 1.0)}


def parse_bvh(path: Path):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    joints, stack, i = [], [], 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(('ROOT', 'JOINT')):
            joints.append({'name': line.split()[1], 'parent': stack[-1] if stack else -1,
                           'offset': np.zeros(3), 'channels': []})
            if line.endswith('{'):                 # some exporters open the brace on the same line
                stack.append(len(joints) - 1)
        elif line.startswith('End Site'):
            while i < len(lines) and lines[i].strip() != '}':
                i += 1
        elif line.startswith('OFFSET'):
            joints[-1]['offset'] = np.array([float(v) for v in line.split()[1:4]])
        elif line.startswith('CHANNELS'):
            parts = line.split()
            joints[-1]['channels'] = parts[2:2 + int(parts[1])]
        elif line == '{':
            stack.append(len(joints) - 1)
        elif line.startswith('}'):
            if stack:
                stack.pop()
        elif line.startswith('MOTION'):
            break
        i += 1
    while not lines[i].strip().startswith('Frames:'):
        i += 1
    count = int(lines[i].split()[1])
    i += 2
    rows = np.array([[float(v) for v in row.split()] for row in lines[i:i + count]])
    return joints, rows


def euler_matrix(channels, values):
    """Compose the joint's rotations in the order its CHANNELS declare."""
    matrix = np.eye(3)
    for channel, value in zip(channels, values):
        if channel not in AXES:
            continue
        axis = np.array(AXES[channel])
        c, s = np.cos(value), np.sin(value)
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        matrix = matrix @ (np.eye(3) * c + s * cross + (1 - c) * np.outer(axis, axis))
    return matrix


def world_track(joints, rows, fps_note):
    """World position and rotation of every joint per frame, in the file's own frame."""
    frames = len(rows)
    positions = np.zeros((frames, len(joints), 3))
    rotations = np.zeros((frames, len(joints), 3, 3))
    for f in range(frames):
        values = rows[f]
        local_p, local_r, read = [None] * len(joints), [None] * len(joints), 0
        for index, joint in enumerate(joints):
            block = values[read:read + len(joint['channels'])]
            read += len(joint['channels'])
            if index == 0:
                local_p[index] = block[0:3] if len(block) >= 6 else joint['offset']
                local_r[index] = euler_matrix(joint['channels'], block[-3:])
            else:
                local_p[index] = joint['offset']
                local_r[index] = euler_matrix(joint['channels'], block)
        for index, joint in enumerate(joints):
            parent = joint['parent']
            if parent < 0:
                positions[f, index] = local_p[index]
                rotations[f, index] = local_r[index]
            else:
                positions[f, index] = positions[f, parent] + rotations[f, parent] @ local_p[index]
                rotations[f, index] = rotations[f, parent] @ local_r[index]
    return positions, rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    joints, rows = parse_bvh(Path(args.src))
    index_of = {}
    for i, joint in enumerate(joints):
        index_of.setdefault(joint['name'], i)
    positions, rotations = world_track(joints, rows, None)
    frames = len(rows)

    missing = [mmd for _, _, mmd in LAFAN1_TREE if mmd is not None and mmd not in index_of]
    if '腰' not in index_of:
        raise SystemExit('source BVH has no 腰 (hips) bone')

    # rest skeleton: the source offsets already describe it
    rest_p = np.zeros((len(joints), 3))
    for i, joint in enumerate(joints):
        if joint['parent'] >= 0:
            rest_p[i] = rest_p[joint['parent']] + joint['offset']
        else:
            rest_p[i] = joint['offset']

    bone_index = {name: index_of[mmd] for name, _, mmd in LAFAN1_TREE if mmd and mmd in index_of}
    offsets = {}
    for name, parent, _ in LAFAN1_TREE:
        if parent is None:
            offsets[name] = np.zeros(3)
        elif name in bone_index and parent in bone_index:
            offsets[name] = rest_p[bone_index[parent]] @ np.eye(3)  # placeholder, recomputed below
    # offsets from the rest skeleton, expressed in the parent's rest frame
    for name, parent, _ in LAFAN1_TREE:
        if parent is None:
            continue
        if name in bone_index and parent in bone_index:
            offsets[name] = rest_p[bone_index[name]] - rest_p[bone_index[parent]]
        else:
            offsets[name] = np.zeros(3)

    lines = ['HIERARCHY']
    children = {name: [n for n, p, _ in LAFAN1_TREE if p == name] for name, _, _ in LAFAN1_TREE}

    def emit(name, depth):
        pad = '\t' * depth
        parent = dict((n, p) for n, p, _ in LAFAN1_TREE)[name]
        lines.append(f'{pad}ROOT {name}' if parent is None else f'{pad}JOINT {name}')
        lines.append(f'{pad}{{')
        if parent is None:
            lines.append(f'{pad}\tOFFSET 0 0 0')
            lines.append(f'{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation')
        else:
            ox, oy, oz = offsets[name]
            lines.append(f'{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}')
            lines.append(f'{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation')
        for child in children[name]:
            emit(child, depth + 1)
        if not children[name]:
            lines.append(f'{pad}\tEnd Site')
            lines.append(f'{pad}\t{{')
            lines.append(f'{pad}\t\tOFFSET 0 0 0')
            lines.append(f'{pad}\t}}')
        lines.append(f'{pad}}}')

    emit('Hips', 0)
    lines.append('MOTION')
    lines.append(f'Frames: {frames}')
    lines.append(f'Frame Time: {1.0 / 30.0:.7f}')

    def matrix_to_euler_zyx(R):
        y = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        z = np.arctan2(R[1, 0], R[0, 0])
        x = np.arctan2(R[2, 1], R[2, 2])
        return z, y, x

    parent_of = dict((n, p) for n, p, _ in LAFAN1_TREE)
    for f in range(frames):
        values = []
        for name, parent, _ in LAFAN1_TREE:
            if name not in bone_index:
                values.extend([0.0, 0.0, 0.0])          # synthetic Spine1
                continue
            i = bone_index[name]
            if parent is None:
                p = positions[f, i]
                values.extend([float(p[0]), float(p[1]), float(p[2])])
                z, y, x = matrix_to_euler_zyx(rotations[f, i])
            else:
                # a synthetic parent (Spine1) contributes identity, so compose from the nearest
                ancestor = parent
                while ancestor is not None and ancestor not in bone_index:
                    ancestor = parent_of[ancestor]
                if ancestor in bone_index:
                    local = rotations[f, bone_index[ancestor]].T @ rotations[f, i]
                else:
                    local = rotations[f, i]
                z, y, x = matrix_to_euler_zyx(local)
            values.extend([np.degrees(z), np.degrees(y), np.degrees(x)])
        lines.append(' '.join(f'{v:.6f}' for v in values))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    info = {'frames': frames, 'joints': len(LAFAN1_TREE),
            'missing_mmd_bones': missing,
            'source_joints': len(joints)}
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
