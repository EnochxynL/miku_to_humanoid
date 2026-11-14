"""Convert an MMD-derived glTF/GLB (PMX model + VMD motion baked into one clip) to a
LaFAN1-topology BVH that GMR's `bvh_lafan1` IK config can consume unchanged.

Why the topology matters: GMR's `bvh_lafan1_to_g1.json` was tuned for the *22 bones of
LaFAN1* (Hips, LeftUpLeg..LeftToe, Spine, Spine1, Spine2, Neck, Head, LeftShoulder..LeftHand,
right side likewise), in Y-up centimetres. An MMD skeleton has 511 bones with Japanese names,
different proportions and intermediate bones everywhere, and earlier attempts to feed it
(either the full chain, or the full chain with hand-tuned rotation offsets) produced a robot
laid out flat or folded at the waist. So this exporter does not try to teach GMR a new
skeleton: it bakes the MMD motion onto LaFAN1's bone set, names and hierarchy, and lets the
stock config do its job.

Usage:
    uv run python src/reference_maker/mmd_glb_to_bvh.py \
        --glb "D:/MY_WORK/archived/py_mmd_anim/assets/星穹铁道—黑塔2.glb" \
        --out runtime/gmr_inputs/herta_mmd.bvh --fps 30

Output convention (verified against the LaFAN1 walk that GMR retargets correctly):
Y-up, centimetres, `Zrotation Yrotation Xrotation` per joint, six channels on the root.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# LaFAN1 hierarchy, in the order that file lists its joints.
LAFAN1_TREE = [
    ('Hips', None),
    ('LeftUpLeg', 'Hips'), ('LeftLeg', 'LeftUpLeg'), ('LeftFoot', 'LeftLeg'),
    ('LeftToe', 'LeftFoot'),
    ('RightUpLeg', 'Hips'), ('RightLeg', 'RightUpLeg'), ('RightFoot', 'RightLeg'),
    ('RightToe', 'RightFoot'),
    ('Spine', 'Hips'), ('Spine1', 'Spine'), ('Spine2', 'Spine1'),
    ('Neck', 'Spine2'), ('Head', 'Neck'),
    ('LeftShoulder', 'Spine2'), ('LeftArm', 'LeftShoulder'),
    ('LeftForeArm', 'LeftArm'), ('LeftHand', 'LeftForeArm'),
    ('RightShoulder', 'Spine2'), ('RightArm', 'RightShoulder'),
    ('RightForeArm', 'RightArm'), ('RightHand', 'RightForeArm'),
]

# LaFAN1 joint -> MMD bone. Spine1 has no MMD counterpart: it is emitted as a zero-length link
# so the hierarchy matches LaFAN1 without disturbing the transforms.
MMD_BONE = {
    'Hips': '腰', 'Spine': '上半身', 'Spine2': '上半身2', 'Neck': '首', 'Head': '頭',
    'LeftUpLeg': '左足', 'LeftLeg': '左ひざ', 'LeftFoot': '左足首', 'LeftToe': '左つま先',
    'RightUpLeg': '右足', 'RightLeg': '右ひざ', 'RightFoot': '右足首', 'RightToe': '右つま先',
    'LeftShoulder': '左肩', 'LeftArm': '左腕', 'LeftForeArm': '左ひじ', 'LeftHand': '左手首',
    'RightShoulder': '右肩', 'RightArm': '右腕', 'RightForeArm': '右ひじ', 'RightHand': '右手首',
}


# ---------------------------------------------------------------- glTF parsing
def load_glb(path: Path):
    data = path.read_bytes()
    _, _, length = struct.unpack_from('<III', data, 0)
    offset, chunks = 12, {}
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from('<II', data, offset)
        chunks[chunk_type] = data[offset + 8: offset + 8 + chunk_len]
        offset += 8 + chunk_len
    return json.loads(chunks[0x4E4F534A].decode('utf-8')), chunks.get(0x004E4942, b'')


_COMPONENTS = {5126: ('f4', 4), 5125: ('u4', 4), 5123: ('u2', 2), 5121: ('u1', 1)}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_accessor(doc, binary, index):
    acc = doc['accessors'][index]
    view = doc['bufferViews'][acc['bufferView']]
    dtype, size = _COMPONENTS[acc['componentType']]
    width = _WIDTH[acc['type']]
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride') or size * width
    out = np.empty((acc['count'], width), dtype=np.float64)
    for i in range(acc['count']):
        out[i] = np.frombuffer(binary, dtype=np.dtype('<' + dtype), count=width,
                               offset=start + i * stride)
    return out.squeeze() if width == 1 else out


# ---------------------------------------------------------------- math helpers
def quat_to_matrix(q):
    """glTF stores quaternions as (x, y, z, w)."""
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) or 1.0)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1 - t) * theta) / s) * q0 + (np.sin(t * theta) / s) * q1


def matrix_to_euler_zyx(R):
    """Angles (z, y, x) with R = Rz(z) @ Ry(y) @ Rx(x)."""
    y = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    z = np.arctan2(R[1, 0], R[0, 0])
    x = np.arctan2(R[2, 1], R[2, 2])
    return z, y, x


class Clip:
    """Per-node sampled local TRS of one glTF animation."""

    def __init__(self, doc, binary, animation_index=0):
        nodes = doc['nodes']
        anim = doc['animations'][animation_index]
        times = []
        for sampler in anim['samplers']:
            times.extend(np.asarray(read_accessor(doc, binary, sampler['input'])).tolist())
        self.t0, self.t1 = float(min(times)), float(max(times))
        self.channels = {}
        for ch in anim['channels']:
            path = ch['target']['path']
            if path not in ('translation', 'rotation'):
                continue
            sampler = anim['samplers'][ch['sampler']]
            t = np.asarray(read_accessor(doc, binary, sampler['input']), dtype=np.float64)
            v = np.asarray(read_accessor(doc, binary, sampler['output']), dtype=np.float64)
            self.channels.setdefault(ch['target']['node'], {})[path] = (t, v)

    def sample(self, node, path, time):
        entry = self.channels.get(node, {}).get(path)
        if entry is None:
            return None
        t, v = entry
        if time <= t[0]:
            return v[0]
        if time >= t[-1]:
            return v[-1]
        i = max(0, min(int(np.searchsorted(t, time)) - 1, len(t) - 2))
        span = t[i + 1] - t[i]
        a = 0.0 if span <= 0 else (time - t[i]) / span
        if path == 'rotation':
            return slerp(v[i], v[i + 1], a)
        return v[i] + a * (v[i + 1] - v[i])


def read_reference_offsets(path: Path):
    """Bone offsets (metres) from a BVH whose proportions we want to borrow."""
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    offsets, names = {}, []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(('ROOT', 'JOINT')):
            names.append(line.split()[1])
        elif line.startswith('End Site'):
            while i < len(lines) and lines[i].strip() != '}':
                i += 1
        elif line.startswith('OFFSET') and names:
            offsets[names[-1]] = np.array([float(v) for v in line.split()[1:4]]) * 0.01
        elif line.startswith('MOTION'):
            break
        i += 1
    return offsets


# ---------------------------------------------------------------- export
def build_bvh(glb: Path, out: Path, fps: float, animation_index: int, proportions_from: Path | None = None):
    doc, binary = load_glb(glb)
    nodes = doc['nodes']
    clip = Clip(doc, binary, animation_index)

    name_to_index = {}
    for i, node in enumerate(nodes):
        name = node.get('name')
        if name and name not in name_to_index:
            name_to_index[name] = i
    parent = {}
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            parent[child] = i

    present = [name for name, _ in LAFAN1_TREE if MMD_BONE.get(name) in name_to_index]
    missing = [name for name, _ in LAFAN1_TREE if name not in present]
    if 'Hips' not in present:
        raise SystemExit('the GLB has no 腰 (hips) bone')

    def world(index, time=None):
        """World position and rotation of a glTF node; time=None gives the bind pose."""
        chain, cur = [], index
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        R, p = np.eye(3), np.zeros(3)
        for cur in reversed(chain):
            node = nodes[cur]
            if time is None:
                t = np.asarray(node.get('translation', [0, 0, 0]), dtype=float)
                q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=float)
            else:
                t = clip.sample(cur, 'translation', time)
                q = clip.sample(cur, 'rotation', time)
                if t is None:
                    t = np.asarray(node.get('translation', [0, 0, 0]), dtype=float)
                if q is None:
                    q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=float)
            p = p + R @ np.asarray(t, dtype=float)
            R = R @ quat_to_matrix(np.asarray(q, dtype=float))
        return p, R

    PARENT = dict(LAFAN1_TREE)
    rest = {name: world(name_to_index[MMD_BONE[name]], None) for name in present}

    offsets = {}
    for name, parent_name in LAFAN1_TREE:
        if parent_name is None:
            offsets[name] = np.zeros(3)
        elif name in rest and parent_name in rest:
            R_parent = rest[parent_name][1]
            offsets[name] = R_parent.T @ (rest[name][0] - rest[parent_name][0])
        else:
            offsets[name] = np.zeros(3)          # synthetic link (Spine1)

    if proportions_from is not None:
        reference = read_reference_offsets(proportions_from)
        for name in offsets:
            if name in reference:
                offsets[name] = reference[name]

    frames = int(round((clip.t1 - clip.t0) * fps)) + 1
    times = clip.t0 + np.arange(frames) / fps
    world_cache = {name: [world(name_to_index[MMD_BONE[name]], t) for t in times] for name in present}

    lines = ['HIERARCHY']
    children_of = {name: [n for n, p in LAFAN1_TREE if p == name] for name, _ in LAFAN1_TREE}

    def emit(index, depth):
        name, parent_name = LAFAN1_TREE[index]
        pad = '\t' * depth
        lines.append(f'{pad}ROOT {name}' if parent_name is None else f'{pad}JOINT {name}')
        lines.append(f'{pad}{{')
        if parent_name is None:
            lines.append(f'{pad}\tOFFSET 0 0 0')
            lines.append(f'{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation')
        else:
            ox, oy, oz = 100.0 * offsets[name]
            lines.append(f'{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}')
            lines.append(f'{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation')
        for child_name in children_of[name]:
            emit([n for n, _ in LAFAN1_TREE].index(child_name), depth + 1)
        if not children_of[name]:
            lines.append(f'{pad}\tEnd Site')
            lines.append(f'{pad}\t{{')
            lines.append(f'{pad}\t\tOFFSET 0 0 0')
            lines.append(f'{pad}\t}}')
        lines.append(f'{pad}}}')

    emit(0, 0)
    lines.append('MOTION')
    lines.append(f'Frames: {frames}')
    lines.append(f'Frame Time: {1.0 / fps:.7f}')

    for f in range(frames):
        values = []
        for name, parent_name in LAFAN1_TREE:
            if name not in present:
                values.extend([0.0, 0.0, 0.0])          # synthetic link keeps identity
                continue
            p, R = world_cache[name][f]
            if parent_name is None:
                values.extend([100.0 * float(p[0]), 100.0 * float(p[1]), 100.0 * float(p[2])])
                z, y, x = matrix_to_euler_zyx(R)
            else:
                # a synthetic link (Spine1) is zero-length with identity rotation, so the
                # parent frame to compose against is the nearest *present* ancestor
                ancestor = parent_name
                while ancestor is not None and ancestor not in present:
                    ancestor = PARENT[ancestor]
                R_parent = world_cache[ancestor][f][1] if ancestor in present else np.eye(3)
                z, y, x = matrix_to_euler_zyx(R_parent.T @ R)
            values.extend([np.degrees(z), np.degrees(y), np.degrees(x)])
        lines.append(' '.join(f'{v:.6f}' for v in values))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'frames': frames, 'seconds': round(clip.t1 - clip.t0, 3), 'joints': len(LAFAN1_TREE),
            'missing_mmd_bones': missing,
            'hips_rest_cm': (100.0 * rest['Hips'][0]).round(2).tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--animation', type=int, default=0)
    # --proportions-from was tried and removed: LaFAN1's bones are laid out along +X with the
    # orientation carried in the channels, so substituting its offsets under our rotations
    # scrambles the skeleton. Fixing proportions needs a pose transfer onto a LaFAN1-shaped
    # skeleton, not an offset swap.
    args = parser.parse_args()
    print(json.dumps(build_bvh(Path(args.glb), Path(args.out), args.fps, args.animation,
                               None), indent=2))
