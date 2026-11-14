"""Convert an MMD-derived glTF/GLB (PMX model + VMD motion already baked into one clip) to BVH.

MMD skeletons are not directly usable by GMR: GMR speaks BVH/FBX/SMPL-X and its `lafan1`
format expects a small set of human bone names. This writes a BVH whose hierarchy is the real
MMD chain (so the offsets stay rigid) but whose bones carry the names GMR looks up:

    Hips, Spine2, LeftUpLeg, LeftLeg, LeftFootMod, RightUpLeg, RightLeg, RightFootMod,
    LeftArm, LeftForeArm, LeftHand, RightArm, RightForeArm, RightHand

Only the ancestors of those bones are exported, so the file stays small. Intermediate MMD
bones keep descriptive names (Spine, LowerBody, LeftShoulder, ...) and are ignored by GMR.

Channel convention follows GMR's lafan1 reader: the root carries six channels
(Xposition Yposition Zposition Zrotation Yrotation Xrotation) and every other joint three
(Zrotation Yrotation Xrotation), composed as R = Rz * Ry * Rx.

Usage:
    uv run python src/reference_maker/mmd_glb_to_bvh.py \
        --glb "D:/MY_WORK/archived/py_mmd_anim/assets/星穹铁道—黑塔2.glb" \
        --out runtime/gmr_inputs/herta_mmd.bvh --fps 30
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# MMD bone name -> GMR human bone name. Only the bones GMR maps are renamed.
MMD_TO_GMR = {
    '腰': 'Hips',
    '上半身2': 'Spine2',
    '左足': 'LeftUpLeg', '左ひざ': 'LeftLeg', '左足首': 'LeftFoot', '左つま先': 'LeftToe',
    '右足': 'RightUpLeg', '右ひざ': 'RightLeg', '右足首': 'RightFoot', '右つま先': 'RightToe',
    '左腕': 'LeftArm', '左ひじ': 'LeftForeArm', '左手首': 'LeftHand',
    '右腕': 'RightArm', '右ひじ': 'RightForeArm', '右手首': 'RightHand',
}
# Readable names for the remaining MMD bones that end up on the exported paths.
EXTRA_NAMES = {
    '上半身': 'Spine', '下半身': 'LowerBody', '首': 'Neck', '頭': 'Head',
    '左肩': 'LeftShoulder', '右肩': 'RightShoulder',
}
REQUIRED = set(MMD_TO_GMR)


# ---------------------------------------------------------------- glTF parsing
def load_glb(path: Path):
    data = path.read_bytes()
    _, _, length = struct.unpack_from('<III', data, 0)
    offset, chunks = 12, {}
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from('<II', data, offset)
        chunks[chunk_type] = data[offset + 8: offset + 8 + chunk_len]
        offset += 8 + chunk_len
    doc = json.loads(chunks[0x4E4F534A].decode('utf-8'))
    return doc, chunks.get(0x004E4942, b'')


_COMPONENTS = {5126: ('f4', 4), 5125: ('u4', 4), 5123: ('u2', 2), 5121: ('u1', 1)}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_accessor(doc, binary, index):
    acc = doc['accessors'][index]
    view = doc['bufferViews'][acc['bufferView']]
    dtype, size = _COMPONENTS[acc['componentType']]
    width = _WIDTH[acc['type']]
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = view.get('byteStride') or size * width
    count = acc['count']
    out = np.empty((count, width), dtype=np.float64)
    for i in range(count):
        base = start + i * stride
        out[i] = np.frombuffer(binary, dtype=np.dtype('<' + dtype), count=width, offset=base)
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


# ---------------------------------------------------------------- clipping
class Clip:
    """Per-node sampled local TRS of one glTF animation."""

    def __init__(self, doc, binary, animation_index=0):
        nodes = doc['nodes']
        anim = doc['animations'][animation_index]
        self.times = []
        for sampler in anim['samplers']:
            self.times.extend(read_accessor(doc, binary, sampler['input']).tolist())
        self.t0, self.t1 = float(min(self.times)), float(max(self.times))
        self.channels = {}                      # node -> {'translation': (t, v), 'rotation': (t, q)}
        for ch in anim['channels']:
            path = ch['target']['path']
            if path not in ('translation', 'rotation'):
                continue                       # morph weights are irrelevant here
            sampler = anim['samplers'][ch['sampler']]
            t = np.asarray(read_accessor(doc, binary, sampler['input']), dtype=np.float64)
            v = np.asarray(read_accessor(doc, binary, sampler['output']), dtype=np.float64)
            self.channels.setdefault(ch['target']['node'], {})[path] = (t, v)
        self.nodes = nodes

    def sample(self, node: int, path: str, time: float):
        entry = self.channels.get(node, {}).get(path)
        if entry is None:
            return None
        t, v = entry
        if time <= t[0]:
            return v[0]
        if time >= t[-1]:
            return v[-1]
        i = int(np.searchsorted(t, time)) - 1
        i = max(0, min(i, len(t) - 2))
        span = t[i + 1] - t[i]
        a = 0.0 if span <= 0 else (time - t[i]) / span
        if path == 'rotation':
            return slerp(v[i], v[i + 1], a)
        return v[i] + a * (v[i + 1] - v[i])


# ---------------------------------------------------------------- export
def build_bvh(glb: Path, out: Path, fps: float, animation_index: int):
    doc, binary = load_glb(glb)
    nodes = doc['nodes']
    clip = Clip(doc, binary, animation_index)

    name_to_index = {}
    for i, node in enumerate(nodes):
        name = node.get('name')
        if name and name not in name_to_index:
            name_to_index[name] = i
    missing = sorted(REQUIRED - set(name_to_index))
    if missing:
        raise SystemExit(f'MMD bones missing from the GLB: {missing}')

    parent = {}
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            parent[child] = i

    # ancestor closure of the bones GMR needs, cut at Hips (腰): everything above the hips
    # (操作中心 / 全ての親 / センター / グルーブ) stays outside the export
    root = name_to_index['腰']
    keep = {root}
    for name in REQUIRED:
        cur = name_to_index[name]
        while cur is not None and cur not in keep:
            keep.add(cur)
            cur = parent.get(cur)

    # rest local TRS -> rest world transforms, used for the constant offsets
    def rest_world(index):
        chain = []
        cur = index
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        R = np.eye(3)
        p = np.zeros(3)
        for cur in reversed(chain):
            node = nodes[cur]
            t = np.asarray(node.get('translation', [0, 0, 0]), dtype=np.float64)
            q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=np.float64)
            p = p + R @ t
            R = R @ quat_to_matrix(q)
        return p, R

    rest_p = {i: rest_world(i)[0] for i in keep}
    offsets = {}
    for i in keep:
        if i == root:
            continue
        p_parent, R_parent = rest_world(parent[i])
        offsets[i] = R_parent.T @ (rest_p[i] - p_parent)

    display = {}
    for i in keep:
        name = nodes[i].get('name', '')
        display[i] = MMD_TO_GMR.get(name) or EXTRA_NAMES.get(name) or f'Bone{i}'

    children = {i: [c for c in nodes[i].get('children', []) if c in keep] for i in keep}
    # a kept child whose parent chain passes through dropped joints is re-attached to the
    # nearest kept ancestor
    for i in keep:
        for c in list(children[i]):
            if c not in keep:
                children[i].remove(c)

    frames = int(round((clip.t1 - clip.t0) * fps)) + 1
    times = clip.t0 + np.arange(frames) / fps

    def world_at(index, time):
        chain = []
        cur = index
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        R = np.eye(3)
        p = np.zeros(3)
        for cur in reversed(chain):
            node = nodes[cur]
            t = clip.sample(cur, 'translation', time)
            if t is None:
                t = np.asarray(node.get('translation', [0, 0, 0]), dtype=np.float64)
            q = clip.sample(cur, 'rotation', time)
            if q is None:
                q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=np.float64)
            p = p + R @ t
            R = R @ quat_to_matrix(q)
        return p, R

    lines = ['HIERARCHY']
    order = []

    def emit(index, depth):
        pad = '\t' * depth
        label = display[index]
        if index == root:
            lines.append(f'{pad}ROOT {label}')
        else:
            lines.append(f'{pad}JOINT {label}')
        lines.append(f'{pad}{{')
        if index == root:
            lines.append(f'{pad}\tOFFSET 0 0 0')
            lines.append(f'{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation')
        else:
            ox, oy, oz = 100.0 * offsets[index]
            lines.append(f'{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}')
            lines.append(f'{pad}\tCHANNELS 3 Zrotation Yrotation Xrotation')
        order.append(index)
        for child in children[index]:
            emit(child, depth + 1)
        if index == root:
            lines.append(f'{pad}}}')
            lines.append(f'End Site')
            lines.append(f'{pad}{{')
            lines.append(f'{pad}\tOFFSET 0 0 0')
            lines.append(f'{pad}}}')
        else:
            lines.append(f'{pad}}}')

    emit(root, 0)
    lines.append('MOTION')
    lines.append(f'Frames: {frames}')
    lines.append(f'Frame Time: {1.0 / fps:.7f}')

    # LAFAN1 convention: Z-up centimetres. The loader applies R (Z-up -> Y-up) and divides by
    # 100, so the skeleton is pre-rotated by R^T and scaled on the root only; every local
    # rotation below the root is then unchanged.
    R_LOADER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    T = R_LOADER.T
    for time in times:
        values = []
        root_pos = None
        root_rot = None
        for index in order:
            p, R = world_at(index, time)
            if index == root:
                root_pos, root_rot = 100.0 * (T @ p), T @ R
                values.extend([float(root_pos[0]), float(root_pos[1]), float(root_pos[2])])
                z, y, x = matrix_to_euler_zyx(root_rot)
                values.extend([np.degrees(z), np.degrees(y), np.degrees(x)])
            else:
                p_parent, R_parent = world_at(parent[index], time) if parent[index] in keep \
                    else rest_world(parent[index])
                R_local = R_parent.T @ R
                z, y, x = matrix_to_euler_zyx(R_local)
                values.extend([np.degrees(z), np.degrees(y), np.degrees(x)])
        lines.append(' '.join(f'{v:.6f}' for v in values))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'frames': frames, 'seconds': clip.t1 - clip.t0, 'joints': len(order),
            'root': display[root], 'root_start': rest_p[root].tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--animation', type=int, default=0)
    args = parser.parse_args()
    info = build_bvh(Path(args.glb), Path(args.out), args.fps, args.animation)
    print(json.dumps(info, indent=2))
