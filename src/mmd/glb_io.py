"""Minimal .glb reader: JSON chunk, accessors, quaternion slerp and animation sampling.

Extracted from the retired BVH converter so the glTF path no longer depends on it.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

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


def mujoco_names(nodes):
    """Node index -> MuJoCo body/joint name, taken from the glTF node name.

    Shared by the model builder and the motion extractor so both agree on the names; a duplicate or an
    empty name is disambiguated here rather than silently colliding inside MuJoCo.
    """
    seen = {}
    names = []
    for index, node in enumerate(nodes):
        raw = (node.get('name') or '').strip() or 'bone_%d' % index
        count = seen.get(raw, 0) + 1
        seen[raw] = count
        names.append(raw if count == 1 else '%s_%d' % (raw, count))
    return names


def xml_attr(value):
    """Escape a name for use inside a double-quoted XML attribute."""
    return (value.replace('&', '&amp;').replace('<', '&lt;')
                 .replace('>', '&gt;').replace('"', '&quot;'))
