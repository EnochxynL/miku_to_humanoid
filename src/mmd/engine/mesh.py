"""PMX mesh -> per-bone geometry groups, plus the two sinks that consume them.

The grouping is the interesting part and it lives here once: triangles are bucketed by their **dominant
bone** (a triangle's three vertices are assigned to the bone each one is most bound to, and the group
takes the majority), because skinning is per-bone and a surface split any other way tears apart when
the rig moves. What leaves this module is either

* ``write_obj`` - OBJ files plus PNG textures on disk (readable, for humans), or
* ``engine.msh`` - binary mesh bytes for the in-memory model, ~60x faster to produce.

Two conventions are load-bearing, and neither is a choice:

* **handedness.** MMD is left-handed; ``(x, y, z) -> (-z, x, y)`` has determinant -1, so triangles are
  emitted with reversed winding and normals go through the same matrix.
* **UV origin.** MMD puts V at the top. The OBJ sink flips it (MuJoCo's OBJ loader flips it back);
  the MSH sink does not (MuJoCo's MSH loader applies no flip, see ``mujoco/msh2obj.py``).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:      # pragma: no cover - PIL is required for textures only
    Image = None

# MMD (left-handed) -> MuJoCo (right-handed); det = -1, hence the winding flip
_BASIS = np.array([[0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])


def to_mujoco(vector, scale):
    return (scale * _BASIS @ np.asarray(vector, dtype=float))


def _dominant_bone(vertex):
    bones = vertex.get('bones') or []
    weights = vertex.get('weights') or []
    if not bones:
        return -1
    if not weights:
        return int(bones[0])
    return int(bones[int(np.argmax(weights))])


def _majority(a, b, c):
    if a == b or a == c:
        return a
    if b == c:
        return b
    return a


def normalise_texture(data):
    """Any MMD texture (bmp/tga/png/jpg) -> PNG bytes; already-PNG data is returned untouched.

    The untouched case matters: three 4k PNGs in this model cost ~2.8 s of pointless decode+encode.
    """
    if Image is None or data[:8] == b'\x89PNG\r\n\x1a\n':
        return data
    with Image.open(io.BytesIO(data)) as image:
        buffer = io.BytesIO()
        image.convert('RGBA').save(buffer, format='PNG')
    return buffer.getvalue()


def groups(model, base_dir=None, bone_positions=None, scale=0.07, flip_v=True, min_alpha=0.5):
    """Geometry grouped by (material, dominant bone), with textures normalised to PNG bytes.

    Positions are expressed **relative to the group's bone**, because that is the frame a MuJoCo mesh
    lives in; without it every patch is displaced by its own bone offset.
    """
    base_dir = Path(base_dir) if base_dir else Path.cwd()
    vertices = model['vertices']
    indices = model['indices']
    materials = model['materials']
    bone_names = [b['name'] for b in model['bones']]

    # ---- textures ------------------------------------------------------------------------------
    textures = {}
    for index, name in enumerate(model['textures']):
        if not name:
            continue
        source = base_dir / str(name).replace('\\', '/')
        try:
            textures[index] = ('tex_%02d.png' % index, normalise_texture(source.read_bytes()))
        except OSError:
            continue

    # ---- geometry ------------------------------------------------------------------------------
    positions = np.array([to_mujoco(v['pos'], scale) for v in vertices])
    normals = np.array([to_mujoco(v['normal'], scale) for v in vertices])
    uvs = np.array([[float(v['uv'][0]), float(v['uv'][1])] for v in vertices], dtype=float)
    uv_msh = uvs.copy()                     # MSH stores the unflipped V
    if flip_v:
        uvs[:, 1] = 1.0 - uvs[:, 1]
    dominant = [_dominant_bone(v) for v in vertices]

    # Some MMD models carry a duplicated surface under a "+" material (肌2 / 肌2+ are the same
    # vertices with different textures). Both are opaque, so they z-fight; keep the plain one.
    vertex_sets, cursor = {}, 0
    for material_index, material in enumerate(materials):
        count = int(material.get('index_count', 0))
        if count:
            vertex_sets[material_index] = frozenset(indices[cursor:cursor + count])
        cursor += count
    duplicates, seen = set(), {}
    for material_index, material in enumerate(materials):
        key = vertex_sets.get(material_index)
        if key is None:
            continue
        if str(material.get('name') or '').endswith('+') and key in seen:
            duplicates.add(material_index)
        else:
            seen.setdefault(key, material_index)

    out, material_meta, skipped = [], [], 0
    cursor = 0
    for material_index, material in enumerate(materials):
        count = int(material.get('index_count', 0))
        slice_indices = indices[cursor:cursor + count]
        cursor += count
        if count == 0:
            continue
        alpha = float(material.get('diffuse', [1, 1, 1, 1])[3])
        material_meta.append({
            'index': material_index,
            'name': material.get('name') or 'mat_%d' % material_index,
            'texture': textures.get(int(material.get('texture', -1)), (None,))[0],
            'diffuse': [float(x) for x in material.get('diffuse', [1, 1, 1, 1])[:3]],
            'alpha': alpha,
        })
        if material_index in duplicates or alpha < min_alpha:
            # duplicate surface, or an MMD overlay (颜+ is alpha 0, 目影 0.3): MuJoCo renders those
            # opaque and they blacken the whole head
            skipped += 1
            continue

        buckets = {}
        for step in range(0, len(slice_indices) - 2, 3):
            a, b, c = (int(slice_indices[step + i]) for i in range(3))
            buckets.setdefault(_majority(dominant[a], dominant[b], dominant[c]), []).append((a, c, b))

        for bone, triangles in buckets.items():
            used = sorted({i for tri in triangles for i in tri})
            remap = {vertex: slot for slot, vertex in enumerate(used)}
            faces = np.array([[remap[a], remap[b], remap[c]] for a, b, c in triangles], dtype=np.int32)
            origin = np.zeros(3)
            if bone_positions is not None and 0 <= bone < len(bone_positions):
                origin = np.asarray(bone_positions[bone], dtype=float)
            out.append({
                'name': 'm%03d_b%s' % (material_index, bone if bone >= 0 else 'root'),
                'material': material_index,
                'bone': bone,
                'bone_name': bone_names[bone] if 0 <= bone < len(bone_names) else None,
                'positions': positions[used] - origin,
                'normals': normals[used],
                'uv_obj': uvs[used],
                'uv_msh': uv_msh[used],
                'faces': faces,
                'triangles': len(triangles),
            })

    padded, split, degenerate = pad_tiny_groups(out)
    return {'groups': out, 'materials': material_meta, 'textures': textures,
            'skipped': skipped, 'padded_groups': padded,
            'split_edges': split, 'degenerate_pads': degenerate}


def pad_tiny_groups(groups):
    """Give every group at least four vertices, so MuJoCo keeps all of them.

    MuJoCo rejects a mesh with fewer than four vertices. Dropping such a group punches a real hole
    (here ``衣1`` and ``衣1+`` each lose six triangles at the hem - both layers at once - and the leg
    shows through), and merging it into another group animates it with the wrong bone.

    The pad splits one triangle at the midpoint of an edge: the surface keeps its shape and its area,
    and the two halves are real triangles rather than degenerate ones. A group whose triangles are all
    degenerate falls back to repeating a vertex, which MuJoCo tolerates only because the mesh declares
    shell inertia.
    """
    padded, split, degenerate = 0, 0, 0
    for group in groups:
        if len(group['positions']) >= 4:
            continue
        padded += 1
        faces = np.asarray(group['faces'], dtype=np.int32)
        guard = 0
        while len(group['positions']) < 4 and guard < 8:
            guard += 1
            target = None
            for index, (a, b, c) in enumerate(faces):
                if len({int(a), int(b), int(c)}) == 3:
                    target = index
                    break
            if target is None:
                break
            a, b, c = (int(v) for v in faces[target])
            position = 0.5 * (group['positions'][a] + group['positions'][b])
            normal = 0.5 * (group['normals'][a] + group['normals'][b])
            uv_obj = 0.5 * (group['uv_obj'][a] + group['uv_obj'][b])
            uv_msh = 0.5 * (group['uv_msh'][a] + group['uv_msh'][b])
            middle = len(group['positions'])
            group['positions'] = np.vstack([group['positions'], position[None, :]])
            group['normals'] = np.vstack([group['normals'], normal[None, :]])
            group['uv_obj'] = np.vstack([group['uv_obj'], uv_obj[None, :]])
            group['uv_msh'] = np.vstack([group['uv_msh'], uv_msh[None, :]])
            faces[target] = [a, middle, c]
            faces = np.vstack([faces, np.array([[middle, b, c]], dtype=np.int32)])
            group['triangles'] += 1
            split += 1

        if len(group['positions']) < 4:
            # every triangle is degenerate: repeat the first vertex, which only works because the mesh
            # declares shell inertia
            repeat = 4 - len(group['positions'])
            group['positions'] = np.vstack([group['positions'],
                                            np.repeat(group['positions'][:1], repeat, axis=0)])
            group['normals'] = np.vstack([group['normals'],
                                          np.repeat(group['normals'][:1], repeat, axis=0)])
            group['uv_obj'] = np.vstack([group['uv_obj'],
                                         np.repeat(group['uv_obj'][:1], repeat, axis=0)])
            group['uv_msh'] = np.vstack([group['uv_msh'],
                                         np.repeat(group['uv_msh'][:1], repeat, axis=0)])
            extra = np.repeat(group['faces'][:1], repeat, axis=0)
            extra[:, 1:] = extra[:, :1]
            faces = np.vstack([faces, extra])
            degenerate += 1
        group['faces'] = faces
    return padded, split, degenerate


def write_obj(built, out_dir):
    """OBJ files plus PNG textures on disk - the readable sink, mostly for debugging."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for _index, (name, png) in built['textures'].items():
        (out_dir / name).write_bytes(png)
    for group in built['groups']:
        with (out_dir / (group['name'] + '.obj')).open('w', encoding='utf-8', newline='\n') as handle:
            handle.write('# %s  material=%s  bone=%s\n'
                         % (group['name'], group['material'], group['bone_name']))
            for x, y, z in group['positions']:
                handle.write('v %.6f %.6f %.6f\n' % (x, y, z))
            for u, v in group['uv_obj']:
                handle.write('vt %.6f %.6f\n' % (u, v))
            for x, y, z in group['normals']:
                handle.write('vn %.6f %.6f %.6f\n' % (x, y, z))
            for a, b, c in group['faces']:
                handle.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n'
                             % (a + 1, a + 1, a + 1, b + 1, b + 1, b + 1, c + 1, c + 1, c + 1))


def export(model, out_dir, base_dir=None, bone_positions=None, scale=0.07,
           flip_v=True, min_alpha=0.5):
    """Convenience: group, then write OBJ + PNG. Returns the group metadata."""
    built = groups(model, base_dir=base_dir, bone_positions=bone_positions, scale=scale,
                   flip_v=flip_v, min_alpha=min_alpha)
    write_obj(built, out_dir)
    return built
