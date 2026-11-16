"""PMX mesh -> MuJoCo meshes, materials and textures.

The PMX side needs no glTF: the model carries everything MuJoCo wants - vertices with normals, UVs and
bone weights, an index buffer concatenated in material order (`materials[i]['index_count']` is the
slice length), a texture name per material, and a diffuse colour with alpha.

Two conversions are load-bearing:

* **handedness.** MMD is left-handed; the skeleton uses ``(x, y, z) -> (-z, x, y) * scale``, whose
  determinant is -1. A reflection flips triangle winding, so faces must be emitted as ``(a, c, b)``
  and normals mapped with the same matrix - otherwise every surface is inside-out.
* **UV origin.** MMD/glTF put V at the top, MuJoCo at the bottom, so ``vt = (u, 1 - v)``.

A group is one (material, bone) pair: geometry is rigid-segmented by the **triangle's majority bone**,
which keeps each surface closed under animation (splitting per-vertex topples the surface apart).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:      # pragma: no cover - PIL is required for textures only
    Image = None

# MMD (left-handed) -> MuJoCo (right-handed); det = -1, hence the winding flip below
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


def _as_png(data, target: Path):
    """Normalise any MMD texture (bmp/tga/png/jpg) to PNG; MuJoCo only reads a few formats."""
    if Image is None:
        target.write_bytes(data)
        return
    with Image.open(io.BytesIO(data)) as image:
        image.convert('RGBA').save(target)


def export(model, out_dir, base_dir=None, bone_positions=None, scale=0.07,
           flip_v=True, min_alpha=0.5):
    """Write OBJ groups plus PNG textures, and return the metadata the MJCF emitter needs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir) if base_dir else Path.cwd()

    vertices = model['vertices']
    indices = model['indices']
    materials = model['materials']
    texture_names = model['textures']
    bone_names = [b['name'] for b in model['bones']]

    # ---- textures ------------------------------------------------------------------------------
    texture_files = {}
    texture_meta = []
    for index, name in enumerate(texture_names):
        raw_name = Path(str(name).replace('\\', '/')).name
        texture_meta.append(None)
        if not name:
            continue
        # PMX stores texture paths relative to the model file
        source = base_dir / str(name).replace('\\', '/')
        target = out_dir / ('tex_%02d.png' % index)
        try:
            _as_png(source.read_bytes(), target)
        except OSError:
            continue
        texture_files[index] = target.name
        texture_meta[index] = target.name

    # ---- geometry ------------------------------------------------------------------------------
    positions = np.array([to_mujoco(v['pos'], scale) for v in vertices])
    normals = np.array([to_mujoco(v['normal'], scale) for v in vertices])
    uvs = np.array([[float(v['uv'][0]), float(v['uv'][1])] for v in vertices], dtype=float)
    if flip_v:
        uvs[:, 1] = 1.0 - uvs[:, 1]
    dominant = [_dominant_bone(v) for v in vertices]

    groups = []
    material_meta = []
    skipped = 0
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
            'texture': texture_files.get(int(material.get('texture', -1))),
            'diffuse': [float(x) for x in material.get('diffuse', [1, 1, 1, 1])[:3]],
            'alpha': alpha,
        })
        if alpha < min_alpha:
            # MMD uses near-transparent materials as overlays (颜+ is a blush layer); MuJoCo blends
            # them far more heavily and they grey out the whole surface
            continue

        buckets = {}
        for step in range(0, len(slice_indices) - 2, 3):
            a, b, c = int(slice_indices[step]), int(slice_indices[step + 1]), int(slice_indices[step + 2])
            bone = _majority(dominant[a], dominant[b], dominant[c])
            buckets.setdefault(bone, []).append((a, b, c))

        for bone, triangles in buckets.items():
            name = 'm%03d_b%s' % (material_index, bone if bone >= 0 else 'root')
            path = out_dir / (name + '.obj')
            # MuJoCo mesh vertices live in the *body's* frame, so a group bound to a bone has to
            # be expressed relative to that bone's bind position - otherwise every patch is
            # displaced by its own bone offset and the model falls apart
            origin = np.zeros(3)
            if bone_positions is not None and 0 <= bone < len(bone_positions):
                origin = np.asarray(bone_positions[bone], dtype=float)
            used = sorted({i for tri in triangles for i in tri})
            if len(used) < 4:
                # MuJoCo rejects a mesh with fewer than four vertices; such a group is a couple of
                # stray triangles and is visually negligible
                skipped += 1
                continue
            remap = {vertex: slot for slot, vertex in enumerate(used)}
            with path.open('w', encoding='utf-8', newline='\n') as handle:
                handle.write('# %s  material=%s  bone=%s\n' % (
                    name, material.get('name'), bone_names[bone] if 0 <= bone < len(bone_names) else '?'))
                for vertex in used:
                    x, y, z = positions[vertex] - origin
                    handle.write('v %.6f %.6f %.6f\n' % (x, y, z))
                for vertex in used:
                    u, v = uvs[vertex]
                    handle.write('vt %.6f %.6f\n' % (u, v))
                for vertex in used:
                    x, y, z = normals[vertex]
                    handle.write('vn %.6f %.6f %.6f\n' % (x, y, z))
                for a, b, c in triangles:
                    # winding reversed: the coordinate change is a reflection
                    fa, fb, fc = remap[a] + 1, remap[b] + 1, remap[c] + 1
                    handle.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n'
                                 % (fa, fa, fa, fc, fc, fc, fb, fb, fb))
            groups.append({
                'name': name,
                'material': material_index,
                'bone': bone,
                'bone_name': bone_names[bone] if 0 <= bone < len(bone_names) else None,
                'file': path.name,
                'triangles': len(triangles),
            })

    return {
        'groups': groups,
        'skipped_groups': skipped,
        'materials': material_meta,
        'textures': [t for t in texture_meta if t],
        'scale': scale,
        'winding': 'reversed (reflection)',
    }
