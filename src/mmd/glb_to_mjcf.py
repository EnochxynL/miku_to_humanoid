"""Build a MuJoCo model straight from the glTF that mmd2gltf-gui produced.

The glTF is the better source for this than the PMX: the exporter already baked MMD's IK and
付与親 into the animation, every bone survives as a node (513 here, including all forty finger
bones), and the skin carries the inverse bind matrices needed to place the mesh.

Layout:
  * one body per node, local TRS unchanged;
  * the *whole* tree hangs under one root body carrying the Y-up -> Z-up rotation, because a proper
    rotation of the scene leaves every local transform invariant - only the root needs it;
  * a ball joint per bone, so the animation can drive any of them; nodes the animation does not
    touch simply stay at their bind transform;
  * ``--mesh`` additionally splits the skinned mesh by (dominant joint, material) into one OBJ per
    group. MuJoCo has no linear blend skinning, and 70% of this model's vertices are influenced by
    2-4 bones, so rigid segmentation is the only option and it leaves seams at bent joints.

    uv run python src/mmd/glb_to_mjcf.py --glb <model.glb> --xml runtime/glb/herta.xml \
        --nodes runtime/glb/herta_nodes.json [--mesh]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROT_Y_UP_TO_Z_UP = '0.7071067811865476 0.7071067811865476 0 0'   # Rx(+90 deg), (w, x, y, z)
STUB_INERTIAL = '<inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />'


def load_gltf_module():
    spec = importlib.util.spec_from_file_location(
        'mmd_glb_to_bvh', Path(__file__).with_name('mmd_glb_to_bvh.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_matrix(q):
    """glTF (x, y, z, w) -> 3x3 rotation."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb', required=True)
    parser.add_argument('--xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--mesh', action='store_true', help='include the textured mesh')
    parser.add_argument('--sphere-radius', type=float, default=0.008,
                        help='when --mesh is off, a sphere per bone keeps the skeleton visible')
    args = parser.parse_args()

    gltf = load_gltf_module()
    doc, binary = gltf.load_glb(Path(args.glb))
    nodes = doc['nodes']
    count = len(nodes)

    parent = [-1] * count
    for i, node in enumerate(nodes):
        for child in node.get('children', []):
            parent[child] = i
    children = {i: [] for i in range(count)}
    roots = []
    for i in range(count):
        if parent[i] >= 0:
            children[parent[i]].append(i)
        else:
            roots.append(i)

    # accumulated node scale, because MuJoCo bodies cannot carry one
    scale_of = [float(np.mean(np.asarray(nodes[i].get('scale', [1, 1, 1]), float))) for i in range(count)]
    accumulated_scale = [1.0] * count
    for i in range(count):
        p = parent[i]
        accumulated_scale[i] = scale_of[i] * (accumulated_scale[p] if p >= 0 else 1.0)
    non_unit = [i for i in range(count) if abs(scale_of[i] - 1.0) > 1e-6]

    xml_dir = Path(args.xml).parent
    asset_dir = xml_dir / 'assets'
    out = ["<?xml version='1.0' encoding='utf-8'?>",
           '<mujoco model="glb_skeleton">',
           '  <compiler angle="radian" autolimits="true" meshdir="assets" texturedir="assets" />',
           '  <visual>',
           '    <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" />',
           '    <map znear="0.01" zfar="20" />',
           '  </visual>',
           '  <statistic center="0 0 0.9" extent="0.9" />']
    assets, body_geoms = [], {}

    if args.mesh:
        asset_dir.mkdir(parents=True, exist_ok=True)
        skin = doc['skins'][0]
        joint_nodes = skin['joints']
        inverse_bind = np.asarray(gltf.read_accessor(doc, binary, skin['inverseBindMatrices']),
                                  dtype=float).reshape(-1, 4, 4)
        # glTF stores matrices column-major
        inverse_bind = inverse_bind.transpose(0, 2, 1)
        joint_slot = {node: slot for slot, node in enumerate(joint_nodes)}

        # ---- textures and materials -------------------------------------------------
        texture_lines = []
        for i, image in enumerate(doc.get('images', [])):
            view = doc['bufferViews'][image['bufferView']]
            start = view.get('byteOffset', 0)
            data = binary[start:start + view['byteLength']]
            # image bytes are not reliably PNG/JPEG (one has an empty mimeType), so normalise
            # everything through PIL; MuJoCo only reads PNG and JPEG
            import io
            from PIL import Image as PILImage
            try:
                with PILImage.open(io.BytesIO(data)) as img:
                    img.convert('RGBA').save(asset_dir / f'tex_{i:02d}.png')
                suffix = '.png'
            except Exception as error:                       # keep the raw bytes if PIL cannot decode
                print(f'  texture {i} could not be decoded ({error}); skipped')
                continue
            texture_lines.append(f'    <texture name="tex_{i:02d}" type="2d" file="tex_{i:02d}{suffix}" />')

        material_lines = []
        for i, material in enumerate(doc.get('materials', [])):
            pbr = material.get('pbrMetallicRoughness', {})
            factor = pbr.get('baseColorFactor', [1, 1, 1, 1])
            texture_ref = ''
            base = pbr.get('baseColorTexture')
            if base is not None:
                source = doc['textures'][base['index']].get('source')
                if source is not None and (asset_dir / f'tex_{source:02d}.png').exists():
                    texture_ref = f' texture="tex_{source:02d}"'
            rgba = f'{factor[0]:.4g} {factor[1]:.4g} {factor[2]:.4g} {factor[3]:.4g}'
            material_lines.append(
                f'    <material name="mat_{i:02d}" rgba="{rgba}"{texture_ref} />')
        assets += texture_lines + material_lines

        # ---- mesh, split by (dominant joint, material) ------------------------------
        pending = {}          # node -> list of (material, vertices, uvs, faces)
        for p_index, primitive in enumerate(doc['meshes'][0]['primitives']):
            attributes = primitive['attributes']
            if 'JOINTS_0' not in attributes:
                continue
            position = np.asarray(gltf.read_accessor(doc, binary, attributes['POSITION']), float)
            uv = (np.asarray(gltf.read_accessor(doc, binary, attributes['TEXCOORD_0']), float)
                  if 'TEXCOORD_0' in attributes else np.zeros((len(position), 2)))
            normal = (np.asarray(gltf.read_accessor(doc, binary, attributes['NORMAL']), float)
                      if 'NORMAL' in attributes else np.zeros((len(position), 3)))
            joints = np.asarray(gltf.read_accessor(doc, binary, attributes['JOINTS_0']), float).astype(int)
            weights = np.asarray(gltf.read_accessor(doc, binary, attributes['WEIGHTS_0']), float)
            index = np.asarray(gltf.read_accessor(doc, binary, primitive['indices']), float).astype(int)
            if weights.ndim != 2:
                weights = weights.reshape(len(position), -1)
                joints = joints.reshape(len(position), -1)
            material = primitive.get('material', -1)

            dominant = joints[np.arange(len(joints)), np.argmax(weights, axis=1)]
            # move each vertex into its bone's bind frame
            local = np.empty_like(position)
            local_normal = np.empty_like(normal)
            for slot in np.unique(dominant):
                mask = dominant == slot
                homogeneous = np.concatenate([position[mask], np.ones((mask.sum(), 1))], axis=1)
                local[mask] = (inverse_bind[slot] @ homogeneous.T).T[:, :3]
                rotated = (inverse_bind[slot][:3, :3] @ normal[mask].T).T
                lengths = np.linalg.norm(rotated, axis=1, keepdims=True)
                local_normal[mask] = rotated / np.where(lengths > 1e-9, lengths, 1.0)

            triangles = index.reshape(-1, 3)
            # a triangle may only join a bone when all three vertices belong to that bone: every
            # vertex above was moved into *its own* bone frame, so a mixed triangle would be
            # attached to the wrong body and fly off as a shard (observed)
            coherent = ((dominant[triangles[:, 0]] == dominant[triangles[:, 1]]) &
                        (dominant[triangles[:, 0]] == dominant[triangles[:, 2]]))
            triangles = triangles[coherent]
            if len(triangles) == 0:
                continue
            # JOINTS_0 indexes the *skin slot*, not the node: map slot -> node before attaching
            slot_to_node = np.asarray(joint_nodes)[dominant]
            tri_node = slot_to_node[triangles[:, 0]]
            for node_index in np.unique(tri_node):
                sel = triangles[tri_node == node_index]
                sel = sel[(sel[:, 0] != sel[:, 1]) & (sel[:, 1] != sel[:, 2]) & (sel[:, 0] != sel[:, 2])]
                if len(sel) == 0 or len(np.unique(sel)) < 4:
                    continue
                used = np.unique(sel)
                remap = {old: new for new, old in enumerate(used)}
                faces = np.vectorize(remap.get)(sel)
                pending.setdefault(int(node_index), []).append(
                    (material, local[used], uv[used], faces, local_normal[used]))

        mesh_index = 0
        for node_index, groups in sorted(pending.items()):
            for material, verts, uvs, faces, normals in groups:
                name = f'mesh_{node_index:03d}_{mesh_index:02d}'
                mesh_index += 1
                obj = []
                for v, t in zip(verts, uvs):
                    obj.append(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}')
                for t in uvs:
                    obj.append(f'vt {t[0]:.6f} {1.0 - t[1]:.6f}')
                for n in normals:
                    obj.append(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}')
                for f in faces + 1:
                    obj.append(f'f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}')
                (asset_dir / f'{name}.obj').write_text('\n'.join(obj) + '\n', encoding='utf-8')
                assets.append(f'    <mesh name="{name}" file="{name}.obj" />')
                material_attr = f' material="mat_{material:02d}"' if material is not None and material >= 0 else ''
                body_geoms.setdefault(int(node_index), []).append(
                    f'      <geom type="mesh" mesh="{name}"{material_attr} '
                    f'contype="0" conaffinity="0" group="2" />')

    out.append('  <asset>')
    out += assets
    out.append('  </asset>')
    out.append('  <worldbody>')
    out.append(f'    <light pos="1 -1 2" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7" />')
    out.append('    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.30 0.32 0.36 1" '
               'contype="0" conaffinity="0" />')
    out.append(f'    <body name="glb_root" pos="0 0 0" quat="{ROT_Y_UP_TO_Z_UP}">')
    out.append('      <freejoint name="glb_root_free" />')
    out.append('      ' + STUB_INERTIAL)

    def emit(index, depth):
        pad = '  ' * depth
        node = nodes[index]
        translation = np.asarray(node.get('translation', [0, 0, 0]), float)
        # node scale is not representable on a MuJoCo body: fold it into the offset
        translation = translation * (accumulated_scale[parent[index]] if parent[index] >= 0 else 1.0)
        rotation = node.get('rotation', [0, 0, 0, 1])
        pos = f'{translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f}'
        quat = ' '.join(f'{float(v):.6f}' for v in [rotation[3], rotation[0], rotation[1], rotation[2]])
        out.append(f'{pad}<body name="glb_body_{index:03d}" pos="{pos}" quat="{quat}">')
        out.append(f'{pad}  {STUB_INERTIAL}')
        out.append(f'{pad}  <joint name="glb_joint_{index:03d}" type="ball" />')
        for geom in body_geoms.get(index, []):
            out.append(geom.replace('      ', pad + '  ', 1))
        if not args.mesh:
            out.append(f'{pad}  <geom type="sphere" size="{args.sphere_radius:g}" '
                       f'rgba="0.75 0.77 0.80 1" contype="0" conaffinity="0" group="1" />')
        for child in children[index]:
            emit(child, depth + 1)
        out.append(f'{pad}</body>')

    for root in roots:
        emit(root, 3)
    out += ['    </body>', '  </worldbody>', '</mujoco>']

    xml_path = Path(args.xml)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text('\n'.join(out) + '\n', encoding='utf-8')

    mapping = {
        'source': Path(args.glb).name,
        'node_count': count,
        'roots': [nodes[i].get('name') for i in roots],
        'scaled_nodes': [nodes[i].get('name') for i in non_unit],
        'mesh_groups': sum(len(v) for v in body_geoms.values()),
        'nodes': [
            {'index': i, 'name': nodes[i].get('name'),
             'body': f'glb_body_{i:03d}', 'joint': f'glb_joint_{i:03d}', 'parent': parent[i]}
            for i in range(count)
        ],
    }
    Path(args.nodes).parent.mkdir(parents=True, exist_ok=True)
    Path(args.nodes).write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'nodes': count, 'roots': mapping['roots'],
                      'scaled_nodes': len(non_unit), 'mesh_groups': mapping['mesh_groups'],
                      'xml': str(xml_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
