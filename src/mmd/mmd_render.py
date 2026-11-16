"""Everything MuJoCo needs in order to *draw* an MMD character: the model and, optionally, its motion.

Neither half is on the retarget path - the pipeline uses ``mmd_pose`` and never touches MuJoCo. What
this produces is a renderable bundle:

    <out>            the skeleton (or with --mesh the textured model) + assets/
    <anim-out>       qpos per frame when --vmd is given, else one frame of the bind pose

The two steps are ordered, not independent: the motion needs the joint layout of the model it was baked
against, and both share the bone names. Baking keeps the root translation (``センター``), so the
character travels instead of dancing in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.pmx import parse_pmx

"""Names for MuJoCo objects built from a source rig.

Every emitter uses the source's own names - a bone is called ``左腕`` in the MJCF because that is what
the model calls it. Nothing downstream needs a translation table, which is why the mapping files are
gone. This module exists so the PMX and glTF emitters share one rule.
"""

def unique_names(raw_names):
    """Make names unique (MuJoCo rejects duplicates) and non-empty."""
    seen = {}
    names = []
    for index, raw in enumerate(raw_names):
        name = (raw or '').strip() or 'bone_%d' % index
        count = seen.get(name, 0) + 1
        seen[name] = count
        names.append(name if count == 1 else '%s_%d' % (name, count))
    return names


def xml_attr(value):
    """Escape a name for use inside a double-quoted XML attribute."""
    return (value.replace('&', '&amp;').replace('<', '&lt;')
                 .replace('>', '&gt;').replace('"', '&quot;'))



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
    parser.add_argument('--tree', default=None,
                        help='optional bone table dump (provenance)')
    parser.add_argument('--scale', type=float, default=0.07,
                        help='MMD units to metres (the player and the glTF exporter both use 0.07)')
    parser.add_argument('--model-name', default='pmx_skeleton')
    parser.add_argument('--all-bones', dest='main_bones_only', action='store_false',
                        help='draw every bone, including hair and skirt chains')
    parser.add_argument('--mesh', action='store_true',
                        help='emit the textured mesh (one OBJ group per material and bone) instead of '
                             'capsules; the geometry is segmented by the triangle majority bone')
    parser.add_argument('--assets', default=None,
                        help='directory for mesh/texture files (default: <xml dir>/assets)')
    parser.add_argument('--no-flip-v', dest='flip_v', action='store_false',
                        help='keep V as-is; by default it is flipped because MMD UVs start at the top')
    parser.add_argument('--keep-overlays', action='store_true',
                        help='keep materials whose alpha < 0.5 (MMD overlays: the blush layer 颜+ is '
                             'alpha 0 and the eye shadow 目影 is alpha 0.3; MuJoCo renders them opaque '
                             'and they blacken the face)')
    parser.add_argument('--vmd', default=None,
                        help='also bake this motion; omitted = the model only')
    parser.add_argument('--anim-out', default=None,
                        help='qpos output; defaults to <model stem>_anim.npz')
    parser.add_argument('--step', type=int, default=1, help='sample every N VMD frames')
    parser.add_argument('--capsule-radius', type=float, default=0.008,
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

    # absolute bone positions in MuJoCo space (offsets are stored parent-relative)
    absolute = [np.zeros(3) for _ in range(count)]

    def _accumulate(index, parent_position):
        absolute[index] = np.asarray(parent_position) + np.asarray(offsets[index])
        for child in children[index]:
            _accumulate(child, absolute[index])

    for root_index in roots:
        _accumulate(root_index, np.zeros(3))

    mesh_info = None
    if args.mesh:
        from engine.mesh import export as export_mesh
        assets = Path(args.assets) if args.assets else Path(args.out).parent / 'assets'
        mesh_info = export_mesh(model, assets, base_dir=Path(args.pmx).parent,
                                bone_positions=absolute, scale=scale,
                                flip_v=args.flip_v, min_alpha=0.0 if args.keep_overlays else 0.5)
        print('mesh: %d groups, %d triangles, %d textures'
              % (len(mesh_info['groups']), sum(g['triangles'] for g in mesh_info['groups']),
                 len(mesh_info['textures'])))
        geoms_by_bone = {}
        for group in mesh_info['groups']:
            geoms_by_bone.setdefault(group['bone'], []).append(group)

    lines = ["<?xml version='1.0' encoding='utf-8'?>",
             f'<mujoco model="{args.model_name}">']
    if mesh_info:
        lines.append('  <compiler angle="radian" autolimits="true" meshdir="assets" texturedir="assets" />')
    else:
        lines.append('  <compiler angle="radian" autolimits="true" />')
    lines += [
             '  <visual>',
             '    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1" />',
             '    <map znear="0.01" zfar="20" />',
             '    <global offwidth="1280" offheight="1024" />',
             '  </visual>',
             '  <statistic center="0 0 0.8" extent="1.1" />',
             ]
    if mesh_info:
        lines.append('  <asset>')
        for index, name in enumerate(mesh_info['textures']):
            lines.append(f'    <texture name="tex{index:02d}" type="2d" file="{name}" />')
        for group in mesh_info['groups']:
            lines.append(f'    <mesh name="{group["name"]}" file="{group["file"]}" />')
        tex_slot = {name: index for index, name in enumerate(mesh_info['textures'])}
        for material in mesh_info['materials']:
            if material['alpha'] < 0.5 and not args.keep_overlays:
                continue
            r, g, b = material['diffuse']
            texture = material['texture']
            reference = (' texture="tex%02d"' % tex_slot[texture]) if texture in tex_slot else ''
            lines.append('    <material name="mat%03d"%s rgba="%.3f %.3f %.3f %.3f" '
                         'specular="0" shininess="0" />'
                         % (material['index'], reference, r, g, b, material['alpha']))
        lines.append('  </asset>')
    lines += [
             '  <worldbody>',
             '    <light pos="1 -1 2" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7" />',
             '    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.28 0.30 0.34 1" contype="0" conaffinity="0" group="0" />',
             '    <body name="pmx_root" pos="0 0 0">',
             '      <freejoint name="pmx_root_free" />',
             '      <inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />']

    names = unique_names([b['name'] for b in bones])

    def emit(index: int, depth: int):
        pad = '  ' * depth
        x, y, z = offsets[index]
        lines.append(f'{pad}<body name="{xml_attr(names[index])}" pos="{x:.9g} {y:.9g} {z:.9g}">')
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />')
        lines.append(f'{pad}  <joint name="{xml_attr(names[index])}" type="ball" />')
        if mesh_info:
            for group in geoms_by_bone.get(index, []):
                lines.append(f'{pad}  <geom type="mesh" mesh="{group["name"]}" '
                             f'material="mat{group["material"]:03d}" contype="0" conaffinity="0" '
                             f'group="1" />')
            cap = None
        else:
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
             'joint': names[i], 'body': names[i],
             'parent': bone['parent'],
             'height': float(to_mujoco_position(bone['pos'], scale)[2])}
            for i, bone in enumerate(bones)
        ],
    }
    if args.tree:
        Path(args.tree).parent.mkdir(parents=True, exist_ok=True)
        Path(args.tree).write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding='utf-8')
    written = [str(out)] + ([str(args.tree)] if args.tree else [])
    anim_out = Path(args.anim_out) if args.anim_out else Path(out).with_name(
        Path(out).stem + '_anim.npz')
    if args.vmd:
        from engine.bake import bake
        qpos, unmatched, times, missing = bake(args.pmx, args.vmd, str(out),
                                               step=args.step, scale=scale)
        anim_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(anim_out, qpos=qpos, times=times)
        summary = {'frames': len(qpos), 'unmatched_vmd_bones': len(unmatched),
                   'bones_without_joint': len(missing), 'anim': str(anim_out)}
    else:
        import mujoco
        robot = mujoco.MjModel.from_xml_path(str(out))
        anim_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(anim_out, qpos=np.array([robot.qpos0]), times=np.zeros(1))
        summary = {'frames': 1, 'anim': str(anim_out)}
    written.append(str(anim_out))
    print(json.dumps({'bones': count, 'roots': tree['roots'][:6], 'motion': summary,
                      'written': written}, ensure_ascii=False))


if __name__ == '__main__':
    main()
