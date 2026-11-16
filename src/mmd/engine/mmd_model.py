"""PMX -> a compiled MuJoCo model, in memory.

This replaces the old "convert to MJCF, then load the MJCF" step. Nothing has to touch the disk: the
XML text and the mesh/texture bytes go straight into ``MjModel.from_xml_string(xml, assets=...)``, with
meshes written as binary MSH (``engine.msh``) because that is ~60x cheaper to produce than OBJ text.

What stays the same is the model itself: every PMX bone becomes a body with a ball joint named after the
bone, so the animation pipeline can address joints by name and the viewer draws the character exactly as
the exporter did.

Costs, measured on the heita model (511 bones, 823 mesh groups, 12 textures):

    parse PMX          0.5 s   (cached per file)
    textures           ~0.4 s  (already-PNG files are reused untouched; converting the 4k PNGs again
                                cost 2.8 s and was pure waste)
    mesh groups        ~0.01 s (MSH)
    compile            ~0.2 s
    -> ~1.1 s first call, and the layer (xml + bytes) is cached, so a second model for the same file
       costs only the compile
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from engine import msh
from engine.mesh import groups as build_groups
from engine.mesh import to_mujoco, write_obj
from engine.pmx import parse_pmx

STUB_INERTIAL = '<inertial pos="0 0 0" mass="0.00001" diaginertia="1e-9 1e-9 1e-9" />'
FLOOR = ('<geom name="floor" type="plane" size="3 3 0.05" rgba="0.28 0.30 0.34 1" '
         'contype="0" conaffinity="0" group="0" />')
_LIGHT = ('<light pos="1 -1 2" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7" />')

_LAYERS = {}            # (path, mtime, scale, mesh, min_alpha) -> (xml, assets, info)


def unique_names(raw_names):
    """Make names unique (MuJoCo rejects duplicates) and non-empty."""
    seen, names = {}, []
    for index, raw in enumerate(raw_names):
        name = (raw or '').strip() or 'bone_%d' % index
        count = seen.get(name, 0) + 1
        seen[name] = count
        names.append(name if count == 1 else '%s_%d' % (name, count))
    return names


def xml_attr(value):
    return (value.replace('&', '&amp;').replace('<', '&lt;')
                 .replace('>', '&gt;').replace('"', '&quot;'))


@dataclass
class Bundle:
    """A compiled model plus what was needed to build it."""

    model: mujoco.MjModel
    info: dict = field(default_factory=dict)
    _xml: str = ''
    _assets: dict = field(default_factory=dict)

    def joint_address(self, bone_name):
        """qpos offset of a bone's ball joint, or None when the rig has no such bone."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, bone_name)
        return int(self.model.jnt_qposadr[joint]) if joint >= 0 else None

    def save(self, out_dir, tree=None):
        """Write the MJCF + OBJ assets to disk - for debugging, or to hand the model to another tool."""
        out_dir = Path(out_dir)
        assets_dir = out_dir / 'assets'
        layer = self.info.get('built')
        if layer is not None:
            write_obj(layer, assets_dir)
        (out_dir / 'model.xml').write_text(self._xml, encoding='utf-8')
        if tree is not None:
            (out_dir / 'tree.json').write_text(
                json.dumps(tree, indent=2, ensure_ascii=False), encoding='utf-8')
        return out_dir / 'model.xml'


def _layer(pmx_path, scale, mesh, min_alpha):
    """(xml text, asset bytes, info) for one PMX - cached, since only the compile is cheap to repeat."""
    pmx_path = Path(pmx_path)
    key = (str(pmx_path.resolve()), pmx_path.stat().st_mtime_ns, scale, mesh, min_alpha)
    if key in _LAYERS:
        return _LAYERS[key]

    model = parse_pmx(str(pmx_path))
    bones = model['bones']
    count = len(bones)
    names = unique_names([b['name'] for b in bones])

    offsets, children, roots = [], {i: [] for i in range(count)}, []
    for index, bone in enumerate(bones):
        parent = bone['parent']
        position = np.asarray(bone['pos'], dtype=float)
        parent_position = (np.asarray(bones[parent]['pos'], dtype=float)
                           if 0 <= parent < count else np.zeros(3))
        offsets.append(to_mujoco(position - parent_position, scale))
        if 0 <= parent < count:
            children[parent].append(index)
        else:
            roots.append(index)

    absolute = [np.zeros(3) for _ in range(count)]

    def accumulate(index, parent_position):
        absolute[index] = np.asarray(parent_position) + np.asarray(offsets[index])
        for child in children[index]:
            accumulate(child, absolute[index])

    for root in roots:
        accumulate(root, np.zeros(3))

    built = None
    if mesh:
        built = build_groups(model, base_dir=pmx_path.parent, bone_positions=absolute,
                             scale=scale, min_alpha=min_alpha)

    assets = {}
    lines = ["<?xml version='1.0' encoding='utf-8'?>", '<mujoco model="pmx_skeleton">',
             '  <compiler angle="radian" autolimits="true" />',
             '  <visual>',
             '    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1" />',
             '    <map znear="0.01" zfar="20" />',
             '    <global offwidth="1280" offheight="1024" />',
             '  </visual>',
             '  <statistic center="0 0 0.8" extent="1.1" />']

    if built is not None:
        lines.append('  <asset>')
        for index, (name, png) in built['textures'].items():
            textures = 'tex%02d' % index
            assets[name] = png
            lines.append('    <texture name="%s" type="2d" file="%s" />' % (textures, name))
        for group in built['groups']:
            filename = group['name'] + '.msh'
            assets[filename] = msh.to_msh(group['positions'], group['normals'],
                                          group['uv_msh'], group['faces'])
            lines.append('    <mesh name="%s" file="%s" />' % (group['name'], filename))
        names_by_png = {png: 'tex%02d' % index for index, (png, _b) in built['textures'].items()}
        for material in built['materials']:
            if material['alpha'] < min_alpha:
                continue
            png = material['texture']
            reference = (' texture="%s"' % names_by_png[png]) if png in names_by_png else ''
            r, g, b = material['diffuse']
            lines.append('    <material name="mat%03d"%s rgba="%.3f %.3f %.3f %.3f" '
                         'specular="0" shininess="0" />'
                         % (material['index'], reference, r, g, b, material['alpha']))
        lines.append('  </asset>')

    lines += ['  <worldbody>', '    ' + _LIGHT, '    ' + FLOOR,
              '    <body name="pmx_root" pos="0 0 0">',
              '      <freejoint name="pmx_root_free" />',
              '      ' + STUB_INERTIAL]

    geoms_by_bone = {}
    if built is not None:
        for group in built['groups']:
            geoms_by_bone.setdefault(group['bone'], []).append(group)

    def emit(index, depth):
        pad = '  ' * depth
        x, y, z = offsets[index]
        lines.append('%s<body name="%s" pos="%.9g %.9g %.9g">'
                     % (pad, xml_attr(names[index]), x, y, z))
        lines.append('%s  %s' % (pad, STUB_INERTIAL))
        lines.append('%s  <joint name="%s" type="ball" />' % (pad, xml_attr(names[index])))
        for group in geoms_by_bone.get(index, []):
            lines.append('%s  <geom type="mesh" mesh="%s" material="mat%03d" '
                         'contype="0" conaffinity="0" group="1" />'
                         % (pad, group['name'], group['material']))
        for child in children[index]:
            emit(child, depth + 1)
        lines.append('%s</body>' % pad)

    for root in roots:
        emit(root, 3)
    lines += ['    </body>', '  </worldbody>', '</mujoco>']
    xml = '\n'.join(lines) + '\n'

    info = {
        'bones': count,
        'roots': [bones[i]['name'] for i in roots],
        'scale': scale,
        'mesh_groups': len(built['groups']) if built else 0,
        'triangles': sum(g['triangles'] for g in built['groups']) if built else 0,
        'textures': len(built['textures']) if built else 0,
        'skipped_groups': built['skipped'] if built else 0,
        'built': built,
        'names': names,
    }
    _LAYERS[key] = (xml, assets, info)
    return _LAYERS[key]


def build(pmx_path, *, scale=0.07, mesh=True, cache=True, min_alpha=0.5):
    """Compile a PMX rig into an MjModel. ``mesh=False`` gives joints only, which is all baking needs."""
    if cache:
        xml, assets, info = _layer(pmx_path, scale, mesh, min_alpha)
    else:
        key = (str(Path(pmx_path).resolve()), Path(pmx_path).stat().st_mtime_ns, scale, mesh, min_alpha)
        _LAYERS.pop(key, None)
        xml, assets, info = _layer(pmx_path, scale, mesh, min_alpha)
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return Bundle(model=model, info=info, _xml=xml, _assets=assets)


def clear_cache():
    _LAYERS.clear()
