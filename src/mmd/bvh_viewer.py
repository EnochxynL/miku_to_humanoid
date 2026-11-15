"""Draw a BVH skeleton in the MuJoCo viewer, using pybvh for the parsing.

MuJoCo cannot load glTF and has no skinning, so a motion source can only be inspected as a
skeleton. Rather than rebuilding the hierarchy as joints, every bone segment is a `mocap`
body carrying a capsule; pybvh supplies the joint positions per frame, and the capsule is
placed at the segment midpoint with its +Z axis along the bone.

    uv run python src/reference_maker/bvh_viewer.py runtime/gmr_inputs/herta_mmd.bvh --loop
    uv run python src/reference_maker/bvh_viewer.py a.bvh b.bvh --loop --speed 0.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import pybvh

PALETTE = [(0.35, 0.62, 0.95, 1.0), (0.95, 0.52, 0.25, 1.0),
           (0.45, 0.85, 0.45, 1.0), (0.90, 0.85, 0.35, 1.0)]


def load(path: Path, colour_index: int):
    bvh = pybvh.read_bvh_file(str(path))
    positions = np.asarray(pybvh.frames_to_node_positions(bvh), dtype=float)   # (F, J, 3)
    names = list(bvh.joint_names) if hasattr(bvh, 'joint_names') else []
    parents = []
    nodes = bvh.nodes if hasattr(bvh, 'nodes') else None
    if nodes is not None:
        index = {n.name: i for i, n in enumerate(nodes)}
        for n in nodes:
            parents.append(index.get(n.parent.name, -1) if getattr(n, 'parent', None) else -1)
    if len(parents) != positions.shape[1]:
        # fall back: connect every joint to the root only (still readable as a cloud)
        parents = [-1] * positions.shape[1]
    return {'positions': positions, 'names': names, 'parents': parents, 'colour': colour_index,
            'scale': 0.01}


def build(spec: mujoco.MjSpec, sources, radius=0.018):
    segments = []
    for source in sources:
        pos, parents = source['positions'], source['parents']
        colour = PALETTE[source['colour'] % len(PALETTE)]
        body = spec.worldbody.add_body(name=f"origin{source['colour']}", pos=[0, 0, 0])
        for j in range(pos.shape[1]):
            p = parents[j] if j < len(parents) else -1
            if p is None or p < 0:
                continue
            length = np.linalg.norm(pos[0, j] - pos[0, p]) * source['scale']
            if length < 1e-4:
                continue
            mocap = body.add_body(name=f"seg{source['colour']}_{j}", pos=[0, 0, 0], mocap=True)
            mocap.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[radius, length / 2],
                           rgba=colour, mass=0)
            segments.append((source, j, p, source['colour']))
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bvh', nargs='+')
    parser.add_argument('--scale', type=float, default=0.01, help='BVH units -> metres')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()

    sources = [load(Path(p), i) for i, p in enumerate(args.bvh)]
    for source in sources:
        source['scale'] = args.scale

    spec = mujoco.MjSpec()
    spec.option.gravity = (0.0, 0.0, 0.0)
    segments = build(spec, sources)
    model = spec.compile()
    data = mujoco.MjData(model)
    # mocap bodies are numbered in creation order: map them back to their segment
    mocap_ids = [b.id for b in model.bodies if b.mocapid >= 0] if hasattr(model, 'bodies') else []
    order = [b.mocapid for b in spec.worldbody.bodies[0].bodies] if hasattr(spec, 'worldbody') else []

    frames = min(s['positions'].shape[0] for s in sources)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for f in range(frames):
                start = time.time()
                slot = 0
                for source, j, p, _ in segments:
                    a = source['positions'][f, p] * args.scale
                    b = source['positions'][f, j] * args.scale
                    direction = b - a
                    length = np.linalg.norm(direction)
                    if length < 1e-6:
                        slot += 1
                        continue
                    direction /= length
                    # quaternion rotating +Z onto the bone direction
                    z = np.array([0.0, 0.0, 1.0])
                    axis = np.cross(z, direction)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm < 1e-8:
                        quat = np.array([1.0, 0.0, 0.0, 0.0]) if direction[2] > 0 else \
                            np.array([0.0, 1.0, 0.0, 0.0])
                    else:
                        axis /= axis_norm
                        angle = np.arccos(np.clip(direction @ z, -1.0, 1.0))
                        quat = np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])
                    data.mocap_pos[slot] = (a + b) / 2
                    data.mocap_quat[slot] = quat
                    slot += 1
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, (1.0 / 30.0) / args.speed - (time.time() - start)))
            if not args.loop:
                break


if __name__ == '__main__':
    main()
