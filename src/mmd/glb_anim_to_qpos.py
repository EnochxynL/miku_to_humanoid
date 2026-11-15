"""Drive the glTF-derived MuJoCo model with the animation that mmd2gltf-gui baked into the GLB.

glTF stores a node's local transform as bind TRS plus animation channels; our MJCF body carries the
bind rotation as its own ``quat`` and a ball joint on top, so the joint value is simply

    q_joint(t) = inverse(bind_rotation) * animated_rotation(t)

and no coordinate conversion is needed anywhere: the whole tree hangs under one root body that
already applies Y-up -> Z-up, so joints stay in glTF space.

    uv run python src/mmd/glb_anim_to_qpos.py --glb <model.glb> --xml runtime/glb/herta.xml \
        --nodes runtime/glb/herta_nodes.json --out runtime/glb/herta_anim.npz --view
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_gltf_module():
    spec = importlib.util.spec_from_file_location(
        'mmd_glb_to_bvh', Path(__file__).with_name('mmd_glb_to_bvh.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb', required=True)
    parser.add_argument('--xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--camera-distance', type=float, default=2.0)
    parser.add_argument('--camera-azimuth', type=float, default=90.0)
    args = parser.parse_args()

    gltf = load_gltf_module()
    doc, binary = gltf.load_glb(Path(args.glb))
    nodes = doc['nodes']
    clip = gltf.Clip(doc, binary, 0)
    duration = clip.t1 - clip.t0

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    bind = {}
    for entry in mapping['nodes']:
        node = nodes[entry['index']]
        q = node.get('rotation', [0, 0, 0, 1])
        bind[entry['index']] = R.from_quat([q[0], q[1], q[2], q[3]])

    # which joints exist, and at which qpos address
    address = {}
    for entry in mapping['nodes']:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry['joint'])
        if joint_id >= 0:
            address[entry['index']] = int(model.jnt_qposadr[joint_id])

    # channels: node -> path; we only use rotation, and the root translation if animated
    animated = sorted(node for node, paths in clip.channels.items() if 'rotation' in paths)
    print(f'animation drives {len(animated)} of {len(nodes)} nodes')

    frames = int(round(duration * args.fps))
    cache = []
    for i in range(frames):
        time = clip.t0 + i / args.fps
        data.qpos[:] = model.qpos0
        for node_index in animated:
            sampled = clip.sample(node_index, 'rotation', time)
            if sampled is None:
                continue
            local = R.from_quat([sampled[0], sampled[1], sampled[2], sampled[3]])
            joint = bind[node_index].inv() * local
            q = joint.as_quat()
            adr = address.get(node_index)
            if adr is None:
                continue
            data.qpos[adr:adr + 4] = [q[3], q[0], q[1], q[2]]     # MuJoCo is (w, x, y, z)
        cache.append(np.array(data.qpos).copy())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=np.array(cache), times=np.arange(len(cache)) / args.fps)
    print(f'wrote {len(cache)} frames to {out}')

    if not args.view:
        return
    from view_model import play
    play(model, data, cache, speed=args.speed,
         distance=args.camera_distance, azimuth=args.camera_azimuth)


if __name__ == '__main__':
    main()
