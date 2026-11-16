"""Draw the source rig as an actual stick skeleton beside the retargeted robot.

GMR's own overlay draws one RGB coordinate frame per *task* bone (14 of them), which reads as a
starburst rather than a skeleton. Here every bone in the MMD hierarchy is connected to its parent, so
the figure that produced the targets is visible as a body and can be compared with the robot directly.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/view_skeleton.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config runtime/gmr_inputs/glb_to_g1.json \
        --loop --speed 0.5 --offset 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import scipy.spatial.transform as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
def load_glb_frames(xml_path, nodes_path, anim_path):
    """World poses of every bone per frame, from a qpos file. Viewer-only: the pipeline uses mmd_pose."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    bodies = {}
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name and name != 'glb_root' and name != 'pmx_root':
            bodies[body] = name
    frames = []
    for qpos in np.load(anim_path)['qpos']:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        frame = {}
        for body, name in bodies.items():
            rotation = st.Rotation.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            q = rotation.as_quat()
            frame[name] = [np.array(data.xpos[body], dtype=float),
                           np.array([q[3], q[0], q[1], q[2]])]
        frames.append(frame)
    return frames   # noqa: E402


def body_tree(model):
    """(parent name, child name) pairs from the MJCF body hierarchy, plus the root names.

    The bodies carry the MMD bone names, so the model itself is the authority - no side table.
    """
    name_of = {body: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
               for body in range(model.nbody)}
    edges, roots = [], []
    for body in range(1, model.nbody):
        name = name_of[body]
        if not name or name == 'glb_root':
            continue
        parent = int(model.body_parentid[body])
        parent_name = name_of.get(parent)
        if parent_name and parent_name not in ('world', 'glb_root'):
            edges.append((parent_name, name))
        else:
            roots.append(name)
    return edges, roots


def connect_subset(names, model):
    """Connect each bone of a subset to its nearest ancestor that is also in the subset."""
    wanted = set(names)
    edges = []
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name not in wanted:
            continue
        parent = int(model.body_parentid[body])
        while parent > 0:
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
            if parent_name in wanted:
                edges.append((parent_name, name))
                break
            parent = int(model.body_parentid[parent])
    return edges


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-xml', '--glb-xml', dest='glb_xml', required=True)
    parser.add_argument('--nodes', default=None, help='unused: the bone tree comes from the MJCF')
    parser.add_argument('--anim', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--offset', type=float, default=0.85,
                        help='shift the skeleton sideways (metres) so it stands beside the robot')
    parser.add_argument('--radius', type=float, default=0.012)
    parser.add_argument('--distance', type=float, default=2.8)
    parser.add_argument('--azimuth', type=float, default=88.1)
    parser.add_argument('--source', choices=('gmr', 'raw'), default='gmr',
                        help='gmr = draw exactly what GMR feeds the IK (scaled_human_data); '
                             'raw = draw the whole MMD rig from the clip')
    parser.add_argument('--exclude', default='下擺,髪,帽,袖,裙,飾',
                        help='comma separated name fragments to leave out (clothing/cage bones flood '
                             'the view: the skirt alone is 224 bones)')
    parser.add_argument('--no-robot', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, args.gmr_root)
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import ROBOT_XML_DICT

    gmrmod.IK_CONFIG_DICT.setdefault('glb', {})[args.robot] = str(Path(args.config).resolve())
    frames = load_glb_frames(args.glb_xml, args.nodes, args.anim)
    # the bone tree belongs to the *source* model; the robot model is only for playback
    source_model = mujoco.MjModel.from_xml_path(args.glb_xml)
    if args.source == 'raw':
        edges, roots = body_tree(source_model)
        skip = [s for s in args.exclude.split(',') if s]
        edges = [(a, b) for a, b in edges if not any(s in a or s in b for s in skip)]
    else:
        edges, roots = [], []
    print(f'{len(frames)} frames, {len(edges)} bone connections')

    retargeter = GMR(src_human='glb', tgt_robot=args.robot,
                     actual_human_height=1.445, verbose=False)
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [args.offset * 0.5, 0.0, 0.95]
        viewer.cam.distance = args.distance
        viewer.cam.elevation = -8
        viewer.cam.azimuth = args.azimuth
        offset = np.array([args.offset, 0.0, 0.0])
        rgba = [0.95, 0.55, 0.2, 1.0]

        while viewer.is_running():
            for frame in frames:
                qpos = retargeter.retarget(frame)
                if args.source == 'gmr':
                    # the data GMR actually solves against, not the raw clip
                    human = {k: v for k, v in retargeter.scaled_human_data.items()}
                    if not edges:
                        edges.extend(connect_subset(list(human.keys()), source_model))
                    frame = human
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)

                scene = viewer.user_scn
                scene.ngeom = 0
                drawn = 0
                for parent_name, child_name in edges:
                    if parent_name not in frame or child_name not in frame:
                        continue
                    a = np.asarray(frame[parent_name][0]) + offset
                    b = np.asarray(frame[child_name][0]) + offset
                    if np.linalg.norm(b - a) < 1e-4:
                        continue
                    if drawn >= scene.maxgeom:
                        break
                    geom = scene.geoms[drawn]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=np.array([args.radius, 0.0, 0.0]),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(rgba),
                    )
                    mujoco.mjv_connector(
                        scene.geoms[drawn],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        width=args.radius,
                        from_=a,
                        to=b,
                    )
                    drawn += 1
                scene.ngeom = drawn
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(0.0, 1.0 / 30.0 / args.speed - 1.0 / 30.0))
            if not args.loop:
                return


if __name__ == '__main__':
    main()
