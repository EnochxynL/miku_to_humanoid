"""Retarget the glTF-driven PMX pose onto a robot through GMR.

Same model as ``pmx_to_gmr.py`` - the source is a rigid-body rig, so a PMX bone carries no rest
rotation and the constant that carries it onto a robot link is ``C = R_rest(link) * align`` - but
the animation now comes from the glTF that mmd2gltf-gui exported, which already has MMD's IK baked
in. That removes the custom VMD bake from the pipeline entirely.

The skeleton MuJoCo model comes from ``glb_to_mjcf.py`` (bodies ``glb_body_NNN``); bone names are the
MMD ones, so the link tables in ``pmx_to_gmr`` apply unchanged.

    <gmr>/.venv/Scripts/python.exe src/mmd/glb_to_gmr.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config-out runtime/gmr_inputs/glb_to_g1.json \
        --bake-out runtime/glb/g1_qpos.npz --view
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmx_to_gmr import LINKS, MEASURE, rotation_between   # noqa: E402  (same link tables)

# link -> (parent robot link) for the constant chain
ROBOT_PARENT = {
    'torso_link': 'pelvis',
    'left_hip_yaw_link': 'pelvis', 'right_hip_yaw_link': 'pelvis',
    'left_knee_link': 'left_hip_yaw_link', 'right_knee_link': 'right_hip_yaw_link',
    'left_ankle_roll_link': 'left_knee_link', 'right_ankle_roll_link': 'right_knee_link',
    'left_shoulder_yaw_link': 'torso_link', 'right_shoulder_yaw_link': 'torso_link',
    'left_elbow_link': 'left_shoulder_yaw_link', 'right_elbow_link': 'right_shoulder_yaw_link',
    'left_wrist_yaw_link': 'left_elbow_link', 'right_wrist_yaw_link': 'right_elbow_link',
}


def load_glb_frames(xml_path, nodes_path, anim_path):
    """World pose of every MMD-named bone per frame, in GMR's human-data shape."""
    mapping = json.loads(Path(nodes_path).read_text(encoding='utf-8'))
    body_of = {}
    for entry in mapping['nodes']:
        if entry.get('name'):
            body_of.setdefault(entry['name'], entry['index'])
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    frames = []
    for qpos in np.load(anim_path)['qpos']:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        frame = {}
        for name, index in body_of.items():
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index:03d}')
            if body < 0:
                continue
            rotation = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            q = rotation.as_quat()
            frame[name] = [np.array(data.xpos[body], dtype=float),
                           np.array([q[3], q[0], q[1], q[2]])]
        frames.append(frame)
    return frames


def build_config(base, robot_xml, glb_xml, nodes_path):
    robot = mujoco.MjModel.from_xml_path(robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)
    src = mujoco.MjModel.from_xml_path(glb_xml)
    src_data = mujoco.MjData(src)
    mujoco.mj_forward(src, src_data)
    mapping = json.loads(Path(nodes_path).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}

    def src_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    base['human_root_name'] = '腰'
    base['human_height_assumption'] = float(src_data.xpos[src_body('頭')][2])
    base['ground_height'] = 0.0
    scales = {'腰': 1.0}
    for link, (bone, child, robot_child) in LINKS.items():
        if link == 'pelvis':
            continue
        robot_length = np.linalg.norm(robot_data.xpos[robot_body(robot_child)] -
                                      robot_data.xpos[robot_body(link)])
        src_length = np.linalg.norm(src_data.xpos[src_body(child)] - src_data.xpos[src_body(bone)])
        if src_length > 1e-6:
            scales.setdefault(bone, float(robot_length / src_length))
    base['human_scale_table'] = scales

    report = []
    for table in ('ik_match_table1', 'ik_match_table2'):
        for link in list(base[table].keys()):
            if link not in LINKS:
                continue
            bone, child, robot_child = LINKS[link]
            rest = R.from_matrix(np.array(robot_data.xmat[robot_body(link)]).reshape(3, 3))
            align = R.identity()
            pair = MEASURE.get(link)
            if link in ('pelvis', 'torso_link'):
                # The spine must not inherit a *tilt* from the source bind pose (this rig binds with
                # its spine leaning), but it does need the yaw: the glTF rig is turned ~90 deg
                # relative to the robot's rest, and leaving that out made every limb target fight
                # the spine orientation (measured 24-120 deg errors).
                if link in LINKS:
                    bone, child, _ = LINKS[link]
                    if bone in index_of and child in index_of:
                        src_dir = src_data.xpos[src_body(child)] - src_data.xpos[src_body(bone)]
                        full = rotation_between(np.array([0.0, 0.0, 1.0]), src_dir)
                        yaw = float(full.as_euler('zyx', degrees=True)[0])
                        align = R.from_euler('z', yaw, degrees=True)
                        report.append((link, yaw))
                pair = None
            if pair is not None:
                src_a, src_b, robot_a, robot_b = pair
                if src_a in index_of and src_b in index_of:
                    robot_dir = (robot_data.xpos[robot_body(robot_b)] -
                                 robot_data.xpos[robot_body(robot_a)])
                    src_dir = src_data.xpos[src_body(src_b)] - src_data.xpos[src_body(src_a)]
                    align = rotation_between(robot_dir, src_dir)
                    report.append((link, float(np.degrees(align.magnitude()))))
            constant = rest * align
            x, y, z, w = constant.as_quat()
            base[table][link][0] = bone
            base[table][link][4] = [float(w), float(x), float(y), float(z)]
    return base, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--config-out', required=True)
    parser.add_argument('--bake-out', default=None)
    parser.add_argument('--posture-cost', type=float, default=1e-2)
    parser.add_argument('--smooth', type=float, default=0.5)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--frames', default='0,40,80,120,160,200')
    args = parser.parse_args()

    gmr_root = Path(args.gmr_root)
    sys.path.insert(0, str(gmr_root))
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    robot_xml = str(gmr_root / 'assets' / 'unitree_g1' / 'g1_mocap_29dof.xml')
    base_path = gmr_root / 'general_motion_retargeting' / 'ik_configs' / 'bvh_lafan1_to_g1.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))

    config, report = build_config(base, robot_xml, args.glb_xml, args.nodes)
    print(f"{'link':<26}{'align':>8}")
    for link, angle in report:
        print(f'{link:<26}{angle:7.1f}')
    config_path = Path(args.config_out).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print('config:', config_path)

    gmrmod.IK_CONFIG_DICT.setdefault('glb', {})[args.robot] = str(config_path)
    frames = load_glb_frames(args.glb_xml, args.nodes, args.anim)
    print(f'source frames: {len(frames)}')
    retargeter = GMR(src_human='glb', tgt_robot=args.robot,
                     actual_human_height=config['human_height_assumption'], verbose=False)
    if args.posture_cost > 0:
        import mink
        posture = mink.PostureTask(retargeter.model, cost=args.posture_cost)
        posture.set_target(retargeter.model.qpos0)
        retargeter.tasks1.append(posture)
        retargeter.tasks2.append(posture)

    robot = mujoco.MjModel.from_xml_path(robot_xml)
    data = mujoco.MjData(robot)

    wanted = [int(v) for v in args.frames.split(',') if int(v) < len(frames)]
    errors = {}
    for index in wanted:
        qpos = retargeter.retarget(frames[index])
        data.qpos[:] = qpos
        mujoco.mj_forward(robot, data)
        for link, (src_a, src_b, robot_a, robot_b) in MEASURE.items():
            human = frames[index][src_b][0] - frames[index][src_a][0]
            human /= (np.linalg.norm(human) or 1.0)
            a = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_a)
            b = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_b)
            if a < 0 or b < 0:
                continue
            got = data.xpos[b] - data.xpos[a]
            got /= (np.linalg.norm(got) or 1.0)
            errors.setdefault(link, []).append(np.degrees(np.arccos(np.clip(got @ human, -1, 1))))
    print(f"\n{'link':<26}{'segment error (deg)':>22}")
    for link, values in errors.items():
        print(f'{link:<26}{np.mean(values):21.1f}')

    if not args.bake_out:
        return
    cache = [np.array(retargeter.retarget(frame)) for frame in frames]
    if args.smooth > 0:
        alpha = float(args.smooth)
        filtered = [cache[0].copy()]
        for previous, current in zip(cache[:-1], cache[1:]):
            filtered.append(alpha * previous + (1.0 - alpha) * current)
        cache = filtered
    out = Path(args.bake_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=np.array(cache), times=np.arange(len(cache)) / 30.0)
    print(f'baked {len(cache)} frames -> {out}')

    if args.view:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from view_model import play
        play(robot, data, cache, speed=args.speed, distance=2.6, azimuth=120.0, lookat=(0.0, 0.0, 0.85))


if __name__ == '__main__':
    main()
