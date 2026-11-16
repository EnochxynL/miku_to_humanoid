"""Retarget the glTF-driven PMX pose onto a robot through GMR.

Same model as ``pmx_to_gmr.py`` - the source is a rigid-body rig, so a PMX bone carries no rest
rotation and the constant that carries it onto a robot link is ``C = R_rest(link) * align`` - but
the animation now comes from the glTF that mmd2gltf-gui exported, which already has MMD's IK baked
in. That removes the custom VMD bake from the pipeline entirely.

The skeleton MuJoCo model comes from ``glb_to_mjcf.py`` (bodies ``glb_body_NNN``); bone names are the
MMD ones, so the link tables in ``pmx_to_gmr`` apply unchanged.

    <gmr>/.venv/Scripts/python.exe src/mmd/glb_to_gmr.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --build-config runtime/gmr_inputs/glb_to_g1.json \
        --config configs/mmd/glb_to_g1.json --bake-out runtime/glb/g1_qpos.npz --view
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
from rig_tables import LINKS, MEASURE, rotation_between   # noqa: E402  (same link tables)

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


def load_glb_frames(xml_path, nodes_path, anim_path, wanted=None):
    """World pose of every bone per frame, in GMR's human-data shape (keyed by bone name)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    bodies = {}
    for body in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name and name != 'glb_root' and (wanted is None or name in wanted):
            bodies[body] = name
    frames = []
    for qpos in np.load(anim_path)['qpos']:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        frame = {}
        for body, name in bodies.items():
            rotation = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            q = rotation.as_quat()
            frame[name] = [np.array(data.xpos[body], dtype=float),
                           np.array([q[3], q[0], q[1], q[2]])]
        frames.append(frame)
    return frames


# Second axis per link. It must be the *next* segment, not a longer super-segment: a super-segment
# (upper arm -> wrist) is nearly collinear with the first axis (upper arm -> elbow), the cross product
# degenerates and the alignment silently collapses to identity - which left limbs rolled arbitrarily.
# link -> (source a, source b, robot a, robot b) for the perpendicular reference
ROLL_REF = {
    'left_shoulder_yaw_link': ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link'),
    'right_shoulder_yaw_link': ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link'),
    'left_elbow_link': ('左手首', '左中指１', 'left_wrist_yaw_link', 'left_wrist_roll_link'),
    'right_elbow_link': ('右手首', '右中指１', 'right_wrist_yaw_link', 'right_wrist_roll_link'),
    'left_hip_yaw_link': ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link'),
    'right_hip_yaw_link': ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link'),
    'left_knee_link': ('左足首', '左つま先', 'left_ankle_roll_link', 'left_toe_link'),
    'right_knee_link': ('右足首', '右つま先', 'right_ankle_roll_link', 'right_toe_link'),
    'left_ankle_roll_link': ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link'),
    'right_ankle_roll_link': ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link'),
    'left_wrist_yaw_link': ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link'),
    'right_wrist_yaw_link': ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link'),
}


def frame_from_pair(first, second):
    """Right-handed basis from two directions (neither needs to be axis aligned)."""
    u = first / (np.linalg.norm(first) or 1.0)
    v = second - (second @ u) * u
    v /= (np.linalg.norm(v) or 1.0)
    return np.column_stack([u, v, np.cross(u, v)])


def build_config(base, robot_xml, glb_xml, nodes_path, two_axis=False):
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
    # human_scale_table is a *uniform body* scale, not a per-segment ratio: GMR computes
    # (bone_pos - root_pos) * scale[bone], so a per-bone ratio collapses whichever chain has a small
    # ratio (the torso got 0.27 and shrank into the hips). Use one factor for the whole body, derived
    # from the robot's standing height over the source's.
    head_height = float(robot_data.xpos[:, 2].max())      # tallest robot body point
    source_head_height = float(src_data.xpos[src_body('頭')][2])
    factor = head_height / source_head_height if source_head_height > 1e-6 else 0.9
    base['human_scale_table'] = {bone: float(factor) for bone in
                                 {v[0] for v in LINKS.values()} | {'腰', '上半身2'}}

    report = []
    for table in ('ik_match_table1', 'ik_match_table2'):
        for link in list(base[table].keys()):
            if link not in LINKS:
                continue
            bone, child, robot_child = LINKS[link]
            rest = R.from_matrix(np.array(robot_data.xmat[robot_body(link)]).reshape(3, 3))
            align = R.identity()
            pair = MEASURE.get(link)
            if pair is None and link in LINKS:
                # wrists have no MEASURE entry, but they still need an alignment: build it from the
                # link's own segment, using LINKS for the source/robot pair
                bone_, child_, robot_child_ = LINKS[link]
                pair = (bone_, child_, link, robot_child_)
            if link in ('pelvis', 'torso_link'):
                pair = None
                # The spine must take the rig's *yaw* (the glTF rig is turned ~90 deg from the
                # robot's rest) but not its tilt (this rig binds with the spine leaning). Yaw has to
                # come from a horizontal axis - deriving it from the near-vertical spine direction is
                # meaningless and silently produced 0 deg, which left the torso fighting every limb.
                src_axis = src_data.xpos[src_body('右足')] - src_data.xpos[src_body('左足')]
                robot_axis = (robot_data.xpos[robot_body('right_hip_yaw_link')] -
                              robot_data.xpos[robot_body('left_hip_yaw_link')])
                # same convention as the limbs: rotate the robot's axis onto the source's axis.
                # Both axes are horizontal, so the minimal rotation between them is exactly the yaw.
                align = rotation_between(robot_axis, src_axis)
                report.append((link, float(np.degrees(align.magnitude()))))
                pair = None
            # The PMX rig has no bind rotation (identity), so ``C = R_rest * align`` worked there.
            # This rig hangs under a root that applies Y-up -> Z-up, so its bind world orientation is
            # Rx90; without dividing it out the target is pitched 90 deg (the robot lay down).
            source_bind = R.from_matrix(np.array(src_data.xmat[src_body(bone)]).reshape(3, 3))
            constant = source_bind.inv() * rest * align
            x, y, z, w = constant.as_quat()
            base[table][link][0] = bone
            base[table][link][4] = [float(w), float(x), float(y), float(z)]
    return base, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', default=None, help='unused: bodies are looked up by bone name')
    parser.add_argument('--anim', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--config', '--config-in', dest='config_in', default=None,
                        help='use this config (the calibrated one) instead of building a new one; '
                             'rebuilding would overwrite the paired-pose offsets')
    parser.add_argument('--build-config', '--config-out', dest='config_out', default=None,
                        help='build a fresh config and write it here; only for setting up, the bake '
                             'wants --config')
    parser.add_argument('--bake-out', default=None)
    parser.add_argument('--two-axis', action='store_true',
                        help='also align the bending plane (experimental: currently collapses to '
                             'identity, so the single-axis alignment is the default)')
    parser.add_argument('--posture-cost', type=float, default=0.0,
                        help='0 is the calibrated default: the posture anchor was a band-aid for a wrong C and now biases the pose back toward rest')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='0 keeps the calibrated result; smoothing was hiding IK jitter caused by the wrong C')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--frames', default='0,40,80,120,160,200')
    args = parser.parse_args()

    gmr_root = Path(args.gmr_root)
    sys.path.insert(0, str(gmr_root))
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    robot_xml = str(gmrmod.ROBOT_XML_DICT[args.robot]) if args.robot in getattr(
        gmrmod, 'ROBOT_XML_DICT', {}) else str(gmr_root / 'assets' / 'unitree_g1' / 'g1_mocap_29dof.xml')

    if args.config_in:
        # Reusing the config matters: building one here would overwrite the paired-pose offsets with
        # the uncalibrated ones, and the bake would silently be a different (worse) motion.
        config = json.loads(Path(args.config_in).read_text(encoding='utf-8'))
        report = []
        config_path = Path(args.config_in).resolve()
        print('using existing config:', config_path)
    else:
        if not args.config_out:
            raise SystemExit('either --build-config (build) or --config (reuse) is required')
        # the stock config is only a template, so it is loaded here rather than unconditionally
        base_path = gmr_root / 'general_motion_retargeting' / 'ik_configs' / 'bvh_lafan1_to_g1.json'
        base = json.loads(base_path.read_text(encoding='utf-8'))
        config, report = build_config(base, robot_xml, args.glb_xml, args.nodes,
                                      two_axis=args.two_axis)
        config_path = Path(args.config_out).resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
        print('config:', config_path)
    print(f"{'link':<26}{'align':>8}")
    for link, angle in report:
        print(f'{link:<26}{angle:7.1f}')

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
