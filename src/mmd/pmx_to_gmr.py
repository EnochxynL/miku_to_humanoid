"""Retarget the baked PMX pose onto a robot through GMR, with the PMX rig as a rigid-body source.

This is the path that avoids asking for a constant that does not exist. The source is not
"MMD bone frames with an unknown correspondence" - it is a **rigid-body model** (the PMX skeleton
built as MuJoCo bodies), so:

* a PMX bone carries **no rest rotation**, which makes ``R_pmx(t)`` literally "the rotation away
  from the bind pose";
* therefore the constant that carries a PMX bone onto a robot link is ``C = R_rest(link) * align``:
  ``R_rest`` replays the delta on the robot's own rest orientation, and ``align`` fixes the one
  genuine pose difference - the PMX rests in an A-pose while the robot hangs its arms down. The
  four alignment angles are the ones measured from the player (humerus 65.4 deg, thighs 9.9/10.5).

Nothing here is tuned: the alignment is derived from the two models' bind directions.

    uv run python src/reference_maker/pmx_to_gmr.py \
        --pmx-xml runtime/pmx/herta.xml --tree runtime/pmx/herta_tree.json \
        --qpos runtime/pmx/herta_qpos.npz --config-out runtime/gmr_inputs/pmx_to_g1.json \
        --retarget --render runtime/pmx/gmr_g1.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# robot link -> (PMX bone, PMX child bone used for the segment direction, robot child link)
# the child bones drive the "align" measurement and mirror what the player aligned.
LINKS = {
    'pelvis': ('腰', '上半身', 'torso_link'),
    'torso_link': ('上半身2', '首', 'head_link'),
    'left_hip_yaw_link': ('左足', '左ひざ', 'left_knee_link'),
    'left_knee_link': ('左ひざ', '左足首', 'left_ankle_roll_link'),
    'left_ankle_roll_link': ('左足首', '左つま先', 'left_toe_link'),
    'right_hip_yaw_link': ('右足', '右ひざ', 'right_knee_link'),
    'right_knee_link': ('右ひざ', '右足首', 'right_ankle_roll_link'),
    'right_ankle_roll_link': ('右足首', '右つま先', 'right_toe_link'),
    'left_shoulder_yaw_link': ('左腕', '左ひじ', 'left_elbow_link'),
    'left_elbow_link': ('左ひじ', '左手首', 'left_wrist_yaw_link'),
    'left_wrist_yaw_link': ('左手首', '左中指１', 'left_wrist_roll_link'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', 'right_elbow_link'),
    'right_elbow_link': ('右ひじ', '右手首', 'right_wrist_yaw_link'),
    'right_wrist_yaw_link': ('右手首', '右中指１', 'right_wrist_roll_link'),
}
# links whose parent is the torso in the robot but a different PMX bone in the model
ROBOT_PARENT = {
    'torso_link': 'pelvis',
    'left_hip_yaw_link': 'pelvis', 'right_hip_yaw_link': 'pelvis',
    'left_knee_link': 'left_hip_yaw_link', 'right_knee_link': 'right_hip_yaw_link',
    'left_ankle_roll_link': 'left_knee_link', 'right_ankle_roll_link': 'right_knee_link',
    'left_shoulder_yaw_link': 'torso_link', 'right_shoulder_yaw_link': 'torso_link',
    'left_elbow_link': 'left_shoulder_yaw_link', 'right_elbow_link': 'right_shoulder_yaw_link',
    'left_wrist_yaw_link': 'left_elbow_link', 'right_wrist_yaw_link': 'right_elbow_link',
}
# measurement pairs, chosen so both sides are the same anatomical segment; the torso is measured
# through pelvis->torso because this MJCF places head_link's origin below torso_link's, and the hand
# is measured through the forearm because the wrist links differ by a lateral roll axis.
MEASURE = {
    'pelvis': ('腰', '上半身', 'pelvis', 'torso_link'),
    'torso_link': ('腰', '上半身', 'pelvis', 'torso_link'),
    'left_hip_yaw_link': ('左足', '左ひざ', 'left_hip_yaw_link', 'left_knee_link'),
    'left_knee_link': ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link'),
    'left_ankle_roll_link': ('左足首', '左つま先', 'left_ankle_roll_link', 'left_toe_link'),
    'right_hip_yaw_link': ('右足', '右ひざ', 'right_hip_yaw_link', 'right_knee_link'),
    'right_knee_link': ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link'),
    'right_ankle_roll_link': ('右足首', '右つま先', 'right_ankle_roll_link', 'right_toe_link'),
    'left_shoulder_yaw_link': ('左腕', '左ひじ', 'left_shoulder_yaw_link', 'left_elbow_link'),
    'left_elbow_link': ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', 'right_shoulder_yaw_link', 'right_elbow_link'),
    'right_elbow_link': ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link'),
}
# kept for reference: the A-pose difference is largest at these joints, but the same mismatch
# exists at every link, so the alignment is applied everywhere now.
ALIGN_AT = {'left_shoulder_yaw_link', 'right_shoulder_yaw_link',
            'left_hip_yaw_link', 'right_hip_yaw_link'}


def rotation_between(u, v) -> R:
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    axis = np.cross(u, v)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(u @ v, -1.0, 1.0))
    if sine < 1e-9:
        return R.identity() if cosine > 0 else R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    return R.from_rotvec(np.arctan2(sine, cosine) * axis / sine)


def load_pmx_frames(xml_path: str, tree_path: str, qpos_path: str):
    """World pose of every PMX bone per frame -> GMR's human-data shape."""
    tree = json.loads(Path(tree_path).read_text(encoding='utf-8'))
    index_of = {entry['name']: entry['index'] for entry in tree['bones']}
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos = np.load(qpos_path)['qpos']
    frames = []
    for values in qpos:
        data.qpos[:] = values
        mujoco.mj_forward(model, data)
        frame = {}
        for name, index in index_of.items():
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'pmx_body_{index:03d}')
            if body < 0:
                continue
            rotation = R.from_matrix(np.array(data.xmat[body]).reshape(3, 3))
            q = rotation.as_quat()          # x, y, z, w
            frame[name] = [np.array(data.xpos[body], dtype=float),
                           np.array([q[3], q[0], q[1], q[2]])]
        frames.append(frame)
    return frames, float(np.load(qpos_path)['qpos'].shape[0]), index_of


def build_config(base: dict, robot_xml: str, pmx_xml: str, tree_path: str) -> tuple[dict, list]:
    """Fill the stock config's tables with PMX bone names and the derived constants."""
    robot = mujoco.MjModel.from_xml_path(robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)
    pmx = mujoco.MjModel.from_xml_path(pmx_xml)
    pmx_data = mujoco.MjData(pmx)
    mujoco.mj_forward(pmx, pmx_data)
    tree = json.loads(Path(tree_path).read_text(encoding='utf-8'))
    pmx_index = {entry['name']: entry['index'] for entry in tree['bones']}

    def pmx_body(name):
        index = pmx_index[name]
        return mujoco.mj_name2id(pmx, mujoco.mjtObj.mjOBJ_BODY, f'pmx_body_{index:03d}')

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    base['human_root_name'] = '腰'
    base['human_height_assumption'] = float(pmx_data.xpos[pmx_body('頭')][2])
    base['ground_height'] = 0.0
    scales = {'腰': 1.0}
    for link, (bone, child_bone, robot_child) in LINKS.items():
        if link == 'pelvis':
            continue
        robot_length = np.linalg.norm(robot_data.xpos[robot_body(robot_child)] -
                                      robot_data.xpos[robot_body(link)])
        pmx_length = np.linalg.norm(pmx_data.xpos[pmx_body(child_bone)] - pmx_data.xpos[pmx_body(bone)])
        if pmx_length > 1e-6:
            scales.setdefault(bone, float(robot_length / pmx_length))
    base['human_scale_table'] = scales

    report = []
    for table in ('ik_match_table1', 'ik_match_table2'):
        for link in list(base[table].keys()):
            if link not in LINKS:
                continue
            bone, child_bone, robot_child = LINKS[link]
            rest = R.from_matrix(np.array(robot_data.xmat[robot_body(link)]).reshape(3, 3))
            align = R.identity()
            # The alignment must use the *anatomically matching* segment on both sides. Falling back
            # to "this link -> its child link" is wrong for the torso: G1 places head_link's origin
            # below torso_link's, so that pair yields a ~180 deg rotation and folds the robot in half.
            # The spine chain must NOT be aligned. This PMX binds with its spine leaning ~35 deg
            # forward and the animation straightens it, so an align computed on the bind pose
            # injects a permanent 40 deg forward tilt into the pelvis - which drags the legs into a
            # sitting posture (observed). The limbs do need it: G1's elbow already sits 75.8 deg
            # bent at qpos 0 while the PMX elbow is straight.
            pair = None if link in ('pelvis', 'torso_link') else MEASURE.get(link)
            if pair is not None:
                pmx_a, pmx_b, robot_a, robot_b = pair
                if all(name in pmx_index for name in (pmx_a, pmx_b)):
                    g1_direction = (robot_data.xpos[robot_body(robot_b)] -
                                    robot_data.xpos[robot_body(robot_a)])
                    pmx_direction = (pmx_data.xpos[pmx_body(pmx_b)] -
                                     pmx_data.xpos[pmx_body(pmx_a)])
                    align = rotation_between(g1_direction, pmx_direction)
                    report.append((link, float(np.degrees(align.magnitude()))))
            constant = rest * align
            x, y, z, w = constant.as_quat()
            entry = base[table][link]
            entry[0] = bone
            entry[4] = [float(w), float(x), float(y), float(z)]
    return base, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx-xml', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--qpos', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--robot-xml', default=None)
    parser.add_argument('--base', default=None, help='stock config to take weights/scales from')
    parser.add_argument('--config-out', default=None)
    parser.add_argument('--retarget', action='store_true')
    parser.add_argument('--render', default=None)
    parser.add_argument('--frames', default='0,40,80,120,160,200')
    parser.add_argument('--bake-out', default=None, help='retarget every frame and save the qpos')
    parser.add_argument('--play', default=None, help='play a baked qpos npz in a viewer')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--max-iter', type=int, default=None, help='GMR IK iteration budget')
    parser.add_argument('--damping', type=float, default=None, help='GMR IK damping')
    parser.add_argument('--posture-cost', type=float, default=1e-2,
                        help='mink PostureTask cost; pins the IK null space down (0 disables)')
    parser.add_argument('--smooth', type=float, default=0.5,
                        help='one-pole low-pass coefficient applied to the baked qpos (0 = off)')
    args = parser.parse_args()

    gmroot = Path(__file__).resolve().parents[2] / '_gmr_path.txt'
    gmr_root = Path(json.loads(gmroot.read_text())) if gmroot.exists() else Path(r'D:\MY_TOOL\GitHub_Tools\GMR')
    import sys
    sys.path.insert(0, str(gmr_root))
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    robot_xml = args.robot_xml or str(
        gmr_root / 'assets' / 'unitree_g1' / 'g1_mocap_29dof.xml')
    base_path = args.base or str(
        gmr_root / 'general_motion_retargeting' / 'ik_configs' / 'bvh_lafan1_to_g1.json')
    base = json.loads(Path(base_path).read_text(encoding='utf-8'))

    config, report = build_config(base, robot_xml, args.pmx_xml, args.tree)
    print(f"{'link':<26}{'align':>8}")
    for link, angle in report:
        print(f'{link:<26}{angle:7.1f}')

    config_path = Path(args.config_out or 'runtime/gmr_inputs/pmx_to_g1.json').resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # GMR opens the config with the system default codec on Windows, so keep it pure ASCII
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print('config:', config_path)

    if not args.retarget:
        return

    gmrmod.IK_CONFIG_DICT.setdefault('pmx', {})[args.robot] = str(config_path)
    frames, _, _ = load_pmx_frames(args.pmx_xml, args.tree, args.qpos)
    retargeter = GMR(src_human='pmx', tgt_robot=args.robot, actual_human_height=1.5, verbose=False)

    robot = mujoco.MjModel.from_xml_path(robot_xml)
    data = mujoco.MjData(robot)

    if args.posture_cost and args.posture_cost > 0:
        # GMR's FrameTasks leave a null space (hip and knee can trade rotation with the shin barely
        # moving), and the solver wanders inside it: measured 7.7 deg of knee jitter while the input
        # was 0.96 deg. A posture term removes the ambiguity; it also improved accuracy here.
        import mink
        posture = mink.PostureTask(retargeter.model, cost=args.posture_cost)
        posture.set_target(retargeter.model.qpos0)
        retargeter.tasks1.append(posture)
        retargeter.tasks2.append(posture)

    if args.play:
        store = np.load(args.play)
        cache = list(store['qpos'])
        print('playing', args.play, len(cache), 'frames')
        play(robot, data, cache, args)
        return

    if args.max_iter is not None:
        retargeter.max_iter = args.max_iter
    if args.damping is not None:
        retargeter.damping = args.damping

    if args.bake_out:
        cache = []
        for index in range(len(frames)):
            cache.append(np.array(retargeter.retarget(frames[index])))
        if args.smooth > 0:
            alpha = float(args.smooth)
            filtered = [cache[0].copy()]
            for previous, current in zip(cache[:-1], cache[1:]):
                filtered.append(alpha * previous + (1.0 - alpha) * current)
            cache = filtered
        out = Path(args.bake_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, qpos=np.array(cache), times=np.arange(len(cache)) / 30.0)
        print('baked', len(cache), 'frames ->', out)
        if args.view:
            play(robot, data, cache, args)
        return

    wanted = [int(v) for v in args.frames.split(',') if int(v) < len(frames)]

    errors = {}
    for index in wanted:
        qpos = retargeter.retarget(frames[index])
        data.qpos[:] = qpos
        mujoco.mj_forward(robot, data)
        for link, (bone, child_bone, robot_a, robot_b) in MEASURE.items():
            if bone not in frames[index] or child_bone not in frames[index]:
                continue
            human = (frames[index][child_bone][0] - frames[index][bone][0])
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

    if args.render:
        import PIL.Image as Image
        renderer = mujoco.Renderer(robot, 460, 400)
        tiles = []
        for index in wanted[:4]:
            data.qpos[:] = retargeter.retarget(frames[index])
            mujoco.mj_forward(robot, data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, 0.85]
            camera.distance = 3.0
            camera.elevation = -6
            camera.azimuth = 120
            renderer.update_scene(data, camera)
            tiles.append(Image.fromarray(renderer.render()))
        width, height = tiles[0].size
        sheet = Image.new('RGB', (width * len(tiles), height), 'white')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (i * width, 0))
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.render)
        print('render:', args.render)


def play(robot, data, cache, args):
    import time
    import mujoco.viewer
    with mujoco.viewer.launch_passive(robot, data) as viewer:
        index = 0
        while viewer.is_running():
            started = time.perf_counter()
            data.qpos[:] = cache[index % len(cache)]
            mujoco.mj_forward(robot, data)
            viewer.sync()
            index += 1
            time.sleep(max(0.0, 1.0 / 30.0 / args.speed - (time.perf_counter() - started)))


if __name__ == '__main__':
    main()
