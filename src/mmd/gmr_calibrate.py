"""Calibrate a PMX rig against a robot: build the IK config and derive its per-link offsets C.

There is one command because there is one valid output. Building the config skeleton alone leaves an
identity orientation offset on every link, which looks like a usable config and bakes a 159 deg wrong
motion - that mistake happened once. So the skeleton is built, calibrated, and only then written.

    source pose   --vpd <vmd>   an MMD pose; omitted = the rig's bind pose
    robot pose    --pose <json> joint angles in degrees; omitted = solve them, then --write-pose saves
                                the result so the next run is a pure read

Why a paired pose: an MMD rig carries no rest rotation, so at the bind pose every bone frame is the
identity and

    target(t) = R_source(t) · C,  at the paired pose  =>  C = R_source(P)^-1 · R_link(P)

which turns C from a fit into a reading. Two links - the pelvis (floating root) and the torso - are not
moved by the pose, so their world-frame yaw has to come from a *bilateral* axis (hip line, shoulder
line); without it the arms are perfect and the body stays twisted.

Source world poses come from ``mmd_pose`` (the vendored deform pipeline), so there is no qpos round
trip and the ``センター`` translation is included.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/gmr_calibrate.py --pmx <model.pmx> \
        --g1-xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml \
        --base <gmr>/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
        --out configs/mmd/pmx_to_g1.json --pose configs/mmd/paired_pose_pmx.json \
        --record configs/mmd/calibration_pmx.json --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmd_pose                                        # noqa: E402
from mmd_bones import LINKS, limb_segments             # noqa: E402

LIMBS = limb_segments()

PELVIS_ROT_WEIGHT = 10.0
OFFICIAL_ARM_WEIGHT = {'left_shoulder_yaw_link': 100.0, 'right_shoulder_yaw_link': 100.0}


def _rotation(quaternion):
    """mmd_pose quaternions are (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quaternion)
    return R.from_quat([x, y, z, w])


def build_skeleton(base_path, g1_xml, source_frame):
    """Bone names, weights and the uniform scale factor. Orientation offsets stay at identity here."""
    config = json.loads(Path(base_path).read_text(encoding='utf-8'))

    if '頭' not in source_frame:
        raise SystemExit('the source rig has no 頭 bone; the scale factor needs a head')

    robot = mujoco.MjModel.from_xml_path(str(g1_xml))
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)
    robot_height = float(data.xpos[:, 2].max())
    source_height = float(source_frame['頭'][0][2])
    factor = robot_height / source_height if source_height > 1e-6 else 0.9

    bones = sorted(set(LINKS.values()) | {'腰', '上半身2'})
    config['human_root_name'] = '腰'
    config['human_height_assumption'] = source_height
    config['ground_height'] = 0.0
    # human_scale_table is a *uniform body* scale, not a per-segment length ratio: GMR computes
    # (position - root) * scale[bone]
    config['human_scale_table'] = {bone: float(factor) for bone in bones}

    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in config.get(table, {}).items():
            if link in LINKS:
                entry[0] = LINKS[link]
                entry[4] = [1.0, 0.0, 0.0, 0.0]
        if 'pelvis' in config.get(table, {}):
            config[table]['pelvis'][2] = PELVIS_ROT_WEIGHT
    return config, factor, source_height, robot_height


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--g1-xml', required=True)
    parser.add_argument('--base', required=True,
                        help="template config: GMR's stock one, or an existing config to re-calibrate")
    parser.add_argument('--out', required=True, help='the calibrated config to write')
    parser.add_argument('--vpd', default=None,
                        help='MMD pose file for the source side; omit = the bind pose')
    parser.add_argument('--pose', default=None,
                        help='robot joint angles in degrees; omit to solve them')
    parser.add_argument('--write-pose', default=None,
                        help='save the robot pose here so the next run is a pure read')
    parser.add_argument('--arm-pos-weight', type=float, default=0.0)
    parser.add_argument('--record', default=None)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    # --- source pose (no MuJoCo) ---------------------------------------------------------------
    wanted = sorted({b for b, *_ in LIMBS} | set(LINKS.values()) | {'腰', '上半身2', '頭'})
    if args.vpd:
        frames, _ = mmd_pose.load(args.pmx, args.vpd, wanted)
        source = frames[0]
        note = 'vpd: %s' % Path(args.vpd).name
    else:
        source = mmd_pose.bind_pose(args.pmx, wanted)[0]
        note = 'bind pose'

    def source_position(name):
        return np.asarray(source[name][0])

    # --- skeleton, calibrated in the same run --------------------------------------------------
    config, factor, source_height, robot_height = build_skeleton(args.base, args.g1_xml, source)

    robot = mujoco.MjModel.from_xml_path(args.g1_xml)
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)

    def body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    addresses, order = {}, []
    for _, _, _, _, joints in LIMBS:
        for name in joints:
            jid = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0 and name not in addresses:
                addresses[name] = (int(robot.jnt_qposadr[jid]), robot.jnt_range[jid])
                order.append(name)

    def set_pose(degrees):
        data.qpos[:] = robot.qpos0
        for name in order:
            data.qpos[addresses[name][0]] = np.radians(degrees[name])
        mujoco.mj_forward(robot, data)

    targets = []
    for src_a, src_b, _ra, _rb, _joints in LIMBS:
        direction = source_position(src_b) - source_position(src_a)
        targets.append(direction / np.linalg.norm(direction))

    def direction_errors():
        values = []
        for target, (_a, _b, robot_a, robot_b, _j) in zip(targets, LIMBS):
            got = np.array(data.xpos[body(robot_b)]) - np.array(data.xpos[body(robot_a)])
            got /= np.linalg.norm(got)
            values.append(np.degrees(np.arccos(np.clip(got @ target, -1.0, 1.0))))
        return values

    if args.pose:
        supplied = json.loads(Path(args.pose).read_text(encoding='utf-8'))
        angles = supplied.get('joints', supplied)
        missing = [n for n in order if n not in angles]
        if missing:
            raise SystemExit('pose file is missing joints: %s' % missing)
        set_pose(angles)
        solve_cost = None
    else:
        def cost(values):
            data.qpos[:] = robot.qpos0
            for name, value in zip(order, values):
                data.qpos[addresses[name][0]] = value
            mujoco.mj_forward(robot, data)
            return float(np.sum(np.square(direction_errors())))

        result = minimize(cost, np.zeros(len(order)), method='L-BFGS-B',
                          bounds=[tuple(addresses[n][1]) for n in order],
                          options={'maxiter': 2000, 'ftol': 1e-12})
        data.qpos[:] = robot.qpos0
        for name, value in zip(order, result.x):
            data.qpos[addresses[name][0]] = value
        mujoco.mj_forward(robot, data)
        angles = {n: float(np.degrees(v)) for n, v in zip(order, result.x)}
        solve_cost = float(result.fun)

    errors = direction_errors()
    mean_error = float(np.mean(errors))

    if args.write_pose:
        payload = {
            'robot_xml': Path(args.g1_xml).name,
            'note': ('paired pose for the C calibration: joint angles that put this robot into the '
                     'source pose (bind pose unless --vpd was given)'),
            'source_pose': args.vpd or '(bind pose)',
            'joints': {n: round(float(v), 4) for n, v in angles.items()},
            'limb_direction_error_deg': round(mean_error, 3),
        }
        out = Path(args.write_pose)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print('pose written:', out)

    # --- the two links the pose does not move need a bilateral axis -----------------------------
    robot_hip = (np.array(data.xpos[body('right_hip_yaw_link')])
                 - np.array(data.xpos[body('left_hip_yaw_link')]))
    robot_hip /= np.linalg.norm(robot_hip)
    source_hip = source_position('右足') - source_position('左足')
    source_hip /= np.linalg.norm(source_hip)
    delta = float(np.arctan2(np.cross(robot_hip, source_hip)[2], robot_hip @ source_hip))
    root_fix = R.from_euler('z', np.degrees(delta), degrees=True)

    # --- C = R_source(P)^-1 · R_link(P) --------------------------------------------------------
    report = []
    for link in list(config['ik_match_table1'].keys()):
        if link not in LINKS:
            continue
        bone = LINKS[link]
        if bone not in source:
            continue
        robot_body = body(link)
        if robot_body < 0:
            continue
        constant = _rotation(source[bone][1]).inv() * R.from_matrix(
            np.array(data.xmat[robot_body]).reshape(3, 3))
        if link in ('pelvis', 'torso_link'):
            constant = constant * root_fix
        q = constant.as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        report.append((link, float(np.degrees(constant.magnitude()))))

    for link in ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
                 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link'):
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][1] = float(args.arm_pos_weight)
                config[table][link][2] = (1.0 if args.arm_pos_weight > 0
                                          else OFFICIAL_ARM_WEIGHT.get(link, 10.0))

    # written only now: a config with identity offsets never reaches the disk
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')

    if args.record:
        record_path = Path(args.record)
        existing = {}
        if record_path.exists():
            try:
                existing = json.loads(record_path.read_text(encoding='utf-8'))
            except Exception:
                existing = {}
        existing['method'] = 'paired pose (configured, not fitted) -> C = R_source(P)^-1 * R_link(P)'
        existing['why'] = ('MMD bones carry no rest rotation, so at the bind pose every bone frame is '
                           'the identity and C collapses to a reading')
        existing['source_bind_note'] = ('source world poses come from the vendored deform pipeline; '
                                        'no qpos round trip, the root translation is included')
        existing['paired_pose'] = {'robot_xml': Path(args.g1_xml).name, 'pose_file': args.pose,
                                   'source_vpd': args.vpd,
                                   'limb_direction_error_deg': round(mean_error, 3)}
        existing['scale_factor'] = round(factor, 6)
        existing['heights'] = {'source': round(source_height, 4), 'robot': round(robot_height, 4)}
        existing['root_calibration'] = {
            'applies_to': ['pelvis', 'torso_link'],
            'value_deg': round(np.degrees(delta), 2),
            'rule': ('links the pose does not move need the world-frame yaw, measured on bilateral '
                     'axes (hip line, shoulder line); limbs get it implicitly'),
        }
        existing['links'] = {link: {'bone': config['ik_match_table1'][link][0],
                                    'pos_w': config['ik_match_table1'][link][1],
                                    'rot_w': config['ik_match_table1'][link][2],
                                    'C_quat_wxyz': [round(x, 6) for x in config['ik_match_table1'][link][4]]}
                             for link in config['ik_match_table1'] if link in LINKS}
        record_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding='utf-8')
        print('record:', record_path)

    if args.report:
        print('source: %s%s' % (note, '' if solve_cost is None else '   solve cost %.6f' % solve_cost))
        print('scale %.4f   source height %.4f m   robot height %.4f m'
              % (factor, source_height, robot_height))
        print('%-8s %10s' % ('limb', 'dir err'))
        for (src_a, _b, _c, _d, _j), err in zip(LIMBS, errors):
            print('%-8s %9.2f' % (src_a, err))
        print('root yaw: %.2f deg' % np.degrees(delta))
        print('%-26s %9s' % ('link', 'C (deg)'))
        for link, angle in report:
            print('%-26s %8.1f' % (link, angle))
    print('written', out_path)


if __name__ == '__main__':
    main()
