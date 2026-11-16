"""Derive the per-link offsets C from a paired pose, and write the IK config.

This file **consumes** the joint angles produced by ``gmr_paired_pose`` - it never solves a pose itself.
That is the whole contract: one file generates the robot pose, this one turns a robot pose into C.

    C = R_source(P)^-1 · R_link(P)

Why that is a reading and not a fit: an MMD rig carries no rest rotation, so at any pose the bone frame
is exactly what the animation says. Two links - the pelvis (floating root) and the torso - are not moved
by a limb pose, so their world-frame yaw comes from a *bilateral* axis (hip line, shoulder line) instead;
without it the arms are perfect and the body stays twisted.

The config skeleton (bone names, weights, the uniform scale factor) is built here too, from the **bind**
pose, so the scale never depends on which pose was used for calibration. Both are written only together:
a config with identity offsets is never left on disk.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/gmr_calibrate.py --pmx <model.pmx> \
        --g1-xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml \
        --base <gmr>/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
        --pose configs/mmd/paired_pose_pmx.json --out configs/mmd/pmx_to_g1.json \
        --record configs/mmd/calibration_pmx.json --report
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmd_pose                                        # noqa: E402
from mmd_bones import LINKS                            # noqa: E402

PELVIS_ROT_WEIGHT = 10.0
OFFICIAL_ARM_WEIGHT = {'left_shoulder_yaw_link': 100.0, 'right_shoulder_yaw_link': 100.0}
ARMS = ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
        'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link')


def _rotation(quaternion):
    """mmd_pose quaternions are (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quaternion)
    return R.from_quat([x, y, z, w])


def build_skeleton(base_path, g1_xml, bind_frame):
    """Bone names, weights and the uniform scale factor. Orientation offsets stay at identity here."""
    config = json.loads(Path(base_path).read_text(encoding='utf-8'))

    if '頭' not in bind_frame:
        raise SystemExit('the source rig has no 頭 bone; the scale factor needs a head')

    robot = mujoco.MjModel.from_xml_path(str(g1_xml))
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)
    robot_height = float(data.xpos[:, 2].max())
    source_height = float(bind_frame['頭'][0][2])
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
    parser.add_argument('--pose', required=True,
                        help='paired pose (joint angles in degrees) from gmr_paired_pose')
    parser.add_argument('--out', required=True, help='the calibrated config to write')
    parser.add_argument('--vpd', default=None,
                        help='MMD pose the paired pose was solved for; omitted = the bind pose')
    parser.add_argument('--arm-pos-weight', type=float, default=0.0)
    parser.add_argument('--record', default=None)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    # --- source pose (no MuJoCo) ---------------------------------------------------------------
    wanted = sorted(set(LINKS.values()) | {'腰', '上半身2', '頭'})
    if args.vpd:
        frames, _ = mmd_pose.load(args.pmx, args.vpd, wanted)
        source = frames[0]
        note = 'vpd: %s' % Path(args.vpd).name
    else:
        source = mmd_pose.bind_pose(args.pmx, wanted)[0]
        note = 'bind pose'
    bind_frame = mmd_pose.bind_pose(args.pmx, wanted)[0]

    def source_position(name):
        return np.asarray(source[name][0])

    # --- skeleton from the *bind* pose, so the scale never depends on the calibration pose -----
    config, factor, source_height, robot_height = build_skeleton(args.base, args.g1_xml, bind_frame)

    # --- the robot pose is data, read from the pose file ---------------------------------------
    supplied = json.loads(Path(args.pose).read_text(encoding='utf-8'))
    angles = supplied.get('joints', supplied)

    robot = mujoco.MjModel.from_xml_path(args.g1_xml)
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)

    def body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    unknown = [name for name in angles
               if mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name) < 0]
    if unknown:
        raise SystemExit('pose file names joints this robot does not have: %s' % unknown)

    data.qpos[:] = robot.qpos0
    for name, value in angles.items():
        joint = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(robot.jnt_qposadr[joint])] = np.radians(value)
    mujoco.mj_forward(robot, data)

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

    for link in ARMS:
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
        existing['why'] = ('MMD bones carry no rest rotation, so at the paired pose every bone frame is '
                           'the identity and C collapses to a reading')
        existing['source_bind_note'] = ('source world poses come from the vendored deform pipeline; '
                                        'no qpos round trip, the root translation is included')
        existing['paired_pose'] = {'robot_xml': Path(args.g1_xml).name, 'pose_file': str(args.pose),
                                   'source_vpd': args.vpd}
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
        print('source: %s   pose: %s' % (note, Path(args.pose).name))
        print('scale %.4f   source height %.4f m   robot height %.4f m'
              % (factor, source_height, robot_height))
        print('root yaw: %.2f deg' % np.degrees(delta))
        print('%-26s %9s' % ('link', 'C (deg)'))
        for link, angle in report:
            print('%-26s %8.1f' % (link, angle))
    print('written', out_path)


if __name__ == '__main__':
    main()
