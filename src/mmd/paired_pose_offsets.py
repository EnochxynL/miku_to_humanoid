"""Derive GMR's per-link offsets from a *configured* pair of poses.

The paired pose is data, the offsets are derived:

    source pose   --vpd <file>    an MMD pose file; omitted = the model's bind pose
    robot pose    --pose <json>   joint angles in degrees; robot-swappable, nothing else changes

Given the two poses the script only reads C off the model. When no robot pose file is supplied the
joint angles are solved against the source pose's limb directions and can be written out with
`--write-pose`, so calibrating a new robot is a one-off step whose result becomes configuration.

Why a paired pose at all: an MMD rig carries no rest rotation, so at the bind pose every bone frame is
the identity, and

    target(t) = R_source(t) · C,  at the paired pose  =>  C = R_source(pose)^-1 · R_link(pose)

which turns C from a fit into a reading. Two links - the pelvis (floating root) and the torso - are not
moved by the pose, so their world-frame yaw has to come from a *bilateral* axis (hip line, shoulder
line); without it the arms are perfect and the body stays twisted.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/paired_pose_offsets.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --config configs/mmd/glb_to_g1.json --pose configs/mmd/paired_pose_g1.json \
        --record configs/mmd/calibration.json --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmd_bones import LINKS   # noqa: E402

# limb (source bone, source child, robot link, robot child body) and the robot joints that move it
LIMBS = [
    ('左腕', '左ひじ', 'left_shoulder_yaw_link', 'left_elbow_link',
     ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint']),
    ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link', ['left_elbow_joint']),
    ('右腕', '右ひじ', 'right_shoulder_yaw_link', 'right_elbow_link',
     ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint']),
    ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link', ['right_elbow_joint']),
    ('左足', '左ひざ', 'left_hip_yaw_link', 'left_knee_link',
     ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint']),
    ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link', ['left_knee_joint']),
    ('右足', '右ひざ', 'right_hip_yaw_link', 'right_knee_link',
     ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']),
    ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link', ['right_knee_joint']),
]

BONE_LINE = re.compile(r'^Bone\d+\{(.*)$')


def load_vpd(path: Path):
    """Bone -> (position, quaternion x,y,z,w) from an MMD .vpd pose file (Shift-JIS)."""
    text = path.read_bytes().decode('cp932', errors='replace')
    lines = text.splitlines()
    poses = {}
    index = 0
    while index < len(lines):
        match = BONE_LINE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        name = match.group(1).strip()
        values = []
        probe = index + 1
        while probe < len(lines) and len(values) < 7:
            line = lines[probe].strip()
            if line and not line.startswith('}') and not line.startswith('Bone'):
                body = line.split('//')[0].strip().replace(';', ',')
                values.extend(float(part) for part in body.split(',') if part.strip())
            probe += 1
        if len(values) >= 7:
            poses[name] = (np.array(values[:3]), np.array(values[3:7]))
        index = probe
    return poses


def apply_vpd(model, data, mapping, poses):
    """Apply MMD local rotations to the glTF-derived model: q_joint = inverse(bind) * vpd_local."""
    applied = 0
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if not name or name not in poses:
            continue
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            continue
        _, quat = poses[name]
        local = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        w, x, y, z = model.body_quat[body]
        bind = R.from_quat([x, y, z, w])
        rotation = bind.inv() * local
        address = int(model.jnt_qposadr[joint])
        kind = model.jnt_type[joint]
        if kind == mujoco.mjtJoint.mjJNT_BALL:
            q = rotation.as_quat()
            data.qpos[address:address + 4] = [q[3], q[0], q[1], q[2]]
        elif kind == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[address] = float(rotation.as_rotvec() @ model.jnt_axis[joint])
        applied += 1
    return applied


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', default=None, help='unused: bodies are looked up by bone name')
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--vpd', default=None,
                        help='MMD pose file for the source side; omit to use the bind pose')
    parser.add_argument('--pose', default=None,
                        help='robot joint angles in degrees (JSON); omit to solve them')
    parser.add_argument('--write-pose', default=None,
                        help='save the robot pose here so a calibration becomes configuration')
    parser.add_argument('--arm-pos-weight', type=float, default=0.0,
                        help="0 keeps GMR's own convention: orientation only for the arms")
    parser.add_argument('--record', default=None,
                        help='audit record (method + per-link C + root yaw + paired pose)')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    # --- source ------------------------------------------------------------------------------
    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    src_data.qpos[:] = src.qpos0
    mapping = (json.loads(Path(args.nodes).read_text(encoding='utf-8'))
               if args.nodes else {'nodes': []})
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}
    if args.vpd:
        poses = load_vpd(Path(args.vpd))
        applied = apply_vpd(src, src_data, mapping, poses)
        print(f'vpd: {Path(args.vpd).name} -> {applied} bones applied')
    mujoco.mj_forward(src, src_data)

    def source_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, name)

    # --- robot -------------------------------------------------------------------------------
    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    def robot_body(name):
        return mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, name)

    joint_address, order = {}, []
    for _, _, _, _, joints in LIMBS:
        for name in joints:
            jid = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0 and name not in joint_address:
                joint_address[name] = (int(robot.jnt_qposadr[jid]), robot.jnt_range[jid])
                order.append(name)

    def set_pose(degrees):
        robot_data.qpos[:] = robot.qpos0
        for name in order:
            robot_data.qpos[joint_address[name][0]] = np.radians(degrees[name])
        mujoco.mj_forward(robot, robot_data)

    targets = []
    for src_a, src_b, _ra, _rb, _joints in LIMBS:
        direction = src_data.xpos[source_body(src_b)] - src_data.xpos[source_body(src_a)]
        targets.append(direction / np.linalg.norm(direction))

    def direction_errors():
        values = []
        for target, (_a, _b, robot_a, robot_b, _joints) in zip(targets, LIMBS):
            got = robot_data.xpos[robot_body(robot_b)] - robot_data.xpos[robot_body(robot_a)]
            got /= np.linalg.norm(got)
            values.append(np.degrees(np.arccos(np.clip(got @ target, -1.0, 1.0))))
        return values

    if args.pose:
        supplied = json.loads(Path(args.pose).read_text(encoding='utf-8'))
        angles = supplied.get('joints', supplied)
        missing = [name for name in order if name not in angles]
        if missing:
            raise SystemExit(f'pose file is missing joints: {missing}')
        set_pose(angles)
        source = 'config: %s' % Path(args.pose).name
        solve_cost = None
    else:
        def cost(values):
            robot_data.qpos[:] = robot.qpos0
            for name, value in zip(order, values):
                robot_data.qpos[joint_address[name][0]] = value
            mujoco.mj_forward(robot, robot_data)
            return float(np.sum(np.square(direction_errors())))

        result = minimize(cost, np.zeros(len(order)), method='L-BFGS-B',
                          bounds=[tuple(joint_address[n][1]) for n in order],
                          options={'maxiter': 2000, 'ftol': 1e-12})
        robot_data.qpos[:] = robot.qpos0
        for name, value in zip(order, result.x):
            robot_data.qpos[joint_address[name][0]] = value
        mujoco.mj_forward(robot, robot_data)
        angles = {name: float(np.degrees(value)) for name, value in zip(order, result.x)}
        source = 'solved'
        solve_cost = float(result.fun)

    errors = direction_errors()
    mean_error = float(np.mean(errors))

    if args.write_pose:
        payload = {
            'robot_xml': args.robot_xml,
            'note': ('paired pose for the C calibration: joint angles that put this robot into the '
                     'source pose (MMD bind pose unless --vpd was given)'),
            'source_pose': args.vpd or '(bind pose)',
            'joints': {name: round(float(value), 4) for name, value in angles.items()},
            'limb_direction_error_deg': round(mean_error, 3),
        }
        out = Path(args.write_pose)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print('pose written:', out)

    # --- the two links the pose does not move need a bilateral axis --------------------------
    def bilateral(model_data, left, right):
        vector = model_data.xpos[right] - model_data.xpos[left]
        return vector / np.linalg.norm(vector)

    robot_hip = bilateral(robot_data, robot_body('left_hip_yaw_link'),
                          robot_body('right_hip_yaw_link'))
    source_hip = (src_data.xpos[source_body('右足')] - src_data.xpos[source_body('左足')])
    source_hip /= np.linalg.norm(source_hip)
    delta = np.arctan2(np.cross(robot_hip, source_hip)[2], robot_hip @ source_hip)
    root_fix = R.from_euler('z', np.degrees(delta), degrees=True)

    # --- C = R_source(pose)^-1 * R_link(pose) -------------------------------------------------
    bone_for_link = dict(LINKS)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    report = []
    for link in list(config['ik_match_table1'].keys()):
        if link not in LINKS:
            continue
        body = robot_body(link)
        if body < 0:
            continue
        source_pose = R.identity()
        if bone_for_link.get(link) and source_body(bone_for_link[link]) >= 0:
            source_pose = R.from_matrix(np.array(
                src_data.xmat[source_body(bone_for_link[link])]).reshape(3, 3))
        constant = source_pose.inv() * R.from_matrix(np.array(robot_data.xmat[body]).reshape(3, 3))
        if link in ('pelvis', 'torso_link'):
            constant = constant * root_fix
        q = constant.as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        report.append((link, float(np.degrees(constant.magnitude()))))

    official = {'left_shoulder_yaw_link': 100.0, 'right_shoulder_yaw_link': 100.0}
    for link in ('left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
                 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link'):
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][1] = float(args.arm_pos_weight)
                config[table][link][2] = (1.0 if args.arm_pos_weight > 0
                                          else official.get(link, 10.0))
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')

    if args.record:
        record_path = Path(args.record)
        existing = {}
        if record_path.exists():
            try:
                existing = json.loads(record_path.read_text(encoding='utf-8'))
            except Exception:
                existing = {}
        existing['method'] = ('paired pose (configured, not fitted) -> '
                              'C = R_source(pose)^-1 * R_link(pose)')
        existing['why'] = ('MMD bones carry no rest rotation, so at the bind pose every bone frame is '
                           'the identity and C collapses to a reading')
        existing['source_bind_note'] = ('the GLB hangs under Rx90, so R_source(pose) carries it and it '
                                        'must be divided out')
        existing['paired_pose'] = {'robot_xml': Path(args.robot_xml).name,
                                   'pose_file': args.pose,
                                   'source_vpd': args.vpd,
                                   'limb_direction_error_deg': round(mean_error, 3)}
        existing['root_calibration'] = {
            'applies_to': ['pelvis', 'torso_link'],
            'value_deg': round(float(np.degrees(delta)), 2),
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
        print('source:', source, '' if solve_cost is None else f'(solve cost {solve_cost:.4f})')
        print('%-8s %10s' % ('limb', 'dir err'))
        for (src_a, _b, _c, _d, _j), err in zip(LIMBS, errors):
            print('%-8s %9.1f' % (src_a, err))
        print('root yaw: %.1f deg' % np.degrees(delta))
        print(f"{'link':<26}{'C (deg)':>9}")
        for link, angle in report:
            print(f'{link:<26}{angle:8.1f}')
    print('written', config_path)


if __name__ == '__main__':
    main()
