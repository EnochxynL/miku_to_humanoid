"""Pick each limb link's roll by making the child hinge plane contain the source's next segment.

The distal-axis constant fixes a limb's *direction* but leaves its roll free. For a link whose child
joint is a hinge that freedom is not harmless: the child can only bend inside the plane perpendicular
to the hinge axis, so if that plane does not contain the source's next segment the child can never
reach it - which is why the forearm stayed 60-90 degrees off even with the upper arm at 1-2 degrees.

The condition is a single scalar per frame:

    h_world(t) . d_src(t)  ~=  0      h_world = R_src(t) · C(phi) · h_local
                                      d_src   = source's next segment direction (world)
                                      h_local = hinge axis in the link's rest frame

so phi is found by a one-dimensional sweep. No IK is run, and no axis sign convention is assumed:
flipping h_local only relabels phi, and the sweep covers the whole circle anyway.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/fit_link_roll.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config runtime/gmr_inputs/glb_to_g1.json
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

# link -> (source bone, source child, source grandchild, robot child body, robot hinge joint)
# the link's own child joint is the hinge whose plane has to contain (child -> grandchild)
SPECS = {
    'left_shoulder_yaw_link': ('左腕', '左ひじ', '左手首', 'left_elbow_link', 'left_elbow_joint'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', '右手首', 'right_elbow_link', 'right_elbow_joint'),
    'left_elbow_link': ('左ひじ', '左手首', '左中指１', 'left_wrist_yaw_link',
                        'left_wrist_pitch_joint'),
    'right_elbow_link': ('右ひじ', '右手首', '右中指１', 'right_wrist_yaw_link',
                         'right_wrist_pitch_joint'),
    'left_hip_yaw_link': ('左足', '左ひざ', '左足首', 'left_knee_link', 'left_knee_joint'),
    'right_hip_yaw_link': ('右足', '右ひざ', '右足首', 'right_knee_link', 'right_knee_joint'),
    'left_knee_link': ('左ひざ', '左足首', '左つま先', 'left_ankle_roll_link',
                       'left_ankle_pitch_joint'),
    'right_knee_link': ('右ひざ', '右足首', '右つま先', 'right_ankle_roll_link',
                        'right_ankle_pitch_joint'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot-xml',
                        default=r'D:\MY_TOOL\GitHub_Tools\GMR\assets\unitree_g1\g1_mocap_29dof.xml')
    parser.add_argument('--step', type=float, default=5.0)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    src = mujoco.MjModel.from_xml_path(args.glb_xml)
    src_data = mujoco.MjData(src)
    mapping = json.loads(Path(args.nodes).read_text(encoding='utf-8'))
    index_of = {e['name']: e['index'] for e in mapping['nodes'] if e.get('name')}
    anim = np.load(args.anim)['qpos']
    sample = list(range(0, len(anim), 6))

    def source_body(name):
        return mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_BODY, f'glb_body_{index_of[name]:03d}')

    robot = mujoco.MjModel.from_xml_path(args.robot_xml)
    robot_data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, robot_data)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))

    print(f"{'link':<26}{'best phi':>10}{'cost before':>13}{'cost after':>12}")
    for link, (bone, child, grandchild, robot_child, joint) in SPECS.items():
        if bone not in index_of or child not in index_of or grandchild not in index_of:
            continue
        entry = config['ik_match_table1'].get(link)
        if entry is None:
            continue
        link_body = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, link)
        link_child = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, robot_child)
        joint_id = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if link_body < 0 or link_child < 0 or joint_id < 0:
            continue
        rot_link = np.array(robot_data.xmat[link_body]).reshape(3, 3)
        limb_local = rot_link.T @ (robot_data.xpos[link_child] - robot_data.xpos[link_body])
        limb_local /= np.linalg.norm(limb_local)
        hinge_local = rot_link.T @ np.array(robot_data.xaxis[joint_id])

        # source world rotations of this bone and the direction of its next segment
        orientations, directions = [], []
        for frame in anim[sample]:
            src_data.qpos[:] = frame
            mujoco.mj_forward(src, src_data)
            orientations.append(R.from_matrix(
                np.array(src_data.xmat[source_body(bone)]).reshape(3, 3)))
            d = (src_data.xpos[source_body(grandchild)] - src_data.xpos[source_body(child)])
            directions.append(d / np.linalg.norm(d))
        orientations = np.array([o.as_matrix() for o in orientations])

        w, x, y, z = entry[4]
        base = R.from_quat([x, y, z, w])

        def cost(phi):
            twist = R.from_rotvec(np.radians(phi) * limb_local)
            constant = (base * twist).as_matrix()
            hinge = np.einsum('tij,j->ti', orientations, constant @ hinge_local)
            proj = np.einsum('ti,ti->t', hinge, np.array(directions))
            return float(np.sqrt(np.mean(proj ** 2)))

        grid = np.arange(-180.0, 180.0, args.step)
        costs = [cost(phi) for phi in grid]
        best = float(grid[int(np.argmin(costs))])
        before = cost(0.0)
        after = cost(best)
        twist = R.from_rotvec(np.radians(best) * limb_local)
        q = (base * twist).as_quat()
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in config.get(table, {}):
                config[table][link][4] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        if args.report:
            print(f'{link:<26}{best:9.0f}{before:13.3f}{after:12.3f}')

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding='utf-8')
    print('written', config_path)


if __name__ == '__main__':
    main()
