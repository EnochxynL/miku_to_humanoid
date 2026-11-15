"""Play the retargeting live with the human skeleton drawn next to the robot.

GMR's RobotMotionViewer can render the source skeleton as tiny frames at each bone, so the mismatch
between the two is visible directly instead of inferred from joint angles:

    robot_motion_viewer.step(root_pos, root_rot, dof_pos, human_motion_data=scaled_human_data)

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/view_retarget.py \
        --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json \
        --anim runtime/glb/herta_anim.npz --config runtime/gmr_inputs/glb_to_g1.json \
        --loop --speed 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glb_to_gmr import load_glb_frames   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb-xml', required=True)
    parser.add_argument('--nodes', required=True)
    parser.add_argument('--anim', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--human-scale', type=float, default=0.35,
                        help='size of the human bone frames drawn next to the robot')
    parser.add_argument('--human-offset', type=float, default=0.7,
                        help='shift the human skeleton sideways (metres) so it does not overlap the robot')
    parser.add_argument('--cross-start', action='store_true',
                        help='cold start every frame instead of warm starting')
    args = parser.parse_args()

    sys.path.insert(0, args.gmr_root)
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer

    gmrmod.IK_CONFIG_DICT.setdefault('glb', {})[args.robot] = str(Path(args.config).resolve())
    frames = load_glb_frames(args.glb_xml, args.nodes, args.anim)
    print(f'{len(frames)} frames')

    retargeter = GMR(src_human='glb', tgt_robot=args.robot,
                     actual_human_height=1.445, verbose=False)
    viewer = RobotMotionViewer(robot_type=args.robot, motion_fps=30)
    rest = np.array(retargeter.model.qpos0)

    while True:
        for index in range(args.start, len(frames)):
            if args.cross_start:
                retargeter.configuration.update(rest)
            qpos = retargeter.retarget(frames[index])
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_point_scale=args.human_scale,
                human_pos_offset=np.array([args.human_offset, 0.0, 0.0]),
                show_human_body_name=False,
            )
            if not viewer.viewer.is_running():
                viewer.close()
                return
            time.sleep(max(0.0, (1.0 / 30.0) / args.speed - 1.0 / 30.0))
        if not args.loop:
            break
        args.start = 0
    viewer.close()


if __name__ == '__main__':
    main()
