"""Retarget an MMD motion onto the robot with GMR's IK.

Inputs are data, not geometry: the calibrated config (which carries C) and the source's world poses
from ``world_poses``. No MJCF of the source rig is involved - the vendored deform pipeline already gives
world poses, including the ``センター`` translation that a qpos round trip used to drop.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/gmr_retarget.py \
        --pmx <model.pmx> --vmd <motion.vmd> --config configs/generated/mmd_to_g1.json \
        --bake-out runtime/pmx/g1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                          # noqa: E402

from engine.bake import wanted_bones, world_poses      # noqa: E402


def load_frames(pmx_path, pose_path, config_path, step=1):
    """GMR human_data frames: {bone name: [position(3), quaternion(w, x, y, z)]}."""
    wanted = wanted_bones(config_path)
    frames, unmatched = world_poses(pmx_path, pose_path, wanted, step=step)
    return frames, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vmd', default=None, help='motion; omit for the bind pose')
    parser.add_argument('--config', required=True, help='the calibrated config (carries C)')
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--bake-out', required=True)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    gmr_root = Path(args.gmr_root)
    sys.path.insert(0, str(gmr_root))
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    config = json.loads(config_path.read_text(encoding='utf-8'))
    gmrmod.IK_CONFIG_DICT.setdefault('mmd', {})[args.robot] = str(config_path)

    frames, unmatched = load_frames(args.pmx, args.vmd, config_path, step=args.step)
    print('source frames: %d (%d unmatched vmd bones)' % (len(frames), len(unmatched)))

    retargeter = GMR(src_human='mmd', tgt_robot=args.robot,
                     actual_human_height=config['human_height_assumption'], verbose=False)
    cache = [np.array(retargeter.retarget(frame)) for frame in frames]

    out = Path(args.bake_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=np.array(cache))
    print('baked %d frames -> %s' % (len(cache), out))

    if not args.view:
        return
    from general_motion_retargeting import ROBOT_XML_DICT
    import mujoco as mj
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    import time
    with mj.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -8
        viewer.cam.azimuth = 90
        while viewer.is_running():
            for qpos in cache:
                data.qpos[:] = qpos
                mj.mj_forward(model, data)
                viewer.sync()
                time.sleep(args.speed / 60.0)


if __name__ == '__main__':
    main()
