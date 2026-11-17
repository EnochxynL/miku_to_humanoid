"""Retarget an MMD motion onto the robot with GMR's IK.

Inputs are data, not geometry: a **neutral calibration** (C per link, plus the body scale, from
``calibration.read_c``) and the source's world poses from ``world_poses``. No MJCF of the source rig is
involved - the vendored deform pipeline already gives world poses, including the ``センター``
translation that a qpos round trip used to drop.

This is the only file that knows GMR's config schema: it turns the neutral calibration into the
``ik_match_table`` shape GMR reads (writing it to a scratch path, because GMR loads configs by path) and
optionally keeps a copy with ``--dump-config``. Swapping the retargeting engine means replacing this file,
not the calibration.

    PYTHONPATH=src;src\\mmd <gmr>/.venv/Scripts/python.exe src/mmd/retarget.py \
        --pmx <model.pmx> --vmd <motion.vmd> --robot configs/robots/g1.json \
        --calibration configs/generated/calibration_g1.json --bake-out runtime/pmx/g1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.bake import world_poses                    # noqa: E402
from config import load as load_profile                # noqa: E402
from engine.constraints import ROOT_BONE, TORSO_BONE   # noqa: E402


def build_config(profile, calibration):
    """GMR's ik config, built from the profile's own template plus the neutral calibration.

    Only GMR's shape lives here: the profile template already carries the bone names, weights and
    offsets for this robot (it is complete on its own), and the calibration supplies C and the body
    scale. Nothing is read from GMR's checkout.
    """
    template = json.loads(json.dumps(profile.template))      # a private copy: callers mutate it
    template['human_root_name'] = ROOT_BONE
    template['human_height_assumption'] = float(calibration['heights']['source'])
    template['ground_height'] = 0.0
    # human_scale_table is a *uniform body* scale, not a per-segment length ratio: GMR computes
    # (position - root) * scale[bone]
    bones = sorted(set(profile.links.values()) | {ROOT_BONE, TORSO_BONE})
    template['human_scale_table'] = {bone: float(calibration['scale']) for bone in bones}

    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in template.get(table, {}).items():
            if link in profile.links:
                entry[0] = profile.links[link]
                entry[4] = [1.0, 0.0, 0.0, 0.0]
        # a profile lists only the places this robot deviates from GMR stock
    for link, info in calibration['links'].items():
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in template.get(table, {}):
                template[table][link][4] = [float(v) for v in info['C_quat_wxyz']]
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vmd', default=None, help='motion; omit for the bind pose')
    parser.add_argument('--robot', required=True,
                        help='robot profile, e.g. configs/robots/g1.json')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--calibration', required=True,
                        help='neutral calibration json from `python -m calibration`')
    parser.add_argument('--bake-out', required=True)
    parser.add_argument('--dump-config', default=None,
                        help='also keep the generated GMR config here (it is always written to a scratch '
                             'path, because GMR loads configs by path)')
    parser.add_argument('--gmr-robot', default=None,
                        help='GMR robot key; defaults to the profile name')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=float, default=0.6)
    args = parser.parse_args()

    profile = load_profile(args.robot)
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    robot_key = args.gmr_robot or profile.name

    config = build_config(profile, calibration)
    config_path = Path(args.dump_config) if args.dump_config else \
        Path('runtime') / ('gmr_config_%s.json' % robot_key)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

    wanted = sorted(set(profile.links.values()) | {ROOT_BONE, TORSO_BONE})
    frames, unmatched = world_poses(args.pmx, args.vmd, wanted, step=args.step)
    print('source frames: %d (%d unmatched vmd bones)' % (len(frames), len(unmatched)))

    gmr_root = Path(args.gmr_root)
    sys.path.insert(0, str(gmr_root))
    import general_motion_retargeting as gmrmod
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    gmrmod.IK_CONFIG_DICT.setdefault('mmd', {})[robot_key] = str(config_path.resolve())
    retargeter = GMR(src_human='mmd', tgt_robot=robot_key,
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
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot_key]))
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
