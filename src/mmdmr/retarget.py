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

import mujoco
import numpy as np


from .mmd.bake import world_poses                    # noqa: E402
from .config import load as load_profile                # noqa: E402
from .mmd.constraints import ROOT_BONE, TORSO_BONE   # noqa: E402


def build_config(engine_config, profile, calibration):
    """GMR's ik config: the engine file plus the neutral calibration.

    The engine file (``configs/gmr/<robot>.json``) carries the weight tables and offsets for this
    robot; the calibration supplies C and the body scale. Nothing is read from GMR's checkout, and the
    robot profile does not know this format exists.
    """
    template = json.loads(json.dumps(engine_config))          # a private copy: callers mutate it
    template.setdefault('human_root_name', ROOT_BONE)
    template['human_height_assumption'] = float(calibration['heights']['source'])
    template.setdefault('ground_height', 0.0)
    # human_scale_table is a *uniform body* scale, not a per-segment length ratio: GMR computes
    # (position - root) * scale[bone]
    bones = sorted(set(profile.links.values()) | {ROOT_BONE, TORSO_BONE})
    template['human_scale_table'] = {bone: float(calibration['scale']) for bone in bones}

    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in template.get(table, {}).items():
            if link in profile.links:
                entry[0] = profile.links[link]
                entry[4] = [1.0, 0.0, 0.0, 0.0]
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
    parser.add_argument('--gmr-config', default=None,
                        help="the engine config for this robot; defaults to the profile's gmr_config")
    parser.add_argument('--calibration', required=True,
                        help='neutral calibration json from `python -m calibration`')
    parser.add_argument('--bake-out', required=True)
    parser.add_argument('--dump-config', default=None,
                        help='also keep the generated GMR config here (it is always written to a scratch '
                             'path, because GMR loads configs by path)')
    parser.add_argument('--gmr-robot', default=None,
                        help='GMR robot key; defaults to the profile name')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--view', action='store_true',
                        help='preview afterwards: character, robot and the intermediate '
                             'target skeleton the engine aims at (green)')
    parser.add_argument('--offset', type=float, default=1.6,
                        help='gap between the two figures in the preview (metres)')
    parser.add_argument('--speed', type=float, default=1,
                        help='playback speed against the 30 fps source')
    parser.add_argument('--loop-pause', type=float, default=0.4,
                        help='pause between loops, in seconds')
    args = parser.parse_args()

    profile = load_profile(args.robot)
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    robot_key = args.gmr_robot or profile.name

    engine_path = Path(args.gmr_config) if args.gmr_config else profile.gmr_config
    if engine_path is None or not Path(engine_path).exists():
        raise SystemExit('no GMR config for this robot: pass --gmr-config or set gmr_config in %s'
                         % args.robot)
    engine_config = json.loads(Path(engine_path).read_text(encoding='utf-8'))

    config = build_config(engine_config, profile, calibration)
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

    cache, intermediate = [], []
    for frame in frames:
        cache.append(np.array(retargeter.retarget(frame)))
        # what GMR actually aims at: the source poses after its scale, the per-link C offset and the
        # ground offset. It keeps that in ``scaled_human_data``, so there is nothing to recompute.
        intermediate.append({bone: np.array(pos, dtype=float)
                             for bone, (pos, _quat) in retargeter.scaled_human_data.items()})

    out = Path(args.bake_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=np.array(cache))
    print('baked %d frames -> %s' % (len(cache), out))

    if not args.view:
        return

    from . import view                                 # noqa: E402  (only the preview needs it)
    from .mmd import mjcf                            # noqa: E402
    from .mmd.bake import bake                       # noqa: E402
    from .mmd.constraints import SKELETON_EDGES      # noqa: E402

    robot_model = mujoco.MjModel.from_xml_path(str(profile.xml))
    bundle = mjcf.build(args.pmx)
    character, _unmatched, _times, _missing = bake(args.pmx, args.vmd, bundle.model, step=args.step)
    model = view.build_scene(bundle.spec, profile.xml, args.offset, None)

    # the engine's target poses live in robot space, so they are drawn where the robot stands
    bones = {}
    for index in range(len(intermediate)):
        for bone, position in intermediate[index].items():
            bones.setdefault(bone, np.zeros((len(intermediate), 3)))[index] = position

    overlays = [
        view.QposOverlay('mmd', bundle.model, character),
        view.QposOverlay('robot', robot_model, np.array(cache),
                         prefix=view.ROBOT_PREFIX, translate=(args.offset, 0.0, 0.0)),
        # the engine's targets sit exactly halfway between the two figures, owned by neither
        view.ChainOverlay(bones, SKELETON_EDGES, radius=0.012,
                          translate=(args.offset * 0.5, 0.0, 0.0)),
    ]
    print('  intermediate: %d bones, %d frames' % (len(bones), len(intermediate)))
    for overlay in overlays:
        overlay.prepare(model)

    # azimuth 90 keeps the offset horizontal on screen: at 180 the figures line up along the view axis
    # and look like they are standing on top of each other even when they are 1.6 m apart
    view.BaseDisplay(model, speed=args.speed, loop=True, distance=3.6, azimuth=90.0,
                     lookat=(args.offset * 0.5, 0.0, 0.9),
                     loop_pause=args.loop_pause).run(overlays)


if __name__ == '__main__':
    main()
