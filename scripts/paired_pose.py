"""Solve the paired pose numerically: the robot's joint angles that best match an MMD pose.

This is the script form of ``scripts/calibrate_gui.py`` - same objective, same solver, from
``mmd.calibration`` - for when the answer just has to be produced rather than dragged into place.

    PYTHONPATH=src;src\\mmd uv run python scripts/paired_pose.py --pmx <model.pmx> \
        --vpd <pose.vpd> --robot configs/robots/g1.json --out configs/generated/paired_pose_g1.json --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
for extra in (ROOT / 'src', ROOT / 'src' / 'mmd'):
    sys.path.insert(0, str(extra))

from calibration import Calibration                 # noqa: E402
from config import load as load_profile, validate   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vpd', default=None, help='the MMD pose to match; omitted = the bind pose')
    parser.add_argument('--robot', required=True,
                        help='robot profile, e.g. configs/robots/g1.json')
    parser.add_argument('--pose', default=None, help='start from this paired pose')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    profile = load_profile(args.robot)
    problems = validate(profile, mujoco.MjModel.from_xml_path(str(profile.xml)), mujoco)
    if problems:
        raise SystemExit('profile does not match the model:\n  ' + '\n  '.join(problems))

    calibration = Calibration(args.pmx, args.vpd, profile, args.pose)
    before = calibration.mean_error()
    after = calibration.solve()
    payload = calibration.save(args.out, args.vpd)

    if args.report:
        print('robot: %s (%s), %d joints, %d segments'
              % (profile.name, Path(profile.xml).name, len(calibration.joints),
                 len(calibration.segments)))
        print('mean direction error: %.3f deg -> %.3f deg' % (before, after))
        for segment, error in zip(calibration.segments, calibration.errors()):
            print('  %-28s %6.2f' % ('%s -> %s' % (segment[2], segment[3]), error))
    print('written %s (mean %.3f deg)' % (args.out, payload['limb_direction_error_deg']))


if __name__ == '__main__':
    main()
