"""MMD pose (VMD or VPD) -> per-frame qpos of the viewer's MJCF.

This exists **for visualisation only**. The retarget path never needs it: ``mmd_pose`` gives the world
poses directly, so no qpos round trip is involved and nothing is lost. What this adds is a model the
renderer can draw - the skeleton from ``pmx_to_mjcf`` or the textured mesh from ``pmx_to_mjcf --mesh``.

A VPD works the same way as a VMD: it is one MMD pose, so the result is a single frame.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/vmd_to_qpos.py --pmx <model.pmx> --vmd <motion.vmd> \
        --xml runtime/pmx/model.xml --out runtime/pmx/anim.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.bake import bake                        # noqa: E402
from engine.pmx import parse_pmx                    # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vmd', default=None, help='motion; omit for the bind pose')
    parser.add_argument('--xml', required=True, help='MJCF from pmx_to_mjcf for this PMX')
    parser.add_argument('--out', required=True)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no-ik', dest='solve_ik', action='store_false')
    args = parser.parse_args()

    import mujoco
    if args.vmd is None:
        # a bind pose is one frame of qpos0
        robot = mujoco.MjModel.from_xml_path(args.xml)
        qpos = np.array([robot.qpos0])
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, qpos=qpos, times=np.zeros(1))
        print('{"frames": 1, "written": "%s"}' % out)
        return

    qpos, unmatched, times, missing = bake(args.pmx, args.vmd, args.xml,
                                           step=args.step, solve_ik=args.solve_ik)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos, times=times)
    print('{"frames": %d, "unmatched_vmd_bones": %d, "bones_without_joint": %d, "written": "%s"}'
          % (len(qpos), len(unmatched), len(missing), out))


if __name__ == '__main__':
    main()
