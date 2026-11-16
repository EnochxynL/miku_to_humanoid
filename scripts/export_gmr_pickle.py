"""Export our retargeted clip as a GMR pickle, which `reference_maker/convert_gmr.py` already reads.

Their converter expects exactly:

    {'root_pos': (N,3), 'root_rot': (N,4) as x,y,z,w, 'dof_pos': (N,29), 'fps': float}

Root rotation must be scalar-LAST here: convert_gmr.py reorders it back to MuJoCo's (w,x,y,z) with
`[:, [3, 0, 1, 2]]`, and feeding it w-first puts the feet a metre in the air (their comment, and the
same mistake I made earlier with the source animation).

The 29 actuated joints are verified to be the same names in the same order as
`assets/unitree_g1/g1_29dof.xml`, so dof_pos maps one-to-one.

    PYTHONPATH=src uv run python src/mmd/export_gmr_pickle.py \
        --npz runtime/glb/g1_final.npz --out runtime/gmr_clips/herta_qqy.pkl --fps 30
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', required=True, help='npz with a qpos array from the retarget')
    parser.add_argument('--out', required=True, help='pickle to write (.pkl)')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--robot-xml',
                        default='assets/unitree_g1/g1_29dof.xml',
                        help='the model whose joint order dof_pos must follow')
    parser.add_argument('--source-name', default='')
    args = parser.parse_args()

    data = np.load(args.npz)
    qpos = np.asarray(data['qpos'] if 'qpos' in data else data['states'][:, :36], dtype=float)
    model = mujoco.MjModel.from_xml_path(args.robot_xml)
    actuated = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if qpos.shape[1] != 7 + len(actuated):
        raise SystemExit(f'qpos has {qpos.shape[1]} columns, expected {7 + len(actuated)}')

    root_pos = qpos[:, :3].copy()
    root_rot_wxyz = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]        # w,x,y,z -> x,y,z,w
    dof_pos = qpos[:, 7:].copy()

    payload = {
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'fps': float(args.fps),
        'source': args.source_name or Path(args.npz).name,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as handle:
        pickle.dump(payload, handle)

    print('wrote %s' % out)
    print('  frames %d   fps %.0f   duration %.2f s' % (len(dof_pos), args.fps,
                                                        len(dof_pos) / args.fps))
    print('  root_pos[0] %s   root_rot[0] (xyzw) %s' % (np.round(root_pos[0], 3),
                                                         np.round(root_rot[0], 3)))
    print('  dof range %.2f .. %.2f rad' % (dof_pos.min(), dof_pos.max()))


if __name__ == '__main__':
    main()
