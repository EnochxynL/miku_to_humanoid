"""Retarget a BVH through GMR using rotation offsets derived from *our* skeleton.

Why this exists: GMR's `bvh_lafan1` IK config carries per-bone rotation offsets tuned for
LAFAN1's bone frames (for the feet, -90 deg about X). A BVH exported from another skeleton
(here: an MMD model converted to glTF, whose bone frames sit near identity) inherits those
offsets and collapses: the solver is asked to put the robot's links at orientations that are
90 deg away from the human's, and the only way to satisfy them is a folded pose (measured:
pelvis 0.07 m, feet -0.45 m).

The fix is to recompute the constant offsets for the skeleton we actually have:

    target = R_human * offset      =>      offset = R_human_rest^-1 * R_robot_rest

`R_human_rest` is the bone's world rotation at the reference frame, `R_robot_rest` the robot
link's rotation in the zero pose. Position offsets from the config are kept as they are.

This has to run inside the GMR environment (it imports mink/torch and the GMR package):

    <gmr-repo>/.venv/Scripts/python.exe src/reference_maker/gmr_mmd_driver.py \
        --bvh runtime/gmr_inputs/herta_mmd.bvh \
        --out runtime/gmr_inputs/herta_g1.pkl \
        --robot unitree_g1 --fps 30

It writes the same pickle layout as GMR's own `bvh_to_robot.py`, so `convert_gmr.py` accepts it.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def robot_rest_rotations(model, body_names):
    """World rotation of each robot body in the zero pose (all joints at 0)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        out[name] = data.xmat[body_id].reshape(3, 3).copy()
    return out


def derive_offsets(retargeter, reference_frame):
    """Replace the config's rotation offsets with ones matched to our skeleton."""
    model = retargeter.model
    frames_needed = set()
    for table in ('human_body_to_task1', 'human_body_to_task2'):
        for task in getattr(retargeter, table).values():
            frames_needed.add(task.frame_name)
    rest_robot = robot_rest_rotations(model, frames_needed)

    updated = 0
    for offset_table, body_table in ((retargeter.rot_offsets1, retargeter.human_body_to_task1),
                                     (retargeter.rot_offsets2, retargeter.human_body_to_task2)):
        for body, task in body_table.items():
            if body not in reference_frame or task.frame_name not in rest_robot:
                continue
            pos, quat = reference_frame[body]
            r_human = R.from_quat([quat[1], quat[2], quat[3], quat[0]])   # scalar-first -> xyzw
            r_robot = R.from_matrix(rest_robot[task.frame_name])
            offset_table[body] = r_human.inv() * r_robot
            updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--robot', default='unitree_g1')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--height', type=float, default=1.75,
                        help='human height GMR assumes when scaling to the robot')
    parser.add_argument('--max-iter', type=int, default=None)
    args = parser.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.kinematics_model import KinematicsModel
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    import torch

    frames, _ = load_bvh_file(args.bvh)
    retargeter = GMR(src_human='bvh_lafan1', tgt_robot=args.robot,
                     actual_human_height=args.height, verbose=False)
    patched = derive_offsets(retargeter, frames[0])

    qpos_list = []
    for frame in frames:
        qpos_list.append(np.array(retargeter.retarget(frame), copy=True))

    qpos = np.array(qpos_list)
    root_pos, root_rot, dof_pos = qpos[:, :3], qpos[:, 3:7].copy(), qpos[:, 7:]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]        # wxyz -> xyzw

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    kinematics = KinematicsModel(retargeter.xml_file, device=device)
    identity_pos = torch.zeros((len(dof_pos), 3), device=device)
    identity_rot = torch.zeros((len(dof_pos), 4), device=device)
    identity_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics.forward_kinematics(
        identity_pos, identity_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float))

    payload = {
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
        'local_body_pos': local_body_pos.detach().cpu().numpy(),
        'fps': args.fps,
        'link_body_list': kinematics.body_names,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as handle:
        pickle.dump(payload, handle)
    print(json.dumps({'frames': int(len(root_pos)), 'offsets_patched': patched,
                      'root_z': [float(root_pos[:, 2].min()), float(root_pos[:, 2].max())],
                      'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
