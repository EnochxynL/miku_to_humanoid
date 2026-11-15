"""Open a MuJoCo window on a model, either at its bind pose or playing a qpos sequence.

    uv run python src/mmd/view_model.py --xml runtime/glb/herta_mesh.xml
    uv run python src/mmd/view_model.py --xml runtime/glb/herta_mesh.xml --qpos runtime/glb/anim.npz --loop

Kept separate from the converters so a model can be inspected without rebuilding anything; the
window stays open until it is closed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--xml', required=True)
    parser.add_argument('--qpos', default=None, help='npz with a qpos array to play')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--camera-distance', type=float, default=2.4)
    parser.add_argument('--camera-lookat', default='0 0 0.95')
    parser.add_argument('--camera-azimuth', type=float, default=200.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    frames = None
    if args.qpos:
        frames = list(np.load(args.qpos)['qpos'])
        print(f'{len(frames)} frames from {args.qpos}')
    mujoco.mj_forward(model, data)

    lookat = [float(v) for v in args.camera_lookat.split()]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = -8
        index = 0
        while viewer.is_running():
            started = time.perf_counter()
            if frames:
                data.qpos[:] = frames[index % len(frames)]
                mujoco.mj_forward(model, data)
                index += 1
                if not args.loop and index >= len(frames):
                    index = 0
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - started)))


if __name__ == '__main__':
    main()
