"""Interactive calibration of the G1 paired pose.

Two windows: the MuJoCo scene (MMD character and robot at the same origin, the character translucent
so the robot is visible inside it) and a control panel with one slider + numeric box per joint, the
eight limb-direction errors that define "the same pose", and Save As.

Why those eight numbers and not the full pose: the paired pose is defined by **limb directions**, which
are convention-free. Orientations cannot be used here - their relationship *is* C, the thing being
calibrated - so the objective and the numeric solver's objective are the same eight angles.

The saved file is exactly what ``gmr_calibrate.py --pose`` reads:

    {"joints": {"left_shoulder_pitch_joint": -53.05, ...}, ...}

    PYTHONPATH=src;src\\mmd uv run python scripts/g1_calibrate_gui.py \
        --pmx <model.pmx> --vpd <pose.vpd> \
        --robot-xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml \
        --pose configs/mmd/paired_pose_pmx.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'src'))
sys.path.insert(0, str(HERE.parents[1] / 'src' / 'mmd'))

import view                                            # noqa: E402
from g1_paired_pose import LIMBS, segments             # noqa: E402

MMD_ALPHA = 0.25
UPDATE_MS = 33


class Calibration:
    """The composed scene plus the joint values being edited."""

    def __init__(self, pmx, vpd, robot_xml, pose_path):
        from engine import mmd_model
        from engine.bake import bake

        self.bundle = mmd_model.build(pmx)
        self.model = view.build_scene(self.bundle.spec, robot_xml, 0.0, None)
        self.data = mujoco.MjData(self.model)

        # the character stays static (it *is* the target); make only it translucent
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            if not name.startswith(view.ROBOT_PREFIX):
                rgba = np.array(self.model.geom_rgba[geom], dtype=float)
                rgba[3] = MMD_ALPHA
                self.model.geom_rgba[geom] = rgba

        self.mmd_slices, _ = view.joint_slices(self.model, self.bundle.model)
        clip, _unmatched, _times, _missing = bake(pmx, vpd, self.bundle.model)
        self.mmd_qpos = clip[0]
        self.apply_mmd()

        self.segments = segments()
        self.joints = []
        for *_rest, joints in self.segments:
            for joint in joints:
                if joint not in self.joints:
                    self.joints.append(joint)
        self.qpos_address = {}
        self.limits = {}
        for joint in self.joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, view.ROBOT_PREFIX + joint)
            self.qpos_address[joint] = int(self.model.jnt_qposadr[jid])
            self.limits[joint] = np.degrees(np.array(self.model.jnt_range[jid]))

        self.values = {joint: 0.0 for joint in self.joints}
        if pose_path and Path(pose_path).exists():
            supplied = json.loads(Path(pose_path).read_text(encoding='utf-8'))
            for joint, value in supplied.get('joints', {}).items():
                if joint in self.values:
                    self.values[joint] = float(value)
        self.apply_robot()

    # ------------------------------------------------------------------ scene
    def apply_mmd(self):
        for destination, source, width in self.mmd_slices:
            self.data.qpos[destination:destination + width] = self.mmd_qpos[source:source + width]

    def apply_robot(self):
        for joint, value in self.values.items():
            self.data.qpos[self.qpos_address[joint]] = np.radians(value)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ objective
    def errors(self):
        out = []
        for source_a, source_b, robot_a, robot_b, _joints in self.segments:
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, source_a)
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, source_b)
            if a < 0 or b < 0:
                out.append(np.nan)
                continue
            want = np.array(self.data.xpos[b]) - np.array(self.data.xpos[a])
            got_start = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          view.ROBOT_PREFIX + robot_a)
            got_end = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        view.ROBOT_PREFIX + robot_b)
            got = np.array(self.data.xpos[got_end]) - np.array(self.data.xpos[got_start])
            want /= np.linalg.norm(want)
            got /= np.linalg.norm(got)
            out.append(float(np.degrees(np.arccos(np.clip(got @ want, -1.0, 1.0)))))
        return out

    def solve(self):
        """Same objective the CLI solver uses, seeded from the current values."""
        x0 = np.radians([self.values[j] for j in self.joints])
        bounds = [tuple(np.radians(self.limits[j])) for j in self.joints]

        def cost(x):
            for joint, value in zip(self.joints, x):
                self.data.qpos[self.qpos_address[joint]] = value
            mujoco.mj_forward(self.model, self.data)
            return float(np.sum(np.square(self.errors())))

        result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12})
        for joint, value in zip(self.joints, result.x):
            self.values[joint] = float(np.degrees(value))
        self.apply_robot()
        return float(np.mean(self.errors()))

    def payload(self, robot_xml, source_pose):
        errors = self.errors()
        return {
            'robot_xml': Path(robot_xml).name,
            'note': ('paired pose written by the interactive calibration tool; consumed by '
                     'gmr_calibrate, which derives C from it'),
            'source_pose': source_pose,
            'joints': {joint: round(value, 4) for joint, value in self.values.items()},
            'limb_direction_error_deg': round(float(np.mean(errors)), 3),
            'per_limb_error_deg': {segment[0]: round(error, 3)
                                   for segment, error in zip(self.segments, errors)},
        }


class Panel:
    """The tkinter side: sliders, numeric boxes, live errors, save."""

    def __init__(self, root, calibration, viewer, robot_xml, source_pose):
        self.root = root
        self.calibration = calibration
        self.viewer = viewer
        self.robot_xml = robot_xml
        self.source_pose = source_pose
        self.sliders = {}
        self.entries = {}
        self.error_labels = []

        root.title('G1 paired pose')
        frame = ttk.Frame(root, padding=6)
        frame.grid(row=0, column=0, sticky='nsew')

        row = 0
        for title, members in self.groups():
            ttk.Label(frame, text=title, font=('TkDefaultFont', 9, 'bold')).grid(
                row=row, column=0, columnspan=3, sticky='w', pady=(6, 2))
            row += 1
            for joint in members:
                row = self.add_joint(frame, row, joint)

        box = ttk.Frame(frame)
        box.grid(row=row, column=0, columnspan=3, sticky='we', pady=(8, 2))
        ttk.Label(box, text='方向误差 (deg)').pack(anchor='w')
        for segment in calibration.segments:
            label = ttk.Label(box, text='%-8s   --' % segment[0])
            label.pack(anchor='w')
            self.error_labels.append(label)

        buttons = ttk.Frame(frame)
        buttons.grid(row=row + 1, column=0, columnspan=3, sticky='we', pady=(8, 0))
        ttk.Button(buttons, text='数值求解', command=self.on_solve).pack(side='left')
        ttk.Button(buttons, text='保存…', command=self.on_save).pack(side='left', padx=6)
        ttk.Button(buttons, text='复位', command=self.on_reset).pack(side='left')

        self.initial = dict(calibration.values)
        self.refresh_errors()
        root.after(UPDATE_MS, self.tick)

    def groups(self):
        # joint names are English: left_*/right_*
        return [('左臂 / 左腿  (left)', [j for j in self.calibration.joints if j.startswith('left')]),
                ('右臂 / 右腿  (right)', [j for j in self.calibration.joints if j.startswith('right')])]

    def add_joint(self, parent, row, joint):
        low, high = self.calibration.limits[joint]
        ttk.Label(parent, text=joint.replace('_joint', '')).grid(row=row, column=0, sticky='w')
        value = tk.DoubleVar(value=self.calibration.values[joint])
        slider = ttk.Scale(parent, from_=low, to=high, variable=value, length=240,
                           command=lambda _v, j=joint: self.on_slide(j))
        slider.grid(row=row, column=1, sticky='we', padx=4)
        entry = ttk.Entry(parent, width=8)
        entry.insert(0, '%.1f' % self.calibration.values[joint])
        entry.grid(row=row, column=2, sticky='e')
        entry.bind('<Return>', lambda _e, j=joint: self.on_entry(j))
        self.sliders[joint] = (slider, value, entry)
        return row + 1

    # ------------------------------------------------------------------ callbacks
    def on_slide(self, joint):
        slider, value, entry = self.sliders[joint]
        self.calibration.values[joint] = float(value.get())
        entry.delete(0, 'end')
        entry.insert(0, '%.1f' % self.calibration.values[joint])
        self.calibration.apply_robot()
        self.refresh_errors()

    def on_entry(self, joint):
        slider, value, entry = self.sliders[joint]
        try:
            typed = float(entry.get())
        except ValueError:
            return
        low, high = self.calibration.limits[joint]
        typed = max(low, min(high, typed))
        value.set(typed)
        self.on_slide(joint)

    def push_values(self):
        for joint, (slider, value, entry) in self.sliders.items():
            value.set(self.calibration.values[joint])
            entry.delete(0, 'end')
            entry.insert(0, '%.1f' % self.calibration.values[joint])

    def refresh_errors(self):
        errors = self.calibration.errors()
        for label, segment, error in zip(self.error_labels, self.calibration.segments, errors):
            label.config(text='%-8s  %5.2f' % (segment[0], error))
        self.current_mean = float(np.mean(errors))

    def on_solve(self):
        mean = self.calibration.solve()
        self.push_values()
        self.refresh_errors()
        self.root.title('G1 paired pose   (solve: %.3f deg)' % mean)

    def on_reset(self):
        self.calibration.values.update(self.initial)
        self.calibration.apply_robot()
        self.push_values()
        self.refresh_errors()

    def on_save(self):
        path = filedialog.asksaveasfilename(
            title='保存配对姿态', defaultextension='.json',
            filetypes=[('pose json', '*.json'), ('all files', '*.*')])
        if not path:
            return
        payload = self.calibration.payload(self.robot_xml, self.source_pose)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        self.root.title('G1 paired pose   (saved %s)' % Path(path).name)
        print('saved', path, 'mean error %.3f deg' % payload['limb_direction_error_deg'])

    def tick(self):
        self.viewer.sync()
        if not self.viewer.is_running():
            self.root.destroy()
            return
        self.root.after(UPDATE_MS, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True, help='the character model, built in memory')
    parser.add_argument('--vpd', required=True, help='the MMD pose to match (a .vpd file)')
    parser.add_argument('--robot-xml', default=view.DEFAULT_ROBOT_XML)
    parser.add_argument('--pose', default=None, help='start from this pose file')
    parser.add_argument('--distance', type=float, default=2.6)
    parser.add_argument('--azimuth', type=float, default=180.0)
    parser.add_argument('--lookat', default='0 0 0.85')
    args = parser.parse_args()

    calibration = Calibration(args.pmx, args.vpd, args.robot_xml, args.pose)
    model, data = calibration.model, calibration.data
    lookat = [float(v) for v in args.lookat.split()]

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = args.distance
    viewer.cam.azimuth = args.azimuth
    viewer.cam.elevation = -6

    root = tk.Tk()
    Panel(root, calibration, viewer, args.robot_xml, args.vpd)
    try:
        root.mainloop()
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
