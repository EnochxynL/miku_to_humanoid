"""Drag the robot's joints until it matches an MMD pose, then save the pose.

Two windows: the MuJoCo scene (character translucent, robot solid, both at the origin) and a panel with
one slider per joint of the loaded model, in model order - no grouping, no joint names in the code. The
objective and the solver live in ``mmd.calibration``, shared with ``scripts/paired_pose.py``.

    PYTHONPATH=src;src\\mmd;scripts uv run python scripts/calibrate_gui.py \
        --pmx <model.pmx> --vpd <pose.vpd> --robot configs/robots/g1.json
    # --robot also takes a GMR key, e.g. booster_t1_29dof: the profile is derived from GMR's stock config
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
for extra in (ROOT / 'src', ROOT / 'src' / 'mmd'):
    sys.path.insert(0, str(extra))

from calibration import Calibration                 # noqa: E402
from engine.profile import load as load_profile, validate   # noqa: E402

UPDATE_MS = 33


class Panel:
    """One slider per joint, in model order, in a scrolling list."""

    def __init__(self, root, calibration, viewer, source_pose):
        self.root = root
        self.cal = calibration
        self.viewer = viewer
        self.source_pose = source_pose
        self.rows = {}
        self.error_labels = []

        root.title('%s - paired pose' % calibration.profile.name)
        outer = ttk.Frame(root, padding=6)
        outer.pack(fill='both', expand=True)

        holder = ttk.Frame(outer)
        holder.pack(fill='both', expand=True)
        canvas = tk.Canvas(holder, width=430, height=520, highlightthickness=0)
        bar = ttk.Scrollbar(holder, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind_all('<MouseWheel>',
                        lambda event: canvas.yview_scroll(int(-event.delta / 120), 'units'))

        header = ttk.Frame(inner)
        header.pack(fill='x', pady=(0, 4))
        for text, width in (('#', 4), ('joint', 28), ('deg', 9), ('range', 15)):
            ttk.Label(header, text=text, width=width).pack(side='left')

        for index, joint in enumerate(calibration.joints):
            self.add_row(inner, index, joint)

        box = ttk.LabelFrame(outer, text='segment direction error (deg)', padding=4)
        box.pack(fill='x', pady=(6, 0))
        for segment in calibration.segments:
            label = ttk.Label(box, text='%-30s  --' % ('%s -> %s' % (segment[2], segment[3])))
            label.pack(anchor='w')
            self.error_labels.append(label)

        buttons = ttk.Frame(outer)
        buttons.pack(fill='x', pady=(6, 0))
        ttk.Button(buttons, text='solve', command=self.on_solve).pack(side='left')
        ttk.Button(buttons, text='save as...', command=self.on_save).pack(side='left', padx=6)
        ttk.Button(buttons, text='revert', command=self.on_revert).pack(side='left')

        self.initial = dict(calibration.values)
        self.refresh()
        root.after(UPDATE_MS, self.tick)

    # ------------------------------------------------------------------ rows
    def add_row(self, parent, index, joint):
        low, high = self.cal.limits[joint]
        row = ttk.Frame(parent)
        row.pack(fill='x')
        ttk.Label(row, text=str(index), width=4).pack(side='left')
        ttk.Label(row, text=joint, width=28).pack(side='left')
        entry = ttk.Entry(row, width=9)
        entry.insert(0, '%.1f' % self.cal.values[joint])
        entry.pack(side='left')
        entry.bind('<Return>', lambda _e, j=joint: self.on_entry(j))
        slider = ttk.Scale(row, from_=low, to=high, length=160,
                           command=lambda _v, j=joint: self.on_slide(j))
        slider.pack(side='left', padx=4)
        ttk.Label(row, text='%6.1f ..%6.1f' % (low, high), width=15).pack(side='left')
        # register before set(): ttk.Scale.set fires the command, and on_slide looks the row up
        self.rows[joint] = (slider, entry)
        slider.set(self.cal.values[joint])

    # ------------------------------------------------------------------ callbacks
    def on_slide(self, joint):
        slider, entry = self.rows[joint]
        self.cal.values[joint] = float(slider.get())
        entry.delete(0, 'end')
        entry.insert(0, '%.1f' % self.cal.values[joint])
        self.cal.apply_robot()
        self.refresh()

    def on_entry(self, joint):
        slider, entry = self.rows[joint]
        try:
            typed = float(entry.get())
        except ValueError:
            return
        low, high = self.cal.limits[joint]
        slider.set(max(low, min(high, typed)))
        self.on_slide(joint)

    def push(self):
        for joint, (slider, entry) in self.rows.items():
            slider.set(self.cal.values[joint])
            entry.delete(0, 'end')
            entry.insert(0, '%.1f' % self.cal.values[joint])

    def refresh(self):
        errors = self.cal.errors()
        for label, segment, error in zip(self.error_labels, self.cal.segments, errors):
            label.config(text='%-30s %6.2f' % ('%s -> %s' % (segment[2], segment[3]), error))

    def on_solve(self):
        mean = self.cal.solve()
        self.push()
        self.refresh()
        moved = sorted(((abs(self.cal.values[j] - self.initial[j]), j) for j in self.cal.joints),
                       reverse=True)[:3]
        self.root.title('%s - paired pose   (solve %.3f deg; biggest moves: %s)'
                        % (self.cal.profile.name, mean,
                           ', '.join('%s %+.1f' % (j, self.cal.values[j] - self.initial[j])
                                     for _d, j in moved)))

    def on_revert(self):
        self.cal.values.update(self.initial)
        self.cal.apply_robot()
        self.push()
        self.refresh()

    def on_save(self):
        path = filedialog.asksaveasfilename(title='save paired pose', defaultextension='.json',
                                            filetypes=[('pose json', '*.json'), ('all files', '*.*')])
        if not path:
            return
        payload = self.cal.save(path, self.source_pose)
        self.root.title('%s - paired pose   (saved %s, %.3f deg)'
                        % (self.cal.profile.name, Path(path).name,
                           payload['limb_direction_error_deg']))
        print('saved %s   mean error %.3f deg' % (path, payload['limb_direction_error_deg']))

    def tick(self):
        self.viewer.sync()
        if not self.viewer.is_running():
            self.root.destroy()
            return
        self.root.after(UPDATE_MS, self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--vpd', required=True, help='the MMD pose to match')
    parser.add_argument('--robot', required=True,
                        help='robot profile, e.g. configs/robots/g1.json')
    parser.add_argument('--pose', default=None, help='start from this paired pose')
    parser.add_argument('--distance', type=float, default=2.6)
    parser.add_argument('--azimuth', type=float, default=180.0)
    parser.add_argument('--lookat', default='0 0 0.85')
    args = parser.parse_args()

    profile = load_profile(args.robot)
    problems = validate(profile, mujoco.MjModel.from_xml_path(str(profile.xml)), mujoco)
    if problems:
        raise SystemExit('profile does not match the model:\n  ' + '\n  '.join(problems))

    calibration = Calibration(args.pmx, args.vpd, profile, args.pose)
    viewer = mujoco.viewer.launch_passive(calibration.model, calibration.data)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [float(v) for v in args.lookat.split()]
    viewer.cam.distance = args.distance
    viewer.cam.azimuth = args.azimuth
    viewer.cam.elevation = -6

    root = tk.Tk()
    Panel(root, calibration, viewer, args.vpd)
    try:
        root.mainloop()
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
