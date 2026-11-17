"""Three-part front end: a launcher, the calibration window, and GMR as a subprocess.

The launcher (a small panel) collects every file location. "开始校准" opens the calibration window - the
MuJoCo preview with one slider per joint - and saving there writes **two** artifacts: the paired pose and
the neutral calibration that the retarget step consumes. Saving fills the launcher's calibration field,
so the retarget step is one click away.

"开始重定向" converts the panel into a command line and runs ``src/mmd/retarget.py`` under **GMR's own
interpreter**: this process cannot import ``general_motion_retargeting`` (it needs ``rich`` and kin, which
live in GMR's venv), so the retarget step is a subprocess and this window is only a parameter loader.

    PYTHONPATH=src;src\\mmd uv run python -m gui --pmx <model.pmx> --vmd <motion.vmd> \
        --vpd <pose.vpd> --robot configs/robots/g1.json --gmr-root <gmr>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]        # src/mmd/gui.py -> repo root
    
from . import config as robot_config                       # noqa: E402
from .calibration import Calibration, read_c, save_c  # noqa: E402

UPDATE_MS = 33
POLL_MS = 300


# ---------------------------------------------------------------------------- the calibration window
class CalibrationWindow(tk.Toplevel):
    """The interactive half: the robot and the character in one scene, a slider per joint."""

    def __init__(self, master, launcher, paths):
        super().__init__(master)
        self.launcher = launcher
        self.paths = paths
        self.profile = robot_config.load(paths['robot'])
        problems = robot_config.validate(
            self.profile, mujoco.MjModel.from_xml_path(str(self.profile.xml)), mujoco)
        if problems:
            raise SystemExit('profile does not match the model:\n  ' + '\n  '.join(problems))

        self.cal = Calibration(paths['pmx'], paths['vpd'], self.profile,
                               paths.get('pose') or None)
        self.viewer = mujoco.viewer.launch_passive(self.cal.model, self.cal.data,
                                                  show_left_ui=False, show_right_ui=False)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.85]
        self.viewer.cam.distance = 2.6
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -6

        self.title('%s - calibration' % self.profile.name)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.rows, self.error_labels = {}, []
        self._build()
        self.initial = dict(self.cal.values)
        self.refresh()
        self.after(UPDATE_MS, self.tick)

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = ttk.Frame(self, padding=6)
        outer.pack(fill='both', expand=True)

        holder = ttk.Frame(outer)
        holder.pack(fill='both', expand=True)
        canvas = tk.Canvas(holder, width=430, height=480, highlightthickness=0)
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
        for index, joint in enumerate(self.cal.joints):
            self._add_row(inner, index, joint)

        box = ttk.LabelFrame(outer, text='segment direction error (deg)', padding=4)
        box.pack(fill='x', pady=(6, 0))
        for segment in self.cal.segments:
            label = ttk.Label(box, text='%-30s  --' % ('%s -> %s' % (segment[2], segment[3])))
            label.pack(anchor='w')
            self.error_labels.append(label)

        buttons = ttk.Frame(outer)
        buttons.pack(fill='x', pady=(6, 0))
        ttk.Button(buttons, text='solve', command=self.on_solve).pack(side='left')
        ttk.Button(buttons, text='revert', command=self.on_revert).pack(side='left', padx=6)
        ttk.Button(buttons, text='save calibration', command=self.on_save).pack(side='left')

        self.status = ttk.Label(outer, text='')
        self.status.pack(fill='x', pady=(4, 0))

    def _add_row(self, parent, index, joint):
        low, high = self.cal.limits[joint]
        row = ttk.Frame(parent)
        row.pack(fill='x')
        ttk.Label(row, text=str(index), width=4).pack(side='left')
        ttk.Label(row, text=joint, width=28).pack(side='left')
        entry = ttk.Entry(row, width=9)
        entry.insert(0, '%.1f' % self.cal.values[joint])
        entry.pack(side='left')
        entry.bind('<Return>', lambda _e, j=joint: self.on_entry(j))
        slider = ttk.Scale(row, from_=low, to=high, length=160)
        slider.pack(side='left', padx=4)
        ttk.Label(row, text='%6.1f ..%6.1f' % (low, high), width=15).pack(side='left')
        # set() fires the command, so assign it only once the row and widgets exist
        self.rows[joint] = (slider, entry)
        slider.set(self.cal.values[joint])
        slider.configure(command=lambda _v, j=joint: self.on_slide(j))

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
            value = float(entry.get())
        except ValueError:
            return
        low, high = self.cal.limits[joint]
        slider.set(max(low, min(high, value)))
        self.on_slide(joint)

    def push(self):
        for joint, (slider, entry) in self.rows.items():
            slider.set(self.cal.values[joint])
            entry.delete(0, 'end')
            entry.insert(0, '%.1f' % self.cal.values[joint])

    def refresh(self):
        if not self.error_labels or not hasattr(self, 'status'):
            return
        errors = self.cal.errors()
        for label, segment, error in zip(self.error_labels, self.cal.segments, errors):
            label.config(text='%-30s %6.2f' % ('%s -> %s' % (segment[2], segment[3]), error))
        self.status.config(text='mean %.3f deg' % self.cal.mean_error())

    def on_solve(self):
        mean = self.cal.solve()
        self.push()
        self.refresh()
        self.title('%s - calibration   (solve %.3f deg)' % (self.profile.name, mean))

    def on_revert(self):
        self.cal.values.update(self.initial)
        self.cal.apply_robot()
        self.push()
        self.refresh()

    def on_save(self):
        pose_path = Path(self.paths['pose'])
        calibration_path = Path(self.paths['calibration'])
        pose_payload = self.cal.save(pose_path, self.paths['vpd'])

        robot_model = mujoco.MjModel.from_xml_path(str(self.profile.xml))
        c = read_c(self.paths['pmx'], robot_model, self.profile, pose_payload, self.paths['vpd'])
        save_c(calibration_path, c)

        self.status.config(text='saved %s + %s' % (pose_path.name, calibration_path.name))
        self.launcher.adopt_calibration(calibration_path, pose_path)

    def tick(self):
        self.viewer.sync()
        if not self.viewer.is_running():
            self.close()
            return
        self.after(UPDATE_MS, self.tick)

    def close(self):
        try:
            self.viewer.close()
        finally:
            self.launcher.calibration_closed()
            self.destroy()


# ---------------------------------------------------------------------------- the launcher
FIELDS = (
    ('pmx', 'MMD model (.pmx)'),
    ('vmd', 'motion to retarget (.vmd)'),
    ('vpd', 'pose to calibrate against (.vpd)'),
    ('robot', 'robot profile (.json)'),
    ('gmr_config', 'GMR config (.json, optional)'),
    ('pose', 'paired pose out (.json)'),
    ('calibration', 'calibration out (.json)'),
    ('qpos', 'retargeted qpos out (.npz)'),
    ('gmr_root', 'GMR checkout'),
)


class Launcher(tk.Tk):
    """Every path in one panel, plus the two buttons that start the two halves."""

    def __init__(self, initial):
        super().__init__()
        self.title('mmd -> humanoid')
        self.values = {name: tk.StringVar(value=str(initial.get(name) or '')) for name, _ in FIELDS}
        self.calibration_window = None
        self.process = None

        outer = ttk.Frame(self, padding=8)
        outer.pack(fill='both', expand=True)
        for row, (name, label) in enumerate(FIELDS):
            ttk.Label(outer, text=label, width=30).grid(row=row, column=0, sticky='w', pady=2)
            ttk.Entry(outer, textvariable=self.values[name], width=64).grid(row=row, column=1, sticky='we')
            ttk.Button(outer, text='...', width=3,
                       command=lambda n=name: self.browse(n)).grid(row=row, column=2, padx=4)

        buttons = ttk.Frame(outer)
        buttons.grid(row=len(FIELDS), column=0, columnspan=3, sticky='we', pady=(8, 0))
        self.calibrate_button = ttk.Button(buttons, text='开始校准', command=self.start_calibration)
        self.calibrate_button.pack(side='left')
        self.retarget_button = ttk.Button(buttons, text='开始重定向', command=self.start_retarget)
        self.retarget_button.pack(side='left', padx=6)

        self.log = tk.Text(outer, height=10, width=96)
        self.log.grid(row=len(FIELDS) + 1, column=0, columnspan=3, sticky='we', pady=(8, 0))
        self.report('fill the paths, calibrate, then retarget')


    # ------------------------------------------------------------------ helpers
    def report(self, message):
        self.log.insert('end', message + '\n')
        self.log.see('end')

    def path(self, name):
        value = self.values[name].get().strip()
        return Path(value) if value else None

    def browse(self, name):
        current = self.values[name].get()
        if name == 'gmr_root' or name == 'robot':
            chosen = filedialog.askdirectory() if name == 'gmr_root' else \
                filedialog.askopenfilename(filetypes=[('json', '*.json')])
        else:
            chosen = filedialog.asksaveasfilename(defaultextension=Path(current).suffix
                                                  if current else '')
        if chosen:
            self.values[name].set(chosen)

    def adopt_calibration(self, calibration_path, pose_path):
        """Saving in the calibration window fills the panel - the retarget step is then one click."""
        self.values['calibration'].set(str(calibration_path))
        self.values['pose'].set(str(pose_path))
        self.report('calibration ready: %s' % calibration_path)

    def calibration_closed(self):
        self.calibration_window = None
        self.calibrate_button.state(['!disabled'])
        self.report('calibration window closed')

    # ------------------------------------------------------------------ calibration
    def start_calibration(self):
        if self.calibration_window is not None:
            self.report('the calibration window is already open')
            return
        for needed in ('pmx', 'vpd', 'robot'):
            if self.path(needed) is None:
                self.report('missing: %s' % needed)
                return
        paths = {name: self.values[name].get() for name, _ in FIELDS}
        stem = Path(paths['robot']).stem or 'robot'
        if not paths['pose']:
            paths['pose'] = str(ROOT / 'configs' / 'generated' / ('paired_pose_%s.json' % stem))
        if not paths['calibration']:
            paths['calibration'] = str(ROOT / 'configs' / 'generated' / ('calibration_%s.json' % stem))
        try:
            self.calibration_window = CalibrationWindow(self, self, paths)
        except SystemExit as error:
            self.report(str(error))
            return
        self.calibrate_button.state(['disabled'])
        self.report('calibration window opened')

    # ------------------------------------------------------------------ retarget (subprocess)
    def retarget_command(self):
        needed = {}
        for name in ('pmx', 'vmd', 'robot', 'calibration', 'qpos', 'gmr_root'):
            value = self.path(name)
            if value is None:
                self.report('missing: %s' % name)
                return None
            needed[name] = value
        python = needed['gmr_root'] / '.venv' / 'Scripts' / 'python.exe'
        if not python.exists():
            self.report('no interpreter under %s' % (needed['gmr_root'] / '.venv'))
            return None
        command = [str(python), '-m', 'mmdmr.retarget',
                   '--pmx', str(needed['pmx']), '--vmd', str(needed['vmd']),
                   '--robot', str(needed['robot']), '--calibration', str(needed['calibration']),
                   '--gmr-root', str(needed['gmr_root']), '--bake-out', str(needed['qpos']),
                   '--view']
        engine = self.path('gmr_config')            # optional: the profile may name its own
        if engine is not None:
            command += ['--gmr-config', str(engine)]
        return command

    def start_retarget(self):
        if self.process is not None and self.process.poll() is None:
            self.report('a retarget is already running')
            return
        command = self.retarget_command()
        if command is None:
            return
        self.report('$ ' + ' '.join(command))
        self.retarget_button.state(['disabled'])
        # GMR's interpreter does not have this package installed, so the child gets src/ on its path
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(ROOT / 'src')
        try:
            self.process = subprocess.Popen(command, cwd=str(ROOT), text=True, env=environment,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            encoding='utf-8', errors='replace')
        except OSError as error:
            self.report('failed to start: %s' % error)
            self.retarget_button.state(['!disabled'])
            return
        self.after(POLL_MS, self.poll_retarget)

    def poll_retarget(self):
        if self.process is None:
            return
        line = self.process.stdout.readline() if self.process.stdout else ''
        if line:
            self.report(line.rstrip())
        if self.process.poll() is None:
            self.after(POLL_MS, self.poll_retarget)
            return
        self.report('retarget finished with %s' % self.process.returncode)
        self.retarget_button.state(['!disabled'])
        self.process = None


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', default=None)
    parser.add_argument('--vmd', default=None)
    parser.add_argument('--vpd', default=None)
    parser.add_argument('--robot', default=str(ROOT / 'configs' / 'robots' / 'g1.json'))
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--pose', default=None)
    parser.add_argument('--calibration', default=None)
    parser.add_argument('--qpos', default=None)
    args = parser.parse_args()

    Launcher(vars(args)).mainloop()


if __name__ == '__main__':
    main()
