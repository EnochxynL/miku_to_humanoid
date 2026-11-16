"""Interactive calibration: drag the robot's joints until it matches an MMD pose, then save the pose.

Two windows. The MuJoCo one shows the character (translucent) and the robot at the same origin; the
tkinter one lists **every joint of the loaded robot model in index order** - no grouping, no hardcoded
joint names - plus the direction error of each profile segment and a Save As.

Nothing here knows what robot it is driving: the links, their MMD partners and the segments come from
``configs/robots/<robot>.json``, and the joint list comes from the XML itself. That is the whole point:
a new robot needs a profile, not a code change.

The saved file is exactly what ``gmr_calibrate.py --pose`` reads:

    {"joints": {"left_shoulder_pitch_joint": -53.05, ...}, ...}

    PYTHONPATH=src;src\\mmd;scripts uv run python scripts/calibrate_gui.py \
        --pmx <model.pmx> --vpd <pose.vpd> --robot configs/robots/g1.json
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

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
for extra in (ROOT / 'src', ROOT / 'src' / 'mmd', ROOT / 'scripts'):
    sys.path.insert(0, str(extra))

import robot_profile                                # noqa: E402
import view                                         # noqa: E402
from engine import mjcf                             # noqa: E402
from engine.bake import bake                        # noqa: E402

CHARACTER_ALPHA = 0.25
UPDATE_MS = 33


class Calibration:
    """The composed scene, the joints being edited, and the direction errors that judge them."""

    def __init__(self, pmx, vpd, profile, pose_path):
        self.profile = profile
        self.bundle = mjcf.build(pmx)
        self.model = view.build_scene(self.bundle.spec, profile.xml, 0.0, None)
        self.data = mujoco.MjData(self.model)

        # the character is the target: keep it, fade it, and stop it from moving
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            if not name.startswith(view.ROBOT_PREFIX):
                rgba = np.array(self.model.geom_rgba[geom], dtype=float)
                rgba[3] = CHARACTER_ALPHA
                self.model.geom_rgba[geom] = rgba

        self.mmd_slices, _ = view.joint_slices(self.model, self.bundle.model)
        clip, _u, _t, _m = bake(pmx, vpd, self.bundle.model)
        self.mmd_qpos = clip[0]

        # every robot joint, in model order - the profile decides the links, the model decides the joints
        self.joints = []
        self.address = {}
        self.limits = {}
        for joint in range(self.model.njnt):
            full = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ''
            if not full.startswith(view.ROBOT_PREFIX):
                continue
            # only one-degree-of-freedom joints: a free joint owns seven qpos slots and a ball joint
            # four, so a single slider writing one slot would corrupt them
            if int(self.model.jnt_type[joint]) not in (mujoco.mjtJoint.mjJNT_HINGE,
                                                       mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            short = full[len(view.ROBOT_PREFIX):]
            low, high = (float(v) for v in self.model.jnt_range[joint])
            if low >= high:                       # unlimited joint: give it a sane sweep
                low, high = -np.pi, np.pi
            self.joints.append(short)
            self.address[short] = int(self.model.jnt_qposadr[joint])
            self.limits[short] = (np.degrees(low), np.degrees(high))

        self.segments = []
        for segment in profile.segments:
            if segment.link in profile.links and segment.to in profile.links:
                self.segments.append((profile.links[segment.link], profile.links[segment.to],
                                      segment.link, segment.to))
        if not self.segments:
            raise SystemExit('the profile defines no usable segments (need both links mapped to bones)')

        self.values = {joint: 0.0 for joint in self.joints}
        if pose_path and Path(pose_path).exists():
            supplied = json.loads(Path(pose_path).read_text(encoding='utf-8'))
            for joint, value in supplied.get('joints', {}).items():
                if joint in self.values:
                    self.values[joint] = float(value)
        self.apply_character()
        self.apply_robot()

    # ------------------------------------------------------------------ scene
    def apply_character(self):
        for destination, source, width in self.mmd_slices:
            self.data.qpos[destination:destination + width] = self.mmd_qpos[source:source + width]

    def apply_robot(self):
        for joint, value in self.values.items():
            self.data.qpos[self.address[joint]] = np.radians(value)
        mujoco.mj_forward(self.model, self.data)

    def body_id(self, name, robot):
        full = (view.ROBOT_PREFIX + name) if robot else name
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, full)

    def direction(self, start, end, robot):
        a, b = self.body_id(start, robot), self.body_id(end, robot)
        if a < 0 or b < 0:
            return None
        vector = np.array(self.data.xpos[b]) - np.array(self.data.xpos[a])
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-9 else None

    # ------------------------------------------------------------------ objective
    def errors(self):
        out = []
        for bone_a, bone_b, link_a, link_b in self.segments:
            want = self.direction(bone_a, bone_b, robot=False)
            got = self.direction(link_a, link_b, robot=True)
            if want is None or got is None:
                out.append(float('nan'))
                continue
            out.append(float(np.degrees(np.arccos(np.clip(got @ want, -1.0, 1.0)))))
        return out

    def solve(self):
        """Same objective the CLI solver uses, seeded from the current slider values."""
        names = list(self.joints)
        x0 = np.radians([self.values[joint] for joint in names])
        bounds = [tuple(np.radians(self.limits[joint])) for joint in names]

        def cost(x):
            for joint, value in zip(names, x):
                self.data.qpos[self.address[joint]] = value
            mujoco.mj_forward(self.model, self.data)
            return float(np.sum(np.square(self.errors())))

        from scipy.optimize import minimize
        result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12})
        for joint, value in zip(names, result.x):
            self.values[joint] = float(np.degrees(value))
        self.apply_robot()
        return float(np.mean(self.errors()))

    def payload(self, source_vpd):
        errors = self.errors()
        return {
            'robot': {'profile': Path(self.profile.name).name, 'name': self.profile.name,
                      'xml': self.profile.xml.name},
            'note': ('paired pose written by the interactive calibration tool; consumed by '
                     'gmr_calibrate, which derives C from it'),
            'source_pose': str(source_vpd),
            'joints': {joint: round(value, 4) for joint, value in self.values.items()},
            'limb_direction_error_deg': round(float(np.mean(errors)), 3),
            'per_segment_error_deg': {
                '%s-%s' % (segment[2], segment[3]): round(error, 3)
                for segment, error in zip(self.segments, errors)},
        }


class Panel:
    """Slider per joint, in model order, inside a scrolling area."""

    def __init__(self, root, calibration, viewer, source_vpd):
        self.root = root
        self.cal = calibration
        self.viewer = viewer
        self.source_vpd = source_vpd
        self.rows = {}
        self.error_labels = []

        root.title('%s - paired pose' % calibration.profile.name)
        outer = ttk.Frame(root, padding=6)
        outer.pack(fill='both', expand=True)

        # --- scrolling joint list ---------------------------------------------------------
        holder = ttk.Frame(outer)
        holder.pack(fill='both', expand=True)
        canvas = tk.Canvas(holder, width=420, height=520, highlightthickness=0)
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
        ttk.Label(header, text='#', width=4).pack(side='left')
        ttk.Label(header, text='joint', width=30).pack(side='left')
        ttk.Label(header, text='deg', width=9).pack(side='left')
        ttk.Label(header, text='range', width=15).pack(side='left')

        for index, joint in enumerate(calibration.joints):
            self.add_row(inner, index, joint)

        # --- live errors -------------------------------------------------------------------
        box = ttk.LabelFrame(outer, text='segment direction error (deg)', padding=4)
        box.pack(fill='x', pady=(6, 0))
        for segment in calibration.segments:
            label = ttk.Label(box, text='%-28s  --' % ('%s -> %s' % (segment[2], segment[3])))
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
        ttk.Label(row, text=joint, width=30).pack(side='left')
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
        self.rows[joint] = (slider, entry)

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
        typed = max(low, min(high, typed))
        slider.set(typed)
        self.on_slide(joint)

    def push(self):
        for joint, (slider, entry) in self.rows.items():
            slider.set(self.cal.values[joint])
            entry.delete(0, 'end')
            entry.insert(0, '%.1f' % self.cal.values[joint])

    def refresh(self):
        errors = self.cal.errors()
        for label, segment, error in zip(self.error_labels, self.cal.segments, errors):
            label.config(text='%-28s %6.2f' % ('%s -> %s' % (segment[2], segment[3]), error))
        self.mean_error = float(np.nanmean(errors))

    def on_solve(self):
        mean = self.cal.solve()
        self.push()
        self.refresh()
        moved = sorted(((abs(self.cal.values[j] - self.initial[j]), j) for j in self.cal.joints),
                       reverse=True)[:3]
        self.root.title('%s - paired pose   (solve %.3f deg, biggest moves: %s)'
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
        payload = self.cal.payload(self.source_vpd)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
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
    parser.add_argument('--robot', required=True, help='robot profile, e.g. configs/robots/g1.json')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--pose', default=None, help='start from this paired pose')
    parser.add_argument('--distance', type=float, default=2.6)
    parser.add_argument('--azimuth', type=float, default=180.0)
    parser.add_argument('--lookat', default='0 0 0.85')
    args = parser.parse_args()

    profile = robot_profile.load(args.robot, args.gmr_root)
    robot_check = mujoco.MjModel.from_xml_path(str(profile.xml))
    problems = robot_profile.validate(profile, robot_check, mujoco)
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
