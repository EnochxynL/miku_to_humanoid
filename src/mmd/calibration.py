"""The calibration itself, without any window: one robot pose judged against one MMD pose.

Two front ends use this - the GUI lets a human drag the sliders, the script solves numerically - so the
objective exists exactly once. Nothing here knows the robot: the links, their MMD partners and the
segments come from a profile, the joints come from the compiled model.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import view
from engine import mjcf
from engine.bake import bake

CHARACTER_ALPHA = 0.25


class Calibration:
    """A composed scene (character + robot at the same origin), the joints, and the segment errors."""

    def __init__(self, pmx, vpd, profile, pose_path=None):
        self.profile = profile
        self.bundle = mjcf.build(pmx)
        self.model = view.build_scene(self.bundle.spec, profile.xml, 0.0, None)
        self.data = mujoco.MjData(self.model)

        # the character is the target: keep it, fade it, and freeze it at the pose being matched
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            if not name.startswith(view.ROBOT_PREFIX):
                rgba = np.array(self.model.geom_rgba[geom], dtype=float)
                rgba[3] = CHARACTER_ALPHA
                self.model.geom_rgba[geom] = rgba

        self.mmd_slices, _ = view.joint_slices(self.model, self.bundle.model)
        if vpd:
            clip, _u, _t, _m = bake(pmx, vpd, self.bundle.model)
            self.mmd_qpos = clip[0]
        else:
            self.mmd_qpos = np.array(self.bundle.model.qpos0)   # no pose given: match the bind pose

        # every one-degree-of-freedom robot joint, in model order: a free joint owns seven qpos slots
        # and a ball joint four, so a single value per joint would corrupt them
        self.joints = []
        self.address = {}
        self.limits = {}
        for joint in range(self.model.njnt):
            full = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ''
            if not full.startswith(view.ROBOT_PREFIX):
                continue
            if int(self.model.jnt_type[joint]) not in (mujoco.mjtJoint.mjJNT_HINGE,
                                                       mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            short = full[len(view.ROBOT_PREFIX):]
            low, high = (float(v) for v in self.model.jnt_range[joint])
            if low >= high:
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
            raise SystemExit('the profile defines no usable segments (both links must map to bones)')

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
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                 (view.ROBOT_PREFIX + name) if robot else name)

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

    def mean_error(self):
        return float(np.nanmean(self.errors()))

    def solve(self):
        """L-BFGS-B over every joint, seeded from the current values, with the slider bounds."""
        from scipy.optimize import minimize

        names = list(self.joints)
        x0 = np.radians([self.values[joint] for joint in names])
        bounds = [tuple(np.radians(self.limits[joint])) for joint in names]

        def cost(x):
            for joint, value in zip(names, x):
                self.data.qpos[self.address[joint]] = value
            mujoco.mj_forward(self.model, self.data)
            return float(np.sum(np.square(self.errors())))

        result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12})
        for joint, value in zip(names, result.x):
            self.values[joint] = float(np.degrees(value))
        self.apply_robot()
        return self.mean_error()

    # ------------------------------------------------------------------ output
    def payload(self, source_pose=None):
        errors = self.errors()
        return {
            'robot': {'name': self.profile.name, 'xml': Path(self.profile.xml).name,
                      'base_config': Path(self.profile.base_config).name},
            'note': ('paired pose: joint angles in degrees, consumed by gmr_calibrate, which derives C '
                     'from them'),
            'source_pose': str(source_pose) if source_pose else None,
            'joints': {joint: round(value, 4) for joint, value in self.values.items()},
            'limb_direction_error_deg': round(float(np.nanmean(errors)), 3),
            'per_segment_error_deg': {'%s-%s' % (segment[2], segment[3]): round(error, 3)
                                      for segment, error in zip(self.segments, errors)},
        }

    def save(self, path, source_pose=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payload(source_pose)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return payload
