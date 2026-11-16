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


# ---------------------------------------------------------------------------- C, the calibration readout
def _rotation(quaternion):
    """world_poses quaternions are (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    w, x, y, z = (float(v) for v in quaternion)
    return Rotation.from_quat([x, y, z, w])


def read_c(pmx, robot_model, profile, pose_path, mdd_pose=None):
    """Read C for every linked pair: ``C = R_source(P)^-1 · R_link(P)``, and the uniform scale.

    This is engine-neutral data - the fixed frame relation between an MMD bone and a robot link at one
    paired pose, plus the body scale. Turning it into a particular engine's config is that engine's job.

    Two links are not moved by a limb pose (the floating root and the torso), so their world-frame yaw
    comes from a *bilateral* axis: the profile names the two hip links, and the source side uses the two
    thigh bones.
    """
    from engine.bake import bind_pose, world_poses
    from engine.profile import BILATERAL_BONES, ROOT_BONE, SCALE_BONE, TORSO_BONE

    wanted = sorted(set(profile.links.values()) | {ROOT_BONE, TORSO_BONE, SCALE_BONE})
    bind_frame = bind_pose(pmx, wanted)[0]
    if mdd_pose:
        source = world_poses(pmx, mdd_pose, wanted)[0][0]
    else:
        source = bind_frame

    def position(bone):
        return np.asarray(source[bone][0], dtype=float)

    data = mujoco.MjData(robot_model)
    mujoco.mj_forward(robot_model, data)
    robot_height = float(data.xpos[:, 2].max())
    source_height = float(bind_frame[SCALE_BONE][0][2])
    scale = robot_height / source_height if source_height > 1e-6 else 0.9

    supplied = json.loads(Path(pose_path).read_text(encoding='utf-8'))
    angles = supplied.get('joints', supplied)
    unknown = [name for name in angles
               if mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0]
    if unknown:
        raise SystemExit('pose file names joints this robot does not have: %s' % unknown)
    data.qpos[:] = robot_model.qpos0
    for name, value in angles.items():
        joint = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(robot_model.jnt_qposadr[joint])] = np.radians(value)
    mujoco.mj_forward(robot_model, data)

    def body(name):
        return mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, name)

    left_hip, right_hip = profile.bilateral
    robot_hip = np.array(data.xpos[body(right_hip)]) - np.array(data.xpos[body(left_hip)])
    robot_hip /= np.linalg.norm(robot_hip)
    source_hip = position(BILATERAL_BONES[1]) - position(BILATERAL_BONES[0])
    source_hip /= np.linalg.norm(source_hip)
    delta = float(np.arctan2(np.cross(robot_hip, source_hip)[2], robot_hip @ source_hip))
    from scipy.spatial.transform import Rotation
    root_fix = Rotation.from_euler('z', np.degrees(delta), degrees=True)

    links = {}
    for link, bone in profile.links.items():
        if bone not in source or body(link) < 0:
            continue
        constant = _rotation(source[bone][1]).inv() * Rotation.from_matrix(
            np.array(data.xmat[body(link)]).reshape(3, 3))
        if link in profile.root_links:
            constant = constant * root_fix
        q = constant.as_quat()                       # scipy: (x, y, z, w)
        links[link] = {'bone': bone,
                       'C_quat_wxyz': [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                       'angle_deg': round(float(np.degrees(constant.magnitude())), 4)}

    return {
        'robot': {'name': profile.name, 'xml': Path(profile.xml).name,
                  'base_config': Path(profile.base_config).name},
        'method': 'paired pose -> C = R_source(P)^-1 * R_link(P)',
        'why': ('MMD bones carry no rest rotation, so at the paired pose every bone frame is the identity '
                'and C collapses to a reading rather than a fit'),
        'scale': float(scale),
        'heights': {'source': float(source_height), 'robot': float(robot_height)},
        'root_yaw_deg': round(float(np.degrees(delta)), 3),
        'root_links': list(profile.root_links),
        'pose_file': str(pose_path),
        'source_pose': str(mdd_pose) if mdd_pose else None,
        'links': links,
    }


def save_c(path, c):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def main():
    import argparse

    import robot_profile

    parser = argparse.ArgumentParser(
        description='Read C from a paired pose and write it as engine-neutral data.')
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--robot', required=True, help='robot profile json, or a GMR robot key')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--pose', required=True, help='paired pose (joint angles in degrees)')
    parser.add_argument('--vpd', default=None, help='the MMD pose the angles were solved for')
    parser.add_argument('--out', required=True, help='where to write the neutral calibration')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()

    profile = robot_profile.load(args.robot, args.gmr_root)
    model = mujoco.MjModel.from_xml_path(str(profile.xml))
    problems = robot_profile.validate(profile, model, mujoco)
    if problems:
        raise SystemExit('profile does not match the model:\n  ' + '\n  '.join(problems))

    c = read_c(args.pmx, model, profile, args.pose, args.vpd)
    save_c(args.out, c)

    if args.report:
        print('robot %s   scale %.4f   root yaw %.2f deg   links %d'
              % (c['robot']['name'], c['scale'], c['root_yaw_deg'], len(c['links'])))
        for link, info in c['links'].items():
            print('  %-26s %-10s C %7.2f deg' % (link, info['bone'], info['angle_deg']))
    print('written %s' % args.out)


if __name__ == '__main__':
    raise SystemExit(main())
