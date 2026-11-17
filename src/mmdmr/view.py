"""One window for looking at the retargeting: the MMD character, the robot that followed it, and
optionally the source skeleton drawn as capsules.

Everything is read from files the pipeline already wrote - this module never imports GMR and never
retargets anything. That separation is the point: ``retarget.py`` produces ``g1.npz``,
``engine.mjcf`` builds the character in memory and ``engine.bake`` turns a pose into qpos;
this file only composes and plays them.

One window shows both figures, which is possible because MuJoCo can hold several kinematic trees in a
single model: the robot is attached into the MMD model's world with its own ``g1_`` prefix
(``MjSpec.attach``). Two *separate* models cannot share a viewer, because a viewer is bound to one
model and mesh geometry is referenced inside that model's own mesh table.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/view.py \
        --pmx <model.pmx> --pose <pose.vpd> --robot-qpos runtime/pmx2/g1.npz --offset 1.6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# The robot model lives in GMR's checkout; keep it as a default so this file needs no GMR import.
DEFAULT_ROBOT_XML = str(Path(__file__).resolve().parents[2] / 'assets' / 'unitree_g1'
                       / 'g1_mocap_29dof.xml')
ROBOT_PREFIX = 'g1_'
MMD_BONE_COLOUR = (0.98, 0.62, 0.16, 1.0)
ROBOT_BONE_COLOUR = (0.20, 0.85, 0.95, 1.0)
WIDTH = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
         mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}


# --------------------------------------------------------------------------------------- scene
def strip_environment(spec):
    """Drop the ground plane and lights of one model so the other one's can be used instead.

    Both MJCFs bring their own floor and light. Two coincident planes z-fight into stripes, and two
    directional lights double the exposure, so exactly one environment survives - the robot's.
    """
    removed = []
    for geom in list(spec.worldbody.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            removed.append(geom.name or 'plane')
            spec.delete(geom)
    for light in list(spec.worldbody.lights):
        removed.append(light.name or 'light')
        spec.delete(light)
    return removed


def build_scene(mmd_xml, robot_xml, offset, ghost_alpha=None):
    """Compose the MMD model and, optionally, the robot into one model.

    Returns the composed model plus the joint slices each source model writes into.
    """
    if isinstance(mmd_xml, mujoco.MjSpec):
        mmd = mmd_xml
    else:
        mmd = mujoco.MjSpec.from_file(str(mmd_xml)) if mmd_xml else mujoco.MjSpec()
    robot = mujoco.MjSpec.from_file(str(robot_xml)) if robot_xml else None
    if robot is not None and mmd_xml:
        # the robot's floor and lights win: its MJCF is the one that describes the working environment
        dropped = strip_environment(mmd)
        print('  stripped from the MMD model: %s' % ', '.join(dropped))
    if robot is not None:
        # attachment position is irrelevant for a free-jointed model: the clip carries it (see
        # QposOverlay.translate), so the frame only gives the subtree a parent
        frame = mmd.worldbody.add_frame()
        mmd.attach(robot, frame=frame, prefix=ROBOT_PREFIX)
    model = mmd.compile()

    if ghost_alpha is not None:
        # translucent skin on both figures so the bones inside stay visible; the ground plane is left
        # alone, and where each figure stands is still decided by --offset
        for geom in range(model.ngeom):
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            rgba = np.array(model.geom_rgba[geom], dtype=float)
            rgba[3] = ghost_alpha
            model.geom_rgba[geom] = rgba
    return model


def joint_slices(composed, source, prefix=''):
    """(destination address, source address, width) for every joint of one source model.

    ``source`` is an MjModel (the in-memory build) or a path to an MJCF.

    A joint can be **unnamed** - ``<freejoint/>`` writes one with no name, and T1's model does exactly
    that. Those still have to be matched, or a whole free base (seven qpos slots) silently drops out: the
    robot would stand at the origin no matter what the clip or the placement says. They are matched by the
    body they belong to, which does carry the prefix.
    """
    if not isinstance(source, mujoco.MjModel):
        source = mujoco.MjModel.from_xml_path(str(source))

    def body_of(model, joint):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint]))

    pairs = []
    for joint in range(composed.njnt):
        name = mujoco.mj_id2name(composed, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name is None:
            body = body_of(composed, joint)
            if body is None:
                continue
            raw = body[len(prefix):] if prefix and body.startswith(prefix) else body
            src = next((other for other in range(source.njnt)
                        if body_of(source, other) == raw
                        and int(source.jnt_type[other]) == int(composed.jnt_type[joint])), -1)
        else:
            if prefix and not name.startswith(prefix):
                continue
            raw = name[len(prefix):] if prefix else name
            src = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, raw)
        if src < 0:
            continue
        pairs.append((int(composed.jnt_qposadr[joint]), int(source.jnt_qposadr[src]),
                      WIDTH[mujoco.mjtJoint(composed.jnt_type[joint])]))
    return pairs, source.nq


# ------------------------------------------------------------------------------------- overlays
class Overlay:
    """Anything that writes part of the composed qpos, or adds geometry, once per frame."""

    frames = 0

    def prepare(self, model):
        pass

    def apply(self, qpos, data, scene, index):
        raise NotImplementedError


class SceneOverlay(Overlay):
    """An overlay that only appends geometry, after forward kinematics."""


class QposOverlay(Overlay):
    """Drive one attached model's subtree from a qpos clip."""

    def __init__(self, label, source_model, clip, prefix='', translate=(0.0, 0.0, 0.0)):
        self.label = label
        self.source_xml = source_model
        self.clip = clip
        self.prefix = prefix
        self.frames = len(self.clip)
        # A model whose root is a free joint gets its world position from qpos, not from where the
        # spec was attached - so placing it beside another figure means translating the clip.
        self.translate = np.asarray(translate, dtype=float)
        self.free_slice = None

    def prepare(self, model):
        self.slices, self.width = joint_slices(model, self.source_xml, self.prefix)
        if not self.slices:
            raise SystemExit('%s: no joints matched in the composed model' % self.label)
        for local, (destination, source, width) in enumerate(self.slices):
            if width == 7:                      # a free joint: 3 positions then a quaternion
                self.free_slice = (destination, source, local)
                break
        print('  %-4s %5d frames, %2d joint slices' % (self.label, self.frames, len(self.slices)))

    def apply(self, qpos, data, scene, index):
        frame = self.clip[index % self.frames]
        for destination, source, width in self.slices:
            qpos[destination:destination + width] = frame[source:source + width]
        if self.free_slice is not None and np.any(self.translate):
            destination, _source, _local = self.free_slice
            qpos[destination:destination + 3] += self.translate


class SkeletonOverlay(SceneOverlay):
    """Draw every bone of one rig: a capsule towards the child, a sphere when there is no length.

    The class describes a single style; which rig it draws is the ``select`` predicate, so two
    instances give the MMD skeleton and the robot skeleton their own thickness and colour. No name
    filtering by default - MMD twists sit exactly on their parent and would vanish otherwise.
    """

    def __init__(self, select=None, exclude='', radius=0.006, rgba=(0.98, 0.62, 0.16, 1.0)):
        self.select = select or (lambda name: True)
        self.skip = [s for s in exclude.split(',') if s]
        self.radius = radius
        self.rgba = np.array(rgba)
        self.edges = []

    def prepare(self, model):
        name_of = {body: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '')
                   for body in range(model.nbody)}
        for body in range(1, model.nbody):
            name = name_of[body]
            if not name or not self.select(name):
                continue
            if any(fragment in name for fragment in self.skip):
                continue
            parent = int(model.body_parentid[body])
            if not name_of.get(parent) or name_of[parent] == 'world':
                continue
            self.edges.append((parent, body))
        print('  skel %d bones (%s)' % (len(self.edges), 'robot' if self.radius >= 0.01 else 'mmd'))

    def apply(self, qpos, data, scene, index):
        # append: user_scn starts empty every frame, but a scene built by the renderer already holds
        # the model's geoms and must not be wiped
        for parent, body in self.edges:
            if scene.ngeom >= scene.maxgeom:
                break
            start = np.array(data.xpos[parent])
            end = np.array(data.xpos[body])
            geom = scene.geoms[scene.ngeom]
            if np.linalg.norm(end - start) < 1e-4:
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.array([self.radius * 1.6, 0.0, 0.0]),
                                    pos=start, mat=np.eye(3).flatten(), rgba=self.rgba)
            else:
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    size=np.array([self.radius, 0.0, 0.0]),
                                    pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=self.rgba)
                mujoco.mjv_connector(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     width=self.radius, from_=start, to=end)
            scene.ngeom += 1


class ChainOverlay(SceneOverlay):
    """Draw a skeleton that has no bodies behind it: explicit world points per frame.

    This is how the retargeting engine's *intermediate* representation is shown - the selected, scaled
    and offset-corrected target poses it actually aims at. They are plain arrays, so a capsule is built
    from two points instead of from ``data.xpos``.
    """

    def __init__(self, points, edges, radius=0.010, rgba=(0.25, 0.95, 0.35, 1.0),
                 translate=(0.0, 0.0, 0.0)):
        self.points = points              # {bone name: array (frames, 3)}
        self.edges = tuple(edges)         # ((bone, bone), ...)
        self.radius = radius
        self.rgba = np.array(rgba)
        self.translate = np.asarray(translate, dtype=float)
        self.frames = len(next(iter(points.values()))) if points else 0

    def prepare(self, model):
        missing = [name for edge in self.edges for name in edge if name not in self.points]
        if missing:
            raise SystemExit('chain overlay is missing bones: %s' % ', '.join(sorted(set(missing))))
        print('  chain %d edges (%d bones, %d frames)'
              % (len(self.edges), len(self.points), self.frames))

    def apply(self, qpos, data, scene, index):
        for start_name, end_name in self.edges:
            if scene.ngeom >= scene.maxgeom:
                break
            start = self.points[start_name][index % self.frames] + self.translate
            end = self.points[end_name][index % self.frames] + self.translate
            geom = scene.geoms[scene.ngeom]
            if np.linalg.norm(end - start) < 1e-4:
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.array([self.radius * 1.6, 0.0, 0.0]),
                                    pos=start, mat=np.eye(3).flatten(), rgba=self.rgba)
            else:
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                    size=np.array([self.radius, 0.0, 0.0]),
                                    pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=self.rgba)
                mujoco.mjv_connector(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                     width=self.radius, from_=start, to=end)
            scene.ngeom += 1


# --------------------------------------------------------------------------------------- display
class BaseDisplay:
    """The window, the camera and the clock. Overlays supply the content."""

    def __init__(self, model, speed=0.6, loop=False, distance=3.0, azimuth=180.0,
                 lookat=(0.6, 0.0, 0.95), show_ui=False, stride=1):
        self.model = model
        self.data = mujoco.MjData(model)
        self.speed = speed
        self.loop = loop
        self.camera = (distance, azimuth, lookat)
        self.show_ui = show_ui
        self.stride = max(1, stride)
        mujoco.mj_forward(self.model, self.data)

    def run(self, overlays):
        model, data = self.model, self.data
        frames = max(1, min((o.frames for o in overlays if o.frames), default=1))
        distance, azimuth, lookat = self.camera
        with mujoco.viewer.launch_passive(model, data, show_left_ui=self.show_ui,
                                          show_right_ui=self.show_ui) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = lookat
            viewer.cam.distance = distance
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = -8
            index = 0
            while viewer.is_running():
                started = time.perf_counter()
                # qpos overlays write first, forward kinematics second, scene drawing last
                for overlay in overlays:
                    if isinstance(overlay, QposOverlay):
                        overlay.apply(data.qpos, data, viewer.user_scn, index)
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                for overlay in overlays:
                    if isinstance(overlay, SkeletonOverlay):
                        overlay.apply(data.qpos, data, viewer.user_scn, index)
                viewer.sync()
                index = (index + self.stride) % frames
                if not self.loop and index == 0:
                    return
                time.sleep(max(0.0, 1.0 / 30.0 / self.speed - (time.perf_counter() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', default=None, help='the character model, built in memory')
    parser.add_argument('--pose', default=None,
                        help='what the character plays: a .vpd (one frame) or a .vmd (a clip)')
    parser.add_argument('--robot', default=None,
                        help='robot profile, e.g. configs/robots/g1.json; '
                             'overrides --robot-xml')
    parser.add_argument('--robot-xml', default=DEFAULT_ROBOT_XML)
    parser.add_argument('--robot-qpos', default=None, help='the retargeted robot clip (retarget)')
    parser.add_argument('--ghost', action='store_true',
                        help='render both figures translucent so the bones inside show through; where '
                             'they stand still comes from --offset')
    parser.add_argument('--ghost-alpha', type=float, default=0.30)
    parser.add_argument('--offset', type=float, default=1.6, help='where the robot stands (metres)')
    parser.add_argument('--skeleton', action='store_true',
                        help='draw every bone of both rigs (capsule to the child, sphere if it has none)')
    parser.add_argument('--skeleton-exclude', default='',
                        help='bone name fragments to skip; empty by default so the skeleton is complete')
    parser.add_argument('--skeleton-radius-mmd', type=float, default=0.002, help='MMD bone radius')
    parser.add_argument('--skeleton-radius-robot', type=float, default=0.010, help='robot bone radius')
    parser.add_argument('--speed', type=float, default=0.6)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--distance', type=float, default=3.2)
    parser.add_argument('--azimuth', type=float, default=180.0)
    parser.add_argument('--lookat', default='0.6 0 0.95')
    parser.add_argument('--show-ui', action='store_true')
    args = parser.parse_args()
    if args.robot:
        from .config import load as load_profile
        args.robot_xml = str(load_profile(args.robot).xml)

    if not args.robot_qpos and not args.pmx:
        raise SystemExit('nothing to play: pass --robot-qpos and/or --pmx')
    if args.pose and not args.pmx:
        raise SystemExit('--pose needs --pmx')

    from .mmd import mjcf
    from .mmd.bake import bake

    bundle = mjcf.build(args.pmx) if args.pmx else None
    if bundle is not None:
        print('  character: %d bones, %d mesh groups, %d textures'
              % (bundle.info['bones'], bundle.info['mesh_groups'], bundle.info['textures']))
    robot_model = (mujoco.MjModel.from_xml_path(args.robot_xml) if args.robot_qpos else None)

    print('composing scene')
    model = build_scene(bundle.spec if bundle else None,
                        args.robot_xml if args.robot_qpos else None,
                        args.offset, args.ghost_alpha if args.ghost else None)
    print('  nbody=%d nmesh=%d nq=%d' % (model.nbody, model.nmesh, model.nq))

    overlays = []
    if bundle is not None:
        if args.pose:
            clip, unmatched, _times, missing = bake(args.pmx, args.pose, bundle.model)
            print('  motion: %d frames from %s (%d bones matched)'
                  % (len(clip), Path(args.pose).name, len(bundle.info['names']) - len(unmatched)))
        else:
            clip = np.array([bundle.model.qpos0])
            print('  motion: bind pose')
        overlays.append(QposOverlay('mmd', bundle.model, clip))
    if args.robot_qpos:
        overlays.append(QposOverlay('g1', robot_model, np.load(args.robot_qpos)['qpos'],
                                    prefix=ROBOT_PREFIX,
                                    translate=(args.offset, 0.0, 0.0)))
    if args.skeleton:
        mmd_only = lambda name: not name.startswith(ROBOT_PREFIX)
        robot_only = lambda name: name.startswith(ROBOT_PREFIX)
        overlays.append(SkeletonOverlay(select=mmd_only, exclude=args.skeleton_exclude,
                                        radius=args.skeleton_radius_mmd, rgba=MMD_BONE_COLOUR))
        overlays.append(SkeletonOverlay(select=robot_only, exclude='',
                                        radius=args.skeleton_radius_robot, rgba=ROBOT_BONE_COLOUR))
    for overlay in overlays:
        overlay.prepare(model)

    lookat = [float(v) for v in args.lookat.split()]
    BaseDisplay(model, speed=args.speed, loop=args.loop, distance=args.distance,
                azimuth=args.azimuth, lookat=lookat, show_ui=args.show_ui,
                stride=args.stride).run(overlays)


if __name__ == '__main__':
    main()
