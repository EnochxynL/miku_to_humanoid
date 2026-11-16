"""One robot's side of the correspondence, loaded from a profile file.

A profile answers the two questions that are genuinely robot-specific:

* ``links``    which of this robot's links tracks which MMD bone (this is what GMR stores in the
               ``bone`` field of its config, so it is the pairing that has to exist somewhere);
* ``segments`` which link pairs and joints define "the same pose". Directions are the only
               convention-free comparison, so a segment is (link, far end, the joints that move it).

Everything else - path resolution, the stock weights, the MMD side of the skeleton - comes from GMR or
from ``mmd_bones``. Paths in the file are relative to the GMR checkout, so a profile is portable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

WEIGHT_FIELD = {'pos_w': 1, 'rot_w': 2}


@dataclass(frozen=True)
class Segment:
    """One limb the paired pose must align."""

    link: str
    to: str
    joints: tuple


@dataclass(frozen=True)
class RobotProfile:
    name: str
    xml: Path
    base_config: Path
    links: dict            # robot link -> MMD bone
    segments: tuple
    root_links: tuple      # links the paired pose cannot move: they need a bilateral yaw reference
    weights: dict          # link -> {field: value}, deviations from the stock config

    def joints(self):
        """The joints a calibration may adjust, in segment order, without repeats."""
        order = []
        for segment in self.segments:
            for joint in segment.joints:
                if joint not in order:
                    order.append(joint)
        return order

    def segments_for(self, link):
        return [segment for segment in self.segments if segment.link == link]


# GMR's configs all name the human side with LaFAN1 bones, so one table serves every robot.
LAFAN1_TO_MMD = {
    'Hips': '腰',
    'Spine2': '上半身2',
    'LeftUpLeg': '左足',
    'LeftLeg': '左ひざ',
    'LeftFootMod': '左足首',
    'RightUpLeg': '右足',
    'RightLeg': '右ひざ',
    'RightFootMod': '右足首',
    'LeftArm': '左腕',
    'LeftForeArm': '左ひじ',
    'LeftHand': '左手首',
    'RightArm': '右腕',
    'RightForeArm': '右ひじ',
    'RightHand': '右手首',
}

# the limb directions the paired pose compares, as (bone, far bone)
SEGMENT_BONES = (
    ('左腕', '左ひじ'), ('左ひじ', '左手首'),
    ('右腕', '右ひじ'), ('右ひじ', '右手首'),
    ('左足', '左ひざ'), ('左ひざ', '左足首'),
    ('右足', '右ひざ'), ('右ひざ', '右足首'),
)


def from_gmr(robot, gmr_root, model=None):
    """Build a profile for any robot GMR already knows, from its stock config.

    ``robot`` is GMR's key (``unitree_g1``, ``booster_t1_29dof`` ...). GMR's XML dictionary lives inside a
    package that imports ``rich``, so the MJCF is found on disk instead, and the config is matched by
    content: every link it names must exist in that MJCF, and every body must be a LaFAN1 bone.
    """
    import mujoco

    gmr = Path(gmr_root)
    assets = gmr / 'assets' / robot
    xmls = sorted(assets.glob('*.xml'), key=lambda path: ('mocap' not in path.name, path.name))
    if not xmls:
        raise SystemExit('no MJCF under %s' % assets)
    xml = xmls[0]
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(xml))

    best, best_score = None, -1
    for config_path in sorted((gmr / 'general_motion_retargeting' / 'ik_configs').glob('*.json')):
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            table = config['ik_match_table1']
        except Exception:
            continue
        if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) < 0 for link in table):
            continue
        if any(str(entry[0]) not in LAFAN1_TO_MMD for entry in table.values()):
            continue
        score = len(table)
        if score > best_score:
            best, best_score, config_path_used = config, score, config_path
    if best is None:
        raise SystemExit('no stock config under %s fits %s'
                         % (gmr / 'general_motion_retargeting' / 'ik_configs', xml.name))

    links = {link: LAFAN1_TO_MMD[str(entry[0])] for link, entry in best['ik_match_table1'].items()}
    by_bone = {bone: link for link, bone in links.items()}
    segments = tuple(Segment(by_bone[start], by_bone[stop], ())
                     for start, stop in SEGMENT_BONES
                     if start in by_bone and stop in by_bone)
    roots = tuple(by_bone[bone] for bone in ('腰', '上半身2') if bone in by_bone)
    return RobotProfile(name=robot, xml=xml, base_config=config_path_used, links=links,
                        segments=segments, root_links=roots, weights={})


def load(path, gmr_root):
    """Read a profile - a JSON file, or a robot key that is generated from GMR's stock config."""
    candidate = Path(str(path))
    if not candidate.exists():
        return from_gmr(str(path), gmr_root)
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    gmr = Path(gmr_root)
    missing = [key for key in ('name', 'xml', 'base_config', 'links', 'segments') if key not in raw]
    if missing:
        raise SystemExit('%s: missing %s' % (path, ', '.join(missing)))

    def resolve(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else gmr / candidate

    segments = tuple(Segment(str(item['link']), str(item['to']), tuple(item['joints']))
                     for item in raw['segments'])
    return RobotProfile(
        name=str(raw['name']),
        xml=resolve(raw['xml']),
        base_config=resolve(raw['base_config']),
        links={str(link): str(bone) for link, bone in raw['links'].items()},
        segments=segments,
        root_links=tuple(raw.get('root_links', ())),
        weights={str(link): {str(field): float(value) for field, value in fields.items()}
                 for link, fields in raw.get('weight_overrides', {}).items()},
    )


def validate(profile, model, mujoco):
    """Problems that would otherwise show up as a silently wrong calibration."""
    problems = []
    for link in profile.links:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) < 0:
            problems.append('link not in the model: %s' % link)
    for link in profile.root_links:
        if link not in profile.links:
            problems.append('root link without a bone: %s' % link)
    for segment in profile.segments:
        if segment.link not in profile.links:
            problems.append('segment start without a bone: %s' % segment.link)
        for name in (segment.link, segment.to):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                problems.append('segment body not in the model: %s' % name)
        for joint in segment.joints:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) < 0:
                problems.append('segment joint not in the model: %s' % joint)
    for link, fields in profile.weights.items():
        if link not in profile.links:
            problems.append('weight override for an unmapped link: %s' % link)
        unknown = [field for field in fields if field not in WEIGHT_FIELD]
        if unknown:
            problems.append('unknown weight field on %s: %s' % (link, ', '.join(unknown)))
    return problems
