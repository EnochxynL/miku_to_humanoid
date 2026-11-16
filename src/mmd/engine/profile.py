"""The MMD side of the correspondence: what "the same pose" means, and how human bone names read.

Nothing here depends on a robot. Two pieces of knowledge live here:

* ``SEGMENT_BONES``  which limb directions define a pose. Directions are the only convention-free
                    comparison available - orientations carry the C being calibrated - so a paired pose
                    is "these eight directions agree". This is a property of the MMD (human) skeleton.
* ``LAFAN1_TO_MMD``  how GMR's human naming (LaFAN1: Hips, LeftUpLeg, LeftArm ...) reads as MMD bone
                    names. Both sides describe the same human, so this bridge is source-side knowledge
                    too; it only ever runs while reading a robot profile.

``NEXT_JOINT`` stays here as reference: the next real joint after a bone, skipping the twist (捩), sleeve
chain and shoulder-helper bones MMD inserts. The segment table below states the resulting pairs directly,
so nothing consumes ``NEXT_JOINT`` any more - but it explains *why* those pairs skip bones, which a new
MMD rig will otherwise have to rediscover.
"""
from __future__ import annotations

# the three bones the calibration needs by name: the root (GMR's human_root_name), the torso (the second
# link a limb pose cannot move) and the scale reference (head height gives the body-size factor)
ROOT_BONE = '腰'
TORSO_BONE = '上半身2'
SCALE_BONE = '頭'

# the two thigh bones: their line gives the world-frame yaw of the links a pose cannot move
BILATERAL_BONES = ('左足', '右足')

# the limb directions a paired pose compares, as (bone, far bone); duplicated corners are deliberate:
# both arms and both legs are checked, a missing bone just drops that segment
SEGMENT_BONES = (
    ('左腕', '左ひじ'), ('左ひじ', '左手首'),
    ('右腕', '右ひじ'), ('右ひじ', '右手首'),
    ('左足', '左ひざ'), ('左ひざ', '左足首'),
    ('右足', '右ひざ'), ('右ひざ', '右足首'),
)

# GMR names every human skeleton with LaFAN1 bones, so one table serves all of its robots
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

# reference only, kept for the explanation above
NEXT_JOINT = {
    '左腕': '左ひじ',
    '右腕': '右ひじ',
    '左足': '左ひざ',
    '右足': '右ひざ',
    '左ひじ': '左手首',
    '右ひじ': '右手首',
    '左ひざ': '左足首',
    '右ひざ': '右足首',
}


# ---------------------------------------------------------------------------- robot profiles
# A profile is an authored file, not something derived at run time: it holds the pairing, which limb
# directions to compare, and the engine template this robot needs. A generator script can produce one by
# reading GMR's checkout (``scripts/get_robot_profile_from_gmr.py``), but nothing here knows GMR exists.
import copy                                            # noqa: E402
import json                                            # noqa: E402
from dataclasses import dataclass                      # noqa: E402
from pathlib import Path                               # noqa: E402


@dataclass(frozen=True)
class Segment:
    """One limb the paired pose must align."""

    link: str
    to: str


@dataclass(frozen=True)
class RobotProfile:
    name: str
    xml: Path                      # the robot's MJCF
    links: dict                    # robot link -> MMD bone
    segments: tuple                # Segment list: which link pairs are compared
    root_links: tuple              # links a limb pose cannot move (they need the bilateral yaw)
    bilateral: tuple               # (left, right) link whose line gives that yaw
    template: dict = None          # the engine config, complete and self-contained


def load(path):
    """Read a profile file. Paths inside it may be absolute or relative to the file."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    missing = [key for key in ('name', 'xml', 'links', 'segments') if key not in raw]
    if missing:
        raise SystemExit('%s: missing %s' % (path, ', '.join(missing)))

    xml = Path(raw['xml'])
    if not xml.is_absolute():
        xml = (path.parent / xml).resolve()

    return RobotProfile(
        name=str(raw['name']),
        xml=xml,
        links={str(link): str(bone) for link, bone in raw['links'].items()},
        segments=tuple(Segment(str(item['link']), str(item['to'])) for item in raw['segments']),
        root_links=tuple(raw.get('root_links', ())),
        bilateral=tuple(raw.get('bilateral_links', ())),
        template=copy.deepcopy(raw.get('template') or {}),
    )


def dump(profile):
    """The complete profile as plain data - what a generator writes and a human reads."""
    return {
        'name': profile.name,
        'xml': str(profile.xml),
        'links': profile.links,
        'segments': [{'link': segment.link, 'to': segment.to} for segment in profile.segments],
        'root_links': list(profile.root_links),
        'bilateral_links': list(profile.bilateral),
        'template': profile.template,
    }


def save(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump(profile), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def validate(profile, model, mujoco):
    """Problems that would otherwise show up as a silently wrong calibration."""
    problems = []
    if len(profile.bilateral) != 2:
        problems.append('bilateral_links needs exactly two links (the hip line), got %d'
                        % len(profile.bilateral))
    for link in profile.bilateral:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) < 0:
            problems.append('bilateral link not in the model: %s' % link)
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
    return problems
