# ---------------------------------------------------------------------------- robot profiles
# A profile is an authored file, not something derived at run time: it holds the pairing, which limb
# directions to compare, and the engine template this robot needs. A generator script can produce one by
# reading GMR's checkout, but nothing here knows GMR exists.
from dataclasses import dataclass                      # noqa: E402
from pathlib import Path                               # noqa: E402
import copy                                            # noqa: E402
import json                                            # noqa: E402


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
    template: dict = None          # engine config, complete and self-contained


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
