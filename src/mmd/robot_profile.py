"""One robot's side of the correspondence, loaded from a profile file.

A profile answers the two questions that are genuinely robot-specific:

* ``links``    which of this robot's links tracks which MMD bone (this is what GMR stores in the
               ``bone`` field of its config, so it is the pairing that has to exist somewhere);
* ``segments`` which link pairs and joints define "the same pose". Directions are the only
               convention-free comparison, so a segment is (link, far end, the joints that move it).

Everything else - path resolution, the stock weights, the MMD side of the skeleton - comes from GMR or
from ``mmd_profile`` (the MMD side). Paths in the file are relative to the GMR checkout, so a profile is portable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.profile import (BILATERAL_BONES, LAFAN1_TO_MMD, ROOT_BONE, SEGMENT_BONES,
                         TORSO_BONE)

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
    bilateral: tuple = ()  # (left link, right link) whose line gives that yaw, e.g. the hips

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
    roots = tuple(by_bone[bone] for bone in (ROOT_BONE, TORSO_BONE) if bone in by_bone)
    bilateral = (tuple(by_bone[bone] for bone in BILATERAL_BONES)
                 if all(bone in by_bone for bone in BILATERAL_BONES) else ())
    return RobotProfile(name=robot, xml=xml, base_config=config_path_used, links=links,
                        segments=segments, root_links=roots, weights={}, bilateral=bilateral)


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
        bilateral=tuple(str(item) for item in raw.get('bilateral_links', ())),
        weights={str(link): {str(field): float(value) for field, value in fields.items()}
                 for link, fields in raw.get('weight_overrides', {}).items()},
    )


def validate(profile, model, mujoco):
    """Problems that would otherwise show up as a silently wrong calibration."""
    problems = []
    for link in profile.links:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) < 0:
            problems.append('link not in the model: %s' % link)
    if len(profile.bilateral) != 2:
        problems.append('bilateral_links needs exactly two links (the hip line), got %d'
                        % len(profile.bilateral))
    for link in profile.bilateral:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) < 0:
            problems.append('bilateral link not in the model: %s' % link)
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


def dump(profile):
    """The whole resolved picture as one dict, with the GMR template inlined.

    ``load()`` answers a robot key out of GMR's checkout, so part of the answer lives in files this repo
    does not own. This shows what was read: the template's *contents*, the derived tables, and where each
    piece came from - which is also the list of what would have to be written by hand for a robot GMR has
    never seen.
    """
    template = {}
    base_config = Path(profile.base_config)
    if base_config.exists():
        template = json.loads(base_config.read_text(encoding='utf-8'))
    return {
        'name': profile.name,
        'provenance': {'xml': str(profile.xml), 'base_config': str(profile.base_config)},
        'links': profile.links,
        'segments': [{'link': segment.link, 'to': segment.to, 'joints': list(segment.joints)}
                     for segment in profile.segments],
        'root_links': list(profile.root_links),
        'bilateral_links': list(profile.bilateral),
        'weight_overrides': profile.weights,
        'derivation': {
            'links': "links 每一项的 bone 字段来自模板，经 LAFAN1_TO_MMD 翻译成 MMD 骨名",
            'segments': 'SEGMENT_BONES 里每个骨对，映射到追踪这两根骨的链接',
            'root_links': '追踪 %s / %s 的链接' % (ROOT_BONE, TORSO_BONE),
            'bilateral_links': '追踪 %s / %s 的链接' % BILATERAL_BONES,
        },
        'template': template,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Show what a robot profile resolved to.')
    parser.add_argument('--robot', required=True, help='profile json, or a GMR robot key')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--out', default=None, help='write the resolved profile here')
    parser.add_argument('--full', action='store_true', help='print the whole resolved profile')
    args = parser.parse_args()

    profile = load(args.robot, args.gmr_root)
    resolved = dump(profile)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('written %s' % path)

    template = resolved['template']
    tables = {key: len(value) for key, value in template.items() if isinstance(value, dict)}
    print('robot: %s' % resolved['name'])
    print('  xml            %s' % resolved['provenance']['xml'])
    print('  base_config    %s' % resolved['provenance']['base_config'])
    print('  links          %d' % len(resolved['links']))
    print('  segments       %d' % len(resolved['segments']))
    print('  root_links     %s' % ', '.join(resolved['root_links']))
    print('  bilateral      %s' % ', '.join(resolved['bilateral_links']))
    print('  weight overrides %d' % len(resolved['weight_overrides']))
    print('  template       %s' % (', '.join('%s (%d)' % item for item in tables.items()) or 'empty'))

    if args.full:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
