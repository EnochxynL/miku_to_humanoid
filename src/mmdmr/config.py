"""The robot side: what a robot profile file contains, and how it is read and checked.

A profile is an **authored** file, not something derived at run time: it pairs this robot's links with
MMD bones, says which link pairs the paired pose compares, names the two links that form the hip line,
and carries the engine template the retargeting engine needs (complete, so the file works with no other
checkout present).

``scripts/get_robot_profile_from_gmr.py`` can write one by reading GMR's IK configs, but that is a
convenience: nothing here knows GMR exists, and a robot GMR has never seen is described by the same file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    """One link pair the paired pose aligns: the direction of ``link -> to``."""

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
    gmr_config: Path = None        # the retargeting engine's own config for this robot, if it has one


def load(path):
    """Read a profile file. Paths inside it may be absolute or relative to the file."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    missing = [key for key in ('name', 'xml', 'links', 'segments') if key not in raw]
    if missing:
        raise SystemExit('%s: missing %s' % (path, ', '.join(missing)))

    def resolve(value):
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (path.parent / candidate).resolve()


    return RobotProfile(
        name=str(raw['name']),
        xml=resolve(raw['xml']),
        links={str(link): str(bone) for link, bone in raw['links'].items()},
        segments=tuple(Segment(str(item['link']), str(item['to'])) for item in raw['segments']),
        root_links=tuple(raw.get('root_links', ())),
        bilateral=tuple(raw.get('bilateral_links', ())),
        gmr_config=resolve(raw['gmr_config']) if raw.get('gmr_config') else None,
    )


def dump(profile, base=None):
    """The profile as plain data. ``base`` makes the paths relative to it, as a reader expects."""
    base = Path(base).resolve() if base else None

    def show(value):
        if value is None:
            return None
        value = Path(value)
        if base:
            try:
                # a sibling directory is a *relative* path, not a subpath: relpath computes "..\\gmr\\x"
                return os.path.relpath(str(value), str(base))
            except ValueError:
                return str(value)                       # another drive: keep it absolute
        return str(value)

    return {
        'name': profile.name,
        'xml': show(profile.xml),
        'gmr_config': show(profile.gmr_config),
        'links': profile.links,
        'segments': [{'link': segment.link, 'to': segment.to} for segment in profile.segments],
        'root_links': list(profile.root_links),
        'bilateral_links': list(profile.bilateral),
    }


def save(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump(profile, base=path.parent), indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')
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
    if profile.gmr_config is not None and not Path(profile.gmr_config).exists():
        problems.append('gmr_config does not exist: %s' % profile.gmr_config)
    for segment in profile.segments:
        if segment.link not in profile.links:
            problems.append('segment start without a bone: %s' % segment.link)
        for name in (segment.link, segment.to):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                problems.append('segment body not in the model: %s' % name)
    return problems
