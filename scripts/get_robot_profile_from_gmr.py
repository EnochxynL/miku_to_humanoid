"""Generate a complete robot profile by borrowing GMR's checkout.

This is a convenience, not a dependency: a profile is an authored file (``configs/robots/<robot>.json``)
that a person can write by hand, and this script only saves that work when the robot happens to ship in
GMR. It reads GMR's stock IK config, rewrites the bone names to MMD's and applies any weight deviations,
producing a self-contained profile - no GMR path survives in the output.

    uv run python scripts/get_robot_profile_from_gmr.py --robot booster_t1_29dof \
        --gmr-root <gmr> --out configs/robots/t1.json
    # a robot whose weights deviate from GMR stock:
    uv run python scripts/get_robot_profile_from_gmr.py --robot unitree_g1 --gmr-root <gmr> \
        --out configs/robots/g1.json --override pelvis.rot_w=10 \
        --override left_elbow_link.pos_w=0 --override left_elbow_link.rot_w=10 ...
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'mmd'))

import mujoco                                          # noqa: E402

import config as config_module                          # noqa: E402
from engine.constraints import (BILATERAL_BONES, ROOT_BONE, SEGMENT_BONES,   # noqa: E402
                                TORSO_BONE)

# How GMR names the human side reads as MMD bone names. This table is only needed while borrowing GMR's
# configs, so it lives here rather than in the profile format: a finished profile already carries MMD
# names and needs no translation.
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

WEIGHT_FIELD = {'pos_w': 1, 'rot_w': 2}


def find_xml(gmr, robot):
    assets = gmr / 'assets' / robot
    xmls = sorted(assets.glob('*.xml'), key=lambda path: ('mocap' not in path.name, path.name))
    if not xmls:
        raise SystemExit('no MJCF under %s' % assets)
    return xmls[0]


def find_config(gmr, model):
    """The stock IK config for this robot, matched by content rather than by filename.

    GMR's robot keys and its config filenames disagree (``booster_t1_29dof`` vs
    ``bvh_lafan1_to_t1_29dof.json``), so the only reliable test is that every link it names exists in the
    MJCF and every body it names is a LaFAN1 bone we can translate.
    """
    best, best_score, best_path = None, -1, None
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
        # GMR's own "handless" production config wins over its bvh variants at equal size
        score = len(table)
        if score > best_score:
            best, best_score, best_path = config, score, config_path
    if best is None:
        raise SystemExit('no stock config under %s fits %s' % (
            gmr / 'general_motion_retargeting' / 'ik_configs', model))
    return best, best_path


def build(gmr, robot, overrides):
    gmr = Path(gmr)
    xml = find_xml(gmr, robot)
    model = mujoco.MjModel.from_xml_path(str(xml))
    config, config_path = find_config(gmr, model)

    links = {link: LAFAN1_TO_MMD[str(entry[0])] for link, entry in config['ik_match_table1'].items()}
    by_bone = {bone: link for link, bone in links.items()}

    template = copy.deepcopy(config)
    template['human_root_name'] = ROOT_BONE
    template['ground_height'] = 0.0
    # the scale and the source height are properties of the *pose*, filled in by calibration
    template.pop('human_scale_table', None)
    template.pop('human_height_assumption', None)

    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in template.get(table, {}).items():
            if link in links:
                entry[0] = links[link]
                entry[4] = [1.0, 0.0, 0.0, 0.0]        # C is applied at retarget time
    for override in overrides:
        target, _, assignment = override.partition('=')
        link, _, field = target.partition('.')
        if field not in WEIGHT_FIELD:
            raise SystemExit('unknown weight field in --override %r' % override)
        for table in ('ik_match_table1', 'ik_match_table2'):
            if link in template.get(table, {}):
                template[table][link][WEIGHT_FIELD[field]] = float(assignment)
            else:
                raise SystemExit('--override names a link that is not in %s: %s' % (table, link))

    return config_module.RobotProfile(
        name=robot,
        xml=xml.resolve(),
        links=links,
        segments=tuple(config_module.Segment(by_bone[item.start], by_bone[item.stop])
                       for item in SEGMENT_BONES
                       if item.start in by_bone and item.stop in by_bone),
        root_links=tuple(by_bone[bone] for bone in (ROOT_BONE, TORSO_BONE) if bone in by_bone),
        bilateral=(tuple(by_bone[bone] for bone in BILATERAL_BONES)
                   if all(bone in by_bone for bone in BILATERAL_BONES) else ()),
    ), template, config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', required=True, help='GMR robot key, e.g. booster_t1_29dof')
    parser.add_argument('--gmr-root', default=r'D:\MY_TOOL\GitHub_Tools\GMR')
    parser.add_argument('--out', required=True, help='the robot profile to write')
    parser.add_argument('--gmr-out', required=True,
                        help='the engine config for this robot, e.g. configs/gmr/g1.json')
    parser.add_argument('--override', action='append', default=[],
                        help='link.field=value, e.g. pelvis.rot_w=10 (applied to both tables)')
    args = parser.parse_args()

    profile, template, config_path = build(args.gmr_root, args.robot, args.override)

    # the engine config goes to its own file; the profile only points at it
    engine_path = Path(args.gmr_out)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    profile = dataclasses.replace(profile, gmr_config=engine_path.resolve())
    config_module.save(profile, args.out)

    print('wrote %s' % args.out)
    print('wrote %s' % engine_path)
    print('  borrowed  %s' % config_path)
    print('  xml       %s' % profile.xml)
    print('  links     %d   segments %d   roots %s   bilateral %s'
          % (len(profile.links), len(profile.segments),
             ', '.join(profile.root_links), ', '.join(profile.bilateral)))
    print('  engine    %s' % ', '.join(sorted(template)))


if __name__ == '__main__':
    main()
