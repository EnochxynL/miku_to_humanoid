"""Build the G1 IK config skeleton from a PMX rig's bind pose.

This produces structure only - bone names per link, weights, the uniform scale factor - and sets the
orientation offsets to identity. The offsets are **not** computed here: they come from the paired-pose
calibration (``calibrate.py``), because building them from an unpaired guess is exactly the mistake
that produced a 159 deg wrong bake once.

    PYTHONPATH=src;src\\mmd uv run python src/mmd/build_config.py --pmx <model.pmx> \
        --g1-xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml \
        --base <gmr>/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
        --out configs/mmd/pmx_to_g1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco                                          # noqa: E402

import mmd_pose                                        # noqa: E402
from mmd_bones import LINKS                            # noqa: E402

PELVIS_ROT_WEIGHT = 10.0      # the floating root carries the whole body's facing; GMR ships 10/5


def build(base_path, g1_xml, pmx_path, out_path, scale=mmd_pose.DEFAULT_SCALE):
    base = json.loads(Path(base_path).read_text(encoding='utf-8'))

    wanted = sorted(set(LINKS.values()) | {'腰', '上半身2', '頭'})
    frame = mmd_pose.bind_pose(pmx_path, wanted)[0]
    if '頭' not in frame:
        raise SystemExit('the source rig has no 頭 bone; the scale factor needs a head')

    robot = mujoco.MjModel.from_xml_path(str(g1_xml))
    data = mujoco.MjData(robot)
    mujoco.mj_forward(robot, data)
    robot_height = float(data.xpos[:, 2].max())
    source_height = float(frame['頭'][0][2])
    factor = robot_height / source_height if source_height > 1e-6 else 0.9

    base['human_root_name'] = '腰'
    base['human_height_assumption'] = source_height
    base['ground_height'] = 0.0
    # human_scale_table is a *uniform body* scale, not a per-segment length ratio: GMR computes
    # (position - root) * scale[bone]
    base['human_scale_table'] = {bone: float(factor) for bone in wanted}

    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in base.get(table, {}).items():
            if link in LINKS:
                entry[0] = LINKS[link]
                entry[4] = [1.0, 0.0, 0.0, 0.0]        # identity: calibrate.py owns the offsets
        if 'pelvis' in base.get(table, {}):
            base[table]['pelvis'][2] = PELVIS_ROT_WEIGHT

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, indent=2, ensure_ascii=True), encoding='utf-8')
    return {
        'config': str(out),
        'links': len([k for k in base['ik_match_table1'] if k in LINKS]),
        'scale_factor': round(factor, 6),
        'source_height': round(source_height, 4),
        'robot_height': round(robot_height, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pmx', required=True)
    parser.add_argument('--g1-xml', required=True)
    parser.add_argument('--base', required=True, help="GMR's stock config, used as the template")
    parser.add_argument('--out', required=True)
    parser.add_argument('--scale', type=float, default=mmd_pose.DEFAULT_SCALE)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.g1_xml, args.pmx, args.out, args.scale)))


if __name__ == '__main__':
    main()
