"""Read MMD .vpd pose files.

A VPD is a static pose: one entry per bone with a translation and a rotation, in the same
left-handed MMD convention and the same shift-jis encoding as VMD. The file is text, not binary, so
this is a small tolerant parser rather than a port of the VMD reader.

    Vocaloid Pose File0001, 1.1, model.pmx;
    1;
    Bone0{センター
      0.000000,0.000000,0.000000;
      0.000000,0.000000,0.000000,1.000000;
    }
"""
from __future__ import annotations

import re
from pathlib import Path

BONE = re.compile(r'^Bone\d+\{(.*)$')


def parse_vpd(path):
    """{bone name: ((x, y, z) translation, (x, y, z, w) rotation)} - rotation only is common."""
    text = Path(path).read_bytes().decode('cp932', errors='replace')
    lines = text.splitlines()
    poses = {}
    index = 0
    while index < len(lines):
        match = BONE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        name = match.group(1).strip()
        values = []
        probe = index + 1
        while probe < len(lines) and len(values) < 7:
            line = lines[probe].strip()
            if line and not line.startswith('}') and not line.startswith('Bone'):
                body = line.split('//')[0].replace(';', ',')
                values.extend(float(part) for part in body.split(',') if part.strip())
            probe += 1
        if len(values) >= 7:
            poses[name] = (tuple(values[:3]), tuple(values[3:7]))
        index = probe
    return poses
