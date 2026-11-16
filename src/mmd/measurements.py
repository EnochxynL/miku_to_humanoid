"""What to measure, and how to compare two directions.

These tables are **diagnostics only**: they decide which anatomical segment is compared on the source
side and on the robot side when reporting accuracy (and, historically, when aligning). Nothing here is
needed to bake a motion - the offsets come from the paired pose.

Kept separate from ``mmd_bones`` on purpose: this is the ruler, not the model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# link -> (source bone, source child, robot link, robot child)
MEASURE = {
    'pelvis': ('腰', '上半身', 'pelvis', 'torso_link'),
    'torso_link': ('腰', '上半身', 'pelvis', 'torso_link'),
    'left_hip_yaw_link': ('左足', '左ひざ', 'left_hip_yaw_link', 'left_knee_link'),
    'left_knee_link': ('左ひざ', '左足首', 'left_knee_link', 'left_ankle_roll_link'),
    'left_ankle_roll_link': ('左足首', '左つま先', 'left_ankle_roll_link', 'left_toe_link'),
    'right_hip_yaw_link': ('右足', '右ひざ', 'right_hip_yaw_link', 'right_knee_link'),
    'right_knee_link': ('右ひざ', '右足首', 'right_knee_link', 'right_ankle_roll_link'),
    'right_ankle_roll_link': ('右足首', '右つま先', 'right_ankle_roll_link', 'right_toe_link'),
    'left_shoulder_yaw_link': ('左腕', '左ひじ', 'left_shoulder_yaw_link', 'left_elbow_link'),
    'left_elbow_link': ('左ひじ', '左手首', 'left_elbow_link', 'left_wrist_yaw_link'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', 'right_shoulder_yaw_link', 'right_elbow_link'),
    'right_elbow_link': ('右ひじ', '右手首', 'right_elbow_link', 'right_wrist_yaw_link'),
}


def rotation_between(u, v) -> R:
    """Minimal rotation taking direction ``u`` onto ``v``."""
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    axis = np.cross(u, v)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(u @ v, -1.0, 1.0))
    if sine < 1e-9:
        return R.identity() if cosine > 0 else R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    return R.from_rotvec(np.arctan2(sine, cosine) * axis / sine)
