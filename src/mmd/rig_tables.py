"""Tables that map MMD bone names onto G1 links, plus the measurement pairs.

Extracted from the retired PMX retargeting module: the glTF path needs exactly these pieces and
nothing else from it. The bone names are Japanese MMD names and are load-bearing - they are the keys
GMR looks up in the human data, so they must not be transliterated.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

LINKS = {
    'pelvis': ('腰', '上半身', 'torso_link'),
    'torso_link': ('上半身2', '首', 'head_link'),
    'left_hip_yaw_link': ('左足', '左ひざ', 'left_knee_link'),
    'left_knee_link': ('左ひざ', '左足首', 'left_ankle_roll_link'),
    'left_ankle_roll_link': ('左足首', '左つま先', 'left_toe_link'),
    'right_hip_yaw_link': ('右足', '右ひざ', 'right_knee_link'),
    'right_knee_link': ('右ひざ', '右足首', 'right_ankle_roll_link'),
    'right_ankle_roll_link': ('右足首', '右つま先', 'right_toe_link'),
    'left_shoulder_yaw_link': ('左腕', '左ひじ', 'left_elbow_link'),
    'left_elbow_link': ('左ひじ', '左手首', 'left_wrist_yaw_link'),
    'left_wrist_yaw_link': ('左手首', '左中指１', 'left_wrist_roll_link'),
    'right_shoulder_yaw_link': ('右腕', '右ひじ', 'right_elbow_link'),
    'right_elbow_link': ('右ひじ', '右手首', 'right_wrist_yaw_link'),
    'right_wrist_yaw_link': ('右手首', '右中指１', 'right_wrist_roll_link'),
}
# links whose parent is the torso in the robot but a different PMX bone in the model
ROBOT_PARENT = {
    'torso_link': 'pelvis',
    'left_hip_yaw_link': 'pelvis', 'right_hip_yaw_link': 'pelvis',
    'left_knee_link': 'left_hip_yaw_link', 'right_knee_link': 'right_hip_yaw_link',
    'left_ankle_roll_link': 'left_knee_link', 'right_ankle_roll_link': 'right_knee_link',
    'left_shoulder_yaw_link': 'torso_link', 'right_shoulder_yaw_link': 'torso_link',
    'left_elbow_link': 'left_shoulder_yaw_link', 'right_elbow_link': 'right_shoulder_yaw_link',
    'left_wrist_yaw_link': 'left_elbow_link', 'right_wrist_yaw_link': 'right_elbow_link',
}
# measurement pairs, chosen so both sides are the same anatomical segment; the torso is measured
# through pelvis->torso because this MJCF places head_link's origin below torso_link's, and the hand
# is measured through the forearm because the wrist links differ by a lateral roll axis.
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
# kept for reference: the A-pose difference is largest at these joints, but the same mismatch
# exists at every link, so the alignment is applied everywhere now.
ALIGN_AT = {'left_shoulder_yaw_link', 'right_shoulder_yaw_link',
            'left_hip_yaw_link', 'right_hip_yaw_link'}


def rotation_between(u, v) -> R:
    u = u / (np.linalg.norm(u) or 1.0)
    v = v / (np.linalg.norm(v) or 1.0)
    axis = np.cross(u, v)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(u @ v, -1.0, 1.0))
    if sine < 1e-9:
        return R.identity() if cosine > 0 else R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    return R.from_rotvec(np.arctan2(sine, cosine) * axis / sine)
