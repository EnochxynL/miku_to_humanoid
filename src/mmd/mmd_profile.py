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
