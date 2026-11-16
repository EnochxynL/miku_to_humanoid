"""The MMD knowledge this pipeline needs, and nothing else.

Two facts, both authored here and both independent of the robot:

* ``LINKS``      which MMD bone each robot link tracks (this is the pairing GMR stores in the ``bone``
                 field of its config, so it is the one table that has to exist somewhere);
* ``NEXT_JOINT`` the *next joint* after a bone. This is where MMD differs from a clean kinematic
                 tree: twists (``左腕捩``), sleeve chains (``左袖1-1``) and shoulder helpers
                 (``左肩C``) sit between real joints, so "the next bone" is almost never the tree
                 child. It is a small, hand-written subset of the MMD standard skeleton.

Swapping the source model means editing this file only: everything else is inferred - the robot-side
child from the MJCF tree, the C offsets from the paired pose.

The bone names are Japanese MMD names and are load-bearing: they are the keys GMR looks up.
"""

# robot link -> MMD bone
LINKS = {
    'pelvis': '腰',
    'torso_link': '上半身2',
    'left_hip_yaw_link': '左足',
    'left_knee_link': '左ひざ',
    'left_ankle_roll_link': '左足首',
    'right_hip_yaw_link': '右足',
    'right_knee_link': '右ひざ',
    'right_ankle_roll_link': '右足首',
    'left_shoulder_yaw_link': '左腕',
    'left_elbow_link': '左ひじ',
    'left_wrist_yaw_link': '左手首',
    'right_shoulder_yaw_link': '右腕',
    'right_elbow_link': '右ひじ',
    'right_wrist_yaw_link': '右手首',
}

# MMD bone -> the next joint (skipping twist/sleeve/helper bones)
NEXT_JOINT = {
    '腰': '上半身',
    '上半身2': '首',
    '左足': '左ひざ',
    '左ひざ': '左足首',
    '左足首': '左つま先',
    '右足': '右ひざ',
    '右ひざ': '右足首',
    '右足首': '右つま先',
    '左腕': '左ひじ',
    '左ひじ': '左手首',
    '左手首': '左中指１',
    '右腕': '右ひじ',
    '右ひじ': '右手首',
    '右手首': '右中指１',
}


def bones_for(link):
    """(bone, next joint bone) for a robot link."""
    bone = LINKS[link]
    return bone, NEXT_JOINT.get(bone)
