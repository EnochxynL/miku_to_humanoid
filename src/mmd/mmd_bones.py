"""The MMD knowledge this pipeline needs, and nothing else.

Three facts, all authored here:

* ``LINKS``      which MMD bone each robot link tracks (this is the pairing GMR stores in the ``bone``
                 field of its config, so it is the one table that has to exist somewhere);
* ``NEXT_JOINT`` the *next joint* after a bone. This is where MMD differs from a clean kinematic
                 tree: twists (``左腕捩``), sleeve chains (``左袖1-1``) and shoulder helpers
                 (``左肩C``) sit between real joints, so "the next bone" is almost never the tree
                 child. Only the bones the limb table consumes are listed.
* ``LIMBS``      which robot segment to measure against which limb, and which joints move it.

Two fields are deliberately **not** stored: the source bone and its child are derived from ``LINKS``
and ``NEXT_JOINT``, so a wrong pairing cannot hide in a third place. What remains is the part that is
genuinely not derivable:

* the far end of the segment is not the MJCF's first child - that would be ``left_wrist_roll_link`` /
  ``left_ankle_pitch_link``, while the anatomical end is ``left_wrist_yaw_link`` /
  ``left_ankle_roll_link``;
* the joint list differs by segment: a shoulder or hip must move all three of its joints (the pose is
  read off the last link of that chain), while an elbow or knee has exactly one.

Swapping the source model means editing this file only: the robot-side child comes from the MJCF tree,
the C offsets from the paired pose.

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

# MMD bone -> the next joint (skipping twist/sleeve/helper bones); only what LIMBS consumes
NEXT_JOINT = {
    '左足': '左ひざ',
    '左ひざ': '左足首',
    '右足': '右ひざ',
    '右ひざ': '右足首',
    '左腕': '左ひじ',
    '左ひじ': '左手首',
    '右腕': '右ひじ',
    '右ひじ': '右手首',
}
