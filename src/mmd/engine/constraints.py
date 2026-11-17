"""MMD facts and the choices this project makes about them. No robot, no GMR, no profile I/O.

Two kinds of thing live here, and they age differently:

* **facts of the format** - MMD's standard bone names. Every rig uses them (``腰``, ``上半身2``, ``頭``,
  ``左足``/``右足``); older rigs may lack ``上半身2``, which is a fallback chain in code, not a setting.
* **choices** - which bone directions define "the same pose". A pose is eight limb directions; that is a
  decision, and changing it changes the calibration objective, so it stays in code next to its use.

Adding hands/fingers later is a matter of appending entries with their own weight and group - see
``SEGMENT_BONES`` - not of redesigning anything.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- standard bone names the calibration reads by name -----------------------------------------
# the root (the bone a human skeleton floats on), the torso (the second link a limb pose cannot move),
# and the head (its height gives the body-size scale)
ROOT_BONE = '腰'
TORSO_BONE = '上半身2'
SCALE_BONE = '頭'

# the two thigh bones: their line gives the world-frame yaw of the links a pose cannot move
BILATERAL_BONES = ('左足', '右足')


@dataclass(frozen=True)
class LimbSegment:
    """One direction the paired pose compares: the MMD bone ``start`` points at ``stop``.

    ``weight`` scales this direction against the others in the solver's cost, so a future hand/finger
    set can be down-weighted instead of dominating the eight limb directions. ``group`` is a label for
    reporting and for a UI that wants to show body and hand errors apart.
    """

    start: str
    stop: str
    group: str = 'body'
    weight: float = 1.0


# These bone pairs are **not** parent-child: MMD inserts twist bones (捩), sleeve chains and shoulder
# helpers between what a human reads as consecutive joints, so "the next real joint" skips them:
#     左腕 -> 左ひじ     跳过 左腕捩 / 左袖 / 左肩C
#     左ひじ -> 左手首   跳过 左ひじ捩 / 左手捩
#     左足 -> 左ひざ     跳过 左足捩 / 左足D
#     左ひざ -> 左足首   跳过 左ひざD / 左足首D 等
SEGMENT_BONES = (
    LimbSegment('左腕', '左ひじ'),
    LimbSegment('左ひじ', '左手首'),
    LimbSegment('右腕', '右ひじ'),
    LimbSegment('右ひじ', '右手首'),
    LimbSegment('左足', '左ひざ'),
    LimbSegment('左ひざ', '左足首'),
    LimbSegment('右足', '右ひざ'),
    LimbSegment('右ひざ', '右足首'),
)

# Hand and finger segments are **not enabled yet**: the G1 model we drive has no finger joints, and GMR's
# IK takes no finger targets (LaFAN1 carries none). When a robot with hands is wired up, add entries in
# the same table - one per phalanx direction, since MMD gives each finger three bones:
#
#     LimbSegment('左中指１', '左中指２', group='hand-left', weight=0.3),
#     LimbSegment('左中指２', '左中指３', group='hand-left', weight=0.3),
#
# A missing bone simply drops that segment (``Calibration`` skips pairs whose bones the profile does not
# map), so adding these entries is safe before any robot can use them. Keep the weight low: 30 hand
# directions at weight 1.0 would drown the eight limb directions that define the pose.


# How the fourteen driven bones hang together, for drawing the retargeting engine's *intermediate*
# representation (the scaled, offset-corrected targets it actually aims at). It is a display-only view of
# the same bones SEGMENT_BONES measures, plus the trunk and the two hip joints.
SKELETON_EDGES = (
    (ROOT_BONE, TORSO_BONE),
    (ROOT_BONE, '左足'), ('左足', '左ひざ'), ('左ひざ', '左足首'),
    (ROOT_BONE, '右足'), ('右足', '右ひざ'), ('右ひざ', '右足首'),
    (TORSO_BONE, '左腕'), ('左腕', '左ひじ'), ('左ひじ', '左手首'),
    (TORSO_BONE, '右腕'), ('右腕', '右ひじ'), ('右ひじ', '右手首'),
)
