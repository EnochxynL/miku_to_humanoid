"""MMD (left-handed) -> MuJoCo (right-handed): one reflection, used everywhere.

MMD is ``[x, y, z]`` with the origin between the feet; MuJoCo is the same space reflected through the
``(-z, x, y)`` matrix, whose determinant is -1. Two consequences bite if they are written in two places:

* positions map by that matrix, times the model scale;
* a **rotation** does not: under a reflection the sense of rotation flips, so a quaternion maps as
  ``(x, y, z, w) -> (w, z, -x, -y)`` rather than by rotating its axis with the matrix.
"""
from __future__ import annotations

import numpy as np

BASIS = np.array([[0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])


def to_vector(vector, scale=1.0):
    """A position or direction in MuJoCo space, scaled to metres."""
    return scale * BASIS @ np.asarray(vector, dtype=float)


def to_quaternion(rotation):
    """MMD quaternion ``(x, y, z, w)`` -> MuJoCo ``(w, x, y, z)`` under the reflection."""
    x, y, z, w = (float(v) for v in rotation)
    return np.array([w, z, -x, -y])
