"""Write MuJoCo's legacy binary .msh.

The format is documented by the reader shipped with MuJoCo (``python/mujoco/msh2obj.py``): four int32
counts followed by float32 vertex data and int32 triangle indices, native little-endian.

    0x00  int32 nvertex, nnormal, ntexcoord, nface
    0x10  float32 vertex positions  [3 * nvertex]
          float32 vertex normals    [3 * nnormal]
          float32 vertex texcoords  [2 * ntexcoord]
          int32   face indices      [3 * nface]

Two details matter:

* MuJoCo's **OBJ loader flips V and its MSH loader does not** (msh2obj undoes that flip when converting
  back), so a caller that already flipped V for OBJ must pass the *unflipped* array here.
* Coordinates go through float32, which costs ~1e-4 m at our scale - far below anything visible.

Writing ~47k vertices takes single-digit milliseconds, against ~0.5 s for the equivalent OBJ text.
"""
from __future__ import annotations

import numpy as np


def to_msh(positions, normals, texcoords, faces) -> bytes:
    """Binary MSH bytes. ``texcoords`` must be unflipped (see the module docstring)."""
    positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3)
    texcoords = np.ascontiguousarray(texcoords, dtype=np.float32).reshape(-1, 2)
    faces = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1, 3)

    header = np.array([len(positions), len(normals), len(texcoords), len(faces)], dtype=np.int32)
    return b''.join([header.tobytes(),
                     positions.tobytes(),
                     normals.tobytes(),
                     texcoords.tobytes(),
                     faces.tobytes()])
