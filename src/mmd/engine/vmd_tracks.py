"""The VMD side of the pipeline, shared by every consumer.

Three pieces of knowledge, none of which should exist twice - ``engine.bake`` writes qpos and
``mmd_pose`` writes world poses, but both have to match a VMD to a PMX the same way:

* **names are truncated to 15 shift-jis bytes** in a VMD, so the lookup table must truncate identically
  or hundreds of tracks silently fail to match;
* **only bones with the IK flag (0x0020) and an ``ik`` block** are driven by IK, and a VMD can switch
  that per frame through ``ik_frames``;
* **frames are sampled 0..max_frame inclusive**, keeping the last frame even when ``step`` overshoots.

A VPD is a pose, not a motion: it carries full bone names and contributes a single frame.
"""
from __future__ import annotations

import numpy as np

from engine.animation import Track
from engine.vpd import parse_vpd

NAME_LIMIT = 15          # shift-jis bytes of a bone name in a VMD
IK_NAME_LIMIT = 20       # ... and in an ik_frames entry


def _key(name, limit=NAME_LIMIT):
    try:
        return name.encode('shift-jis', errors='replace')[:limit]
    except Exception:
        return name.encode('utf-8', errors='replace')[:limit]


def name_map(bones):
    """Truncated name -> bone index; the first bone wins, as MMD names may repeat."""
    mapping = {}
    for index, bone in enumerate(bones):
        mapping.setdefault(_key(bone['name']), index)
    return mapping


def match_tracks(bones, vmd):
    """({bone index: Track}, [unmatched VMD bone names])."""
    lookup = name_map(bones)
    tracks, unmatched = {}, []
    for name, keys in vmd['bones'].items():
        index = lookup.get(_key(name))
        if index is None:
            unmatched.append(name)
        else:
            tracks[index] = Track(keys)
    return tracks, unmatched


def ik_schedule(bones, vmd, solve_ik=True):
    """A function frame -> the set of IK bones enabled at that frame."""
    all_ik = [i for i, bone in enumerate(bones) if bone['flags'] & 0x0020 and 'ik' in bone]
    base = set(all_ik) if solve_ik else set()

    states = []
    if vmd.get('ik_frames'):
        lookup = {}
        for index in all_ik:
            lookup.setdefault(_key(bones[index]['name'], IK_NAME_LIMIT), index)
        for frame in vmd['ik_frames']:
            values = {}
            for name, on in frame['ik'].items():
                index = lookup.get(_key(name, IK_NAME_LIMIT))
                if index is not None:
                    values[index] = on
            states.append((frame['frame'], values))

    def enabled_at(frame):
        if not states:
            return base
        state = {}
        for start, values in states:
            if start > frame:
                break
            state.update(values)
        return {i for i in base if state.get(i, True)}

    return enabled_at


def frame_range(vmd, step=1):
    """0..max_frame inclusive, keeping the final frame when step does not land on it."""
    last = vmd['max_frame']
    frames = list(range(0, last + 1, step))
    if frames[-1] != last:
        frames.append(last)
    return frames


def vpd_frame(bones, pose_path):
    """({bone index: (translation, rotation)}, [unmatched names]) for a one-frame pose."""
    lookup = name_map(bones)
    frame, unmatched = {}, []
    for name, (translation, rotation) in parse_vpd(str(pose_path)).items():
        index = lookup.get(_key(name))
        if index is None:
            unmatched.append(name)
        else:
            frame[index] = (np.array(translation, dtype=float), np.array(rotation, dtype=float))
    return frame, unmatched
