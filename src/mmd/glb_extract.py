"""Extract both halves of a .glb in one pass: the MuJoCo model and the motion.

A .glb is two assets in one container - the node/mesh/skin hierarchy and a baked animation - so the
model extractor and the motion extractor are **siblings**, not a pipeline:

    .glb ─┬─ glb_to_mjcf.extract()      ->  <stem>.xml (+ <stem>_mesh.xml and assets)
          └─ glb_anim_to_qpos.extract() ->  <stem>_anim.npz  (qpos per frame)

This script parses the container once and drives both, which is the only ordering that exists: the
motion needs the joint layout of the model it was baked against.

MuJoCo has no official trajectory file format. Its state is exposed as a flat vector
(``mj_stateSize``/``mj_getState``/``mj_setState`` for checkpointing qpos+qvel+act+ctrl+time), and
MJCF ``<keyframe>`` holds sparse named poses - neither is meant for a 319-frame clip, and both are
model-specific because the qpos layout depends on the model. So the convention is what this writes:
``npz`` with ``qpos`` (N, nq) and ``times`` (N,) plus the model xml it belongs to.

    PYTHONPATH=src uv run python src/mmd/glb_extract.py --glb <model.glb> --out-dir runtime/glb
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glb_anim_to_qpos import run as extract_motion      # noqa: E402
from glb_to_mjcf import run as extract_model            # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glb', required=True)
    parser.add_argument('--out-dir', default='runtime/glb')
    parser.add_argument('--stem', default=None, help='output basename; defaults to the .glb stem')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--mesh', action='store_true', help='also write the textured model')
    parser.add_argument('--no-rollup', dest='rollup', action='store_false')
    parser.add_argument('--drop-overlays', action='store_true')
    parser.add_argument('--no-flip-v', dest='flip_v', action='store_false')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.glb).stem
    xml_path = out_dir / f'{stem}.xml'
    nodes_path = out_dir / f'{stem}_nodes.json'
    anim_path = out_dir / f'{stem}_anim.npz'

    # --- model -----------------------------------------------------------------------------------
    model_args = Namespace(glb=args.glb, xml=str(xml_path), nodes=str(nodes_path),
                           mesh=False, flip_v=args.flip_v, sphere_radius=0.008,
                           rollup=args.rollup, drop_overlays=args.drop_overlays)
    extract_model(model_args)
    if args.mesh:
        extract_model(Namespace(**{**vars(model_args),
                                   'xml': str(out_dir / f'{stem}_mesh.xml'),
                                   'nodes': str(out_dir / f'{stem}_mesh_nodes.json'),
                                   'mesh': True}))

    # --- motion ----------------------------------------------------------------------------------
    motion_args = Namespace(glb=args.glb, xml=str(xml_path), nodes=str(nodes_path),
                            out=str(anim_path), fps=args.fps,
                            view=False, speed=0.6, camera_distance=2.0, camera_azimuth=90.0)
    qpos = extract_motion(motion_args)

    print()
    print('model  :', xml_path)
    print('nodes  :', nodes_path)
    print('motion :', anim_path, qpos.shape if qpos is not None else '')


if __name__ == '__main__':
    main()
