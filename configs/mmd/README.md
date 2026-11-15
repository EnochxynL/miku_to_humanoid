# MMD → G1 retargeting: the configuration that works

This directory is the **versioned copy** of everything the working retargeting configuration depends
on. `runtime/` is gitignored (generated assets), so without this copy the configuration that produced
the good result exists only on one machine.

## Files

| file | what it is |
|---|---|
| `glb_to_g1.json` | GMR IK config: per-link rotation offsets + weights + scale table |
| `herta_nodes.json` | GLB node index → MMD bone name → body/joint/parent mapping |
| `herta_mesh_nodes.json` | same mapping for the textured mesh model |
| `herta_anim.npz` | the baked source animation, 319 frames of joint values |
| `README.md` | this file |

## Why the offsets look the way they do

`C = rotation_between(a_local, a_src)` — the minimal rotation taking **the robot link's limb axis**
onto **the source bone's limb axis** (`src/mmd/distal_offsets.py`). That fixes the limb *direction*;
the third degree of freedom (roll about the limb) is left free, which a hinge does not care about.

Two earlier constructions were wrong, and both were hidden by a lucky property of the PMX models:

* `C = rest · align` (PMX route) worked only because the PMX rig binds with **identity** bone
  orientation, so a world-frame rotation could be used as a local-frame one;
* `C = src_bind⁻¹ · rest · align` (first GLB attempt) divided the bind rotation out in the wrong
  place and left a ~180° yaw error: the targets pointed the arms *inward*, the shoulder roll hit its
  lower limit (-34.4°) and looked "locked at the factory".

Measured arm direction error at three frames: **43°/65° before → 2°/1° after**.

## Arms are driven by position, not orientation

A hinge elbow cannot satisfy a full 3-dof orientation target; with orientation-only weights the
forearm stayed 60-92° off. Driving the arm chain by position (`pos_w=300, rot_w=1`) brings the
forearm to **11/38°**. `distal_offsets.py --arm-pos-weight 300` (the default) writes this.

## Reproduce from scratch

```powershell
$env:PYTHONPATH='src'

# 1. GLB -> MuJoCo (skeleton + textured mesh)
uv run python src/mmd/glb_to_mjcf.py --glb <model.glb> `
    --xml runtime/glb/herta_mesh.xml --nodes runtime/glb/herta_mesh_nodes.json --mesh
uv run python src/mmd/glb_to_mjcf.py --glb <model.glb> `
    --xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json

# 2. animation -> qpos for the source model
uv run python src/mmd/glb_anim_to_qpos.py --glb <model.glb> `
    --xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json `
    --out runtime/glb/herta_anim.npz

# 3. base config (bone names, weights, scale table)
<gmr>/.venv/Scripts/python.exe src/mmd/glb_to_gmr.py `
    --glb-xml runtime/glb/herta.xml --nodes runtime/glb/herta_nodes.json `
    --anim runtime/glb/herta_anim.npz --config-out runtime/gmr_inputs/glb_to_g1.json --frames 0

# 4. the offsets that matter
uv run python src/mmd/distal_offsets.py --glb-xml runtime/glb/herta.xml `
    --nodes runtime/glb/herta_nodes.json --config runtime/gmr_inputs/glb_to_g1.json

# 5. play robot + source skeleton side by side
<gmr>/.venv/Scripts/python.exe src/mmd/view_skeleton.py --glb-xml runtime/glb/herta.xml `
    --nodes runtime/glb/herta_nodes.json --anim runtime/glb/herta_anim.npz `
    --config runtime/gmr_inputs/glb_to_g1.json --loop --offset 0.85 --source gmr
```

## Known gaps (measured, not guessed)

* **right forearm**: 33-38° off, with a 127° outlier mid-clip;
* **legs**: 9-54°, and the robot stands **0.22 m lower** than the skeleton;
* **shoulder width**: the source's arm-root bones are 0.16 m apart, G1's shoulder links 0.31 m — a
  structural mismatch that a rigid bone→link mapping cannot absorb;
* **hands**: GMR receives no palm bones, so finger/palm orientation is not retargeted at all.
