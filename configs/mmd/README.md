# MMD → G1 retargeting: the configuration that works

This directory is the **versioned copy** of everything the working retargeting configuration depends
on. `runtime/` is gitignored (generated assets), so without this copy the configuration that produced
the good result exists only on one machine.

Only the **glTF path** is kept in `src/mmd`; the PMX and BVH routes were removed.

## Files

| file | what it is |
|---|---|
| `glb_to_g1.json` | GMR IK config: per-link rotation offsets + weights + scale table |
| `calibration.json` | how every constant was derived, the measured errors, and the retractions |
| `herta_nodes.json` | GLB node index → MMD bone name → body/joint/parent mapping |
| `herta_mesh_nodes.json` | same mapping for the textured mesh model |
| `herta_anim.npz` | the baked source animation, 319 frames of joint values |
| `README.md` | this file |

## The scripts (all glTF)

| script | role |
|---|---|
| `glb_io.py` | minimal .glb reader (JSON chunk, accessors, animation sampling) |
| `mmd_bones.py` | **the MMD knowledge**: robot link → MMD bone, and MMD bone → next joint (twist/sleeve bones skipped). The only file to edit when the source model changes |
| `measurements.py` | which segment to compare when reporting accuracy - the ruler, not the model |
| `glb_to_mjcf.py` | GLB → MJCF skeleton and textured mesh |
| `glb_anim_to_qpos.py` | GLB animation → per-frame qpos of that MJCF |
| `glb_to_gmr.py` | source frames → GMR config + baked robot qpos |
| `paired_pose_offsets.py` | **the calibration**: derives every constant from the MMD bind pose |
| `export_gmr_pickle.py` | our qpos → the GMR pickle `reference_maker/convert_gmr.py` reads |
| `view_model.py` | plain MuJoCo playback of a qpos sequence |
| `view_skeleton.py` | robot + source skeleton side by side (`--source gmr` shows exactly what GMR gets) |

## What C is, and why it took so long

GMR asks for one constant per link:

```
target_orientation(robot link, t) = R_source_bone(t) · C
```

`C` is the **frame convention** between one bone in the source rig and one link in the robot rig. It is
not part of the motion; both frames are rigidly attached to the same physical segment.

Two things made it hard, and both were my own errors:

* **It is under-determined from geometry.** A pose only observes two of its three degrees of freedom
  (the limb direction); the roll about the limb is invisible. Every "two-axis" attempt failed because
  the second axis degenerates when the limb is straight - and at the MMD bind pose the limbs are.
* **The GLB hides a 90° rotation.** The whole tree hangs under an `Rx90` root (Y-up → Z-up), so
  `R_source_bind = Rx90`. Forgetting to divide it out either pitches the robot flat or leaves the arms
  48° off. The same `Rx90` cost two separate debugging rounds.

**The resolution is the paired pose.** MMD bones carry no rest rotation, so at the bind pose every
bone frame is the identity and

```
C = R_source_bind^-1 · R_link(robot posed to match)
```

becomes a *reading* instead of a fit. `paired_pose_offsets.py` solves the robot's arm and leg joints
to match the bind pose limb directions (0.0–0.3° residual) and reads C off the model.

The two links that solve does **not** move - `pelvis` (the floating root) and `torso_link` - also need
the world-frame yaw, measured on **bilateral axes** (hip line, shoulder line) because those are the
only convention-free measures of body facing. They were 70–75° off before that correction and 8–11°
after. This is why "the arms were perfect but the body was twisted" for so long: every child target is
an absolute world orientation, so the spine silently absorbed a wrong root.

## Weights are GMR's, not tuned

Everything runs at the shipped convention: `pelvis rot_w=10`, `torso 100`, arms orientation-only
(`pos_w=0`), ankle `pos_w=50`. Two workarounds were tried and removed, because both were hiding the
missing root yaw rather than fixing it:

* `pos_w=300` on the arms (position-driven limbs): it bypasses the frame correspondence entirely;
* `pelvis rot_w=300`: the target was wrong, so pushing harder only made the twist more rigid.

`pos_w` is a quadratic cost weight; only ratios matter. Measured effect of the arm weight once the
offsets are right: from 10 to 3000 the forearm error stays on the same 4–22° plateau - the cliff is
*whether a position term exists at all*, not its value.

## Measured accuracy

| quantity | value |
|---|---|
| upper arm direction | **0.0°** |
| forearm direction | 2–6° (13–20° mid-clip) |
| shoulder line (torso facing) | 3–7° |
| hip line (body facing) | 8–11° |
| knee bend amount | 3.9° median (the knee sweeps 0–69° correctly) |
| thigh / shin direction | 6.7° / 5.6° median |

The leg figure is a **trade-off, not a bug**: `ankle pos_w=50` holds the feet on the ground. Dropping
it to 0 sharpens the shin to 1.3° but slides the feet 8–15 cm. Keep 50.

## Reproduce from scratch

```powershell
$env:PYTHONPATH='src'

# 1+2. the .glb carries both assets, and the two extractors are siblings, not a sequence -
#      one pass over the container produces the model *and* the motion
uv run python src/mmd/glb_extract.py --glb <model.glb> --out-dir runtime/glb --mesh
#      -> herta.xml (skeleton) + herta_mesh.xml with assets/ (textured) + herta_anim.npz (motion)
#      the individual scripts still work alone: glb_to_mjcf.py / glb_anim_to_qpos.py

# 3. base config (bone names, weights, scale table)
<gmr>/.venv/Scripts/python.exe src/mmd/glb_to_gmr.py `
    --glb-xml runtime/glb/herta.xml `
    --anim runtime/glb/herta_anim.npz --build-config runtime/gmr_inputs/glb_to_g1.json --frames 0

# 4. the calibration that matters
uv run python src/mmd/paired_pose_offsets.py --glb-xml runtime/glb/herta.xml `
    --config runtime/gmr_inputs/glb_to_g1.json --report

# 5. bake the robot motion WITH the calibrated config (rebuilding it here would overwrite the
#    paired-pose offsets and silently produce a worse motion - use --config-in)
<gmr>/.venv/Scripts/python.exe src/mmd/glb_to_gmr.py --config configs/mmd/glb_to_g1.json 
    --glb-xml runtime/glb/herta.xml 
    --anim runtime/glb/herta_anim.npz --frames 0 --bake-out runtime/glb/g1_final.npz

# 6. play robot + source skeleton side by side
<gmr>/.venv/Scripts/python.exe src/mmd/view_skeleton.py --glb-xml runtime/glb/herta.xml `
    --anim runtime/glb/herta_anim.npz `
    --config runtime/gmr_inputs/glb_to_g1.json --loop --offset 0.85 --source gmr

# 7. hand it to the control stack: GMR pickle -> the imitation clip format
uv run python src/mmd/export_gmr_pickle.py --npz runtime/glb/g1_final.npz `
    --out runtime/gmr_clips/herta_qqy.pkl --fps 30
uv run python src/reference_maker/convert_gmr.py --pickle runtime/gmr_clips/herta_qqy.pkl `
    --out runtime/refs/herta_qqy/signals.npz --robot g1
```

## Naming

The MJCF does not carry generic names: every body and joint is called what the model calls it
(左腕, 左ひじ, …), taken from the glTF node name (the exporter writes 
odes[i].name and the names
are unique for this model). That is why no mapping table sits in the data path any more - a bone's
name *is* the interface. The optional *_nodes.json is kept only as provenance.

## Known gaps

* **hands**: GMR receives no palm bones, so finger and palm orientation are not retargeted at all;
* **legs** 5–7°: the ankle position weight above, not an unknown twist;
* **flight frames**: `convert_gmr.py` labels 23% of this dance as flight (both feet up), which the
  WBC cannot support - start from a double-support frame or trim the airborne section;
* the **PMX and BVH routes are gone on purpose**. They live in git history if ever needed; the BVH
  route failed for the same C reason as everything else, not because BVH is a bad format.
