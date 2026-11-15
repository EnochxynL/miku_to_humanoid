# src/mmd —— MMD 到机器人的流水线

链路与用法见 [`docs/pmx_rigidbody_pipeline.md`](../../docs/pmx_rigidbody_pipeline.md)。

| 文件 | 作用 |
|---|---|
| `pmx_to_mjcf.py` | PMX 骨架 → MuJoCo 刚体模型（每骨一个 body + ball joint）|
| `bake_pmx_vmd.py` | VMD → PMX qpos，含付与親与 CCD IK；`--view` 开窗口 |
| `pmx_to_gmr.py` | PMX 位姿 → GMR → 目标机器人 qpos；`--play` 播放 |
| `vendor_pmx.py` / `vendor_vmd.py` | PMX / VMD 解析器（从作者自己的仓库引入）|
| `configs/` | 早期尝试的 GMR 配置（线路已被 PMX 刚体路线取代，仅存档）|

诊断工具（`check_*.py`、`port_pmx_mapping.py`、`run_player_dump.py`）用于复算播放器的
对齐角、核对时间轴、检验 GMR 的每链路常量是否成立——都是这条流水线的验收依据。

`mmd_glb_to_bvh.py`、`bvh_retopologize.py`、`bvh_viewer.py` 属于**已否决**的
"MMD → BVH → GMR" 路线，保留作为记录，见 `docs/gmr_offset_derivation.md`。

所有命令都需要 `PYTHONPATH=src`。
