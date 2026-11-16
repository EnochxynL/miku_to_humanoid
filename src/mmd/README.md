# src/mmd — MMD 到机器人

三层，**依赖方向只能从上往下**：`mmd_*` 只依赖引擎，`gmr_*` 才依赖 GMR 和机器人模型。

| 层 | 文件 | 作用 |
|---|---|---|
| **MMD 原生** | `mmd_pose.py` | `pmx + (vmd\|vpd)` → 逐帧骨骼世界位姿。源侧唯一入口，不碰 MuJoCo |
| | `mmd_bones.py` | MMD 知识：链接↔骨名、骨→下一关节、测量段 |
| | `mmd_render.py` | 可视化素材：MJCF（`--mesh` 带贴图）+ qpos |
| | `view_model.py` / `view_skeleton.py` | 播放 / 机器人+源骨架对照 |
| **GMR 桥接** | `gmr_calibrate.py` | 成对姿态 → **已带 C 的 IK 配置**（骨架与 C 同一次产生）|
| | `gmr_retarget.py` | 源位姿 + C → G1 qpos |
| | `export_gmr_pickle.py` | G1 qpos → GMR pickle |
| **引擎** | `engine/` | vendored 自 mmd2gltf：`pmx` `vmd` `mathutil` `animation` `bake` `mesh` |

## 数据流

```
pmx + (vmd|vpd) ── mmd_pose ──→ 世界位姿 ──┐
                                          ├─ gmr_retarget ──→ G1 qpos ──→ export_gmr_pickle ──→ .pkl
g1.xml + 成对姿态 ── gmr_calibrate ──→ C ──┘

pmx + vmd ── mmd_render ──→ MJCF + qpos ──→ view_model（只为看）
```

数学只有一条：`T_link(t) = S_bone(t) · C`，其中 `C = S_bone(P)⁻¹ · T_link(P)` 由一个配对姿态读出。

## 约定

- `PYTHONPATH=src;src\mmd`
- **body/joint 名就是 MMD 骨名**（`左腕`），没有映射表
- 生产线不用 MuJoCo 算源侧：`Skeleton.world()` 直接给世界位姿，避免 qpos 往返丢掉 `センター` 平移
- 未校准的配置**从不落盘**：`gmr_calibrate.py` 一次跑完骨架与 C

细节与踩过的坑见 [`configs/mmd/README.md`](../../configs/mmd/README.md)。
