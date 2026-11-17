# src/mmd — MMD 到机器人

三层，**依赖方向只能从上往下**：`mmd_*` 只依赖引擎，`gmr_*` 才依赖 GMR 和机器人模型。

| 层 | 文件 | 作用 |
|---|---|---|
| **MMD 原生** | `engine/bake.py` | `pmx + (vmd\|vpd)` → 逐帧骨骼世界位姿。源侧唯一入口，不碰 MuJoCo |
| | `mmd_profile.py` | **MMD 侧知识**：哪几条肢体方向定义姿态、LaFAN1→MMD 一览、捩骨参考 |
| | `engine/mjcf.py` | **PMX → MjModel（纯内存）**：MJCF 文本 + MSH 网格 + PNG 贴图字节 |
| | `view.py` | **一个窗口**：MMD 角色 + 机器人 + 可选骨架胶囊（三层 overlay，不跑 GMR）|
| **GMR 桥接** | `gmr_paired_pose.py` | 解出"让机器人摆成 MMD 姿态"的关节角 → `paired_pose_*.json` |
| | `calibration.py` | **消费**关节角文件 → **中性 C**（每链接旋转 + 尺度），不认任何引擎格式 |
| | `gmr_retarget.py` | 源位姿 + C → G1 qpos |
| | `export_gmr_pickle.py` | G1 qpos → GMR pickle |
| **引擎** | `engine/` | vendored 自 mmd2gltf：`pmx` `vpd` `vmd` `mathutil` `animation` `bake` `mesh` `msh` `mmd_model` |

## 数据流

```
pmx + (vmd|vpd) ── engine/bake.py ──→ 世界位姿 ──┐
                                          ├─ gmr_retarget ──→ G1 qpos ──→ export_gmr_pickle ──→ .pkl
pmx + pose ── calibration ──→ 关节角 → C（中性） ──┘

pmx + (vpd|vmd) ── mmd_model ──→ MjModel ─┐
                                         ├─ view（一个窗口并排显示，全在内存里）
g1 qpos + robot xml ─────────────────────┘
```

数学只有一条：`T_link(t) = S_bone(t) · C`，其中 `C = S_bone(P)⁻¹ · T_link(P)` 由一个配对姿态读出。

## 约定

- `PYTHONPATH=src;src\mmd`
- **body/joint 名就是 MMD 骨名**（`左腕`），没有映射表
- 生产线不用 MuJoCo 算源侧：`Skeleton.world()` 直接给世界位姿，避免 qpos 往返丢掉 `センター` 平移
- GMR 配置**不再是产物**：`gmr_retarget.py` 用中性 C 现场构建（`--dump-config` 才留一份）

细节与踩过的坑见 [`configs/README.md`](../../configs/README.md)。

## 成对姿态不是唯一的（重要）

`calibration.py` 只用**肢体方向**判定"同一姿态"，而绕肢体轴的**扭转**不改变方向，所以解的零空间
是自由的实际自由：同一组方向可以对应相差 70° 以上的关节角，且残差都是 0.00°。

两条推论：

1. 求解加了极小正则项（`1e-6·Σq²`）让结果**确定且最小扭转**，否则两次运行会落到零空间的不同点；
2. 扭转的取值**会改变最终动作**（每个肢体一个常数扭转），所以 `paired_pose_*.json` 是**需要人工/视觉
   确认的数据**，不是可以随手重算的中间量。当前发布的姿态已通过目视验收，重新生成会得到另一个
   "同样合法但不同"的解。
