# PMX → G1 动作重定向

把 MMD 模型（PMX）的动作（VMD）搬到 Unitree G1 上，产出 GMR 能读的 pickle，最终变成 MPC-WBC 的
500 Hz 参考。**中间不经过 glTF，源侧也不经过 MuJoCo**。

## 目录：源配置 vs 产物

| 目录 | 内容 | 谁维护 |
|---|---|---|
| `robots/` | **机器人描述**（引擎无关）：链接↔MMD 骨、段、根链接、髋线 | 手写，或由生成器脚本写出 |
| `gmr/` | **GMR 的引擎配置**：权值表 `ik_match_table1/2`、偏移、根名、开关 | 生成器脚本写出；换引擎时整个作废 |
| `generated/` | 命令产物：配对姿态、中性 C | **不要手改**，随时可重算 |

`robots/<robot>.json` 里只有一个 `gmr_config: "../gmr/<robot>.json"` 指向引擎配置，因此机器人描述
**不认识任何引擎格式**；`retarget.py` 是唯一读 `gmr/` 的文件。

`generated/` 里的两个文件各自的重算方式：

```powershell
uv run mmgmr-calibration --pmx $pmx --robot configs/robots/g1.json `
    --solve --pose-out configs/generated/paired_pose_g1.json          # 配对姿态（关节角）
uv run mmgmr-calibration --pmx $pmx --robot configs/robots/g1.json `
    --pose configs/generated/paired_pose_g1.json `
    --out configs/generated/calibration_g1.json       # 中性 C
```

换机器人时用生成器一次写出两个文件（GMR 出厂配置里的人类骨骼统一是 LaFAN1 命名）：

```powershell
uv run python scripts/get_robot_profile_from_gmr.py --robot booster_t1_29dof `
    --gmr-root $gmr --out configs/robots/t1.json --gmr-out configs/gmr/t1.json
```

`t1.json` 与 `gmr/t1.json` 都是这次生成的快照，便于肉眼核对；GMR 若不在，profile 仍然可用，只是
`retarget.py` 需要的 `gmr/` 那份得另外手写。

## 数据流

```
                         engine/  （vendored mmd2gltf 的形变管线）
                              │
PMX + (VMD|VPD) ── engine.bake ──┤──→ 逐帧骨骼世界位姿 {骨名: [位置, 四元数]}    ← 生产线用这个
                              │
                              └── engine.mjcf ──→ MjModel（纯内存）──→ view（只为看）

成对姿态 + robots/<robot>.json ── calibration ──→ generated/calibration_<robot>.json（中性 C + 尺度）
源位姿 + C ──────────────────── retarget  ──→ 机器人 qpos ── export_gmr_pickle ──→ .pkl
```

**关键点：源侧不经过 qpos。** `Skeleton.world()` 直接给出世界位姿（朝向 0.0000° 与 MJCF 前向运动学一致），
而把局部旋转烘成 qpos 再跑 FK 反而会**丢掉 `センター` 的平移**——那正是旧路径的问题。

## 数学（整套系统的全部内容）

```
目标:  T_link(t) = S_bone(t) · C

其中 C = S_bone(P)⁻¹ · T_link(P)          ← P 是成对姿态（VPD + G1 关节角）

展开: T_link(t) = [S_bone(t) S_bone(P)⁻¹] · T_link(P)
                  └ 源相对校准姿态的增量 ┘   └ 机器人的校准姿态 ┘
```

**只需在一个姿态上校准一次**，因为运动以"相对校准姿态的增量"形式流动。`·` 是旋转复合，`C` 是唯一的
约定交接点。骨盆与躯干不被姿态改变，其世界系偏航必须用**双侧轴**（髋线、肩线）单独测量——漏了它，
手臂完美而身体是拧的。

## 文件

**生产线**

| 文件 | 输入 → 输出 |
|---|---|
| `src/mmgmr/mmd/` | vendored MMD 引擎：`pmx` `vmd` `mathutil` `animation`（形变管线）`bake` `mesh` |
| `src/mmgmr/mmd/bake.py` | `pmx + (vmd\|vpd)` → 逐帧世界位姿（已换算到机器人空间）|
| `uv run mmgmr-calibration --solve` | 解出"机器人摆成 MMD 姿态"的关节角（肢体方向匹配）|
| src/mmgmr/calibration.py | **消费**该关节角 → **中性 C**（不认引擎格式）|
| `src/mmgmr/retarget.py` | + 中性 C → 构建 GMR 配置并重定向（唯一认识 GMR 格式的地方）|
| `scripts/export_gmr_pickle.py` | G1 qpos → GMR pickle |
| `src/mmgmr/mmd/constraints.py` | **MMD 侧知识**：肢体方向段（`SEGMENT_BONES`）、LaFAN1→MMD 桥 |

**可视化专用**

| 文件 | 用途 |
|---|---|
| `src/mmgmr/mmd/mjcf.py` | **PMX → MjModel（纯内存）**；qpos 由 `engine.bake` 现算 |
| `src/mmgmr/view.py` | 播放 / 机器人+源骨架对照 |

## 复现

```powershell
$pmx = '<model.pmx>'; $vmd = '<motion.vmd>'
$gmr = 'D:\MY_TOOL\GitHub_Tools\GMR'
$g1  = "$gmr\assets\unitree_g1\g1_mocap_29dof.xml"

# 1. 配置骨架 → 2. 校准（没有机器人姿态文件时先解一次：去掉 --pose 加 --write-pose）
uv run mmgmr-calibration --pmx $pmx --robot configs/robots/g1.json `
    --base "$gmr\general_motion_retargeting\ik_configs\bvh_lafan1_to_g1.json" `
    --out configs/generated/mmd_to_g1.json
uv run mmgmr-calibration --pmx $pmx --robot configs/robots/g1.json `
    --calibration configs/generated/calibration_g1.json --pose configs/generated/paired_pose_g1.json `
    --record configs/generated/calibration_g1.json --report

# 3. 重定向（源侧无 MuJoCo，无 qpos）
$env:PYTHONPATH="src"; <gmr>/.venv/Scripts/python.exe -m mmgmr.retarget --pmx $pmx --vmd $vmd `
    --calibration configs/generated/calibration_g1.json --bake-out runtime/pmx/g1.npz

# 4. 交付
uv run python scripts/export_gmr_pickle.py --qpos runtime/pmx/g1.npz --out runtime/gmr_clips/pmx.pkl
```

可视化（可选）：

```powershell
uv run python src/mmgmr/view_model.py --xml runtime/pmx/model.xml --qpos runtime/pmx/anim.npz --loop
```

## 换模型 / 换动作 / 换机器人

| 换什么 | 只改 |
|---|---|
| **换动作** | `--vmd`（模型与校准都不动）|
| **换 MMD 模型** | 骨名不同才需要动 `mmd/constraints.py` 的 `SEGMENT_BONES`，然后重跑 1–3 |
| **换机器人** | `--robot <机器人键或配置>`；配对姿态来自 `configs/generated/paired_pose_<robot>.json` |

## 必须知道的坑（都踩过）

| 坑 | 现象 | 处理 |
|---|---|---|
| MMD 左手系 → MuJoCo 是**反射**（det = −1）| 每个面内外翻 | 三角形按 `(a,c,b)` 写出 |
| UV 原点在**上**，MuJoCo 在**下** | 整片采样错位、发灰 | `vt = (u, 1−v)` |
| mesh 顶点是模型绝对坐标，MuJoCo 要**相对所属 body** | 模型碎成一圈圈 | 减去该骨的绑定位置 |
| `颜+`（alpha 0）、`目影`（alpha 0.3）是叠加层 | 脸上黑块 | 默认丢弃 alpha < 0.5 |
| `肌2`/`肌2+` 顶点完全相同且不透明 | 胸口发灰、随视角闪 | 丢弃重复的 `+` 层（z-fighting）|
| VMD 骨名**截断到 15 字节** | 大量轨道静默丢失 | 按同样截断建映射（vendored 实现已处理）|
| 只烘局部旋转再跑 FK | **丢掉 `センター` 的整体位移** | 源侧直接用 `Skeleton.world()` |
| 离屏渲染高度 > 480 | 报错提到 `my_height`（那是提示文案占位符）| MJCF 加 `<global offwidth="1312" offheight="1024"/>` |

## 验证数字

| 检查 | 结果 |
|---|---|
| `Skeleton.world()` vs MJCF 前向运动学（朝向）| **0.0000°**（位置差 = 旧路径丢掉的根平移）|
| 校准肢体方向残差 | **0.0°**（8 段）|
| 合并后 C vs 拆分版 C | 四元数偏差 **2.2e-16**（逐位相同）|
| 新路径关节角 vs 旧路径 | 中位 **0.000°**，最大 1.759° |
| 根平移跨度（旧 → 新）| x 0.108→0.135，y 0.123→0.226，**z 0.011→0.150 m** |

## 已知缺口

- VMD 里 358 条轨道在该 PMX 上找不到对应骨（模型作者差异），属正常；
- 腿部朝向跟踪 5–7°（踝关节位置权重抢了朝向）；
- MMD 的 `toon`/`sphere` 贴图不参与渲染（MuJoCo 无对应模型）；
- 物理摇晃（`physics.py` 的刚体抖动）未启用。
