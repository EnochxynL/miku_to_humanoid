# PMX → G1 动作重定向

把 MMD 模型（PMX）的动作（VMD）搬到 Unitree G1 上，产出 GMR 能读的 pickle，最终变成 MPC-WBC 的
500 Hz 参考。**中间不经过 glTF**：PMX 是模型、VMD/VPD 是动作，二者天然解耦。

## 数据流

```
PMX ─┬─ pmx_to_mjcf.py ──→ 骨架 MJCF（body/joint 名 = MMD 骨名）+ 可选贴图模型
     │
     ├─ VMD + engine.bake ─→ 逐帧 qpos（参考烘焙：vendored mmd2gltf 的形变管线）
     │
     └─ VPD + G1 关节角 ──→ calibrate.py ──→ C（每链接一个常量）

骨架 MJCF + qpos + G1 模型 + C ──→ retarget.py ──→ G1 qpos ──→ export_gmr_pickle.py ──→ .pkl
```

## 数学（整套系统的全部内容）

```
目标:  T_link(t) = S_bone(t) · C

其中 C = S_bone(P)⁻¹ · T_link(P)          ← P 是成对姿态（VPD + G1 关节角）

展开: T_link(t) = [S_bone(t) S_bone(P)⁻¹] · T_link(P)
                  └ 源相对校准姿态的增量 ┘   └ 机器人的校准姿态 ┘
```

**只需在一个姿态上校准一次**，因为运动以"相对校准姿态的增量"形式流动。`·` 是旋转复合，`C` 是
唯一的约定交接点。两个链接（骨盆、躯干）不被姿态改变，其世界系偏航必须用**双侧轴**（髋线、肩线）
单独测量——漏了它，手臂完美而身体是拧的。

## 文件

| 文件 | 作用 |
|---|---|
| `src/mmd/engine/pmx.py` | PMX 解析（骨、顶点、材质、贴图）|
| `src/mmd/engine/vmd.py` | VMD 解析（贝塞尔曲线、IK 帧）|
| `src/mmd/engine/mathutil.py` | 四元数/向量工具 |
| `src/mmd/engine/animation.py` | **烘焙管线**（逐字 vendored 自 mmd2gltf）|
| `src/mmd/engine/mesh.py` | 网格 → OBJ 组 + PNG + 材质（反射、UV、叠层全处理）|
| `src/mmd/engine/bake.py` | 参考烘焙的 `local_q` → MuJoCo qpos |
| `src/mmd/pmx_to_mjcf.py` | PMX 骨架 → MJCF（`--mesh` 出贴图模型）|
| `src/mmd/mmd_bones.py` | **MMD 知识**：链接→骨名、骨→下一关节（跳过捩/袖骨）|
| `src/mmd/measurements.py` | 量尺：报告误差用哪段（不参与烘焙）|
| `src/mmd/calibrate.py` | 成对姿态 → C（`--pose` 读机器人侧，`--vpd` 读源侧）|
| `src/mmd/retarget.py` | 源 qpos + C → G1 qpos |
| `src/mmd/export_gmr_pickle.py` | G1 qpos → GMR pickle |
| `src/mmd/view_model.py` / `view_skeleton.py` | 播放 / 机器人+源骨架对照 |

## 复现

```powershell
$env:PYTHONPATH='src;src\mmd'
$pmx = '<model.pmx>'; $vmd = '<motion.vmd>'

# 1. 骨架（--mesh 同时出贴图模型，给 viewer 看）
uv run python src/mmd/pmx_to_mjcf.py --pmx $pmx --out runtime/pmx/skeleton.xml --tree runtime/pmx/tree.json
uv run python src/mmd/pmx_to_mjcf.py --pmx $pmx --out runtime/pmx/model.xml --tree runtime/pmx/tree2.json --mesh

# 2. 动作
uv run python -m engine.bake --pmx $pmx --vmd $vmd --xml runtime/pmx/skeleton.xml --out runtime/pmx/anim.npz

# 3. 校准：没有机器人姿态文件就先解一次并存成数据
<gmr>/.venv/Scripts/python.exe src/mmd/retarget.py --build-config configs/mmd/pmx_to_g1.json `
    --source-xml runtime/pmx/skeleton.xml --anim runtime/pmx/anim.npz --frames 0
uv run python src/mmd/calibrate.py --source-xml runtime/pmx/skeleton.xml `
    --config configs/mmd/pmx_to_g1.json --write-pose configs/mmd/paired_pose_pmx.json `
    --record configs/mmd/calibration_pmx.json --report

# 4. 重定向
<gmr>/.venv/Scripts/python.exe src/mmd/retarget.py --config configs/mmd/pmx_to_g1.json `
    --source-xml runtime/pmx/skeleton.xml --anim runtime/pmx/anim.npz --bake-out runtime/pmx/g1.npz

# 5. 交付
uv run python src/mmd/export_gmr_pickle.py --qpos runtime/pmx/g1.npz --out runtime/gmr_clips/pmx.pkl
```

## 换模型 / 换机器人

| 换什么 | 只改 |
|---|---|
| **换 MMD 模型** | `src/mmd/mmd_bones.py`（骨名不同才需要），然后重跑 1–4 |
| **换动作** | `--vmd`（模型与步态无关，不需要重新生成模型）|
| **换机器人** | `--robot-xml` + `--pose`（那台机器人摆进成对姿态的关节角）|

## 必须知道的坑（都踩过）

| 坑 | 现象 | 处理 |
|---|---|---|
| MMD 左手系 → MuJoCo 是**反射**（det = −1）| 每个面内外翻 | 三角形按 `(a,c,b)` 写出 |
| UV 原点在**上**，MuJoCo 在**下** | 整片采样错位、发灰 | `vt = (u, 1−v)` |
| mesh 顶点是模型绝对坐标，MuJoCo 要**相对所属 body** | 模型碎成一圈圈 | 减去该骨的绑定位置 |
| `颜+`（alpha 0）、`目影`（alpha 0.3）是叠加层 | 脸上黑块 | 默认丢弃 alpha < 0.5 |
| `肌2`/`肌2+` 顶点完全相同且都不透明 | 胸口发灰、随视角闪 | 丢弃重复的 `+` 层（z-fighting）|
| VMD 骨名**截断到 15 字节** | 大量轨道静默丢失 | 按同样截断建映射（vendored 实现已处理）|
| 离屏渲染高度 > 480 | 报错提到 `my_height`（那是提示文案占位符）| MJCF 加 `<global offwidth="1312" offheight="1024"/>` |

## 验证数字

| 检查 | 结果 |
|---|---|
| 校准肢体方向残差 | **0.0°**（8 段）|
| 与 glTF 路线的 C 关系 | `C_pmx = Rx90 · C_glb`，12 条肢体链接 **差 0.0°** |
| 端到端 vs 已验收 glTF 结果 | 关节角 **中位 1.05°、最大 16.96°** |

## 已知缺口

- VMD 里 358 条轨道在该 PMX 上找不到对应骨（模型作者差异），属正常；
- 腿部朝向跟踪 5–7°（踝关节位置权重抢了朝向）；
- MMD 的 `toon`/`sphere` 贴图不参与渲染（MuJoCo 无对应模型）；
- 物理摇晃（`physics.py` 的刚体抖动）未启用。
