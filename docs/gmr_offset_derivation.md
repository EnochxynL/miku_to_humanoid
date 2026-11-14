# GMR 偏移（rot_offset）成因排查

> 目标：回答"全身扭转、连 rest 朝向都要调参，是不是 bug"。本轮不靠观感，全部用可复现的测量。

## 1. GMR 这一层的真实语义（代码审计）

| 事实 | 位置 |
|---|---|
| 配置里 `rot_offset` 读入即 `R.from_quat(rot_offset, scalar_first=True)`，**标量在前 (w,x,y,z)** | `motion_retarget.py:129` |
| 目标朝向 = `R_human · rot_offset`，即偏移在**人骨局部坐标系**里右乘 | `offset_human_data()` |
| 配置坐标系为 **Z-up**（`ground = ground_height * [0,0,1]`）| `motion_retarget.py:85` |
| 每个任务 = mink `FrameTask(position_cost, orientation_cost)`，同一条人骨同时给位置和朝向 | `setup_retarget_configuration` |
| **仓库里没有任何推导/标定 `rot_offset` 的工具** | 全库 grep `calibrat/align_vectors/rot_offset` 无生成器 |

出厂 `bvh_lafan1_to_g1.json` 的权重分布：pelvis/髋/膝/肘/腕 `[0,10]`，踝 `[50,10]`，躯干/肩 `[0,100]`——**几乎纯朝向驱动**，位置只在踝。

因此：偏移是**逐 (源骨架, 机器人) 手工给出**的常量。SMPL-X 有 17 份配置、LaFAN1 有 8 份，都是各自标的。这解释了"为什么换骨架感觉要重调"——但**不代表只能靠肉眼**。

## 2. 本轮做的测量与结论

### 2.1 数据本身是自洽的（排除"导出坏了"）

| 检查 | 结果 |
|---|---|
| GLB bind pose 的骨朝向 | 全部 `[0,0,0]` ✓ |
| BVH 中骨朝向 vs GLB 的世界朝向（逐帧对比）| **0.0°** ✓ |
| GLB 里"髋→脊柱"位置方向 vs 该骨局部 Y 轴 | 差 **12.6°** ✓ 自洽 |
| BVH 中 `LeftUpLeg` / `RightUpLeg` 的 x 坐标 | `+0.079` / `−0.061` ✓ 左右没认错 |
| pybvh / MuJoCo 渲染 | 骨架站直 ✓ |

### 2.2 `rot_offset` 是不是可以推导：能，但要选对方程

| 推导方式 | 目标误差（目标朝向 vs 人骨）| 说明 |
|---|---|---|
| 用户手调值 | 57.8° | 早期在四元数顺序有 bug 时调的 |
| **指向轴（distal）几何推导** | **4.6°** | 每根骨用"链接指向子链接"的局部轴 → 人骨"指向子骨"的局部轴 |
| bind pose 全对应（`C = R_bind⁻¹·R_rest`）| 88.2° | **错**：bind pose 与机器人 rest 是两个不同姿态，强行对应把 88° 的差烘焙进去 |
| mesh PCA 求指向轴 | 54.4° | **错**：MJCF 的 body 原点在关节处，相邻原点仅 4.6–8 cm，PCA 主轴与骨方向不是一回事 |

### 2.3 "目标对了"≠"姿态对了"——本轮的负结果

用 distal 推导（目标误差 4.6°）渲染：机器人能站，但**手臂/脚的姿态仍然不对**。

逐帧量到的残差（frame 0）：

| 骨 | 目标误差 | 实际误差 |
|---|---|---|
| LeftUpLeg | 0.0 | 30.0 |
| LeftFootMod | 0.0 | 89.4 |
| LeftArm | 0.0 | 4.5 |
| LeftForeArm | 0.1 | 58.1 |
| RightForeArm | 0.1 | 95.1 |

放宽求解预算（`max_iter` 10→200，`damping` 0.5→0.05）后只从 40.8° 降到 38.7°——**已收敛**，不是算不够。

### 2.4 为什么"指向轴对了"还会歪：指向只定了 2/3 个自由度

`C` 是 3 个自由度。指向轴只约束 2 个，**剩下的滚转决定了子关节的弯曲平面**。滚转选错时，前臂够不到目标，求解器只能折中——这正是 35–140° 残差的来源。

补第二个轴（子关节铰链）的尝试：

| 做法 | 结果 |
|---|---|
| 机器人铰链轴 = 子关节 `xaxis` 在父链接局部系下的表示 | ✓ 可以直接拿到 |
| 人骨铰链轴 = 父→子相对旋转的转轴，在整段 clip 上加权平均 | ✓ 可以量出来 |
| 两轴正交化后构造 `C = B_human·B_robotᵀ` | **更差**（前臂 35°→140°）|

原因待查：MMD 骨的相对旋转不是纯铰链，平均出来的轴可能被其他分量的数值噪声主导；且符号需要按弯曲方向对齐。

## 3. 目前可复现的产物

| 文件 | 作用 |
|---|---|
| `src/reference_maker/gmr_derive_offsets.py` | 从几何推导 `rot_offset`（distal / distal+hinge 两种模式）|
| `src/reference_maker/score_config.py` | 给任意配置打分：**目标误差**（配置对不对）与**实际误差**（求解器跟不跟得上）分离 |
| `runtime/gmr_inputs/_derived.json` | distal 推导产物（目标误差 4.6°，姿态仍不对）|
| `src/reference_maker/offset_editor.py` | 逐骨调整的可视化编辑器（含 facing 读数）|

### 2.5 又排除一个：转换器的左右手没问题

真正生成该 GLB 的工具是 `py_mmd_anim`（不是 `mmd2gltf-gui`；后者注释里写着
"position / normal : (x, y, z) -> (x, y, -z) …… rotations unaffected"，但**不是**产出该文件的工具）。
`py_mmd_anim` 的 Rust 端有 `Conversion::rotation` 与单测 `mirror_conjugation_flips_x_and_y`，
说明它会镜像旋转。

实测检验（绑定姿态，左右成对骨）：

| 检查 | 结果 |
|---|---|
| `R_right` vs `M · R_left · M`（M = diag(−1,1,1)）| **0.0°**（左腕/右腕、左ひじ/右ひじ、左足/右足、左ひざ/右ひざ、左足首/右足首）|
| 位置是否镜像 | 精确相等 ✓ |

→ **GLB 是自洽且正确镜像的**，转换器不是问题。

## 4. 下一步（未做）

1. **滚转的正确来源**。最可能成立的定义：让人骨相对旋转的**主轴**与机器人对应关节的**铰链轴**一致——本轮实现有 bug（符号/正交化/平均方式），需要重做并用 `score_config.py` 的"实际误差"验证。
2. **参考姿态的选择**：不要用舞蹈首帧，也不要用 bind pose 直接对应；正确做法可能是"用一个已知关节角全为 0 的姿态"。
3. **换动作验证**：用第二个 VMD（`pmx-mjcf/motions/胡桃待機モーション.vmd`，经 `mmd2gltf-gui` 转 GLB）走同一流程，确认推导是**动作无关**的；这是"换个 vmd 会不会又出问题"的答案。
4. ~~转换器审计~~ **已完成并排除**（见 2.5）：GLB 左右对称 0.0°，镜像不是原因。
