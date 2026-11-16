# MMD → 机器人：以 PMX 刚体为源的流水线

> 2026-11-15 打通并验证。取代此前"MMD → BVH → GMR"的路线（那条路已判定不成立，见
> `gmr_offset_derivation.md`）。

## 为什么换路线

旧路线把 MMD 塞进 LaFAN1 的 21 个骨名，再让 GMR 用"每根链接一个固定常量"去描述人。
实测证明**那个常量不存在**：用播放器的真实姿态量 `R_human⁻¹·R_link`，跨帧抖动 12–32°（前臂 100°+），
因为 MMD 的绑定语义（多骨合成一个关节、`flexion` 标量、付与親、A-pose rest）无法用逐链接常量表达。

新路线把 **PMX 骨架本身当成一台机器人**（刚体模型），于是：

- 源数据就是"每根链接的世界位姿"，不存在"人骨帧 ↔ 机器人链接帧"的对应问题；
- PMX 骨**没有 rest 旋转**，`R_pmx(t)` 本身就是相对 bind 的增量 → 常量只需 `C = R_rest(link) · align`；
- PMX 没有机器人限位，中间姿态不被压扁。

## 四段流水线

```
① PMX ──► MuJoCo 刚体模型          engine/mmd_model.py
      每骨一个 body + ball joint，根一个 free joint，零位 FK 误差 9.9e-10 m
      坐标：(-z, x, y) * scale，scale=0.07
      可视化：仅主干 24 根骨画胶囊（否则裙摆/头发淹没画面）

② VMD ──► PMX qpos（带 IK）        bake_pmx_vmd.py
      world_R(i) = world_R(parent) · local_q[i]      # PMX 无 rest 旋转
      ① 付与親：local_q[i] = RotVec(ratio·rotvec(src)) · local_q[i]
      ② IK：CCD，goal = IK 骨自身世界位置；每 link 每次 ≤ limit·(index+1)，
         随后按 link 的欧拉上下限钳制；按 (after_physics, layer, index) 顺序
      ③ 骨名按 shift-jis 截断 15 字节匹配 VMD 键
      输出 qpos = mmd_quat_to_mujoco(local_q)，即 (x,y,z,w) → (w,z,-x,-y)

③ PMX qpos ──► GMR 输入            pmx_to_gmr.py
      每帧 FK → 每根 PMX 骨的世界位姿 {骨名: [位置, 朝向(wxyz)]}
      常量 C = R_G1rest(link) · align
        · align 由两模型**解剖对应段**的绑定方向算出（rotation_between）
        · 脊柱链（pelvis / torso_link）**不加 align**，见下"踩过的坑"
      比例表：每段 robot_len / pmx_len（GMR 的 human_scale_table）

④ GMR IK ──► 目标机器人 qpos
      mink 加权最小二乘；追加 mink.PostureTask(cost=1e-2, target=qpos0)
      输出再做一阶时间平滑 α=0.5
```

## 最终精度（左/右，同解剖段方向误差）

| 段 | 误差 |
|---|---|
| 肩（上臂）| **0.1°** |
| 踝 | 6.9 / 7.0° |
| 髋（大腿）| 7.4 / 9.3° |
| 前臂 | 10.6 / 10.9° |
| 膝（小腿）| 12.2 / 13.3° |

膝抖动：7.73° → **1.65°**（最大单帧跳 22.5° → 7.84°）。

## 踩过的三个坑（都会重现，值得记住）

### 1. 脊柱绝不能加 align

这个 PMX 的 `腰 → 上半身` 在 bind 时**前倾 35°**（建模习惯，`腰` 在 `上半身` 后方），
而动画第 0 帧把它纠正成竖直。用 bind 方向算 align 会得到 **40.1° 的前倾常量**，
加到**骨盆**上 → 整个人折成坐姿，双腿跟着歪。

修：`pelvis` / `torso_link` 不参与 align。四肢必须参与，因为 G1 的肘在 qpos 0 就已经弯 75.8°，
而 PMX 的肘是直的（肘的 align 达 86.6°）。

### 2. align 的对照段必须解剖对应

第一版用"本链接 → 子链接"直接算，但 G1 的 `torso_link → head_link` 方向**朝下**
（`[0.09, 0, −0.996]`），与 PMX 的 `上半身2 → 首`（朝上）配对 → 166° 旋转，机器人对折。
正确做法：用 `MEASURE` 表（`pelvis→torso_link` 对 `腰→上半身`、`前臂` 对 `前臂` 等）。

### 3. 膝抖动是零空间漂移，不是没算够

`FrameTask` 只约束链接位姿，髋和膝可以反向交换（大腿 11° / 膝 20°）而在膝关节下方几乎抵消——
约束误差不变，求解器没有理由停下，肉眼却看到膝在抽。加大迭代（10→100）**完全无效**。

修：`mink.PostureTask` 把零空间钉住（精度还变好：大腿 7.77→5.08），再叠 α=0.5 的时间平滑。
平滑是滤波器，含约 33 ms 滞后；α=0.7 反而更差。

## 复现

```powershell
# ① ② PMX → qpos（--view 开窗口）
uv run python src/mmd/engine/mmd_model.py --pmx <model.pmx> --out runtime/pmx/herta.xml --tree runtime/pmx/herta_tree.json
uv run python src/mmd/bake_pmx_vmd.py --pmx <model.pmx> --vmd <motion.vmd> --xml runtime/pmx/herta.xml --tree runtime/pmx/herta_tree.json --out runtime/pmx/herta_qpos.npz --view

# ③ ④ 重定向到 G1 并播放（在 GMR venv 下）
<gmr>/.venv/Scripts/python.exe src/mmd/pmx_to_gmr.py \
  --pmx-xml runtime/pmx/herta.xml --tree runtime/pmx/herta_tree.json --qpos runtime/pmx/herta_qpos.npz \
  --config-out runtime/gmr_inputs/pmx_to_g1.json --retarget --bake-out runtime/pmx/g1_qpos.npz --view
```

## 换机器人

`pmx_to_gmr.py` 的常量全部由两个模型的**绑定几何**算出，没有手工参数；换目标机器人只需
`--robot <name> --robot-xml <它的 MJCF>`，并确认 `LINKS` / `MEASURE` 里的链接名存在于该机型。
GMR 自带的 17 个机器人配置可作为参考，但这里的常量生成是独立的。

## 已知限制

- **手腕**：PMX 的旋前旋后住在 `腕捩/手捩`，这段舞里它们是 identity，所以手腕细节未被驱动；
  腕链接当前也不加 align（无对应段）。
- **膝 12–13°、髋 7–9°**：再压小需要改约束设计（例如对膝只约束方向、放弃滚转），不是调参。
- 脊柱指标（37.7°）**不是误差**：它量的是本模型脊柱骨 35° 的建模倾角，不该被复现。
