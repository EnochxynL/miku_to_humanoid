# configs/

三种东西，别混。总览与命令见仓库根目录的 [README](../README.md)。

| 目录 | 内容 | 谁维护 |
|---|---|---|
| `robots/` | **引擎无关**的机器人描述：`links`（链接↔MMD 骨）、`segments`、`root_links`、`bilateral_links`、`xml`，以及一行 `gmr_config` 指向引擎配置 | 手写，或用生成器写出 |
| `gmr/` | **GMR 专属**：`ik_match_table1/2`（`[骨名, pos_w, rot_w, pos_offset, C_quat]`）、`robot_root_name`、`human_root_name`、`ground_height`、`use_ik_match_table1/2` | 生成器写出；换引擎时整个作废 |
| `generated/` | **产物**：`paired_pose_<robot>.json`（关节角）、`calibration_<robot>.json`（中性 C + 尺度） | 不要手改，随时可重算 |

## 重算产物

```powershell
uv run mmdmr-calibration --pmx <model.pmx> --robot configs/robots/<robot>.json `
    --vpd <pose.vpd> --solve `
    --pose-out configs/generated/paired_pose_<robot>.json `
    --out configs/generated/calibration_<robot>.json
```

## 换机器人 / 新增机器人

GMR 支持的机器人不需要手写，生成器一次写出两个文件（机器人的 MJCF 优先取 `assets/<robot>/` 里的副本）：

```powershell
uv run python scripts/get_robot_profile_from_gmr.py --robot <gmr-key> `
    --gmr-root <gmr> --out configs/robots/<robot>.json --gmr-out configs/gmr/<robot>.json
```

GMR 出厂权重需要偏离时（例如 G1 的 `pelvis.rot_w=10`、手臂 `pos_w=0`）用 `--override link.field=value`
把它写进引擎配置——**偏离必须是数据，不能是代码里的常量**。
