# python/ 脚本职责

按"做什么"分成四组。PC 侧和板端共用的只有 `audio_frontend.py`。

## 前端（PC ↔ 板端必须一致）

| 文件 | 职责 | 在哪跑 |
| --- | --- | --- |
| `audio_frontend.py` | Log-Mel 参考实现（float）+ 定点模拟实现 | 两边 |
| `mel_parity.py` | 一致性归因、排障路径、C dump 比对 | PC |
| `build_eval_set.py` | 合成评测集（含 4 类硬负例） | PC |

`audio_frontend.py` 是唯一真源：训练、PC 验证、板端 C 实现、量化校准全部对齐它。

## 模型

| 文件 | 职责 | 依赖 |
| --- | --- | --- |
| `train_tiny_cnn.py` | BC-ResNet 6 候选 + 蒸馏 + 帕累托 | torch |
| `convert_adla.py` | ONNX → `.adla`，写版本指纹 | amlnn-toolkit |
| `probe_operators.py` | 算子可行性体检 | onnx（可选） |

## 板端

| 文件 | 职责 | 依赖 |
| --- | --- | --- |
| `board_infer.py` | 流式推理 + 分层延迟 + npu_duty | numpy / amlnn_edge_toolkit_lite |
| `board_power_budget.py` | 功耗账本（cpuidle / INA226 双口径） | 标准库 / smbus2 |
| `fsm.py` | 非对称事件状态机（可独立自检） | 无 |

## 决策与体检

| 文件 | 职责 |
| --- | --- |
| `check_env.py` | 环境自检；板端查 ALSA / cpuidle / NPU 库 |
| `fsm.py --demo` | 用人为构造的分数序列验证非对称策略 |

## 约定

- 所有脚本 `--out` 默认写 `bench/logs/*.json`，**结论只认这些文件**，口头估算不算。
- 延迟一律 P50/P99，且 fastpath 与 confirm 分开报。
- 任何脚本在依赖缺失时打印 `[skip]` / `[fail]` 并给出安装提示，不静默失败。
