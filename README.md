# llm-eval · 轻量化 LLM 容量测评

## 工具定位

本仓库提供一套**轻量级大模型容量测评**工具：在 **OpenAI 兼容接口**（如 vLLM 暴露的 Chat / Completions）上做**阶梯并发压测**，结合**本机资源采样**与请求明细，用于评估 **当前单机硬件与部署条件下** 服务的承载区间、延迟与吞吐表现，并辅助判断**与采样数据相关的**资源瓶颈信号。

工具**不替代**完整的可观测性平台；产出以 **CSV 原始数据**、`summary.json` 聚合与 **Markdown 报告**为主，便于对内评估或工程交付沟通。

---

## 推荐使用方式

资源监控模块采集的是 **运行本工具进程所在主机** 的指标。为使 **GPU / 显存 / CPU / 内存 / 网络** 与正在承担推理负载的进程一致，建议：

**将本工具部署在与被测 vLLM（或同等推理服务）相同的服务器上执行压测。**

在同一台机器上运行时，典型可观测项包括（视环境与驱动而定）：

- GPU 利用率  
- GPU 显存占用（及利用率等导出字段）  
- CPU 利用率  
- 内存占用  
- 网络 I/O 累计（本机网卡视角）

---

## 重要说明：本机采样 ≠ base_url 所在机器

- 资源采样始终基于 **工具运行所在主机**，**不会**根据 `base_url` 自动登录或采集远端服务器上的资源。
- 若压测在 **客户端笔记本 / 跳板机** 上执行，而 vLLM 在 **远端服务器**：
  - **吞吐、时延、成功率** 等仍反映「客户端 → 服务」这一路径上的体验，**可参考**；
  - **GPU / 显存 / CPU / 内存** 等曲线反映的是 **客户端本机**，**一般不对应**远端推理节点；
  - 报告中 **资源瓶颈类推断可能失真**；工具会在控制台与 `report.md` 中对远端 `base_url` 场景给出 **明确警告**，结论需结合服务端监控交叉验证。

---

## 当前支持范围与扩展

| 项目 | 说明 |
|------|------|
| 部署形态 | **主要面向单机部署**（例如单机多卡、常见 8 卡推理节点） |
| 多机汇总 | **当前版本不支持** 多机多卡资源的统一采集与跨机汇总分析 |
| 自定义 | 若有多机或自定义指标需求，需在 **`monitor` / `analyzer` / `report`**（及配置）上 **自行扩展**；本仓库保持核心路径简单可改 |

---

## 适用场景 / 不适用场景

**适用：**

- 单机部署 vLLM（或兼容接口的推理服务）的 **容量摸底**、阶梯加压与稳定性粗判  
- **本机发压 + 本机资源采样** 对齐时的结论交付（资源与请求在同一节点）  
- 需要 **自动化 Markdown 报告** 与可复现 CSV / JSON 产物的工程流程  

**不适用（或需自研扩展）：**

- **多机多卡** 统一资源测评与跨节点汇总  
- **复杂分布式推理** 全链路拆解与归因  
- **跨广域网** 的完整端到端性能归因（本工具不替代链路上各段的独立监控）  

---

## 功能概览

| 环节 | 说明 |
|------|------|
| 配置 | YAML + Pydantic 校验，路径相对配置文件目录解析 |
| 样本 | short / medium / long 三池 JSONL，按配比随机抽样 |
| 压测 | 异步并发、`baseline` / `step` / `stability` 三种模式 |
| 分析 | 分档成功率、TTFT/延迟分位、吞吐、资源窗口统计、稳定并发与瓶颈推断（受「本机采样」约束） |
| 报告 | 面向汇报的 `report.md`（Jinja2 模板），含适用范围与远端服务警告 |

## 环境要求

- Python **3.10+**（推荐 3.11+）  
- 被测 HTTP 服务可达；本机采集 GPU 时需 NVIDIA 驱动与可用 NVML / `nvidia-smi`（无 GPU 或采集失败时相关字段可能为空，见实现与测试环境）

## 安装

### 使用 uv（推荐）

```bash
git clone <repo-url> llm-evaluation
cd llm-evaluation
uv sync --all-groups   # 含 pytest 等开发依赖
```

### 使用 pip

```bash
pip install -r requirements.txt
pip install pytest    # 仅运行测试时需要
```

安装后可使用入口脚本（若已通过 `pip install -e .` 或等价方式安装包）：

```bash
llm-eval --help
```

或直接：

```bash
python runner.py --help
```

## 配置文件说明

默认示例：`config/config.yaml`。另提供可复制模板：`examples/config.example.yaml`（绑定 `examples/prompts/` 小样本，便于试跑）。

| 块 | 作用 |
|----|------|
| `server` | `base_url`、`model`、`api_key`、`endpoint_type`（`chat_completions` / `completions`） |
| `test` | `mode`、`concurrency`、`duration_sec`、`ramp_up_sec`、`stream`、`timeout_sec`、`max_tokens` |
| `dataset` | short/medium/long 比例与三份 JSONL 路径 |
| `sampling` | 资源采样间隔、`gpu_indices`（空=自动监控本机全部 GPU）、`random_seed` |
| `threshold` | 分析用阈值（成功率、P95、GPU 等） |
| `output` | 输出根目录、`run_name`（空则自动生成时间戳子目录） |

**路径规则**：配置中的相对路径均相对于 **该 YAML 文件所在目录** 解析。

## Prompts 数据格式

每行一个 JSON 对象：`id`、`category`（short/medium/long）、`messages`（OpenAI 风格）、可选 `expected_output_tokens`。示例见 `prompts/*.jsonl`、`examples/prompts/*.jsonl`。

## 运行方式

**校验配置 + 加载样本（不发起请求）**

```bash
uv run python runner.py --config examples/config.example.yaml --dry-run
```

**完整测评流程**（压测 → CSV → `summary.json` → `report.md`）

```bash
uv run python runner.py --config config/config.yaml
```

常用参数：`--config` / `-c`、`--dry-run`、`--log-level`。

## 输出结果说明

每次运行写入 `output.base_dir` 下子目录（如 `run_YYYYMMDD_HHMMSS/`）：

| 文件 | 内容 |
|------|------|
| `raw_requests.csv` | 请求耗时、token、成功与否等 |
| `resource_usage.csv` | 本机资源按间隔采样（多 GPU 时为每卡多行） |
| `requests_*.jsonl` | 各档位可选明细 |
| `summary.json` | 聚合指标、`conclusions`、`environment_assumptions`（采样范围与远端警告标记） |
| `report.md` | Markdown 报告（含适用范围与瓶颈表述限定） |

## 运行测试

```bash
uv run pytest tests/ -q
```

## 仓库结构（摘）

```
config/           # 示例配置
examples/         # 模板配置与极简 prompts
prompts/          # 示例样本
runner.py         # CLI 入口
config_loader.py  # 配置
sampler.py        # 样本
client.py         # HTTP 客户端
benchmark.py      # 压测与落盘
monitor.py        # 本机资源采样
analyzer.py       # 统计与结论
report.py         # 报告
utils/            # 工具函数
tests/            # 单元测试
```

## 许可证

若对外交付，请根据组织要求补充 `LICENSE` 文件。
