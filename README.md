# llm-eval · 轻量化 LLM 容量测评

面向推理服务（OpenAI 兼容 Chat/Completions）的**阶梯并发压测**工具：按配置逐级加压、采集请求明细与本机/GPU 资源采样，自动产出 **CSV 原始数据**、**summary.json** 统计与 **Markdown 汇报文档**，便于对内评估或向客户交付容量结论。

## 功能概览

| 环节 | 说明 |
|------|------|
| 配置 | YAML + Pydantic 校验，路径相对配置文件目录解析 |
| 样本 | short / medium / long 三池 JSONL，按配比随机抽样 |
| 压测 | 异步并发、`baseline` / `step` / `stability` 三种模式 |
| 分析 | 分档成功率、TTFT/延迟分位、吞吐、GPU 窗口统计、稳定并发与瓶颈推断 |
| 报告 | 面向汇报的 `report.md`（Jinja2 模板） |

## 环境要求

- Python **3.10+**（推荐 3.11+）
- 被测服务可访问；本机采集 GPU 时需 NVIDIA 驱动（无 GPU 时监控字段可能为空或降级，视环境而定）

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

默认示例：`config/config.yaml`。另提供**可复制的模板**：`examples/config.example.yaml`（绑定 `examples/prompts/` 小样本，便于试跑）。

主要块说明：

| 块 | 作用 |
|----|------|
| `server` | `base_url`、`model`、`api_key`、`endpoint_type`（`chat_completions` / `completions`） |
| `test` | `mode`、`concurrency` 档位列表、`duration_sec` 单档时长、`ramp_up_sec` 预热、`stream`、`timeout_sec`、`max_tokens` |
| `dataset` | `short_ratio` / `medium_ratio` / `long_ratio`（和需≈1）、三份 `*_file` JSONL 路径 |
| `sampling` | 资源采样间隔、`random_seed`（样本抽样可复现） |
| `threshold` | 分析用：成功率与 P95 TTFT/延迟上限、GPU 阈值、安全并发比例等 |
| `output` | `base_dir` 输出根目录、`run_name` 子目录（空则自动生成时间戳） |

**路径规则**：配置中出现的相对路径（如 `dataset.*_file`、`output.base_dir`）均相对于**该 YAML 文件所在目录**解析。

## Prompts 数据格式

每行一个 JSON 对象，字段包括：

- `id`：字符串，唯一
- `category`：`short` | `medium` | `long`，须与所在文件语义一致
- `messages`：OpenAI 风格 `{"role","content"}` 列表
- `expected_output_tokens`（可选）：若存在则覆盖该样本的 `max_tokens` 上限

示例见仓库内 `prompts/*.jsonl` 或 `examples/prompts/*.jsonl`。

## 运行方式

**校验配置 + 加载样本（不发起请求）**

```bash
uv run python runner.py --config examples/config.example.yaml --dry-run
```

**完整测评流程**（压测 → CSV → `summary.json` → `report.md`）

```bash
uv run python runner.py --config config/config.yaml
```

常用参数：

- `--config` / `-c`：配置文件路径（默认 `config/config.yaml`）
- `--dry-run`：仅校验与样本加载
- `--log-level`：如 `DEBUG`

## 输出结果说明

每次运行写入 `output.base_dir` 下的子目录（如 `run_20260417_153000/`），典型产物：

| 文件 | 内容 |
|------|------|
| `raw_requests.csv` | 每条请求的耗时、token、成功与否等明细 |
| `resource_usage.csv` | 按间隔采样的 CPU/内存/GPU 等 |
| `requests_*.jsonl` | 各档位可选留存的结构化请求记录 |
| `summary.json` | 分档聚合指标与 `conclusions`（最大稳定并发、安全区间、瓶颈类型等） |
| `report.md` | 面向汇报的 Markdown 报告 |

控制台结束时会摘要**输出目录**与**关键结论**（瓶颈、稳定并发、建议区间等）。

## 运行测试

```bash
uv run pytest tests/ -q
```

测试覆盖：配置加载与校验、JSONL 样本解析、`analyzer` 基于 CSV 的统计与稳定并发判定等。

## 仓库结构（摘）

```
config/           # 本地/演示用配置
examples/         # 可复制的配置模板与极简 prompts
prompts/          # 默认示例样本（可按项目替换）
runner.py         # CLI 入口
config_loader.py  # 配置模型与加载
sampler.py        # 样本加载与抽样
client.py         # HTTP 客户端
benchmark.py      # 压测编排与落盘
analyzer.py       # 统计与结论
report.py         # Markdown 报告
utils/            # 日志、时间等
tests/            # 单元测试
```

## 许可证

若对外交付，请根据组织要求补充 `LICENSE` 文件。
