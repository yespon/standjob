# 岗标辅导系统 — Standjob

> 基于 **LangGraph** 构建的引导式、交互式岗标表格辅导系统。  
> 当用户上传"岗位价值与岗位任务"Excel 时，系统不直接给出答案，而是结合 14 项评审标准和教材内容，以苏格拉底式追问逐条引导用户自我反思、完善表格。

---

## 目录结构

```
standjob/
├── backend/                    # Python 后端
│   ├── graph/                  # LangGraph 核心模块
│   │   ├── state.py            # 全局状态定义（CoachState + 14 项评分标准 + 引导武器表）
│   │   ├── nodes.py            # 所有 LangGraph Node 实现（7 个功能节点 + 2 个等待节点）
│   │   └── builder.py          # Graph 组装、条件边、send_user_message 工具函数
│   ├── references/             # 内嵌参考文档
│   │   ├── rubric.md           # 14 项评审打分标准
│   │   └── textbook.md         # 辅导教材精要（含自检十一问）
│   ├── scripts/                # 辅助脚本
│   │   ├── validate_sheets.py  # Excel 结构校验 + 内容提取
│   │   └── progress.py         # 进度持久化管理（init/show/update/next/reset）
│   ├── loaders.py              # 文件读取工具（xlsx / pdf → 结构化数据）
│   ├── api.py                  # FastAPI REST API + SSE 流式端点
│   ├── cli.py                  # 交互式 CLI 演示入口
│   └── __init__.py
├── web/                        # Next.js 前端
├── data/                       # 原始数据文件（评分标准 xlsx + 教材 pdf）
├── scripts/
│   └── e2e_regression.py       # 端到端回归测试
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
├── pyproject.toml
└── README.md
```

---

## Graph 架构

系统完整实现了 gangbiao-coach 技能的 **阶段 0-4 生命周期**：

```
阶段 0: 文件采集                   阶段 4: 收尾
───────────────                   ──────────────
START → load_standards            generate_closure
  │ (检查进度)                       │
  ▼                                  ▼
wait_for_file ⏸                  wait_for_reply ⏸
  │ (用户提供文件)                    │ (用户可继续提问)
  ▼                                  │
阶段 1: 结构校验                    │
validate_structure ──(失败)──→ wait_for_file
  │ (通过)
  ▼
阶段 2: 全面评审
review_item ←───────────────────┐
  │ (phase=guiding)              │
  ▼                              │
阶段 3: 分步引导                   │
guide_reflection                 │
  │                              │
  ▼                              │
wait_for_reply ⏸                │
  │ (用户消息)                    │
  ▼                              │
detect_user_intent               │
  ├─ question → answer_user_question ──→ wait_for_reply
  └─ reply → process_response
       ├─ continue → guide_reflection ─┘
       ├─ next_item → review_item
       └─ done → generate_closure → END
```

### 节点说明

| Node | 阶段 | 职责 |
|------|------|------|
| `load_standards` | 0 | 加载内嵌的 rubric.md + textbook.md；检查是否有已保存进度，提供恢复选项 |
| `wait_for_file` | 0 | ⏸ interrupt：等待用户上传文件 |
| `validate_structure` | 1 | 调用 validate_sheets.py 校验文件结构，通过后提取结构化数据；失败则回退到内部 load_submission |
| `review_item` | 2 | 对当前条目逐条对照 14 项评分标准进行评审；三问法筛选 + 灰区放过；岗位价值问题优先 |
| `guide_reflection` | 3 | 基于引导武器表 + 自检十一问，生成苏格拉底式引导问题；遵循 肯定→切入→一问→等待 结构 |
| `wait_for_reply` | 3 | ⏸ interrupt：等待用户回复 |
| `detect_user_intent` | 3 | 规则优先意图识别，低置信度走 LLM 二判（reply / question） |
| `process_response` | 3 | 回判三步：肯定进步→给出判断→追或不追；卡住 3 次（9 轮）自动放过 |
| `answer_user_question` | 3 | 被动答疑模式，回答后自然过渡回引导流程 |
| `generate_closure` | 4 | 自然总结：亮点 + 待打磨 + 教材收束；不给打分；清理进度 |

---

## 核心设计

### 五条铁律

1. **不直接给答案** — 用提问引导用户自己想，永远不说"应该写什么"
2. **严格依据评分标准** — 逐条对照 14 项评分标准，不用自定义维度
3. **教材仅作背景** — 基于教材但用人话表达，不报章节号、不报条目编号
4. **每次聚焦一点** — 一轮只谈一个问题项
5. **正向激励** — 先认可具体优点，再指出改进点

### 三问法（严格度校准）

每条价值/任务先用三问法筛选，三问都过则放过不追问：

1. 能圈出明确的"客户主语"吗？
2. 客户能看到自己拿到的"具体好处"吗？
3. 最终落到 收入/成本/风险/品牌 中至少一个吗？

**灰色地带默认放过** — 标准没明文覆盖的不主动找事。

### 回判三步

用户回复后的判断流程：

1. **肯定进步** — 必须指出比上版好在哪个具体点
2. **给出判断** — 自然地告诉用户这一项过了没
3. **追或不追** — 三问法过了就放过；未过则更具体地再问；卡住超 3 轮升级提示，超 9 轮自动放过

> "还能更好"不是追问的理由，"评委会扣分"才是。

### 状态流转

```
init → loaded → validating → reviewing → guiding → ... → done → closure
```

| Phase | 含义 |
|-------|------|
| `init` | 初始状态 |
| `loaded` | 评审标准已加载，等待文件 |
| `validating` | 结构校验中 |
| `reviewing` | 评审当前条目 |
| `guiding` | 分步引导中 |
| `done` | 所有条目辅导完成 |
| `closure` | 收尾总结，可继续答疑 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/session/start` | 创建新会话（返回 thread_id + 初始 AI 消息） |
| POST | `/api/session/start/stream` | 同上，SSE 流式返回 |
| POST | `/api/upload` | 上传 xlsx 文件，返回服务端路径 |
| POST | `/api/chat` | 发送用户消息，获取 AI 回复 |
| POST | `/api/chat/stream` | 同上，SSE 流式返回 |
| GET | `/api/session/{thread_id}/state` | 获取当前会话状态 |

---

## 快速开始

### 1. 安装依赖

```bash
# 后端
pip install -e .
# 或使用 uv
uv sync

# 前端
cd web && npm install
```

### 2. 配置环境变量

复制 `.env` 并填入实际值：

```bash
# LLM 配置（二选一，支持 DeepSeek / OpenAI / Anthropic 兼容接口）
OPENAI_BASE_URL=https://your-llm-endpoint/v1
OPENAI_API_KEY=sk-xxx

# 可选：不使用 LLM 时启用规则兜底模式
# STANDJOB_DISABLE_LLM=1

# 可选：覆盖内嵌的参考文档路径
# SCORING_CRITERIA_PATH=/path/to/评分标准.xlsx
# TEACHING_MATERIAL_PATH=/path/to/教材.pdf
```

> 不设置 `SCORING_CRITERIA_PATH` / `TEACHING_MATERIAL_PATH` 时，系统自动使用 `backend/references/` 下的内嵌文档。

### 3. 运行

```bash
# CLI 模式
python -m backend.cli

# API 模式
uvicorn backend.api:app --reload

# Docker
docker compose up -d --build
# 前端访问 http://127.0.0.1:8080
# API 文档 http://127.0.0.1:8080/api/session/start (POST)
```

> **端口说明**：Docker 通过 nginx 在 `127.0.0.1:8080` 提供服务（仅 IPv4）。如果 8080 端口被占用，修改 `docker-compose.yml` 中的 `ports` 映射即可。首次启动后需 `--force-recreate` 确保使用最新镜像：
> ```bash
> docker compose up -d --build --force-recreate
> ```

### 4. 代码集成

```python
from backend.graph import build_graph, send_user_message

graph = build_graph()
thread_id = "session-001"
config = {"configurable": {"thread_id": thread_id}}

# 启动会话
for chunk in graph.stream({"messages": []}, config, stream_mode="values"):
    ...

# 上传文件
outputs = send_user_message(
    graph, thread_id,
    user_input="我上传了文件",
    file_path="/uploads/岗标价值与岗标任务.xlsx",
)

# 回复引导问题
outputs = send_user_message(
    graph, thread_id,
    user_input="对产品经理，减少缺陷流出，避免口碑拖累市场推广",
)
```

---

## 辅助脚本

### validate_sheets.py — 结构校验

```bash
python backend/scripts/validate_sheets.py <xlsx_path>           # 校验结构
python backend/scripts/validate_sheets.py <xlsx_path> --extract  # 校验+提取结构化数据
```

### progress.py — 进度管理

```bash
python backend/scripts/progress.py init <xlsx_path>             # 初始化进度
python backend/scripts/progress.py show                         # 查看当前进度
python backend/scripts/progress.py update <item> <status> [note]  # 更新条目状态
python backend/scripts/progress.py next                         # 推进到下一个待处理项
python backend/scripts/progress.py reset                        # 清除进度
```

---

## 常见问题

**Q: `docker compose up` 后无法访问？**
- 确认使用 `http://127.0.0.1:8080`（IPv4），而非 `localhost`（可能解析到 IPv6 导致连接重置）
- 检查本地 8080 端口是否被占用：`lsof -i :8080`
- 确保容器正常运行：`docker compose ps`
- 如果改过前端代码，需 `docker compose build --no-cache frontend && docker compose up -d --force-recreate`

**Q: 本地已有进程占用 3000 端口？**
- Docker 的 nginx 已改用 8080 端口映射，不影响本地开发服务
- 若需改回 3000 端口，先停止本地进程再修改 `docker-compose.yml` 中的 `ports`

**Q: 不想配 LLM API Key？**
- 设置 `STANDJOB_DISABLE_LLM=1` 启用纯规则兜底模式，无需 API Key 即可运行

---

## 技术栈

- **后端**: Python 3.13 · FastAPI · LangGraph · LangChain · OpenPyXL · PyMuPDF
- **前端**: Next.js
- **部署**: Docker · Nginx
- **LLM**: 支持 DeepSeek / OpenAI / Anthropic 兼容接口
