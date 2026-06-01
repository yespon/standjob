# 岗标辅导系统 — LangGraph 实现

> 基于 **LangGraph** + **Claude** 构建的引导式、交互式岗标表格辅导工具。  
> 当用户上传表格时，系统不直接给出答案，而是结合评审标准和教材，逐条引导用户自我反思、完善表格。

---

## 目录结构

```
gang_biao_coach/
├── state.py      # 全局状态定义（CoachState + ReviewItem）
├── loaders.py    # 文件读取工具（xlsx / pdf → 文本）
├── nodes.py      # 所有 LangGraph Node 实现
├── graph.py      # Graph 组装、条件边、send_user_message 工具函数
├── main.py       # CLI 交互入口（演示 / 测试用）
└── README.md     # 本文档
```

---

## Graph 架构

```
START
  │
  ▼
load_standards          ← 预加载评分标准.xlsx + 教材.pdf
  │
  ▼
wait_for_file           ← ⏸ interrupt：等待用户上传文件
  │  (用户提供文件路径)
  ▼
parse_submission        ← 解析 xlsx，提取表头 + 行数据
  │
  ▼
review_item             ← AI 评审当前条目（不向用户展示原始评审结果）
  │
  ▼
guide_reflection        ← 构造引导性问题 + 评分概览，发送给用户
  │
  ▼
wait_for_reply          ← ⏸ interrupt：等待用户回复
  │  (用户输入)
  ▼
process_response        ← 分析用户理解深度，决定：
  │                        • continue_reflection → guide_reflection（追问）
  │                        • next_item          → review_item（下一条）
  │                        • done               → END
  └──────────────────────────────────────────────────────┘
```

### 核心节点说明

| Node | 职责 |
|------|------|
| `load_standards` | 启动时一次性加载评分标准和教材，写入 State |
| `parse_submission` | 解析用户上传的 xlsx，提取 columns + rows |
| `review_item` | 调用 LLM 对当前行进行结构化评审（内部使用，不直接展示） |
| `guide_reflection` | 将评审结果转化为引导性问题发给用户 |
| `process_response` | 判断用户理解程度，决定继续追问还是推进下一条 |

---

## 状态流转（CoachState.phase）

```
init → loaded → reviewing → guiding → reviewing → ... → done
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph langchain-anthropic langchain openpyxl pymupdf
```

### 2. 设置环境变量

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# 可选：指定真实文件路径
export SCORING_CRITERIA_PATH=/path/to/评分标准.xlsx
export TEACHING_MATERIAL_PATH=/path/to/岗标价值与岗标任务教材.pdf
```

### 3. 运行 CLI 演示

```bash
cd gang_biao_coach
python main.py
```

> 不设置文件路径时，系统会自动使用内置 Mock 数据演示完整流程。

---

## 集成到 Web 应用

```python
from graph import build_graph, send_user_message

# 初始化（全局单例）
graph = build_graph()
thread_id = "session-xxx"
config = {"configurable": {"thread_id": thread_id}}

# 启动会话（load_standards 自动执行）
for chunk in graph.stream(None, config, stream_mode="values"):
    ai_msgs = [m.content for m in chunk.get("messages", []) if isinstance(m, AIMessage)]

# 用户上传文件后
outputs = send_user_message(
    graph, thread_id,
    user_input="我上传了文件",
    file_path="/uploads/岗标价值与岗标任务-软件工程师.xlsx"
)

# 用户回复引导问题
outputs = send_user_message(
    graph, thread_id,
    user_input="我觉得这里的价值描述确实太宽泛了，应该更聚焦在具体业务影响上..."
)
```

---

## 辅导原则（编码在系统提示中）

1. **不直接给答案** — 通过苏格拉底式追问引导用户自己得出结论
2. **关注逻辑链** — 检查「岗标价值 → 任务 → 关键行为 → 输出成果」的完整性
3. **每次聚焦一点** — 每轮对话只引导一个维度的改进
4. **引用教材** — 引导时结合教材概念和示例，让用户有据可依
5. **正向激励** — 先认可做得好的地方，再指出改进点
