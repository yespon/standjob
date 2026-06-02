# 后端架构

## 模块总览

```
backend/
├── graph/                  # LangGraph 核心
│   ├── state.py            # CoachState 状态定义 + 常量
│   ├── nodes.py            # 7 个功能节点 + 2 个等待节点
│   └── builder.py          # Graph 组装 + send_user_message
├── references/             # 内嵌参考文档（rubric.md + textbook.md）
├── scripts/                # 辅助脚本（validate_sheets.py + progress.py）
├── loaders.py              # xlsx/pdf 文件解析
├── api.py                  # FastAPI REST + SSE
└── cli.py                  # 交互式 CLI
```

## Graph 节点

| Node | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load_standards` | — | scoring_criteria, teaching_material, phase=loaded | 加载内嵌参考文档；检查进度恢复 |
| `wait_for_file` | — | — | ⏸ interrupt 占位节点 |
| `validate_structure` | submission_path | submission_rows, structure_valid | 调用 validate_sheets.py 或回退到 load_submission |
| `review_item` | submission_rows[idx] | review_items, issue_queue, score | 逐条对照 14 项标准；三问法筛选；灰区放过 |
| `guide_reflection` | review_items[idx] | AIMessage, phase=guiding | 引导武器表 + 自检十一问生成苏格拉底式问题 |
| `wait_for_reply` | — | — | ⏸ interrupt 占位节点 |
| `detect_user_intent` | last HumanMessage | last_user_intent | 规则优先 + LLM 二判 |
| `process_response` | last HumanMessage | 判断结果 + 下一步 | 回判三步；卡住升级提示；自动放过 |
| `answer_user_question` | last HumanMessage | AIMessage | 被动答疑 + 过渡回引导 |
| `generate_closure` | all review_items | closure_summary, highlights | 自然总结 + 教材收束 + 清理进度 |

## 条件边

| 源节点 | 路由函数 | 分支 |
|--------|----------|------|
| `validate_structure` | `route_after_validate` | 通过→review_item / 失败→wait_for_file |
| `review_item` | `route_after_review` | 有问题→guide_reflection / 完成→generate_closure |
| `detect_user_intent` | `route_after_intent` | reply→process_response / question→answer_user_question |
| `process_response` | `route_after_process_response` | reviewing→review_item / guiding→guide_reflection / done→generate_closure |

## 状态字段（CoachState）

### 对话
- `messages: Annotated[list[BaseMessage], add_messages]` — 对话历史（追加模式）
- `phase: str` — 当前阶段
- `active_mode: str` — proactive / reactive_qa
- `last_user_intent: str` — reply / question

### 预加载
- `scoring_criteria: str` — 14 项评审标准文本
- `teaching_material: str` — 教材精要文本

### 进度恢复
- `has_saved_progress: bool` — 是否有已保存进度
- `progress_snapshot: dict` — 进度快照
- `resume_confirmed: bool` — 用户确认恢复

### 提交数据
- `submission_path: str` — 文件路径
- `submission_rows: list[dict]` — 解析后的表格数据
- `structure_valid: bool` — 结构校验结果
- `structure_errors: list[str]` — 校验错误

### 评审结果
- `review_items: list[ReviewItem]` — 每条目的评审结果
- `current_item_index: int` — 当前辅导条目
- `current_issue_index: int` — 当前问题项
- `issue_status_map: dict` — issue_id → status
- `coaching_queue_order: list[str]` — 辅导队列顺序
- `current_focus_id: str` — 当前聚焦问题项
- `stuck_counter: int` — 卡住计数
- `hint_level: int` — 提示升级级别

### 收尾
- `closure_summary: str` — 收尾总结
- `highlights: list[str]` — 改得好的地方
- `remaining_polish: list[str]` — 待打磨的地方

## 评审标准（14 项）

| ID | 类别 | 等级 | 描述 |
|----|------|------|------|
| 1 | 岗位价值 | A | 站在自身视角而非客户视角提炼岗位价值 |
| 2 | 岗位价值 | A | 岗位价值并非源于对客户最在意、最深层需求的提炼 |
| 3 | 岗位价值 | A | 没有从不同客户的视角出发提炼岗位价值 |
| 4 | 岗位价值 | A | 岗位价值描述空泛，指导性不强 |
| 5 | 岗位效能 | A | 岗位效能不能直接、有效地衡量岗位价值 |
| 6 | 岗位任务 | A | 核心任务的目的集合无法完整覆盖岗位价值 |
| 7 | 岗位任务 | B | 任务命名未按动词+修饰语+名词 |
| 8 | 岗位任务 | B | 直接用目的命名任务 |
| 9 | 任务目的与成果 | A | 任务目的模糊，导致为何而做不清 |
| 10 | 任务目的与成果 | A | 成果标准与任务目的脱节 |
| 11 | 任务目的与成果 | A | 任务成果评估周期设计过长 |
| 12 | 任务目的与成果 | A | 成果标准不符合SMART原则 |
| 13 | 任务目的与成果 | A | 把交付物当做成果 |
| 14 | 任务目的与成果 | A | 成果标准未在完成度、交期、预算上设计挑战目标 |

> A 级扣 10 分，B 级扣 5 分；满分 100，≥85 通关

## LLM 配置

系统按优先级检测 API Key：`OPENAI_API_KEY` → `DEEPSEEK_API_KEY` → `ANTHROPIC_API_KEY`

设置 `STANDJOB_DISABLE_LLM=1` 可启用纯规则兜底模式（无需 API Key）。

---

## Docker 部署

### 容器架构

```
127.0.0.1:8080 → nginx (80) ─┬→ frontend:3000  (Next.js standalone)
                              └→ backend:8000   (FastAPI + Uvicorn)
```

- `nginx` 对外暴露 `127.0.0.1:8080`（仅 IPv4），反向代理前端和后端
- `frontend` 容器内运行 Next.js standalone，需 `HOSTNAME=0.0.0.0` 确保监听所有接口
- `backend` 容器内运行 Uvicorn，默认监听 `0.0.0.0:8000`
- SSE 流式端点 `/api/session/start/stream` 和 `/api/chat/stream` 已在 nginx 中配置 `proxy_buffering off`

### 关键配置

| 文件 | 关键项 | 说明 |
|------|--------|------|
| `docker-compose.yml` | `ports: "127.0.0.1:8080:80"` | 仅绑定 IPv4 8080，避免 IPv6 连接重置 |
| `Dockerfile.frontend` | `ENV HOSTNAME=0.0.0.0` | Next.js standalone 默认绑定容器 hostname，需覆盖 |
| `nginx.conf` | `proxy_buffering off` | SSE 流式支持 |
| `.env` | `OPENAI_BASE_URL` / `OPENAI_API_KEY` | LLM 接口配置 |
| `.env` | `STANDJOB_DISABLE_LLM=1` | 可选：纯规则兜底模式 |

### 常见问题

- **容器启动但无法访问**：使用 `http://127.0.0.1:8080`（IPv4），不要用 `localhost`（可能解析到 IPv6）
- **前端改动后未生效**：`docker compose build --no-cache frontend && docker compose up -d --force-recreate`
- **端口冲突**：修改 `docker-compose.yml` 中的 `ports` 映射到其他端口
