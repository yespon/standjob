# 岗标辅导系统 — 前端

Next.js 前端，对接后端 FastAPI + LangGraph 辅导引擎。

## 功能

- 上传"岗位价值与岗位任务"Excel 文件
- 实时对话式辅导（SSE 流式输出）
- 显示辅导进度（当前条目、问题项、阶段）
- 支持主动引导 + 被动答疑两种模式

## API 对接

| 端点 | 用途 |
|------|------|
| `POST /api/session/start/stream` | 创建会话，SSE 流式返回初始化消息 |
| `POST /api/upload` | 上传 xlsx 文件 |
| `POST /api/chat/stream` | 发送消息，SSE 流式返回 AI 回复 |
| `GET /api/session/{id}/state` | 获取会话状态（phase / current_item / focus_id / stuck_counter） |

## 开发

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

## Docker 部署

```bash
docker compose up -d --build
# 访问 http://127.0.0.1:8080
```

> 注意：Docker 环境通过 nginx 反向代理，前端和后端 API 统一在 `127.0.0.1:8080` 访问。
> 前端代码中 `API_BASE` 为空字符串，API 请求自动走同域 nginx 代理到后端。

## 状态字段

前端通过 `/api/session/{id}/state` 获取的会话状态：

| 字段 | 类型 | 说明 |
|------|------|------|
| `phase` | string | `loaded` / `reviewing` / `guiding` / `done` / `closure` |
| `current_item_index` | int | 当前辅导的条目索引（从 0 开始） |
| `total_items` | int | 总条目数 |
| `active_mode` | string | `proactive`（主动引导）/ `reactive_qa`（被动答疑） |
| `current_focus_id` | string? | 当前聚焦的问题项 ID |
| `stuck_counter` | int | 用户卡住次数 |
| `hint_level` | int | 提示升级级别（0-3） |
| `closure_summary` | string? | 收尾总结文本 |
