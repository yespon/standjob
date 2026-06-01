"""
岗标辅导系统 - FastAPI 后端

提供 REST API 供 Next.js 前端调用，替代原 CLI 交互。
支持 SSE 流式输出。
"""
from __future__ import annotations
import json
import os
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

from .graph import build_graph, send_user_message

app = FastAPI(title="岗标辅导系统 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局 Graph 实例 ──────────────────────────────────────────
graph = build_graph()

UPLOAD_DIR = Path("/tmp/standjob_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Request / Response 模型 ──────────────────────────────────

class StartSessionResponse(BaseModel):
    thread_id: str
    messages: list[dict]
    phase: str
    next_nodes: list[str]
    active_mode: str = "proactive"


class SendMessageRequest(BaseModel):
    thread_id: str
    message: str
    file_path: str | None = None


class ChatResponse(BaseModel):
    messages: list[dict]
    phase: str
    next_nodes: list[str]
    active_mode: str = "proactive"


class StateResponse(BaseModel):
    phase: str
    next_nodes: list[str]
    current_item_index: int
    total_items: int
    reflection_round: int
    active_mode: str = "proactive"


# ── API 端点 ────────────────────────────────────────────────

@app.post("/api/session/start", response_model=StartSessionResponse)
def start_session():
    """创建新会话，启动 Graph 直到第一个 interrupt"""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    ai_messages = []
    for chunk in graph.stream({"messages": []}, config, stream_mode="values"):
        msgs = chunk.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": m.content}
                if msg_dict not in ai_messages:
                    ai_messages.append(msg_dict)

    state = graph.get_state(config)
    phase = state.values.get("phase", "init")
    next_nodes = list(state.next) if state.next else []
    active_mode = state.values.get("active_mode", "proactive")

    return StartSessionResponse(
        thread_id=thread_id,
        messages=ai_messages,
        phase=phase,
        next_nodes=next_nodes,
        active_mode=active_mode,
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传 xlsx 文件，返回服务端路径"""
    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="仅支持 .xlsx 文件")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"file_path": str(save_path), "filename": file.filename}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: SendMessageRequest):
    """发送用户消息，获取 AI 回复"""
    config = {"configurable": {"thread_id": req.thread_id}}

    # 检查会话是否存在
    try:
        state = graph.get_state(config)
        if not state.values:
            raise HTTPException(status_code=404, detail="会话不存在")
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    ai_contents = send_user_message(graph, req.thread_id, req.message, req.file_path)

    ai_messages = [{"role": "assistant", "content": c} for c in ai_contents]

    state = graph.get_state(config)
    phase = state.values.get("phase", "init")
    next_nodes = list(state.next) if state.next else []
    active_mode = state.values.get("active_mode", "proactive")

    return ChatResponse(
        messages=ai_messages,
        phase=phase,
        next_nodes=next_nodes,
        active_mode=active_mode,
    )


@app.get("/api/session/{thread_id}/state", response_model=StateResponse)
def get_session_state(thread_id: str):
    """获取当前会话状态"""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        if not state.values:
            raise HTTPException(status_code=404, detail="会话不存在")
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    values = state.values
    return StateResponse(
        phase=values.get("phase", "init"),
        next_nodes=list(state.next) if state.next else [],
        current_item_index=values.get("current_item_index", 0),
        total_items=len(values.get("submission_rows", [])),
        reflection_round=values.get("reflection_round", 0),
        active_mode=values.get("active_mode", "proactive"),
    )


# ── SSE 流式端点 ────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/session/start/stream")
def start_session_stream():
    """创建新会话，SSE 流式返回初始化过程中的 AI 消息"""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def generate():
        yield _sse_event("session", {"thread_id": thread_id})

        seen_contents: set[str] = set()
        for chunk in graph.stream({"messages": []}, config, stream_mode="values"):
            msgs = chunk.get("messages", [])
            for m in msgs:
                if isinstance(m, AIMessage) and m.content not in seen_contents:
                    seen_contents.add(m.content)
                    yield _sse_event("message", {
                        "role": "assistant",
                        "content": m.content,
                    })

        state = graph.get_state(config)
        phase = state.values.get("phase", "init")
        next_nodes = list(state.next) if state.next else []
        active_mode = state.values.get("active_mode", "proactive")
        yield _sse_event("done", {"phase": phase, "next_nodes": next_nodes, "active_mode": active_mode})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/chat/stream")
def chat_stream(req: SendMessageRequest):
    """发送用户消息，SSE 流式返回 AI 回复"""
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        state = graph.get_state(config)
        if not state.values:
            raise HTTPException(status_code=404, detail="会话不存在")
    except Exception:
        raise HTTPException(status_code=404, detail="会话不存在")

    def generate():
        # 更新状态
        update: dict = {
            "messages": [HumanMessage(content=req.message)],
            "awaiting_user_input": False,
        }
        if req.file_path:
            update["submission_path"] = req.file_path

        graph.update_state(config, update)

        # 预填充已有 AI 消息，避免重复发送历史消息
        pre_state = graph.get_state(config)
        seen_contents: set[str] = set()
        for m in pre_state.values.get("messages", []):
            if isinstance(m, AIMessage):
                seen_contents.add(m.content)

        try:
            for chunk in graph.stream(None, config, stream_mode="values"):
                msgs = chunk.get("messages", [])
                for m in msgs:
                    if isinstance(m, AIMessage) and m.content not in seen_contents:
                        seen_contents.add(m.content)
                        yield _sse_event("message", {
                            "role": "assistant",
                            "content": m.content,
                        })
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse_event("error", {"detail": f"处理出错：{type(e).__name__}: {str(e)[:200]}"})

        state = graph.get_state(config)
        phase = state.values.get("phase", "init")
        next_nodes = list(state.next) if state.next else []
        active_mode = state.values.get("active_mode", "proactive")
        yield _sse_event("done", {"phase": phase, "next_nodes": next_nodes, "active_mode": active_mode})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
