"""
岗标辅导 Graph 组装

Graph 结构：

  START
    │
    ▼
  load_standards ──────────────────────────────┐
    │ (phase="loaded")                          │
    ▼                                           │
  [等待用户上传文件]                             │
    │ (用户消息中包含文件路径)                   │
    ▼                                           │
  parse_submission                              │
    │ (phase="reviewing")                       │
    ▼                                           │
  review_item ←────────────────────────────────┤
    │ (phase="guiding")                         │
    ▼                                           │
  guide_reflection                              │
    │ (awaiting_user_input=True)                │
    ▼                                           │
  [等待用户回复]                                │
    │ (用户消息)                                │
    ▼                                           │
  process_response                              │
    │                                           │
    ├── next_action="continue_reflection" ──→ guide_reflection
    │
    └── next_action="next_item"
          │
          ├── 还有条目 ──→ review_item
          │
          └── 全部完成 ──→ END
"""
from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import CoachState
from .nodes import (
    load_standards,
    parse_submission,
    review_item,
    guide_reflection,
    process_response,
    answer_user_question,
    detect_user_intent,
)


# ─────────────────────────────────────────────────────────────
# 条件边：判断当前应走哪个分支
# ─────────────────────────────────────────────────────────────

def route_after_load(state: CoachState) -> Literal["wait_for_file"]:
    """load_standards 之后：等待用户上传文件"""
    return "wait_for_file"


def route_after_human(state: CoachState) -> Literal[
    "parse_submission", "process_response", "guide_reflection"
]:
    """
    收到用户消息后的路由：
    - 如果是初始阶段（loaded），检测到文件路径 → parse_submission
    - 如果是 guiding 阶段 → process_response
    - 其他 → guide_reflection（兜底）
    """
    phase = state.get("phase", "init")
    submission_path = state.get("submission_path")
    messages = state.get("messages", [])

    if phase == "loaded" and submission_path:
        return "parse_submission"
    elif phase == "guiding":
        return "process_response"
    else:
        return "guide_reflection"


def route_after_intent(state: CoachState) -> Literal["process_response", "answer_user_question"]:
    """基于 detect_user_intent 节点结果进行路由。"""
    intent = state.get("last_user_intent", "reply")
    if intent == "question":
        return "answer_user_question"
    return "process_response"


def route_after_process_response(state: CoachState) -> Literal[
    "review_item", "guide_reflection", "wait_for_reply"
]:
    """
    process_response 之后的路由：
    - phase="reviewing" → 有新条目待评审 → review_item
    - phase="guiding"  → 继续当前条目引导 → guide_reflection
    - phase="done"     → 进入答疑等待态
    """
    phase = state.get("phase", "guiding")
    if phase == "done":
        return "wait_for_reply"
    elif phase == "reviewing":
        return "review_item"
    else:
        return "guide_reflection"


def route_after_review(state: CoachState) -> Literal["guide_reflection", "wait_for_reply"]:
    """review_item 之后路由"""
    phase = state.get("phase", "guiding")
    if phase == "done":
        return "wait_for_reply"
    return "guide_reflection"


# ─────────────────────────────────────────────────────────────
# 虚拟 Node：等待用户输入（interrupt_before 实现）
# ─────────────────────────────────────────────────────────────

def wait_for_file(state: CoachState) -> dict:
    """占位 Node，实际由 interrupt_before 暂停在此处等待用户上传文件"""
    return {}  # 不修改状态；图会被 interrupt 暂停


def wait_for_reply(state: CoachState) -> dict:
    """占位 Node，等待用户对引导问题的回复"""
    return {}


# ─────────────────────────────────────────────────────────────
# 构建 Graph
# ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    构建并编译岗标辅导图。
    使用 MemorySaver 保持多轮对话状态。
    """
    builder = StateGraph(CoachState)

    # ── 注册节点 ──────────────────────────────────────────────
    builder.add_node("load_standards", load_standards)
    builder.add_node("wait_for_file", wait_for_file)        # 等待用户上传文件
    builder.add_node("parse_submission", parse_submission)
    builder.add_node("review_item", review_item)
    builder.add_node("guide_reflection", guide_reflection)
    builder.add_node("wait_for_reply", wait_for_reply)      # 等待用户回复
    builder.add_node("detect_user_intent", detect_user_intent)
    builder.add_node("process_response", process_response)
    builder.add_node("answer_user_question", answer_user_question)

    # ── 连接边 ────────────────────────────────────────────────

    # 启动 → 加载标准
    builder.add_edge(START, "load_standards")

    # 加载完成 → 等待文件
    builder.add_edge("load_standards", "wait_for_file")

    # 等待文件 → 解析（用户提供文件后）
    builder.add_edge("wait_for_file", "parse_submission")

    # 解析完成 → 评审第一条
    builder.add_edge("parse_submission", "review_item")

    # 评审完成 → 引导反思
    builder.add_conditional_edges(
        "review_item",
        route_after_review,
        {
            "guide_reflection": "guide_reflection",
            "wait_for_reply": "wait_for_reply",
        }
    )

    # 发出引导问题 → 等待用户回复
    builder.add_edge("guide_reflection", "wait_for_reply")

    # 收到回复：先识别意图
    builder.add_edge("wait_for_reply", "detect_user_intent")

    # 按意图路由
    builder.add_conditional_edges(
        "detect_user_intent",
        route_after_intent,
        {
            "process_response": "process_response",
            "answer_user_question": "answer_user_question",
        }
    )

    # 被动答疑后回到等待输入
    builder.add_edge("answer_user_question", "wait_for_reply")

    # 处理回复后的分支
    builder.add_conditional_edges(
        "process_response",
        route_after_process_response,
        {
            "review_item": "review_item",
            "guide_reflection": "guide_reflection",
            "wait_for_reply": "wait_for_reply",
        }
    )

    # ── 编译（interrupt_before 让 wait 节点暂停等待用户输入） ──
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["wait_for_file", "wait_for_reply"],
    )

    return graph


# ─────────────────────────────────────────────────────────────
# 便捷方法：向 Graph 注入用户消息并恢复执行
# ─────────────────────────────────────────────────────────────

def send_user_message(
    graph,
    thread_id: str,
    user_input: str,
    file_path: str | None = None,
) -> list[str]:
    """
    向正在等待的 Graph 发送用户消息，恢复执行并收集 AI 回复。

    Parameters
    ----------
    graph       : 已编译的 CompiledGraph
    thread_id   : 会话 ID
    user_input  : 用户文本输入
    file_path   : 用户上传的文件路径（可选）

    Returns
    -------
    list[str] : 本次执行产生的所有 AI 消息内容
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}

    # 构建状态更新
    update: dict = {
        "messages": [HumanMessage(content=user_input)],
        "awaiting_user_input": False,
    }
    if file_path:
        update["submission_path"] = file_path

    # 更新状态并恢复执行
    graph.update_state(config, update)

    # 预填充已有 AI 消息，避免重复返回历史消息
    from langchain_core.messages import AIMessage
    pre_state = graph.get_state(config)
    seen: set[str] = set()
    for m in pre_state.values.get("messages", []):
        if isinstance(m, AIMessage):
            seen.add(m.content)

    ai_outputs: list[str] = []
    for chunk in graph.stream(None, config, stream_mode="values"):
        msgs = chunk.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage) and m.content not in seen:
                seen.add(m.content)
                ai_outputs.append(m.content)

    return ai_outputs
