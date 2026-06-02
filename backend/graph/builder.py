"""
岗标辅导 Graph 组装

完全实现 gangbiao-coach skill v2 的阶段 0-4 生命周期：

  阶段 0 (文件采集)                           阶段 4 (收尾)
  ───────────────                             ──────────────
  START → load_standards                      generate_closure
    │ (检查进度)                                  │
    ▼                                              ▼
  [等待用户上传文件]                           [等待用户提问或结束]
    │ (用户提供文件)                              │
    ▼                                              │
  阶段 1 (结构校验)                              │
  validate_structure                             │
    │ (校验通过)                                  │
    ▼                                              │
  阶段 2 (全面评审)                              │
  review_item ←─────────────────────────────┐    │
    │ (phase="guiding")                      │    │
    ▼                                        │    │
  阶段 3 (分步引导)                           │    │
  guide_reflection                           │    │
    │ (awaiting_user_input=True)             │    │
    ▼                                        │    │
  [等待用户回复]                              │    │
    │ (用户消息)                              │    │
    ▼                                        │    │
  detect_user_intent                         │    │
    ├─ intent=question → answer_user_question ┘   │
    └─ intent=reply → process_response            │
         ├─ next_action="continue" → guide_reflection
         ├─ next_action="next_item" → review_item
         └─ next_action="done"     → generate_closure → END
"""
from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import CoachState
from .nodes import (
    load_standards,
    validate_structure,
    review_item,
    guide_reflection,
    process_response,
    answer_user_question,
    detect_user_intent,
    generate_closure,
)


# ─────────────────────────────────────────────────────────────
# 条件边：判断当前应走哪个分支
# ─────────────────────────────────────────────────────────────

def route_after_load(state: CoachState) -> Literal["wait_for_file"]:
    """load_standards 之后：等待用户上传文件"""
    return "wait_for_file"


def route_after_validate(state: CoachState) -> Literal["wait_for_file", "review_item"]:
    """
    validate_structure 之后：
    - 校验失败 → 回到等待文件
    - 校验通过 → 进入评审
    """
    if not state.get("structure_valid", True):
        return "wait_for_file"
    return "review_item"


def route_after_review(state: CoachState) -> Literal["guide_reflection", "generate_closure", "review_item"]:
    """review_item 之后路由"""
    phase = state.get("phase", "guiding")
    if phase == "done":
        return "generate_closure"
    if phase == "reviewing":
        # 当前条目无问题，自动推进到下一条
        return "review_item"
    return "guide_reflection"


def route_after_guide(state: CoachState) -> Literal["wait_for_reply", "review_item", "generate_closure"]:
    """guide_reflection 之后路由"""
    phase = state.get("phase", "guiding")
    if phase == "done":
        return "generate_closure"
    if phase == "reviewing":
        # 当前条目无问题，自动推进到下一条
        return "review_item"
    return "wait_for_reply"


def route_after_intent(state: CoachState) -> Literal["process_response", "answer_user_question"]:
    """基于 detect_user_intent 节点结果进行路由"""
    intent = state.get("last_user_intent", "reply")
    if intent == "question":
        return "answer_user_question"
    return "process_response"


def route_after_process_response(state: CoachState) -> Literal[
    "review_item", "guide_reflection", "generate_closure", "wait_for_reply"
]:
    """
    process_response 之后的路由：
    - phase="done"        → 收尾
    - phase="reviewing"   → 有新条目待评审 → review_item
    - phase="guiding"     → 继续当前条目引导 → guide_reflection
    - phase="closure"     → 已在收尾流程
    """
    phase = state.get("phase", "guiding")
    if phase == "done":
        return "generate_closure"
    elif phase == "reviewing":
        return "review_item"
    elif phase in {"closure"}:
        return "wait_for_reply"
    else:
        return "guide_reflection"


def route_after_closure(state: CoachState) -> Literal["wait_for_reply", "END"]:
    """收尾之后：等待用户继续提问或结束"""
    return "wait_for_reply"


# ─────────────────────────────────────────────────────────────
# 虚拟 Node：等待用户输入（interrupt_before 实现）
# ─────────────────────────────────────────────────────────────

def wait_for_file(state: CoachState) -> dict:
    """占位 Node，实际由 interrupt_before 暂停在此处等待用户上传文件"""
    return {}


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

    # 阶段 0: 初始化 + 文件采集
    builder.add_node("load_standards", load_standards)
    builder.add_node("wait_for_file", wait_for_file)

    # 阶段 1: 结构校验
    builder.add_node("validate_structure", validate_structure)

    # 阶段 2: 全面评审
    builder.add_node("review_item", review_item)

    # 阶段 3: 分步引导
    builder.add_node("guide_reflection", guide_reflection)
    builder.add_node("wait_for_reply", wait_for_reply)
    builder.add_node("detect_user_intent", detect_user_intent)
    builder.add_node("process_response", process_response)
    builder.add_node("answer_user_question", answer_user_question)

    # 阶段 4: 收尾
    builder.add_node("generate_closure", generate_closure)

    # ── 连接边 ────────────────────────────────────────────────

    # 阶段 0: 启动 → 加载标准 → 等待文件
    builder.add_edge(START, "load_standards")
    builder.add_edge("load_standards", "wait_for_file")

    # 等待文件 → 结构校验（用户提供文件后）
    builder.add_edge("wait_for_file", "validate_structure")

    # 阶段 1: 结构校验后路由
    builder.add_conditional_edges(
        "validate_structure",
        route_after_validate,
        {
            "wait_for_file": "wait_for_file",
            "review_item": "review_item",
        }
    )

    # 阶段 2: 评审后路由
    builder.add_conditional_edges(
        "review_item",
        route_after_review,
        {
            "guide_reflection": "guide_reflection",
            "generate_closure": "generate_closure",
            "review_item": "review_item",
        }
    )

    # 阶段 3: 引导后路由（可能自动推进到下一条或收尾）
    builder.add_conditional_edges(
        "guide_reflection",
        route_after_guide,
        {
            "wait_for_reply": "wait_for_reply",
            "review_item": "review_item",
            "generate_closure": "generate_closure",
        }
    )

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
            "generate_closure": "generate_closure",
            "wait_for_reply": "wait_for_reply",
        }
    )

    # 阶段 4: 收尾 → 等待用户继续提问或结束
    builder.add_edge("generate_closure", "wait_for_reply")

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
    """
    from langchain_core.messages import HumanMessage, AIMessage

    config = {"configurable": {"thread_id": thread_id}}

    update: dict = {
        "messages": [HumanMessage(content=user_input)],
        "awaiting_user_input": False,
    }
    if file_path:
        update["submission_path"] = file_path

    graph.update_state(config, update)

    # 预填充已有 AI 消息，避免重复返回历史消息
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
