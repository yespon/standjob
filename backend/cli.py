"""
岗标辅导系统 - 主入口 & 交互式 CLI 演示

使用方式：
  python -m backend.cli

环境变量（可选）：
  SCORING_CRITERIA_PATH  = 评分标准.xlsx 的路径
  TEACHING_MATERIAL_PATH = 岗标价值与岗标任务教材.pdf 的路径
  ANTHROPIC_API_KEY      = Anthropic API Key
"""
from __future__ import annotations
import os
import sys
import uuid

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from .graph import build_graph, send_user_message


def print_ai(content: str):
    """格式化输出 AI 消息"""
    print("\n" + "─" * 60)
    print("🤖 辅导助手：")
    print(content)
    print("─" * 60)


def print_user(content: str):
    """格式化输出用户消息"""
    print(f"\n👤 您：{content}")


def get_all_ai_messages(graph, config) -> list[str]:
    """从当前状态提取所有 AI 消息（去重，只取最新产生的）"""
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    return [m.content for m in messages if isinstance(m, AIMessage)]


def run_initial_stream(graph, config, *, initial_input=None) -> list[str]:
    """运行初始流程直到第一个 interrupt，收集输出"""
    outputs = []
    for chunk in graph.stream(initial_input, config, stream_mode="values"):
        msgs = chunk.get("messages", [])
        for m in msgs:
            if isinstance(m, AIMessage) and m.content not in outputs:
                outputs.append(m.content)
    return outputs


def main():
    print("=" * 60)
    print("   📋 岗标价值与岗标任务 - 智能辅导系统")
    print("   基于 LangGraph + Claude")
    print("=" * 60)

    # 初始化 Graph
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # ── Phase 1: 启动 → load_standards → 停在 wait_for_file ──
    print("\n正在初始化，加载评审标准...")
    initial_outputs = run_initial_stream(graph, config, initial_input={"messages": []})

    for msg in initial_outputs:
        print_ai(msg)

    # ── Phase 2: 主交互循环 ──────────────────────────────────
    while True:
        state = graph.get_state(config)
        phase = state.values.get("phase", "init")
        next_nodes = state.next

        # 检查是否完成
        if phase == "done" or not next_nodes:
            print("\n✅ 辅导会话结束，感谢使用！")
            break

        # 判断当前等待状态
        waiting_for_file = "wait_for_file" in next_nodes
        waiting_for_reply = "wait_for_reply" in next_nodes

        if waiting_for_file:
            # 等待用户上传文件
            print("\n📎 请输入 xlsx 文件路径（直接回车使用演示文件）：", end="")
            file_input = input().strip()

            if not file_input:
                file_path = None  # nodes.py 会自动创建 mock 文件
                user_text = "我已上传文件，请开始评审"
            else:
                file_path = file_input
                user_text = f"文件路径：{file_input}"

            print_user(user_text)
            outputs = send_user_message(graph, thread_id, user_text, file_path)
            for msg in outputs:
                print_ai(msg)

        elif waiting_for_reply:
            # 等待用户对引导问题的回复
            print("\n💬 请输入您的回答（输入 'skip' 跳过当前条目，'quit' 退出）：", end="")
            user_text = input().strip()

            if not user_text:
                user_text = "我需要再想想"

            if user_text.lower() == "quit":
                print("\n👋 已退出辅导系统")
                break

            if user_text.lower() == "skip":
                user_text = "好的，我明白了，我们继续下一条"

            print_user(user_text)
            outputs = send_user_message(graph, thread_id, user_text)
            for msg in outputs:
                print_ai(msg)

        else:
            # 非 interrupt 状态，继续执行
            outputs = run_initial_stream(graph, config)
            for msg in outputs:
                print_ai(msg)
            if not outputs:
                # 无输出且无 interrupt，可能已完成
                break


if __name__ == "__main__":
    # 检查 API Key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠ 警告：未设置 ANTHROPIC_API_KEY 环境变量")
        print("请运行：export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    main()
