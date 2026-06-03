#!/usr/bin/env python3
"""In-process deterministic regression harness for the coaching graph.

Runs the LangGraph offline (STANDJOB_DISABLE_LLM forced on) so behavior is
fully rule-based and reproducible. Drives a fixed conversation script and
captures a trace (AI message prefixes + key state fields after each turn).

Usage:
    python3 scripts/graph_regression.py          # print trace
    python3 scripts/graph_regression.py --save    # write golden baseline
    python3 scripts/graph_regression.py --check    # diff against baseline
"""
from __future__ import annotations

import os
import sys
import json
import uuid
from pathlib import Path

# Force deterministic, LLM-free execution before importing the graph.
os.environ["STANDJOB_DISABLE_LLM"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from backend.graph import build_graph, send_user_message  # noqa: E402

BASELINE = ROOT / "scripts" / "graph_regression_baseline.json"

# Fixed user-turn script. Mix of unresolved replies (drive escalation),
# a question (intent routing), and concrete replies (resolve issues).
SCRIPT: list[str] = [
    "我再想想",            # unresolved reply
    "什么是三问法？",        # question -> answer_user_question
    "我会从客户视角改，帮产品经理减少故障、降低成本",  # concrete reply, may resolve
    "我再想想",
    "我再想想",
    "我打算把任务名改成动宾结构，明确成果率指标",
    "我再想想",
    "我再想想",
    "我再想想",
]

# State fields to snapshot after each turn (deterministic, no message content).
STATE_FIELDS = [
    "phase",
    "active_mode",
    "last_user_intent",
    "current_item_index",
    "current_issue_index",
    "issue_round",
    "stuck_counter",
    "hint_level",
    "current_focus_id",
    "current_line",
    "line1_completed",
]


def _snapshot(graph, cfg) -> dict:
    st = graph.get_state(cfg)
    v = st.values
    snap = {f: v.get(f) for f in STATE_FIELDS}
    snap["next"] = list(st.next) if st.next else []
    snap["total_items"] = len(v.get("submission_rows", []))
    return snap


def run_trace() -> list[dict]:
    # Ensure a clean global progress file each run.
    pf = Path("/tmp/gangbiao-coach-progress.json")
    if pf.exists():
        pf.unlink()

    graph = build_graph()
    tid = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}

    trace: list[dict] = []

    # Turn 0: init
    outs: list[str] = []
    for chunk in graph.stream({"messages": []}, cfg, stream_mode="values"):
        for m in chunk.get("messages", []):
            if isinstance(m, AIMessage) and m.content not in outs:
                outs.append(m.content)
    trace.append({"turn": "init", "ai": [c[:40] for c in outs], "state": _snapshot(graph, cfg)})

    # Turn 1: upload (mock file)
    outs = send_user_message(graph, tid, "我已上传文件，请开始", None)
    trace.append({"turn": "upload", "ai": [c[:40] for c in outs], "state": _snapshot(graph, cfg)})

    # Scripted reply turns
    for i, msg in enumerate(SCRIPT):
        st = graph.get_state(cfg)
        if not st.next:
            trace.append({"turn": f"reply{i}", "skipped": "graph ended", "state": _snapshot(graph, cfg)})
            break
        outs = send_user_message(graph, tid, msg)
        trace.append({"turn": f"reply{i}:{msg[:8]}", "ai": [c[:40] for c in outs], "state": _snapshot(graph, cfg)})

    return trace


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--print"
    trace = run_trace()
    rendered = json.dumps(trace, ensure_ascii=False, indent=2)

    if mode == "--save":
        BASELINE.write_text(rendered, encoding="utf-8")
        print(f"baseline saved: {BASELINE} ({len(trace)} turns)")
        return

    if mode == "--check":
        if not BASELINE.exists():
            print("NO BASELINE — run with --save first", file=sys.stderr)
            sys.exit(2)
        expected = BASELINE.read_text(encoding="utf-8")
        if expected.strip() == rendered.strip():
            print(f"PASS: trace matches baseline ({len(trace)} turns)")
            return
        # Show first divergence.
        exp = json.loads(expected)
        for idx, (a, b) in enumerate(zip(exp, trace)):
            if a != b:
                print(f"DIVERGENCE at turn index {idx}:", file=sys.stderr)
                print("  expected:", json.dumps(a, ensure_ascii=False), file=sys.stderr)
                print("  actual:  ", json.dumps(b, ensure_ascii=False), file=sys.stderr)
                break
        else:
            print(f"LENGTH DIFF: baseline={len(exp)} actual={len(trace)}", file=sys.stderr)
        sys.exit(1)

    print(rendered)


if __name__ == "__main__":
    main()
