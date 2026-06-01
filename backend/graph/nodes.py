"""
岗标辅导 Graph Nodes
每个 node 接收 CoachState，返回部分状态更新（dict）。
"""
from __future__ import annotations
import os
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .state import CoachState, ReviewItem
from ..loaders import (
    load_scoring_criteria,
    load_submission,
    load_pdf_as_text,
    MOCK_SCORING_CRITERIA,
    MOCK_TEACHING_MATERIAL,
)

# ─────────────────────────────────────────────────────────────
# LLM 实例（共享）
# ─────────────────────────────────────────────────────────────
# llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=2048)
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    max_tokens=2048,
    request_timeout=8,
    max_retries=0,
)

QUESTION_HINTS = ("?", "？", "什么", "为何", "为什么", "如何", "怎么", "请问", "能否", "可以", "是否")
REPLY_HINTS = ("我觉得", "我理解", "我会", "我打算", "我修改", "可以改成", "因为", "我的回答", "我先")
LLM_DISABLED_BY_ENV = os.environ.get("STANDJOB_DISABLE_LLM", "").lower() in {"1", "true", "yes", "on"}
LLM_ENABLED = (not LLM_DISABLED_BY_ENV) and any(
    os.environ.get(k)
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")
)

# ─────────────────────────────────────────────────────────────
# 辅助：构建系统提示
# ─────────────────────────────────────────────────────────────
def _build_system_prompt(scoring_criteria: str, teaching_material: str) -> str:
    return f"""你是一名专业的岗标辅导专家，帮助员工完善"岗标价值与岗标任务"表格。

你的辅导原则：
1. 【不直接给答案】永远不直接告诉用户"应该写什么"，而是通过提问引导用户自己思考
2. 【严格依据评分标准】评审时必须逐条对照下方"评审打分标准"中的问题项进行检查，不要使用自定义的评分维度
3. 【教材仅作背景】引导和回答都必须基于教材内容，但在场景一不要直接贴出教材原文
4. 【每次聚焦一点】每轮对话只聚焦一个问题，避免信息轰炸
5. 【正向激励】先认可做得好的地方，再指出需要完善的地方

评审打分规则：
- 满分100分
- 每发现一个问题项，根据扣分等级扣分：A级扣10分，B级扣5分
- 得分≥85分为通关

--- 评审打分标准 ---
{scoring_criteria}

--- 教材内容 ---
{teaching_material}
"""


def _safe_json_load(raw: str, fallback: dict) -> dict:
    """兼容模型输出 markdown 代码块。"""
    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned.strip())
    except Exception:
        return fallback


def _is_question_intent(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(h in stripped for h in QUESTION_HINTS)


def _is_reply_intent(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(h in stripped for h in REPLY_HINTS)


def _rule_based_intent(text: str) -> tuple[str, float]:
    """规则优先：明显问句/明显作答直接判定，其余交给 LLM。"""
    stripped = (text or "").strip()
    if not stripped:
        return "reply", 0.6

    q = _is_question_intent(stripped)
    r = _is_reply_intent(stripped)

    if q and not r:
        return "question", 0.85
    if r and not q:
        return "reply", 0.8
    if q and r:
        return "uncertain", 0.4

    # 陈述句兜底为作答倾向，但置信度较低，触发 LLM 二判
    return "reply", 0.55


def _llm_intent_fallback(state: CoachState, user_text: str) -> str:
    """二判：当规则不稳定时，用轻量 LLM 分类 reply/question。"""
    system = _build_system_prompt(
        state.get("scoring_criteria", MOCK_SCORING_CRITERIA),
        state.get("teaching_material", MOCK_TEACHING_MATERIAL),
    )
    prompt = f"""请判断用户最新输入意图。

当前阶段: {state.get('phase', 'guiding')}
当前待引导问题: {state.get('pending_question', '')}
用户输入: {user_text}

分类标准:
1. intent=reply: 用户在回答当前引导问题或给出修改思路
2. intent=question: 用户在主动发起新的咨询问题（解释/方法/示例等）

仅输出 JSON:
{{"intent":"reply或question","confidence":0.0}}
"""
    if not LLM_ENABLED:
        return "reply"

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    except Exception:
        return "reply"

    parsed = _safe_json_load(resp.content, {"intent": "reply", "confidence": 0.5})
    intent = str(parsed.get("intent", "reply")).strip().lower()
    if intent not in {"reply", "question"}:
        return "reply"
    return intent


def _format_value_for_view(value: Any) -> str:
    if isinstance(value, list):
        items = []
        for idx, task in enumerate(value, 1):
            task_parts = [f"{k}: {v}" for k, v in task.items() if v]
            if task_parts:
                items.append(f"{idx}. {' | '.join(task_parts)}")
        return "\n".join(items)
    return str(value)


def _pick_related_fields(row_data: dict[str, Any], issue: dict[str, Any]) -> list[str]:
    """根据问题描述抽取最相关字段，至少返回 1 个字段。"""
    desc = f"{issue.get('category', '')} {issue.get('issue_desc', '')} {issue.get('explanation', '')}".lower()
    non_empty_keys = [k for k, v in row_data.items() if v]
    if not non_empty_keys:
        return []

    ranked: list[str] = []
    keywords = [
        ("岗位价值", ["岗位价值", "价值", "客户", "商业"]),
        ("岗位效能", ["岗位效能", "效能", "指标", "量化"]),
        ("核心任务", ["核心任务", "任务", "目的", "成果"]),
        ("辅助任务", ["辅助任务", "任务", "目的", "成果"]),
    ]
    for key in non_empty_keys:
        score = 0
        for _, hints in keywords:
            if any(h in key for h in hints):
                score += 1
            if any(h in desc for h in hints) and any(h in key for h in hints):
                score += 2
        if "资源投入" in key:
            score -= 1
        if score > 0:
            ranked.append((score, key))

    if ranked:
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return [k for _, k in ranked[:3]]

    return non_empty_keys[:2]


def _generate_issue_question(
    scoring_criteria: str,
    teaching_material: str,
    row_data: dict[str, Any],
    issue: dict[str, Any],
) -> str:
    """为当前问题项生成单一聚焦引导问题。"""
    system = _build_system_prompt(scoring_criteria, teaching_material)
    prompt = f"""请针对当前问题项生成一个苏格拉底式引导问题。

当前条目：
{json.dumps(row_data, ensure_ascii=False)}

当前问题项：
{json.dumps(issue, ensure_ascii=False)}

要求：
1. 只问一个问题
2. 不给答案
3. 问题要具体，可直接促使用户修改当前问题项

仅输出 JSON：
{{"question":"..."}}"""
    if not LLM_ENABLED:
        return "如果你现在就改这一处，你会先补充哪条最具体、可验证的信息？"

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    except Exception:
        return "如果你现在就改这一处，你会先补充哪条最具体、可验证的信息？"

    parsed = _safe_json_load(resp.content, {"question": "你觉得这一处如果要让评审更容易通过，最先应补充哪条具体信息？"})
    return parsed.get("question", "你觉得这一处如果要让评审更容易通过，最先应补充哪条具体信息？")


def _mock_review_result(row_data: dict[str, Any]) -> dict[str, Any]:
    """无外部模型时的轻量评审兜底，保证流程可运行。"""
    issues: list[dict[str, Any]] = []
    value_text = str(row_data.get("岗位价值", "") or row_data.get("岗位价值_", ""))
    eff_text = str(row_data.get("岗位效能", "") or row_data.get("岗位效能_", ""))

    if ("负责" in value_text) or (not value_text.strip()):
        issues.append(
            {
                "issue_id": "V-1",
                "issue_desc": "岗位价值表达偏动作化，客户价值不够明确",
                "category": "岗位价值",
                "level": "B",
                "deduction": 5,
                "explanation": "当前描述更像职责罗列，缺少“为谁创造什么价值”的表达。",
                "user_facing_desc": "当前岗位价值更像在描述你做了什么，还没有清晰体现你为谁创造了什么价值。",
            }
        )

    if (eff_text.strip() == "") or (eff_text.strip() in {"无", "N/A", "NA"}):
        issues.append(
            {
                "issue_id": "E-1",
                "issue_desc": "岗位效能缺少可衡量指标",
                "category": "岗位效能",
                "level": "B",
                "deduction": 5,
                "explanation": "当前效能栏位没有体现可观察结果。",
                "user_facing_desc": "当前岗位效能还看不出可衡量结果，建议补充可验证指标。",
            }
        )

    return {
        "matched_issues": issues,
        "score": max(0, 100 - sum(i.get("deduction", 0) for i in issues)),
        "teaching_refs": ["离线模式：使用本地规则评审。"],
    }


def _build_proactive_message(
    idx: int,
    total: int,
    row_data: dict[str, Any],
    issue: dict[str, Any],
    question: str,
) -> str:
    """场景一固定四段式输出。"""
    related_fields = issue.get("related_fields", [])

    original_lines: list[str] = []
    for key, value in row_data.items():
        if value and "资源投入" not in key:
            original_lines.append(f"- {key}: {_format_value_for_view(value)}")
    if not original_lines:
        original_lines = ["- （当前条目暂无可展示内容）"]

    related_lines: list[str] = []
    for key in related_fields:
        if key in row_data and row_data.get(key):
            related_lines.append(f"- {key}: {_format_value_for_view(row_data[key])}")
    if not related_lines:
        related_lines = ["- （当前问题项暂无明确关联字段，建议先补充关键描述）"]

    original_block = "\n".join(original_lines)
    related_block = "\n".join(related_lines)

    return (
        f"---\n"
        f"📋 第 {idx+1}/{total} 条（主动引导）\n\n"
        f"【原内容】\n"
        f"{original_block}\n\n"
        f"【关联内容】\n"
        f"{related_block}\n\n"
        f"【问题描述】\n"
        f"- {issue.get('user_facing_desc', '当前条目存在一个需要进一步澄清的问题项。')}\n\n"
        f"【引导性问题】\n"
        f"💬 {question}"
    )


def _advance_to_next_item(state: CoachState, feedback: str) -> dict:
    """当前条目问题全部解决后，切到下一条或完成态。"""
    idx = state["current_item_index"]
    rows = state["submission_rows"]
    next_idx = idx + 1
    is_last = next_idx >= len(rows)

    if is_last:
        msg = AIMessage(content=(
            f"✅ {feedback}\n\n"
            "🎉 当前评审条目已全部完成。\n"
            "后续若你有问题，可继续直接提问，我会进入被动答疑模式。"
        ))
        return {
            "messages": [msg],
            "current_item_index": next_idx,
            "phase": "done",
            "active_mode": "reactive_qa",
            "awaiting_user_input": True,
            "pending_question": None,
        }

    msg = AIMessage(content=(
        f"✅ {feedback}\n\n"
        f"---\n继续进入第 {next_idx + 1}/{len(rows)} 条，我们保持一次只聚焦一个问题。"
    ))
    return {
        "messages": [msg],
        "current_item_index": next_idx,
        "phase": "reviewing",
        "active_mode": "proactive",
        "reflection_round": 0,
        "issue_round": 0,
        "current_issue_index": 0,
        "awaiting_user_input": False,
    }


# ─────────────────────────────────────────────────────────────
# NODE 1: load_standards
# 启动时预加载评审标准（评分标准xlsx + 教材PDF）
# ─────────────────────────────────────────────────────────────
def load_standards(state: CoachState) -> dict:
    """
    预加载评分标准 xlsx 和教材 PDF。
    路径从环境变量读取，不存在则用 Mock 数据（便于测试）。
    """
    print("[Node] load_standards: 加载评审标准...")

    criteria_path = os.environ.get("SCORING_CRITERIA_PATH", "")
    teaching_path = os.environ.get("TEACHING_MATERIAL_PATH", "")

    # 加载评分标准
    if criteria_path and os.path.exists(criteria_path):
        scoring_criteria = load_scoring_criteria(criteria_path)
        print(f"  ✓ 已加载评分标准: {criteria_path}")
    else:
        scoring_criteria = MOCK_SCORING_CRITERIA
        print("  ⚠ 使用 Mock 评分标准（设置 SCORING_CRITERIA_PATH 可指定真实文件）")

    # 加载教材 PDF
    if teaching_path and os.path.exists(teaching_path):
        teaching_material = load_pdf_as_text(teaching_path)
        print(f"  ✓ 已加载教材: {teaching_path}")
    else:
        teaching_material = MOCK_TEACHING_MATERIAL
        print("  ⚠ 使用 Mock 教材（设置 TEACHING_MATERIAL_PATH 可指定真实文件）")

    welcome_msg = AIMessage(content=(
        "👋 欢迎使用岗标辅导系统！\n\n"
        "请上传您的 **岗标价值与岗标任务** 表格（.xlsx 格式），"
        "我将逐项引导您完善每个条目 📋"
    ))

    return {
        "scoring_criteria": scoring_criteria,
        "teaching_material": teaching_material,
        "phase": "loaded",
        "active_mode": "proactive",
        "last_user_intent": "reply",
        "messages": [welcome_msg],
        "review_items": [],
        "current_item_index": 0,
        "reflection_round": 0,
        "issue_round": 0,
        "current_issue_index": 0,
        "issue_status_map": {},
        "awaiting_user_input": True,
    }


# ─────────────────────────────────────────────────────────────
# NODE 2: parse_submission
# 解析用户上传的表格，提取结构化数据
# ─────────────────────────────────────────────────────────────
def parse_submission(state: CoachState) -> dict:
    """解析用户提交的 xlsx，提取列和行数据"""
    path = state.get("submission_path", "")
    print(f"[Node] parse_submission: 解析文件 {path}")

    if not path or not os.path.exists(path):
        # 使用 Mock 数据测试
        path = _create_mock_submission()
        print(f"  ⚠ 使用 Mock 提交文件: {path}")

    columns, rows, submission_text = load_submission(path)

    # 根据是否为分组结构生成不同的消息
    has_tasks = any(isinstance(row.get("核心任务"), list) for row in rows)
    if has_tasks:
        core_count = sum(len(row.get("核心任务", [])) for row in rows)
        aux_count = sum(len(row.get("辅助任务", [])) for row in rows)
        msg = AIMessage(content=(
            f"✅ 已成功解析您的表格！\n\n"
            f"📊 共识别到 **{len(rows)} 个岗位价值条目**，包含：\n"
            f"  • 核心任务 **{core_count}** 项\n"
            f"  • 辅助任务 **{aux_count}** 项\n\n"
            "接下来我将逐条引导您完善，我们一起来把这份表格打磨得更完整 💪\n\n"
            "准备好了吗？我们从第一条开始👇"
        ))
    else:
        msg = AIMessage(content=(
            f"✅ 已成功解析您的表格！\n\n"
            f"📊 共识别到 **{len(rows)} 个条目**，列结构如下：\n"
            f"`{'` → `'.join(columns)}`\n\n"
            "接下来我将逐条引导您完善，我们一起来把这份表格打磨得更完整 💪\n\n"
            "准备好了吗？我们从第一条开始👇"
        ))

    return {
        "submission_path": path,
        "submission_text": submission_text,
        "submission_columns": columns,
        "submission_rows": rows,
        "phase": "reviewing",
        "active_mode": "proactive",
        "messages": [msg],
        "current_item_index": 0,
        "current_issue_index": 0,
        "issue_round": 0,
        "review_items": [],
    }


# ─────────────────────────────────────────────────────────────
# NODE 3: review_item
# 对当前条目进行 AI 评审，生成结构化评审结果
# ─────────────────────────────────────────────────────────────
def review_item(state: CoachState) -> dict:
    """
    对 current_item_index 对应的行进行评审，
    生成 ReviewItem（含评分、问题、建议），存入 review_items。
    不向用户直接展示评审结果，仅供内部使用。
    """
    idx = state["current_item_index"]
    rows = state["submission_rows"]
    columns = state["submission_columns"]

    if idx >= len(rows):
        print(f"[Node] review_item: 所有条目已评审完毕")
        return {"phase": "done"}

    row_data = rows[idx]
    print(f"[Node] review_item: 评审第 {idx+1}/{len(rows)} 条...")

    # 构建评审 Prompt
    system = _build_system_prompt(
        state["scoring_criteria"],
        state["teaching_material"]
    )
    review_prompt = f"""请对以下岗标条目进行专业评审。

## 评审规则
请严格按照"评审打分标准"中的问题项逐条检查当前条目。
- 满分100分
- 每命中一个问题项，根据扣分等级扣分：A级扣10分，B级扣5分
- 只检查与当前条目相关的问题项（根据"通关阶段"和"评审内容"匹配）

## 评审要求
- 必须引用"评审打分标准"中的具体问题项编号和原始描述
- 必须引用"教材内容"中的具体原则、概念或案例来支撑评审意见
- 不要使用自定义的评分维度（如"价值清晰度""任务完整性"等），只用评审打分标准中的问题项

## 当前条目
表格列定义：{' | '.join(columns)}

当前条目（第{idx+1}行）：
{json.dumps(row_data, ensure_ascii=False, indent=2)}

请严格按照如下 JSON 格式输出评审结果（不要输出任何其他内容）：
{{
  "matched_issues": [
    {{
      "issue_id": "问题项编号",
      "issue_desc": "从评审打分标准中引用的问题项原始描述",
      "category": "评审内容类别（如岗位价值、岗位效能、岗位任务等）",
      "level": "A或B",
      "deduction": 10,
            "explanation": "具体说明为什么此条目命中了该问题项",
            "user_facing_desc": "给用户看的问题描述（不包含分数、扣分、等级）"
    }}
  ],
    "score": 100,
    "teaching_refs": ["内部教材参考（不展示给用户）"]
}}

注意：score = 100 - 所有命中问题项的扣分之和。每个问题项只能命中一次。"""

    if LLM_ENABLED:
        try:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=review_prompt),
            ])
            result = _safe_json_load(
                response.content,
                {
                    "matched_issues": [],
                    "score": 100,
                    "teaching_refs": ["请参考教材相关章节"],
                },
            )
        except Exception:
            result = _mock_review_result(row_data)
    else:
        result = _mock_review_result(row_data)

    matched_issues = result.get("matched_issues", [])
    total_deduction = sum(item.get("deduction", 0) for item in matched_issues)
    score = max(0, 100 - total_deduction)

    issue_queue: list[dict[str, Any]] = []
    issue_status_map = dict(state.get("issue_status_map", {}))
    for item in matched_issues:
        issue_id = item.get("issue_id", "unknown")
        related_fields = _pick_related_fields(row_data, item)
        user_desc = item.get("user_facing_desc") or item.get("issue_desc") or "当前条目存在需完善的问题。"
        issue_queue.append(
            {
                "issue_id": issue_id,
                "issue_desc": item.get("issue_desc", ""),
                "category": item.get("category", ""),
                "status": "pending",
                "related_fields": related_fields,
                "user_facing_desc": user_desc,
            }
        )
        issue_status_map[issue_id] = "pending"

    first_question = ""
    if issue_queue:
        issue_queue[0]["status"] = "in_progress"
        issue_status_map[issue_queue[0]["issue_id"]] = "in_progress"
        first_question = _generate_issue_question(
            state["scoring_criteria"],
            state["teaching_material"],
            row_data,
            issue_queue[0],
        )
    else:
        first_question = "这一条目目前没有命中问题项。你是否想主动优化其中某个部分？"

    review_item_obj: ReviewItem = {
        "row_index": idx + 1,
        "row_data": row_data,
        "score": score,
        "dimension_scores": {"matched_issues": matched_issues},
        "issues": [
            f"[{item.get('category', '')}] {item.get('issue_desc', '')} ({item.get('level', '')}级 -{item.get('deduction', 0)}分)"
            for item in matched_issues
        ],
        "suggestions": result.get("teaching_refs", []),
        "standard_ref": result.get("teaching_refs", [""])[0] if result.get("teaching_refs") else "",
        "status": "reflecting",
        "issue_queue": issue_queue,
    }

    # 将评审结果追加到列表
    existing = list(state.get("review_items", []))
    existing.append(review_item_obj)

    return {
        "review_items": existing,
        "pending_question": first_question,
        "phase": "guiding",
        "active_mode": "proactive",
        "current_issue_index": 0,
        "issue_round": 0,
        "issue_status_map": issue_status_map,
        "reflection_round": 0,
    }


# ─────────────────────────────────────────────────────────────
# NODE 4: guide_reflection
# 根据评审结果，向用户提出引导性问题，启动反思对话
# ─────────────────────────────────────────────────────────────
def guide_reflection(state: CoachState) -> dict:
    """
    发出引导性问题，引导用户自我反思当前条目。
    构建一段展示当前条目概况 + 引导问题的消息。
    """
    idx = state["current_item_index"]
    review_items = state.get("review_items", [])
    rows = state["submission_rows"]

    if idx >= len(rows):
        return {"phase": "done"}

    current_review = review_items[idx] if idx < len(review_items) else None
    row_data = rows[idx]
    columns = state["submission_columns"]
    total = len(rows)
    question = state.get("pending_question", "")
    issue_idx = state.get("current_issue_index", 0)
    issue_queue = current_review.get("issue_queue", []) if current_review else []

    print(f"[Node] guide_reflection: 条目{idx+1}/{total}，问题项{issue_idx+1}")

    if not current_review:
        content = "当前条目尚未生成评审结果，请继续输入。"
    elif not issue_queue:
        content = (
            f"📋 第 {idx+1}/{total} 条\n\n"
            "当前条目未命中问题项。若你愿意，我也可以继续主动帮你做优化提问。"
        )
    else:
        issue = issue_queue[min(issue_idx, len(issue_queue) - 1)]
        content = _build_proactive_message(
            idx=idx,
            total=total,
            row_data=row_data,
            issue=issue,
            question=question,
        )

    msg = AIMessage(content=content)
    return {
        "messages": [msg],
        "awaiting_user_input": True,
        "phase": "guiding",
        "active_mode": "proactive",
    }


def answer_user_question(state: CoachState) -> dict:
    """场景二：用户主动提问时的被动答疑。"""
    messages = state.get("messages", [])
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    question = user_messages[-1].content if user_messages else ""

    system = _build_system_prompt(
        state.get("scoring_criteria", MOCK_SCORING_CRITERIA),
        state.get("teaching_material", MOCK_TEACHING_MATERIAL),
    )
    qa_prompt = f"""用户主动提问如下，请进入被动答疑模式给出专业回答。

用户问题：{question}

要求：
1. 直接回答问题，语言专业但简洁
2. 可基于教材原则解释，但不要贴教材原文大段引用
3. 如果当前仍在主动引导流程（phase=guiding/reviewing），回答后加一句"我们回到当前问题项"的过渡

只输出 JSON：
{{
  "answer": "...",
  "follow_back": "..."
}}"""

    if LLM_ENABLED:
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=qa_prompt)])
            parsed = _safe_json_load(
                resp.content,
                {
                    "answer": "这个问题很关键。建议你先从服务对象、可衡量成果和业务结果三个维度拆开检查。",
                    "follow_back": "我们回到当前问题项，继续逐项完善。",
                },
            )
        except Exception:
            parsed = {
                "answer": "这是一个很好的问题。可以先从服务对象、关键行为和可衡量成果三层去梳理。",
                "follow_back": "我们回到当前问题项，继续逐项完善。",
            }
    else:
        parsed = {
            "answer": "这是一个很好的问题。可以先从服务对象、关键行为和可衡量成果三层去梳理。",
            "follow_back": "我们回到当前问题项，继续逐项完善。",
        }

    follow_back = ""
    if state.get("phase") in {"guiding", "reviewing"}:
        follow_back = parsed.get("follow_back", "我们回到当前问题项，继续逐项完善。")
    elif state.get("phase") == "done":
        follow_back = "你还可以继续提问，我会持续答疑。"

    content = f"💡 针对你的问题“{question}”：\n{parsed.get('answer', '')}"
    if follow_back:
        content += f"\n\n{follow_back}"

    return {
        "messages": [AIMessage(content=content)],
        "active_mode": "reactive_qa",
        "last_user_intent": "question",
        "awaiting_user_input": True,
    }


def detect_user_intent(state: CoachState) -> dict:
    """意图识别节点：规则优先，低置信度再走 LLM 二判。"""
    phase = state.get("phase", "guiding")
    messages = state.get("messages", [])
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    last_user_msg = user_messages[-1].content if user_messages else ""

    if phase == "done":
        return {
            "last_user_intent": "question",
            "active_mode": "reactive_qa",
        }

    intent, confidence = _rule_based_intent(last_user_msg)
    if intent == "uncertain" or confidence < 0.7:
        intent = _llm_intent_fallback(state, last_user_msg)

    return {
        "last_user_intent": intent,
        "active_mode": "reactive_qa" if intent == "question" else "proactive",
    }


# ─────────────────────────────────────────────────────────────
# NODE 5: process_response
# 处理用户回复，决定继续追问 or 进入下一条目
# ─────────────────────────────────────────────────────────────
def process_response(state: CoachState) -> dict:
    """
    分析用户对引导问题的回复：
    - 理解不深：继续追问（最多3轮）
    - 理解到位 or 已达最大轮次：给予总结性反馈，移向下一条目
    """
    idx = state["current_item_index"]
    review_items = state.get("review_items", [])
    current_review = review_items[idx] if idx < len(review_items) else None
    messages = state.get("messages", [])
    issue_round = state.get("issue_round", 0)
    rows = state["submission_rows"]
    issue_idx = state.get("current_issue_index", 0)

    print(f"[Node] process_response: 条目{idx+1}，问题项{issue_idx+1}，已追问{issue_round+1}轮")

    # 获取用户最后一条消息
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    last_user_msg = user_messages[-1].content if user_messages else ""

    if _is_question_intent(last_user_msg):
        return answer_user_question(state)

    if not current_review:
        return {"phase": "reviewing", "active_mode": "proactive", "last_user_intent": "reply"}

    issue_queue = current_review.get("issue_queue", [])
    if not issue_queue:
        return _advance_to_next_item(state, "这一条目无需调整，我们继续下一条。")

    current_issue = issue_queue[min(issue_idx, len(issue_queue) - 1)]

    system = _build_system_prompt(
        state["scoring_criteria"],
        state["teaching_material"]
    )

    # 构建上下文：最近几轮对话
    recent = messages[-8:] if len(messages) > 8 else messages

    decide_prompt = f"""当前正在辅导第{idx+1}条岗标条目的当前问题项。

当前问题项：
{json.dumps(current_issue, ensure_ascii=False)}

用户刚才回复："{last_user_msg}"
已进行轮次：{issue_round + 1}

请判断：
1. 该问题项是否已经解决（用户已给出可执行、可落地的改进方向）？
2. 如果未解决，请继续给一个更聚焦的问题；如果已解决，给一句过渡反馈。

请仅输出 JSON：
{{
    "issue_resolved": true,
    "feedback_to_user": "...",
    "next_question": "..."
}}"""

    if LLM_ENABLED:
        try:
            response = llm.invoke([
                SystemMessage(content=system),
                *recent,
                HumanMessage(content=decide_prompt),
            ])
            decision = _safe_json_load(
                response.content,
                {
                    "issue_resolved": False,
                    "feedback_to_user": "你的方向是对的，我们再把关键点说得更具体一些。",
                    "next_question": "如果你现在就改写这段内容，你会先增加哪一个可量化或可验证的信息？",
                },
            )
        except Exception:
            decision = {
                "issue_resolved": any(k in last_user_msg for k in ["我会", "我改", "我补充", "可量化", "指标", "客户"]),
                "feedback_to_user": "你的思考方向不错。",
                "next_question": "你愿意把这条改成一句可被直接评审验证的表达吗？",
            }
    else:
        decision = {
            "issue_resolved": any(k in last_user_msg for k in ["我会", "我改", "我补充", "可量化", "指标", "客户"]),
            "feedback_to_user": "你的思考方向不错。",
            "next_question": "你愿意把这条改成一句可被直接评审验证的表达吗？",
        }

    feedback = decision.get("feedback_to_user", "谢谢你的思考。")
    issue_resolved = bool(decision.get("issue_resolved", False))
    next_question = decision.get("next_question", "你觉得还差哪一步可以让这个问题真正解决？")

    issue_status_map = dict(state.get("issue_status_map", {}))
    queue_copy = list(issue_queue)

    if issue_resolved:
        current_issue_id = current_issue.get("issue_id", f"issue_{issue_idx}")
        issue_status_map[current_issue_id] = "resolved"
        queue_copy[issue_idx]["status"] = "resolved"

        next_issue_idx = issue_idx + 1
        if next_issue_idx < len(queue_copy):
            queue_copy[next_issue_idx]["status"] = "in_progress"
            next_issue = queue_copy[next_issue_idx]
            issue_status_map[next_issue.get("issue_id", f"issue_{next_issue_idx}")] = "in_progress"
            next_issue_question = _generate_issue_question(
                state["scoring_criteria"],
                state["teaching_material"],
                rows[idx],
                next_issue,
            )
            review_items[idx]["issue_queue"] = queue_copy
            return {
                "messages": [AIMessage(content=f"✅ {feedback}\n\n我们进入下一个关注点，继续逐项完善。")],
                "review_items": review_items,
                "issue_status_map": issue_status_map,
                "current_issue_index": next_issue_idx,
                "issue_round": 0,
                "pending_question": next_issue_question,
                "phase": "guiding",
                "active_mode": "proactive",
                "last_user_intent": "reply",
                "awaiting_user_input": False,
            }

        review_items[idx]["issue_queue"] = queue_copy
        return {
            **_advance_to_next_item(state, feedback),
            "review_items": review_items,
            "issue_status_map": issue_status_map,
            "last_user_intent": "reply",
        }

    review_items[idx]["issue_queue"] = queue_copy
    content = f"✅ {feedback}\n\n💬 {next_question}"
    return {
        "messages": [AIMessage(content=content)],
        "review_items": review_items,
        "pending_question": next_question,
        "issue_round": issue_round + 1,
        "reflection_round": state.get("reflection_round", 0) + 1,
        "awaiting_user_input": True,
        "phase": "guiding",
        "active_mode": "proactive",
        "last_user_intent": "reply",
    }


# ─────────────────────────────────────────────────────────────
# 辅助：创建 Mock 提交文件（测试用）
# ─────────────────────────────────────────────────────────────
def _create_mock_submission() -> str:
    """创建一个测试用的岗标提交文件（含多级表头和左右分表结构）"""
    import openpyxl
    path = "/tmp/mock_submission.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "岗标价值与岗标任务"

    # Row 1: 大类表头
    ws.append(["岗位价值", "岗位效能", "核心任务", None, None, "资源投入",
               "辅助任务", None, None, "资源投入"])
    ws.merge_cells("C1:E1")   # 核心任务 横跨3列
    ws.merge_cells("G1:I1")   # 辅助任务 横跨3列

    # Row 2: 子列表头
    ws.append([None, None, "任务名称", "任务目的", "成果标准", None,
               "任务名称", "任务目的", "成果标准", None])
    ws.merge_cells("A1:A2")   # 岗位价值 纵向合并
    ws.merge_cells("B1:B2")   # 岗位效能 纵向合并
    ws.merge_cells("F1:F2")   # 资源投入 纵向合并
    ws.merge_cells("J1:J2")   # 资源投入 纵向合并

    # ── 条目1：有问题的内容 ──
    ws.append([
        "负责软件开发工作", "无",           # 价值描述太宽泛，效能无指标
        "写代码", "完成功能开发", "完成任务", "60%",    # 核心任务1
        "开会", "信息同步", "参加会议", "40%",          # 辅助任务1
    ])
    ws.append([
        None, None,
        "修bug", "减少缺陷", "bug数量", None,           # 核心任务2
        "写周报", "汇报工作", "提交周报", None,          # 辅助任务2
    ])
    ws.merge_cells("A3:A4")

    # ── 条目2：较好的内容 ──
    ws.append([
        "通过技术架构优化提升系统稳定性", "系统可用率≥99.9%",
        "主导核心模块技术方案设计", "提升模块质量与可维护性", "方案评审通过率100%", "70%",
        "技术文档编写", "知识沉淀与传承", "文档完整率90%", "30%",
    ])
    ws.append([
        None, None,
        "性能瓶颈分析与优化", "提升系统响应速度", "P99延迟≤200ms", None,
        "技术分享", "团队能力提升", "季度至少2次", None,
    ])
    ws.merge_cells("A5:A6")

    # ── 条目3：有问题的内容 ──
    ws.append([
        "支撑业务快速迭代", "无",
        "需求分析和功能开发", "满足业务需求", "功能上线", "80%",
        "协助测试", "保障质量", "配合完成", "20%",
    ])

    wb.save(path)
    print(f"  已生成 Mock 提交文件: {path}")
    return path
