"""
岗标辅导 Graph Nodes
每个 node 接收 CoachState，返回部分状态更新（dict）。
"""
from __future__ import annotations
import os
import json
from typing import Any, Optional

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
# LLM 实例（延迟初始化）
# ─────────────────────────────────────────────────────────────
_llm_instance = None

def _get_llm():
    """延迟初始化 LLM 实例，避免导入时立即初始化"""
    global _llm_instance
    if _llm_instance is None and _is_llm_enabled():
        try:
            _llm_instance = ChatOpenAI(
                model="deepseek-v4-pro",
                max_tokens=2048,
                request_timeout=8,
                max_retries=0,
            )
        except Exception as e:
            print(f"[警告] LLM 初始化失败: {e}")
            _llm_instance = None
    return _llm_instance


def _is_llm_enabled() -> bool:
    """检查 LLM 是否启用"""
    disabled = os.environ.get("STANDJOB_DISABLE_LLM", "").lower() in {"1", "true", "yes", "on"}
    if disabled:
        return False
    has_key = any(os.environ.get(k) for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"))
    return has_key


# 保持向后兼容
QUESTION_HINTS = ("?", "？", "什么", "为何", "为什么", "如何", "怎么", "请问", "能否", "可以", "是否")
REPLY_HINTS = ("我觉得", "我理解", "我会", "我打算", "我修改", "可以改成", "因为", "我的回答", "我先")
LLM_DISABLED_BY_ENV = os.environ.get("STANDJOB_DISABLE_LLM", "").lower() in {"1", "true", "yes", "on"}


@property
def LLM_ENABLED() -> bool:
    """LLM 是否可用（延迟检查）"""
    return _is_llm_enabled()

RUBRIC_ITEMS: list[dict[str, Any]] = [
    {"id": 1, "category": "岗位价值", "level": "A", "desc": "站在自身视角而不是站在客户视角提炼岗位价值"},
    {"id": 2, "category": "岗位价值", "level": "A", "desc": "岗位价值并非源于对客户最在意、最深层需求的提炼"},
    {"id": 3, "category": "岗位价值", "level": "A", "desc": "没有从不同客户的视角出发，去提炼岗位价值"},
    {"id": 4, "category": "岗位价值", "level": "A", "desc": "岗位价值描述空泛，指导性不强"},
    {"id": 5, "category": "岗位效能", "level": "A", "desc": "岗位效能并不能直接、有效地衡量岗位价值"},
    {"id": 6, "category": "岗位任务", "level": "A", "desc": "核心任务的目的集合无法完整覆盖岗位价值"},
    {"id": 7, "category": "岗位任务", "level": "B", "desc": "任务命名未按动词+修饰语+名词"},
    {"id": 8, "category": "岗位任务", "level": "B", "desc": "直接用目的命名任务"},
    {"id": 9, "category": "任务目的与成果", "level": "A", "desc": "任务目的模糊，导致为何而做不清"},
    {"id": 10, "category": "任务目的与成果", "level": "A", "desc": "成果标准与任务目的脱节"},
    {"id": 11, "category": "任务目的与成果", "level": "A", "desc": "任务成果评估周期设计过长"},
    {"id": 12, "category": "任务目的与成果", "level": "A", "desc": "成果标准不符合SMART原则"},
    {"id": 13, "category": "任务目的与成果", "level": "A", "desc": "把交付物当做成果"},
    {"id": 14, "category": "任务目的与成果", "level": "A", "desc": "成果标准未在完成度、交期、预算上设计挑战目标"},
]

RUBRIC_INDEX = {item["id"]: item for item in RUBRIC_ITEMS}
WHO_HINTS = ("客户", "用户", "产品经理", "研发", "业务", "销售", "团队", "测试", "运营")
BENEFIT_HINTS = ("减少", "提升", "降低", "提高", "保障", "避免", "缩短", "稳定", "改善")
BIZ_HINTS = ("收入", "成本", "风险", "品牌", "口碑", "市场", "利润", "效率")

# 教材示例引用（用于引导时引用具体案例）
TEXTBOOK_EXAMPLES = {
    "岗位价值": {
        "正例": "对产品经理的价值是'减少缺陷流出，避免口碑拖累市场推广，夯实产品质量竞争力'",
        "反例": "充分理解用户和产品需求，设计并实现易用、体验好、高性价比解法...（动作偏多，但包含客户、收获、商业结果——不需要死磕）",
        "认知陷阱": [
            "视角偏差：固守'我能做什么'，而非传递'你能得到什么'",
            "深度偏差：轻信'客户开的药方'，没挖到真实问题",
        ],
    },
    "岗位效能": {
        "测试工程师例子": "价值是'减少缺陷流出'，对应效能是'缺陷泄漏率≤0.5%；线上重大故障：0次'。价值是'提供清晰质量反馈'，对应效能是'缺陷描述一次通过率≥95%'",
    },
    "岗位任务": {
        "命名格式": "动词+修饰语+名词，例如'整理客户拜访纪要'",
        "反例": "'降低产品成本'（直接用目的命名，只标出终点，未指引行动）",
    },
    "任务目的与成果": {
        "客服例子": "目的从'5分钟内首次响应'校准为'一次性彻底解决用户疑问，避免二次进线'后，完成度变为'用户回复已解决且24小时内未就同一问题再次咨询'",
        "交付陷阱": "低效委派是'你开发一套产品培训课件'，高效委派要明确'目的：解决因一线工程师不熟悉产品内部原理导致故障处理时长超标'，'成果：一个月内将平均故障处理时长从6小时降至3小时以内'",
    },
    "服务客户": {
        "财务BP": "客户是业务部门负责人，而非财务总监",
        "服务团队": "把'工程商'细分为'中大型集成商'和'小型承包商'",
    },
}


def _rubric_catalog_text() -> str:
    lines = []
    for item in RUBRIC_ITEMS:
        lines.append(
            f"{item['id']}. [{item['category']}] {item['desc']} (等级{item['level']})"
        )
    return "\n".join(lines)


def _coerce_rubric_item_id(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    item_id = int(digits)
    if item_id in RUBRIC_INDEX:
        return item_id
    return None


def _sanitize_user_desc(text: str) -> str:
    sanitized = (text or "").strip()
    blocked = ["评分标准", "教材第", "问题项", "第1项", "第2项", "第3项", "第4项", "第5项", "第6项", "第7项", "第8项", "第9项", "第10项", "第11项", "第12项", "第13项", "第14项"]
    for token in blocked:
        sanitized = sanitized.replace(token, "")
    return sanitized.strip("：:，,。 ") or "当前表达还有一层关键信息不够具体。"


def _extract_value_text(row_data: dict[str, Any]) -> str:
    texts: list[str] = []
    for key, value in row_data.items():
        if "岗位价值" in key and value:
            texts.append(str(value))
    return " ".join(texts)


def _passes_three_questions(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    has_who = any(k in candidate for k in WHO_HINTS)
    has_benefit = any(k in candidate for k in BENEFIT_HINTS)
    has_biz = any(k in candidate for k in BIZ_HINTS)
    return has_who and has_benefit and has_biz


def _should_relax_issue(item_id: int, row_data: dict[str, Any], issue: dict[str, Any]) -> bool:
    """
    灰区放过逻辑 - 基于 skill 中的严格度校准对照表

    判定原则：三问法都过 → 合格放过；灰色地带默认放过
    """
    # 岗位价值相关项（1-4）：若已通过三问法，则不进入引导清单
    if item_id in {1, 2, 3, 4}:
        value_text = _extract_value_text(row_data)
        if _passes_three_questions(value_text):
            return True

    # 岗位效能（5）：若效能与价值关键词能对上，放过
    if item_id == 5:
        value_text = _extract_value_text(row_data)
        eff_text = str(row_data.get("岗位效能", "") or row_data.get("岗位效能_", ""))
        # 简单检查：效能不为空且包含量化指标关键词
        if eff_text and any(k in eff_text for k in ("%", "率", "次", "个", "小时", "天", "数量", "≤", ">=")):
            return True

    # 任务命名 B级问题（7-8）：边界问题描述轻微时放过
    if item_id in {7, 8}:
        desc = (issue.get("issue_desc") or "") + (issue.get("explanation") or "")
        # 描述中有"轻微"、"偶有"等词，或任务名仍符合动宾结构
        if "轻微" in desc or "偶有" in desc:
            return True
        task_names = []
        for k, v in row_data.items():
            if "任务" in k and v and isinstance(v, str):
                task_names.append(v)
        # 如果任务名包含动词，基本可用
        verbs = ("负责", "完成", "制定", "整理", "分析", "设计", "开发", "测试", "评审", "编写")
        if task_names and any(v in task_names[0] for v in verbs):
            return True

    # 任务目的与成果（9-14）：主要目的有支撑则放过
    if item_id in {9, 10, 11, 12, 13, 14}:
        # 检查是否有明确的成果指标
        outcome_text = ""
        for k, v in row_data.items():
            if "成果" in k and v:
                outcome_text += str(v)
        # 有具体指标则放过
        if any(k in outcome_text for k in ("%", "率", "次", "个", "≤", ">=", "内", "以上")):
            return True

    return False


def _normalize_matched_issues(matched_issues: list[dict[str, Any]], row_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    normalized: list[dict[str, Any]] = []
    relaxed_ids: list[int] = []

    for raw in matched_issues:
        item_id = _coerce_rubric_item_id(raw.get("issue_id"))
        if item_id is None:
            continue
        canonical = RUBRIC_INDEX[item_id]
        level = str(raw.get("level", canonical["level"]))
        deduction = int(raw.get("deduction", 10 if level == "A" else 5))
        issue = {
            "issue_id": str(item_id),
            "rubric_item_id": item_id,
            "issue_desc": raw.get("issue_desc") or canonical["desc"],
            "category": raw.get("category") or canonical["category"],
            "level": "A" if level.upper() == "A" else "B",
            "deduction": deduction,
            "explanation": raw.get("explanation", ""),
            "user_facing_desc": _sanitize_user_desc(raw.get("user_facing_desc") or raw.get("issue_desc") or canonical["desc"]),
        }

        if _should_relax_issue(item_id, row_data, issue):
            relaxed_ids.append(item_id)
            continue
        normalized.append(issue)

    normalized.sort(key=lambda x: int(x.get("rubric_item_id", 99)))
    return normalized, sorted(set(relaxed_ids))


def _build_escalation_question(issue: dict[str, Any], row_data: dict[str, Any], hint_level: int) -> str:
    related_fields = issue.get("related_fields", [])
    field_name = related_fields[0] if related_fields else "当前描述"
    current_text = str(row_data.get(field_name, "")) if field_name in row_data else ""

    if hint_level <= 1:
        return f"如果只改 {field_name} 这一处，你会先补哪一个可验证信息，能让评审一眼看到变化？"
    if hint_level == 2:
        return (
            f"给你一个更具体的抓手：围绕 {field_name}，先补'服务对象'、'可观察结果'、'业务影响'三者中的哪一项？"
        )
    return (
        f"我们再收窄一步：基于你现在这句“{current_text[:60]}”，你愿意先把它改成“为谁带来什么具体结果”的一句话吗？"
    )

# ─────────────────────────────────────────────────────────────
# 辅助：构建系统提示（五条铁律）
# ─────────────────────────────────────────────────────────────
def _build_system_prompt(scoring_criteria: str, teaching_material: str) -> str:
    """
    构建系统提示，遵循五条铁律：
    1. 【不直接给答案】永远不直接告诉用户"应该写什么"
    2. 【严格依据评分标准】逐条对照14项评分标准
    3. 【教材仅作背景】引导和回答基于教材，但场景一不直接贴原文
    4. 【每次聚焦一点】每轮对话只聚焦一个问题
    5. 【正向激励】先认可做得好的地方，再指出需完善的地方
    """
    return f"""你是一名专业的岗标辅导专家，帮助员工完善"岗标价值与岗标任务"表格。

## 你的辅导原则（五条铁律）

1. **【不直接给答案】** 永远不直接告诉用户"应该写什么"，而是通过提问引导用户自己思考
2. **【严格依据评分标准】** 评审时必须逐条对照14项评分标准进行检查，不要使用自定义评分维度
3. **【教材仅作背景】** 引导和回答基于教材内容，但在主动引导场景不要直接贴教材原文
4. **【每次聚焦一点】** 每轮对话只聚焦一个问题，避免信息轰炸
5. **【正向激励】** 先认可做得好的地方，再指出需要完善的地方

## 角色纪律

- **始终以教练身份对话**——不说"我来评审一下""根据我的分析"这类暴露模型思考过程的话，自然地聊
- **引用教材和标准时，用人话表达核心意思，不报章节号、不报条目编号**
- **不要展示评审过程**——用户不需要知道你对着14条逐条检查了

### 正确示例
❌ "根据评分标准第7项，你的任务命名未按'动词+修饰语+名词'"
❌ "教材第5章提到'动词+修饰语+名词'格式"
✅ "你写的'AI应用开发与业务落地'——'业务落地'听起来更像你想要的结果，而不是一个能直接动手做的动作。如果把任务名字换成'一看就知道要干什么'的写法，你觉得该怎么改？"

## 评审打分规则

- 满分100分
- 每发现一个问题项，根据扣分等级扣分：A级扣10分，B级扣5分
- 得分≥85分为通关

## 严格度校准（三问法）

每条价值/任务先用此筛选，三问都过则放过不追问：
1. 能圈出明确的"客户主语"吗？
2. 客户能看到自己拿到的"具体好处"吗？
3. 最终落到 收入/成本/风险/品牌 中至少一个吗？

**灰色地带默认放过**——标准没明文覆盖的不主动找事。

---

### 评审打分标准
{scoring_criteria}

### 教材内容
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
    if not _is_llm_enabled():
        return "reply"

    try:
        llm = _get_llm()
        if llm is None:
            return "reply"
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
    """
    为当前问题项生成单一聚焦引导问题。
    根据问题类别引用教材示例，用苏格拉底式提问引导用户。
    """
    system = _build_system_prompt(scoring_criteria, teaching_material)

    # 获取问题类别和ID
    category = issue.get("category", "")
    rubric_item_id = issue.get("rubric_item_id", 0)

    # 构建教材示例引用
    example_context = ""
    if category in TEXTBOOK_EXAMPLES:
        examples = TEXTBOOK_EXAMPLES[category]
        if "正例" in examples:
            example_context += f"\n教材正例：{examples['正例']}\n"
        if "测试工程师例子" in examples:
            example_context += f"\n教材示例：{examples['测试工程师例子']}\n"
        if "命名格式" in examples:
            example_context += f"\n格式要求：{examples['命名格式']}\n"
        if "客服例子" in examples:
            example_context += f"\n教材案例：{examples['客服例子']}\n"

    prompt = f"""请针对当前问题项生成一个苏格拉底式引导问题。

当前条目：
{json.dumps(row_data, ensure_ascii=False)}

当前问题项：
{json.dumps(issue, ensure_ascii=False)}
{example_context}
要求：
1. **只问一个问题** - 每次聚焦一个点
2. **不给答案** - 用提问"钓"出答案
3. **先肯定再切入** - 每轮开头认可一个具体优点
4. **引用教材示例** - 用教材中的具体案例帮助理解（但用人话表达，不标出处）
5. **问题要具体** - 可直接促使用户修改当前问题项

生成策略：
- 岗位价值问题：引导用户明确"为谁创造什么价值"
- 岗位效能问题：引导用户找到与价值对应的衡量指标
- 岗位任务问题：引导用户按"动词+修饰语+名词"格式命名
- 任务目的与成果问题：引导用户区分交付物和实际成果

仅输出 JSON：
{{"question":"..."}}"""

    if not _is_llm_enabled():
        return _generate_fallback_question(category, rubric_item_id)

    try:
        llm = _get_llm()
        if llm is None:
            return _generate_fallback_question(category, rubric_item_id)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    except Exception:
        return _generate_fallback_question(category, rubric_item_id)

    parsed = _safe_json_load(resp.content, {"question": _generate_fallback_question(category, rubric_item_id)})
    return parsed.get("question", _generate_fallback_question(category, rubric_item_id))


def _generate_fallback_question(category: str, rubric_item_id: int) -> str:
    """根据问题类别返回兜底引导问题"""
    if category == "岗位价值":
        if rubric_item_id in {1, 2}:
            return "这个价值描述是从客户视角出发的吗？你最直接服务的客户是谁，他能从你这里获得什么具体好处？"
        elif rubric_item_id == 3:
            return "不同客户的需求是否都一样？如果不一样，你觉得需要怎么区分？"
        else:
            return "如果只改这一处，你会先补哪一个可验证信息，能让评审一眼看到变化？"
    elif category == "岗位效能":
        return "对应这个岗位价值，客户视角下能观察到的结果是什么？怎么衡量？"
    elif category == "岗位任务":
        if rubric_item_id == 7:
            return "任务命名格式是：动词+修饰语+名词。比如'整理客户拜访纪要'——你现在的任务名能改成这种格式吗？"
        elif rubric_item_id == 8:
            return "这个名字说的是'想要什么结果'，而不是'要做什么动作'——能感觉到区别吗？"
        else:
            return "核心任务是否完整覆盖了岗位价值？还有哪个价值点没有对应到任务？"
    elif category == "任务目的与成果":
        if rubric_item_id == 9:
            return "如果有人问你这个任务到底'为啥要做'，你能一句话说清吗？"
        elif rubric_item_id == 10:
            return "你写的成果，真的能证明这个目的达到了吗？"
        elif rubric_item_id == 13:
            return "交了文档和达到目的，是一回事吗？"
        else:
            return "这个指标，换个人来量，能得出一样的数吗？"
    return "如果你现在就改这一处，你会先补充哪条最具体、可验证的信息？"


def _mock_review_result(row_data: dict[str, Any]) -> dict[str, Any]:
    """无外部模型时的轻量评审兜底，保证流程可运行。"""
    issues: list[dict[str, Any]] = []
    value_text = str(row_data.get("岗位价值", "") or row_data.get("岗位价值_", ""))
    eff_text = str(row_data.get("岗位效能", "") or row_data.get("岗位效能_", ""))

    if ("负责" in value_text) or (not value_text.strip()):
        issues.append(
            {
                "issue_id": "1",
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
                "issue_id": "5",
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
    """
    场景一固定四段式输出。
    遵循每轮对话结构：
    1. 【肯定】指出做得好的具体点
    2. 【切入】用自然的话点出问题
    3. 【引用教材示例】用人话表达，不标出处
    4. 【一问】抛出引导问题
    """
    related_fields = issue.get("related_fields", [])
    category = issue.get("category", "")
    user_facing_desc = issue.get("user_facing_desc", "当前条目存在需完善的问题。")

    # 提取关联内容展示
    related_lines: list[str] = []
    for key in related_fields:
        if key in row_data and row_data.get(key):
            related_lines.append(f"- {key}: {_format_value_for_view(row_data[key])}")
    if not related_lines:
        related_lines = ["- （当前问题项暂无明确关联字段，建议先补充关键描述）"]
    related_block = "\n".join(related_lines)

    # 根据问题类别生成肯定语和教材引用
    affirmation = _generate_affirmation(category, row_data)
    textbook_ref = _generate_textbook_reference(category, issue)

    return (
        f"---\n"
        f"📋 第 {idx+1}/{total} 条\n\n"
        f"【肯定】\n"
        f"{affirmation}\n\n"
        f"【关联内容】\n"
        f"{related_block}\n\n"
        f"【需要关注的点】\n"
        f"{user_facing_desc}\n\n"
        f"【教材参考】\n"
        f"{textbook_ref}\n\n"
        f"【引导性问题】\n"
        f"💬 {question}"
    )


def _generate_affirmation(category: str, row_data: dict[str, Any]) -> str:
    """生成肯定语，指出做得好的具体点"""
    if category == "岗位价值":
        # 检查是否有客户视角
        value_text = _extract_value_text(row_data)
        if any(k in value_text for k in WHO_HINTS):
            return "你已经明确了服务对象，这是很好的起点。"
        return "你开始思考岗位价值了，这个方向是对的。"
    elif category == "岗位效能":
        return "你有意识地为岗位价值寻找衡量方式，这是值得肯定的。"
    elif category == "岗位任务":
        return "你的任务命名已经有了动作感，下一步可以让它更具体。"
    elif category == "任务目的与成果":
        return "你开始思考任务的目的了，这比只列动作要深入。"
    return "你在这个条目上已经有了基础框架，我们继续打磨。"


def _generate_textbook_reference(category: str, issue: dict[str, Any]) -> str:
    """生成教材引用，用人话表达，不标出处"""
    rubric_item_id = issue.get("rubric_item_id", 0)

    if category == "岗位价值":
        if rubric_item_id in {1, 2}:
            return "岗位价值的本质是为谁创造价值 + 创造何种具体价值。测试工程师的例子值得参考：对产品经理的价值是'减少缺陷流出，避免口碑拖累市场推广'——先说服务对象，再说能让对方获得什么具体的商业结果。"
        elif rubric_item_id == 3:
            return "多个岗位混在一起时，需要思考每个岗位对这个岗位的需求是不是都一样。如果不一样，可能需要拆分成不同的客户群体。"
        else:
            return "岗位价值要避免空泛的描述，需要体现这个岗位的独一无二的客户期待。"
    elif category == "岗位效能":
        return "岗位效能是岗位价值的可视化脉络，必须与价值一一对应、同频共振。比如价值是'减少缺陷流出'，对应效能可以是'缺陷泄漏率≤0.5%'。"
    elif category == "岗位任务":
        if rubric_item_id == 7:
            return "任务命名格式建议用'动词+修饰语+名词'，比如'整理客户拜访纪要'——这样可以让大家直接理解'做什么'和'如何入手'。"
        elif rubric_item_id == 8:
            return "用目的命名任务（比如'降低产品成本'）只标出了终点，但未指引行动。任务名应该让人一看就知道要做什么动作。"
        else:
            return "核心任务是对岗位价值有直接和强力支撑的任务。"
    elif category == "任务目的与成果":
        if rubric_item_id == 9:
            return "任务目的要能说清'为啥要做'。客服的例子：从'5分钟内首次响应'校准为'一次性彻底解决用户疑问'后，效果会更清晰。"
        elif rubric_item_id == 13:
            return "交付物和实际成果不是一回事。高效委派要明确目的（解决什么问题）和成果（达到什么指标），而不是只说要交付什么文档。"
        else:
            return "成果要符合SMART原则，避免把交付物当成果。"
    return "可以参考教材中的相关案例来理解这个问题。"


def _advance_to_next_item(state: CoachState, feedback: str) -> dict:
    """当前条目问题全部解决后，切到下一条或完成态。"""
    idx = state["current_item_index"]
    rows = state["submission_rows"]
    next_idx = idx + 1
    is_last = next_idx >= len(rows)

    if is_last:
        closure_summary = "本轮已完成全部条目引导，后续进入答疑模式。"
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
            "stuck_counter": 0,
            "hint_level": 0,
            "closure_summary": closure_summary,
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
        "stuck_counter": 0,
        "hint_level": 0,
        "current_focus_id": None,
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
        "rubric_eval_summary": {
            "checked_item_ids": [item["id"] for item in RUBRIC_ITEMS],
            "matched_item_ids": [],
            "relaxed_item_ids": [],
            "coverage_ok": False,
        },
        "coaching_queue_order": [],
        "current_focus_id": None,
        "stuck_counter": 0,
        "hint_level": 0,
        "closure_summary": None,
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
        "stuck_counter": 0,
        "hint_level": 0,
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
    review_prompt = f"""请对以下岗标条目做内部评审（用户不可见）。

必须完整检查如下 14 项，不得自定义维度：
{_rubric_catalog_text()}

评审约束：
1. 逐条检查 14 项，并输出 checked_item_ids（必须覆盖 1-14）
2. 命中问题项输出 matched_issues（仅保留明确会扣分的硬伤）
3. 对于灰区问题，宁可放过，不进入 matched_issues
4. user_facing_desc 只能是自然语言，不要出现编号、章节号、分数

当前条目（第{idx+1}行）
表格列定义：{' | '.join(columns)}
{json.dumps(row_data, ensure_ascii=False, indent=2)}

仅输出 JSON：
{{
    "checked_item_ids": [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    "matched_issues": [
        {{
            "issue_id": "1-14",
            "issue_desc": "问题描述",
            "category": "岗位价值/岗位效能/岗位任务/任务目的与成果",
            "level": "A或B",
            "deduction": 10,
            "explanation": "命中证据",
            "user_facing_desc": "口语化问题描述，不含编号"
        }}
    ],
    "teaching_refs": ["内部参考"]
}}"""

    if _is_llm_enabled():
        try:
            llm = _get_llm()
            if llm is None:
                result = _mock_review_result(row_data)
            else:
                response = llm.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=review_prompt),
                ])
                result = _safe_json_load(
                    response.content,
                    {
                        "checked_item_ids": [item["id"] for item in RUBRIC_ITEMS],
                        "matched_issues": [],
                        "teaching_refs": ["请参考教材相关章节"],
                    },
                )
        except Exception:
            result = _mock_review_result(row_data)
    else:
        result = _mock_review_result(row_data)

    checked_item_ids = sorted({
        _coerce_rubric_item_id(v)
        for v in result.get("checked_item_ids", [item["id"] for item in RUBRIC_ITEMS])
        if _coerce_rubric_item_id(v) is not None
    })
    if len(checked_item_ids) != len(RUBRIC_ITEMS):
        checked_item_ids = [item["id"] for item in RUBRIC_ITEMS]

    matched_issues_raw = result.get("matched_issues", [])
    matched_issues, relaxed_item_ids = _normalize_matched_issues(matched_issues_raw, row_data)
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
                "rubric_item_id": int(item.get("rubric_item_id", 0) or 0),
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
            f"[{item.get('category', '')}] {item.get('issue_desc', '')} ({item.get('level', '')}级)"
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

    matched_item_ids = [int(item.get("rubric_item_id", 0) or 0) for item in matched_issues if int(item.get("rubric_item_id", 0) or 0) in RUBRIC_INDEX]
    coaching_queue_order = [issue.get("issue_id", "") for issue in issue_queue]

    return {
        "review_items": existing,
        "pending_question": first_question,
        "phase": "guiding",
        "active_mode": "proactive",
        "current_issue_index": 0,
        "issue_round": 0,
        "issue_status_map": issue_status_map,
        "reflection_round": 0,
        "rubric_eval_summary": {
            "checked_item_ids": checked_item_ids,
            "matched_item_ids": matched_item_ids,
            "relaxed_item_ids": relaxed_item_ids,
            "coverage_ok": len(checked_item_ids) == len(RUBRIC_ITEMS),
        },
        "coaching_queue_order": coaching_queue_order,
        "current_focus_id": coaching_queue_order[0] if coaching_queue_order else None,
        "stuck_counter": 0,
        "hint_level": 0,
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

    if _is_llm_enabled():
        try:
            llm = _get_llm()
            if llm is None:
                raise Exception("LLM not available")
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
# 遵循用户修改后的回判三步：1.肯定进步 2.给出判断 3.追或不追
# ─────────────────────────────────────────────────────────────
def process_response(state: CoachState) -> dict:
    """
    分析用户对引导问题的回复：
    - 理解不深：继续追问（最多3轮）
    - 理解到位 or 已达最大轮次：给予总结性反馈，移向下一条目

    回判三步：
    1. 肯定进步 - 必须指出当前版比上版好在哪个具体点
    2. 给出判断 - 用自然的话告诉用户这一项过了没
    3. 追或不追 - 跨过三问法合格线则放过，未跨过则再问
    """
    idx = state["current_item_index"]
    review_items = state.get("review_items", [])
    current_review = review_items[idx] if idx < len(review_items) else None
    messages = state.get("messages", [])
    issue_round = state.get("issue_round", 0)
    rows = state["submission_rows"]
    issue_idx = state.get("current_issue_index", 0)

    max_round = 3
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
    rubric_item_id = int(current_issue.get("rubric_item_id", 0) or 0)
    category = current_issue.get("category", "")

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

请按照以下三步进行判断：

**第一步：肯定进步**
指出用户回复比上一轮好在哪个具体点（不能只说"很好"）。

**第二步：给出判断**
- 是否解决：用户是否给出可执行、可落地的改进方向？
- 三问法检查（仅岗位价值类问题）：是否包含客户主语、具体好处、商业落点？

**第三步：追或不追**
- 问题解决或三问法合格 → 明确认可，跳下一项
- 未解决但方向对 → 更具体地再问一次
- 卡住超3轮 → 给候选项A/B/C，但仍以问题形式呈现

**重要原则**："还能更好"不是追问的理由。三问法过了就放过。

请仅输出 JSON：
{{
    "affirmation": "具体的肯定语...",
    "issue_resolved": true,
    "judgment": "自然语言判断...",
    "feedback_to_user": "...",
    "next_question": "..."
}}"""

    if _is_llm_enabled():
        try:
            llm = _get_llm()
            if llm is None:
                raise Exception("LLM not available")
            response = llm.invoke([
                SystemMessage(content=system),
                *recent,
                HumanMessage(content=decide_prompt),
            ])
            decision = _safe_json_load(
                response.content,
                {
                    "affirmation": "你这轮思考有进展。",
                    "issue_resolved": False,
                    "judgment": "方向对了，但还可以更具体。",
                    "feedback_to_user": "你的方向是对的，我们再把关键点说得更具体一些。",
                    "next_question": "如果你现在就改写这段内容，你会先增加哪一个可量化或可验证的信息？",
                },
            )
        except Exception:
            decision = _fallback_decision(last_user_msg, rubric_item_id)
    else:
        decision = _fallback_decision(last_user_msg, rubric_item_id)

    # 提取判断结果
    affirmation = decision.get("affirmation", "你这轮思考有进展。")
    feedback = decision.get("feedback_to_user", "谢谢你的思考。")
    issue_resolved = bool(decision.get("issue_resolved", False))
    next_question = decision.get("next_question", "你觉得还差哪一步可以让这个问题真正解决？")

    # 三问法检查（仅对岗位价值类问题）
    if not issue_resolved and rubric_item_id in {1, 2, 3, 4}:
        issue_resolved = _passes_three_questions(last_user_msg)
        if issue_resolved:
            feedback = f"{affirmation} 你已经补齐了服务对象、具体收益和业务落点，这一项可以先放过。"
        else:
            feedback = f"{affirmation} {feedback}"
    else:
        feedback = f"{affirmation} {feedback}"

    issue_status_map = dict(state.get("issue_status_map", {}))
    queue_copy = list(issue_queue)

    if issue_resolved:
        # 问题解决，进入下一个
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
                "current_focus_id": next_issue.get("issue_id"),
                "stuck_counter": 0,
                "hint_level": 0,
            }

        review_items[idx]["issue_queue"] = queue_copy
        return {
            **_advance_to_next_item(state, feedback),
            "review_items": review_items,
            "issue_status_map": issue_status_map,
            "last_user_intent": "reply",
        }

    # 未解决，继续追问
    review_items[idx]["issue_queue"] = queue_copy
    stuck_counter = state.get("stuck_counter", 0)
    hint_level = state.get("hint_level", 0)
    next_issue_round = issue_round + 1

    # 卡住超3轮，升级提示
    if next_issue_round >= max_round:
        stuck_counter += 1
        hint_level = min(hint_level + 1, 3)
        next_question = _build_escalation_question(current_issue, rows[idx], hint_level)
        next_issue_round = 0

    content = f"✅ {feedback}\n\n💬 {next_question}"
    return {
        "messages": [AIMessage(content=content)],
        "review_items": review_items,
        "pending_question": next_question,
        "issue_round": next_issue_round,
        "reflection_round": state.get("reflection_round", 0) + 1,
        "awaiting_user_input": True,
        "phase": "guiding",
        "active_mode": "proactive",
        "last_user_intent": "reply",
        "current_focus_id": current_issue.get("issue_id"),
        "stuck_counter": stuck_counter,
        "hint_level": hint_level,
    }


def _fallback_decision(user_msg: str, rubric_item_id: int) -> dict:
    """无LLM时的兜底判断逻辑"""
    has_action = any(k in user_msg for k in ["我会", "我改", "我补充", "改成", "增加"])
    has_metric = any(k in user_msg for k in ["可量化", "指标", "%", "率", "次", "个"])
    has_customer = any(k in user_msg for k in WHO_HINTS)

    # 三问法检查
    passes_three = _passes_three_questions(user_msg)

    if rubric_item_id in {1, 2, 3, 4}:
        # 岗位价值问题
        if passes_three:
            return {
                "affirmation": "你这轮给出了很具体的改进方向，",
                "issue_resolved": True,
                "judgment": "三问法通过，可以放过。",
                "feedback_to_user": "你已经明确了客户、好处和商业落点，这一项可以过了。",
                "next_question": "",
            }
        elif has_customer and has_action:
            return {
                "affirmation": "你开始从客户视角思考了，",
                "issue_resolved": False,
                "judgment": "还需要再具体一点。",
                "feedback_to_user": "方向对了，能不能再说说客户能获得什么具体好处？",
                "next_question": "如果只改一处，你会先补充哪条最具体的信息？",
            }
    elif rubric_item_id == 5:
        # 岗位效能问题
        if has_metric:
            return {
                "affirmation": "你找到了衡量方式，",
                "issue_resolved": True,
                "judgment": "效能指标已明确。",
                "feedback_to_user": "这个指标很具体，可以过了。",
                "next_question": "",
            }
    elif rubric_item_id in {7, 8}:
        # 任务命名问题
        if "动词" in user_msg or "名词" in user_msg or any(k in user_msg for k in ["负责", "完成", "制定", "整理"]):
            return {
                "affirmation": "你开始调整任务命名了，",
                "issue_resolved": True,
                "judgment": "符合动宾结构即可。",
                "feedback_to_user": "这个命名更清晰了，可以过了。",
                "next_question": "",
            }

    # 默认返回
    return {
        "affirmation": "你这轮思考有进展，",
        "issue_resolved": has_action and has_metric,
        "judgment": "方向对了" if has_action else "还可以更具体",
        "feedback_to_user": "你的思考方向不错。" if has_action else "能不能再说说你具体会怎么改？",
        "next_question": "如果你现在就改写这段内容，你会先增加哪一个可量化或可验证的信息？",
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
