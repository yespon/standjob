"""
岗标辅导 Graph State 定义
"""
from __future__ import annotations
from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ─────────────────────────────────────────────────────────────
# 评审标准项定义（14项评分标准）
# ─────────────────────────────────────────────────────────────
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

# 三问法关键词（用于严格度校准）
WHO_HINTS = ("客户", "用户", "产品经理", "研发", "业务", "销售", "团队", "测试", "运营")
BENEFIT_HINTS = ("减少", "提升", "降低", "提高", "保障", "避免", "缩短", "稳定", "改善")
BIZ_HINTS = ("收入", "成本", "风险", "品牌", "口碑", "市场", "利润", "效率")


class IssueItem(TypedDict):
    """评审发现的单个问题项（内部结构）"""
    issue_id: str
    issue_desc: str
    category: str
    level: str
    deduction: int
    explanation: str
    teaching_ref: str
    guided: bool


class MatchedIssue(TypedDict):
    """review_item 中单条命中问题项结构。"""
    issue_id: str
    issue_desc: str
    category: str
    level: str
    deduction: int
    explanation: str


class IssueProgress(TypedDict):
    """问题项逐项引导进度。"""
    issue_id: str
    rubric_item_id: int
    issue_desc: str
    category: str
    status: str              # pending | in_progress | resolved
    related_fields: list[str]
    user_facing_desc: str


class ReviewItem(TypedDict):
    """单个提交条目的评审结果"""
    row_index: int
    row_data: dict[str, Any]
    score: int
    dimension_scores: dict[str, Any]  # 包含 matched_issues
    issues: list[str]
    suggestions: list[str]
    standard_ref: str
    status: str
    issue_queue: list[IssueProgress]


class RubricEvalSummary(TypedDict):
    """单条目评审覆盖情况（内部追踪，不对用户展示）。"""
    checked_item_ids: list[int]      # 已检查的评分项ID列表
    matched_item_ids: list[int]      # 命中问题的评分项ID列表
    relaxed_item_ids: list[int]      # 灰区放过的评分项ID列表
    coverage_ok: bool                # 是否完整覆盖所有评分项


class CoachState(TypedDict):
    """LangGraph 全局状态"""

    # ── 对话消息（支持追加） ──────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 预加载的评审标准（启动时一次性写入，后续只读） ─────────────
    scoring_criteria: Optional[str]    # 评分标准.xlsx 的文本摘要
    teaching_material: Optional[str]   # 教材 PDF 的文本摘要
    phase: str                         # init | loaded | reviewing | guiding | done
    active_mode: str                   # proactive | reactive_qa
    last_user_intent: str              # reply | question

    # ── 用户提交的表格 ──────────────────────────────────────────
    submission_path: Optional[str]     # 用户上传文件路径
    submission_text: Optional[str]     # 表格转换后的文本内容
    submission_columns: list[str]      # 表格列定义
    submission_rows: list[dict[str, Any]]  # 解析后的表格数据

    # ── 评审结果 ────────────────────────────────────────────────
    all_issues: list[IssueItem]        # 一次性评审发现的所有问题
    score: int                         # 总分（100 - 所有扣分之和）
    current_issue_index: int           # 兼容旧版流程的当前问题索引
    issue_round: int                   # 当前问题项追问轮次
    issue_status_map: dict[str, str]   # issue_id -> pending/in_progress/resolved
    review_items: list[ReviewItem]     # 当前实现按条目生成的评审结果
    current_item_index: int            # 当前正在引导的条目索引
    rubric_eval_summary: RubricEvalSummary  # 评分标准覆盖情况追踪
    coaching_queue_order: list[str]    # 辅导问题队列顺序
    current_focus_id: Optional[str]    # 当前聚焦的问题项ID
    stuck_counter: int                 # 用户卡住计数器（用于升级提示）
    hint_level: int                    # 当前提示级别（1-3）
    closure_summary: Optional[str]     # 收尾总结

    # ── 当前轮次辅导上下文 ──────────────────────────────────────
    pending_question: Optional[str]    # 待发给用户的引导问题
    awaiting_user_input: bool          # 是否等待用户回复
    reflection_round: int              # 当前问题已追问轮次（最多 3 轮）
