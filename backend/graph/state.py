"""
岗标辅导 Graph State 定义
"""
from __future__ import annotations
from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


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

    # ── 当前轮次辅导上下文 ──────────────────────────────────────
    pending_question: Optional[str]    # 待发给用户的引导问题
    awaiting_user_input: bool          # 是否等待用户回复
    reflection_round: int              # 当前问题已追问轮次（最多 3 轮）
