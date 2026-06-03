"""
岗标辅导 Graph State 定义

完全对应 gangbiao-coach skill v2 的阶段 0-4 生命周期：
  阶段 0 (file_collection)  → 文件采集
  阶段 1 (structure_check)  → 结构校验 + 内容提取
  阶段 2 (full_review)      → 全面评审（后台，不暴露给用户）
  阶段 3 (step_guidance)    → 分步引导（核心环节）
  阶段 4 (closure)          → 收尾总结
"""
from __future__ import annotations
from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ─────────────────────────────────────────────────────────────
# 评审标准项定义（14 项评分标准 + 问题项 0 客户定义，来自 rubric.md）
# ─────────────────────────────────────────────────────────────
RUBRIC_ITEMS: list[dict[str, Any]] = [
    # 问题项 0：关键客户定义（优先于岗位价值，来自 chapter-map.md）
    {"id": 0, "category": "客户定义", "level": "A", "desc": "关键客户画像不清晰，无法明确服务对象的特征与需求"},
    # 问题项 1-4：岗位价值
    {"id": 1, "category": "岗位价值", "level": "A", "desc": "站在自身视角而不是站在客户视角提炼岗位价值"},
    {"id": 2, "category": "岗位价值", "level": "A", "desc": "岗位价值并非源于对客户最在意、最深层需求的提炼"},
    {"id": 3, "category": "岗位价值", "level": "A", "desc": "没有从不同客户的视角出发，去提炼岗位价值"},
    {"id": 4, "category": "岗位价值", "level": "A", "desc": "岗位价值描述空泛，指导性不强"},
    # 问题项 5：岗位效能
    {"id": 5, "category": "岗位效能", "level": "A", "desc": "岗位效能并不能直接、有效地衡量岗位价值"},
    # 问题项 6-8：岗位任务
    {"id": 6, "category": "岗位任务", "level": "A", "desc": "核心任务的目的集合无法完整覆盖岗位价值"},
    {"id": 7, "category": "岗位任务", "level": "B", "desc": "任务命名未按动词+修饰语+名词"},
    {"id": 8, "category": "岗位任务", "level": "B", "desc": "直接用目的命名任务"},
    # 问题项 9-14：任务目的与成果
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
BIZ_HINTS = ("收入", "成本", "风险", "品牌", "口碑", "市场", "利润", "效率", "竞争力")


# ─────────────────────────────────────────────────────────────
# 引导武器表（按问题类型对应的苏格拉底式提问风格）
# 来自 textbook.md 案例 + 自检十一问，用人话表达
# ─────────────────────────────────────────────────────────────
GUIDANCE_WEAPONS: dict[str, dict[str, str]] = {
    # 问题项 0：关键客户定义
    "0": {
        "question_style": "先别急着写价值——我问你个实在的：你这岗位最主要是伺候谁？用一句话说说这人的角色、他每天烦什么，行吗？",
        "textbook_ref": "第1-2章：锁定关键客户的三步决策——明确客户画像、价值排序、精准洞察",
    },
    # 问题项 1：视角偏差
    "1": {
        "question_style": "你看啊，你写的这段更像是在说'我干了啥'。但如果我是你客户，我读完后能直接看到'我得到了啥'吗？",
        "textbook_ref": "第3章：视角偏差陷阱——从'我能做什么'转向'你能得到什么'",
    },
    # 问题项 2：需求偏差
    "2": {
        "question_style": "客户跟你说'我要这个'，你确定这就是他真正缺的？还是他自己开的药方？比如他说要培训，真问题会不会是新人开不了单？",
        "textbook_ref": "第3章：深度偏差——错把客户药方当问题症结",
    },
    # 问题项 3：价值粗放
    "3": {
        "question_style": "你列了好几个客户，但给每个人的价值听起来都差不多。试着给每个客户分别说一句——如果听起来像一个模子刻的，可能还没区分到位。",
        "textbook_ref": "第3章：多客户岗位必须分别定义价值",
    },
    # 问题项 4：描述空泛
    "4": {
        "question_style": "'助力业务成功'这种话听起来挺对，但具体到你是谁、帮谁、帮成什么样，就有点虚了。你能把它换成一句更实在的吗？",
        "textbook_ref": "第3章：标准表述——为谁创造价值 + 创造何种具体价值",
    },
    # 问题项 5：效能脱节
    "5": {
        "question_style": "你写的这些指标，跟你上面说的价值能对上号吗？比如你说减少缺陷，那指标里有缺陷率吗？咱们一过一过？",
        "textbook_ref": "第4章：效能必须与价值一一对应、同频共振",
    },
    # 问题项 6：价值任务脱钩
    "6": {
        "question_style": "来，咱们玩个双向对账：你列的每项价值，背后都有任务撑着吗？反过来，每项任务又具体撑了哪项价值？",
        "textbook_ref": "第5章：核心任务的目的集合必须完整覆盖岗位价值",
    },
    # 问题项 7：命名格式
    "7": {
        "question_style": "你这个任务名——'详细设计'、'需求评审'——新人看了能直接知道要干啥吗？还是得像猜谜一样琢磨？",
        "textbook_ref": "第5章：动词+修饰语+名词——如'整理客户拜访纪要'",
    },
    # 问题项 8：用目的命名
    "8": {
        "question_style": "'降低产品成本'——这是你想要的结果，不是你能直接动手做的动作。如果把任务名改成'一看就知道要干啥'，你会怎么写？",
        "textbook_ref": "第5章：终结习惯性执行——停摆推演法",
    },
    # 问题项 9：目的模糊
    "9": {
        "question_style": "如果有人问你'做这个任务到底图啥'，你能一句话说清吗？不说'完成任务'这种虚的。",
        "textbook_ref": "第5章/第9章自检六：目的必须包含'旨在…''为了…'",
    },
    # 问题项 10：成果脱节
    "10": {
        "question_style": "你写的成果，能直接证明目的达到了吗？比如目的是'一次性解决'，成果却是'响应时效'，这俩就对不上。",
        "textbook_ref": "第7章：先定目的，再定成果——成果必须回应目的关键词",
    },
    # 问题项 11：反馈迟滞
    "11": {
        "question_style": "你这个成果，得等多久才能知道有没有问题？能不能做到当天或者当周就发现问题？",
        "textbook_ref": "第9章自检十一：高频任务必须配短周期评估",
    },
    # 问题项 12：不符合 SMART
    "12": {
        "question_style": "这个指标换个人来量，能得出一样的数吗？比如'质量好'就挺主观，但'缺陷率≤0.5%'谁量都一样。",
        "textbook_ref": "第7章：成果标准需具体、可衡量、可达成、相关、有时限",
    },
    # 问题项 13：交付物当成果
    "13": {
        "question_style": "'交付XX文档'——交了文档不等于目的达到了。客户真正要的是文档里的东西解决了他的问题，对吧？",
        "textbook_ref": "第8章：交付陷阱——从'提交文件'到'实现目的'",
    },
    # 问题项 14：无挑战目标
    "14": {
        "question_style": "完成度、交期、预算这三块，你设的目标够有挑战性吗？还是保守到肯定能完成？",
        "textbook_ref": "第7章：三维坐标（完成度/交期/预算）均应设计为挑战目标",
    },
}

# 严格度校准对照表（来自 skill v2）
STRICTNESS_TABLE: dict[str, dict[str, str]] = {
    "视角偏差": {"扣分": "全文无客户收获", "放过": "动作偏多但有客户视角"},
    "描述空泛": {"扣分": "'助力业务成功'纯口号", "放过": "有具体名词，动词偏多"},
    "效能脱节": {"扣分": "效能完全无对应价值", "放过": "大部分能对上"},
    "命名格式": {"扣分": "'详细设计''减少故障影响'", "放过": "偶有缺失但仍是动宾"},
    "用目的命名": {"扣分": "'减少故障影响'", "放过": "带一点目的但仍可执行"},
    "成果脱节": {"扣分": "成果完全没回应目的", "放过": "主要目的有支撑"},
    "交付物当成果": {"扣分": "'交付XX文档'唯一成果", "放过": "既有交付物也有效果指标"},
}

# 真实反例（来自 skill v2，用于三问法校准）
REAL_EXAMPLE = (
    "充分理解用户和产品需求，设计并实现易用、体验好、高性价比解法，"
    "按时保质提供软件产品，帮助事业部提升市场份额和口碑。"
    "\n动作偏多，但包含客户、收获、商业结果——评委不会扣分。绝对不要拉着用户死磕。"
)

# 自检十一问（来自 textbook.md 第9章）
SELF_CHECK_QUESTIONS: list[dict[str, str]] = [
    {"id": "一", "mistake": "价值错位", "check": "因为本岗位解决了（某客户）在（某场景）下关于（某问题）的难题，所以不可或缺。"},
    {"id": "二", "mistake": "需求偏差", "check": "当客户提出（某药方）时，我诊断出其试图攻克的真实堡垒是（某业务症结）。"},
    {"id": "三", "mistake": "价值粗放", "check": "列出所有关键客户，分别一句话陈述核心价值；若听起来大同小异，则未完成细分。"},
    {"id": "四", "mistake": "价值任务脱钩", "check": "双向验证：①每项价值→必须打赢的关键战役有哪些？②每项任务→直接支撑了哪一个价值？"},
    {"id": "五", "mistake": "任务虚化", "check": "每个任务名称是否遵循'动词+修饰语+对象'？员工看名称能否直接知道做什么？"},
    {"id": "六", "mistake": "目的错位", "check": "目的描述是否包含'旨在…''为了…'等明确目的性表述？"},
    {"id": "七", "mistake": "目的虚焦", "check": "目的是否与岗位价值表述过于相似？'停摆推演法'能否推出独特贡献？"},
    {"id": "八", "mistake": "目的偏移", "check": "①归因性检验：主要由本任务贡献？②验证性检验：能否在任务周期内被直接验证？"},
    {"id": "九", "mistake": "目的悬空", "check": "'要证明这个目的达到，需要拿出哪些可验证证据（即成果标准）？'"},
    {"id": "十", "mistake": "评价异化", "check": "①成果标准是否主要依据客观数据（≥70%）？②若采用评审，标准是否透明、前置？"},
    {"id": "十一", "mistake": "反馈迟滞", "check": "①反馈即时性：数据能否在当天/当周被捕获？②异常可见性：能否实时识别偏差？"},
]


class IssueItem(TypedDict):
    """评审发现的单个问题项（内部结构）"""
    issue_id: str
    issue_desc: str
    category: str
    level: str              # A=重点关注，B=可优化（仅内部排序，不给用户展示）
    explanation: str
    teaching_ref: str
    guided: bool


class MatchedIssue(TypedDict):
    """review_item 中单条命中问题项结构。"""
    issue_id: str
    issue_desc: str
    category: str
    level: str              # A=重点关注，B=可优化
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


class ProgressSnapshot(TypedDict):
    """进度持久化快照，用于恢复会话。"""
    file_path: str
    current_item: int
    items: dict[str, Any]       # item_id → {"status": ..., "evidence": ...}
    history: list[dict[str, Any]]


class CoachState(TypedDict):
    """LangGraph 全局状态"""

    # ── 对话消息（支持追加） ──────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 预加载的评审标准（启动时一次性写入，后续只读） ─────────────
    scoring_criteria: Optional[str]    # 评分标准文本（来自 rubric.md）
    teaching_material: Optional[str]   # 教材文本（来自 textbook.md）
    phase: str                         # init | loaded | validating | reviewing | guiding | done | closure
    active_mode: str                   # proactive | reactive_qa
    last_user_intent: str              # reply | question

    # ── 会话初始化 & 进度恢复 ──────────────────────────────────────
    has_saved_progress: bool           # 是否存在已保存的进度
    progress_snapshot: Optional[ProgressSnapshot]  # 进度快照
    resume_confirmed: bool             # 用户确认恢复进度

    # ── 用户提交的表格 ──────────────────────────────────────────
    submission_path: Optional[str]     # 用户上传文件路径
    submission_text: Optional[str]     # 表格转换后的文本内容
    submission_columns: list[str]      # 表格列定义
    submission_rows: list[dict[str, Any]]  # 解析后的表格数据
    structure_valid: Optional[bool]    # 结构校验是否通过
    structure_errors: list[str]        # 结构校验错误信息

    # ── 评审结果 ────────────────────────────────────────────────
    all_issues: list[IssueItem]        # 一次性评审发现的所有问题（内部诊断用，不给用户展示）
    # 注意：skill 要求不给用户打分，所以没有 score 字段
    current_issue_index: int           # 当前问题项索引
    issue_round: int                   # 当前问题项追问轮次
    issue_status_map: dict[str, str]   # issue_id -> pending/in_progress/resolved
    review_items: list[ReviewItem]     # 按条目生成的评审结果
    current_item_index: int            # 当前正在引导的条目索引
    rubric_eval_summary: RubricEvalSummary  # 评分标准覆盖情况追踪（内部用）
    coaching_queue_order: list[str]    # 辅导问题队列顺序
    current_focus_id: Optional[str]    # 当前聚焦的问题项ID
    stuck_counter: int                 # 用户卡住计数器（用于升级提示）
    hint_level: int                    # 当前提示级别（1-3）

    # ── 双线推进状态（skill 要求：线1完成后才进入线2）──────────────────
    current_line: int                  # 1 = 线1(客户-价值-效能), 2 = 线2(任务-目的-成果)
    line1_completed: bool              # 线1是否全部完成

    # ── 当前轮次辅导上下文 ──────────────────────────────────────
    pending_question: Optional[str]    # 待发给用户的引导问题
    awaiting_user_input: bool          # 是否等待用户回复
    # issue_round 追踪当前问题已追问轮次（最多 3 轮）

    # ── 收尾 ──────────────────────────────────────────────────
    closure_summary: Optional[str]     # 收尾总结
    highlights: list[str]              # 改得好的地方
    remaining_polish: list[str]        # 还可以继续打磨的地方
