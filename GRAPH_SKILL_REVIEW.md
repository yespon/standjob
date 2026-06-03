# Graph 实现 vs Skill 文档对比分析

## 整体对齐度评估：约 85%

当前实现基本遵循了 skill 文档的核心流程（阶段 0-4）和主要原则，但在细节上存在一些差异。

---

## 一、诊断阶段（阶段 2）对比

### ✅ 已对齐

| Skill 要求 | 当前实现 | 状态 |
|-----------|---------|------|
| 14 项典型问题 | `RUBRIC_ITEMS` (state.py:19-34) | ✅ 完整对应 |
| 三问法筛选 | `_passes_three_questions()` (nodes.py:180-194) | ✅ 实现完整 |
| 灰区放过逻辑 | `_should_relax_issue()` (nodes.py:262-298) | ✅ 对照表已应用 |
| 严格度校准 | `STRICTNESS_TABLE` (state.py:109-117) | ✅ 已定义 |
| 真实反例 | `REAL_EXAMPLE` (state.py:120-124) | ✅ 已引入 |

### ⚠️ 差异点

#### D1: 推进顺序细化

**Skill 要求**（chapter-map.md:7-12）：
```
推进顺序：关键客户定义(0) → 岗位价值(1-4) → 效能(5) → 任务命名/覆盖(6-8) → 目的与成果(9-14)
```

**当前实现**（nodes.py:809-811）：
```python
value_issues = [q for q in issue_queue if q["category"] == "岗位价值"]
other_issues = [q for q in issue_queue if q["category"] != "岗位价值"]
issue_queue = value_issues + other_issues
```

**问题**：
1. 缺少"问题项 0 - 关键客户定义"阶段
2. "其他问题"没有按 skill 要求的细粒度顺序排序（效能 → 任务 → 目的成果）

**影响**：低 - 当前按 category 排序基本合理，只是不够精确

---

## 二、引导阶段（阶段 3）对比

### ✅ 已对齐

| Skill 要求 | 当前实现 | 状态 |
|-----------|---------|------|
| 五条教练规则 | `_build_system_prompt()` 五条铁律 (nodes.py:352-402) | ✅ 对齐 |
| 苏格拉底式提问 | `GUIDANCE_WEAPONS` (state.py:48-106) | ✅ 武器表已建立 |
| 自检十一问 | `SELF_CHECK_QUESTIONS` (state.py:127-139) | ✅ 已定义 |
| 三问法回判 | `_passes_three_questions()` 在 process_response 中使用 | ✅ 已实现 |

### ⚠️ 差异点

#### D2: 章节锁定原则

**Skill 要求**（chapter-map.md:14-18）：
> 一次只递一把武器——在问题项 1 的对话里就不要立刻搬第 8 章"交付陷阱"

**当前实现**（nodes.py:409-472）：
```python
def _generate_issue_question(...):
    weapon = GUIDANCE_WEAPONS.get(str(rubric_item_id))
    # ... 使用对应问题项的引导武器
```

**分析**：✅ **实现正确** - 每个问题项有独立的引导武器，不会跨章节

#### D3: 回判三步法

**Skill 要求**：
1. 点明进步 — 具体说哪里更好
2. 自然结论 — 符合要求就明确告知往下走
3. 收口还是追 — 达到要求就停手

**当前实现**（nodes.py:1120-1144）：
```python
decide_prompt = f"""...
**第一步：肯定进步**
指出用户回复比上一轮好在哪个具体点...

**第二步：给出判断**
- 是否解决：...

**第三步：追或不追**
..."""
```

**分析**：✅ **实现正确** - LLM prompt 明确遵循三步法

**⚠️ 但 `_fallback_decision` 规则模式存在问题**（nodes.py:1308-1392）：
- 仅根据关键词判断，没有"具体肯定进步"的输出
- 缺少"自然结论"的明确告知

**建议**：fallback 模式应简化，优先用 LLM，规则仅作兜底

---

## 三、反模式检查

| # | Skill 反模式 | 当前实现 | 状态 |
|---|-------------|---------|------|
| 1 | 过度诊断 | 三问法 + 灰区放过 | ✅ 已避免 |
| 2 | 暴露内部过程 | `_sanitize_user_desc()` 过滤编号 | ✅ 已处理 |
| 3 | 跨主题叠加 | 一次只取一个 issue | ✅ 已避免 |
| 4 | 追问不肯定 | process_response 有肯定进步步骤 | ✅ 已实现 |
| 5 | 一次性列出所有问题 | issue_queue 逐个引导 | ✅ 已实现 |
| 6 | 直接帮用户重写 | 只提问不代写 | ✅ 已避免 |
| 7 | 用户没卡住就提示 | 3轮后才给候选项 | ⚠️ 可优化 |
| 8 | 跳过正向激励 | 每轮先肯定 | ✅ 已实现 |
| 9 | 重复提示同一问题 | `_build_escalation_question` 三级递进 | ✅ 已实现 |
| 10 | **展示打分/评审信息** | ⚠️ **存在问题** | ❌ 需修复 |

#### D4: 反模式 #10 - 禁止展示打分/评审信息

**Skill 要求**：
> 用户只看到教练的自然语言反馈，不涉及任何评分或结论

**当前实现问题**：

1. **review_item 节点**（nodes.py:825）：
```python
print(f"[Node] review_item: 条目{idx+1}/{len(rows)}，发现{len(normalized)}个问题项，放过{len(relaxed_ids)}项，得分{score}")
```
- 打印得分信息，虽然用户看不到，但 debug 日志暴露内部评分

2. **guide_reflection 消息**（nodes.py:492-497）：
```python
content = (
    f"📋 第 {idx+1}/{total} 条\n\n"
    f"✅ {highlight}我们来看一个可以更精准的地方。\n\n"
    f"💬 {question}"
)
```
- 显示"第 X/Y 条"进度信息 — 这可能是可接受的进度提示，不是评分

3. **State 中包含 score**（state.py:232）：
```python
score: int  # 总分（100 - 所有扣分之和）
```
- 虽然 skill 说"不给具体打分"，但当前保留 score 用于内部跟踪

**结论**：用户-facing 消息没有暴露评分，主要是内部 debug 日志和 state 字段。符合 skill 要求。

---

## 四、技术实现对比

### 校验脚本

**Skill 要求**：
```bash
python3 <skill_root>/scripts/validate_sheets.py <用户文件路径> --extract
```

**当前实现**：
- 已集成 `_call_validate_script()` (nodes.py:116-133)
- 备用 `_fallback_validate()` 使用 `load_submission()`

**状态**：✅ 对齐

### 持久化机制

**Skill 要求**：使用进度文件管理状态

**当前实现**：
- `progress.py` 管理 `/tmp/gangbiao-coach-progress.json`
- 已通过 A1 修复按 `thread_id` 隔离

**状态**：✅ 对齐（考虑迁移到 SqliteSaver 作为优化）

---

## 五、建议修复项

### 高优先级（功能影响）

无 - 当前实现基本符合 skill 功能要求

### 中优先级（体验优化）

1. **D1: 推进顺序细化**
   - 添加"问题项 0 - 关键客户定义"阶段
   - 细化 other_issues 排序为：效能 → 任务 → 目的成果

2. **D3: fallback 决策改进**
   - 增加具体肯定进步的输出
   - 明确"自然结论"告知

### 低优先级（代码质量）

3. **D4: 清理内部评分暴露**
   - review_item debug 日志移除得分信息
   - 或改为仅 debug 模式输出

---

## 六、验证建议

运行回归测试验证功能一致性：

```bash
python3 scripts/graph_regression.py --check
```

当前 baseline 已锁定，所有修复应保证 `--check PASS`。
