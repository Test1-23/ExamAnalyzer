# Exam Knowledge Point Analyzer

从配对试卷 PDF（Question Paper + Mark Scheme）中提取知识点，构建自组织知识图谱，提供对话式 AI 学习助手。

> **系统仅支持文本层 PDF。** 扫描件、图片型 PDF、内嵌图片文字、图表均无法处理。如果你的 PDF 是扫描版，需先用 OCR 工具转为可搜索 PDF 后再使用。

## 介绍

本工具自动解析试卷 PDF，配对题目与答案，利用 LLM 提取知识点并构建行为驱动的自组织知识图谱。内置对话助手支持多轮问答和学习路径推荐。

核心流程：PDF 提取 → QA 配对 → 知识库积累 → MS 得分点提取 → 行为驱动的 Topic 自组织 → 自动 KP 生成 → 离线分析 → 持续进化。

## 环境配置

- Python 3.10+
- DeepSeek API 访问权限

```bash
pip install -r requirements.txt
```

API 可通过三种方式配置（优先级从高到低）：

```bash
# 方式一：环境变量
export DEEPSEEK_API_URL="https://api.deepseek.com/v1"
export DEEPSEEK_API_KEY="sk-..."

# 方式二：Web 界面（启动后在浏览器中填写）
# 打开 http://127.0.0.1:5000 → 在页面顶部填写 API URL 和 Key → 保存

# 方式三：配置文件 exam_analyzer/config.json
```
```json
{
  "api_url": "https://api.deepseek.com/v1",
  "api_key": "sk-..."
}
```

## 文件命名规则

试卷 PDF 放入 `input/` 目录，文件名需遵循以下格式以便系统自动配对：

```
{subject}_{season}{year}_{type}_{variant}.pdf
```

- **subject**: 科目编号（如 `1145`）
- **season**: 考试季字母（`s` = 夏季, `w` = 冬季, `m` = 春季）
- **year**: 两位年份（如 `24` = 2024）
- **type**: 文件类型（`qp` = Question Paper, `ms` = Mark Scheme）
- **variant**: 试卷变体编号

**配对示例**：同一份试卷的题目和答案需使用相同的前缀和变体编号：

```
0721_s24_qp_1.pdf  +  0721_s24_ms_1.pdf   →  配对为一组
0721_s24_qp_2.pdf  +  0721_s24_ms_2.pdf   →  配对为另一组
0721_w24_qp_1.pdf  +  0721_w24_ms_1.pdf   →  配对
```

## 运行分析

**方式一：Web 界面（推荐）**

```bash
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

在网页顶部填写 API URL 和 Key（自动保存，下次无需重填）→ 点击「开始分析」即可自动运行全流程。页面实时显示进度、日志和输出结果。分析完成后自动加载知识库，可立即开始对话问答。

**方式二：命令行**

```bash
cd exam_analyzer
python main.py
```

系统按年份升序处理所有配对的 PDF，最早的一对初始化知识库，后续逐步扩展。分析完成后启动 `python app.py` 即可使用对话助手。

### 测试

```bash
python tests/test_suite.py          # 交互选择模式
python tests/test_suite.py check    # 即时验证现有 DB
python tests/test_suite.py light    # 轻量测试
python tests/test_suite.py full     # 全量测试
python tests/test_suite.py chat     # 聊天端点测试
```

报告输出到 `TestReport/`。

## 技术栈与架构

### 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek Flash（全流程，纯 Flash 无 Pro 依赖） |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| 数据库 | SQLite WAL 模式，版本化迁移框架，BaseStore 抽象层 |
| PDF 提取 | pdfplumber (主) + PyMuPDF (fallback)，仅文本层 |
| Web | Flask + Jinja2 |
| 事件总线 | Node.js + WebSocket（跨模块解耦，manifest 合约） |
| 向量编码器 | 可插拔接口（EmbeddingEncoder 当前，SaeEncoder 预留） |

### 目录结构

```
exam_analyzer/
├── main.py                    # CLI 入口
├── app.py                     # Flask Web 入口
├── src/
│   ├── db/                    # 数据库层 (Mygo refactor)
│   │   ├── connection.py      #   ConnectionMgr: 连接池 + WAL + 迁移 + transaction()
│   │   ├── query_builder.py   #   QueryBuilder: %-格式化 SQL 构建器
│   │   ├── dialect.py         #   SqlDialect ABC + SqliteDialect (未来 DB 可移植)
│   │   └── repository.py      #   BaseRepository: 泛型 CRUD 基类
│   ├── schema/                # 33 表 DDL 按域拆分 (Mygo refactor)
│   │   ├── _tables_core.py    #   qa_pairs, exam_sessions, chat_history, checkpoints
│   │   ├── _tables_kp.py      #   knowledge_points, kp_edges, qa_kp_membership, vectors
│   │   ├── _tables_topic.py   #   dynamic_topics, topic_links, dependencies, difficulty
│   │   ├── _tables_fragment.py#   ms_fragments, help_map, membership, centrality
│   │   ├── _tables_student.py #   student_memory, knowledge_state, confusions, trajectory
│   │   ├── _tables_analysis.py#   api_call_log, feedback, trends, signatures, cache...
│   │   ├── _indexes.py        #   4 个索引定义
│   │   └── _migrations.py     #   3 个版本化迁移
│   ├── stores/                # 8 个领域 Store (继承 BaseStore)
│   │   ├── base.py            #   BaseStore: _write / _read_one / _read_all / _write_locked
│   │   ├── qa_store.py        #   qa_pairs (delete → insert → rename)
│   │   ├── topic_store.py     #   dynamic_topics + links + difficulty
│   │   ├── kp_store.py        #   knowledge_points + edges + membership
│   │   ├── fragment_store.py  #   ms_fragments + help_map + centrality
│   │   ├── chat_store.py      #   chat_history
│   │   ├── student_store.py   #   student memory/state/confusions/trajectory
│   │   ├── analysis_store.py  #   api_call_log + feedback + cache + deps + 诊断查询
│   │   └── vector_store.py    #   kp_vectors + qa_kp_scores + topic_vectors
│   ├── knowledge/             # 知识点子系统 (Mygo 新建)
│   │   ├── _encoding.py       #   VectorEncoder ABC + EmbeddingEncoder
│   │   ├── _clustering.py     #   EmergentClusterer: 涌现聚类, 自适应阈值
│   │   └── system.py          #   KpSystem: 统一入口 facade
│   ├── analysis/              # 后处理分析 (Mygo refactor: diagnostics + offline 合并)
│   │   ├── _cascade.py        #   Topic 分裂/合并 + 向量级联
│   │   ├── _cross_paper.py    #   论文签名 + 基线 + 异常检测
│   │   ├── _migration.py      #   Fragment 迁移 + Topic 统计
│   │   ├── _pitfalls.py       #   陷阱发现 + 考试趋势
│   │   ├── _student.py        #   学生反馈闭环
│   │   ├── _verbs.py          #   命令动词提取
│   │   ├── _difficulty.py     #   难度评估
│   │   ├── _dependencies.py   #   主题依赖发现
│   │   └── _report.py         #   报告生成 + run_offline_analysis()
│   ├── events/                # Node.js 事件总线客户端 (Mygo 新建)
│   │   ├── events.py          #   EventType 常量 (28 种事件)
│   │   ├── bus_client.py      #   EventBusClient: HTTP 发布, fire-and-forget
│   │   └── manifest.py        #   ModuleManifest: 5 模块事件合约
│   ├── web/                   # Flask Blueprint Web 层
│   │   ├── state.py           #   全局共享状态 + 线程安全访问器
│   │   ├── app_factory.py     #   create_app() 工厂
│   │   └── routes_*.py        #   4 Blueprint, 21 路由
│   ├── pipeline/              # 流水线编排 (P4 拆分)
│   │   ├── orchestrator.py    #   run_pipeline(): 支持 emergent 聚类路径
│   │   ├── context.py         #   PipelineContext
│   │   ├── counters.py        #   StageCounters
│   │   ├── result.py          #   PipelineResult
│   │   └── tracker.py         #   ProgressTracker
│   ├── chat/                  # 5-agent 聊天管线
│   │   ├── agents.py          #   Query Analyst → Answer Gen → Critic → Suggest
│   │   └── context.py         #   KP 缓存 + 分析上下文
│   ├── prompts/               # Prompt 模板
│   │   ├── prompt_factory.py  #   PromptType 枚举 + PromptBuilder (17 种 prompt)
│   │   └── pipeline_prompts.py#   QA 配对 + 摘要 + 答题 + 评分 prompt
│   ├── knowledge_graph.py     # [遗留] QA 聚类 → KP 生成 → 边融合
│   ├── adversarial_refiner.py # [遗留] KP 拆分/合并 (待迁移至 knowledge/)
│   ├── distiller.py           # [遗留] 蒸馏器 (待迁移至 knowledge/)
│   ├── evolution.py           # [遗留] 自进化循环 (待迁移至 knowledge/)
│   ├── topic_merger.py        # [遗留] 主题合并 (待迁移至 knowledge/)
│   ├── retriever.py           # [遗留] QARetriever (待迁移至 knowledge/)
│   └── [其他模块]             # deepseek_client, embedding_cluster, models, utils 等
├── eval/                      # 质量评估
│   └── feedback_agent.py      # 6 维度聊天质量评估
├── tests/                     # 自动化测试 (77+ 单元测试)
│   ├── conftest.py            # Mock 基础设施
│   ├── test_store_crud.py     # Store CRUD 测试
│   ├── test_pure_functions.py # 纯函数测试
│   └── test_*.py              # 单元 + mock + 检索测试
├── input/                     # 试卷 PDF (gitignored)
├── intermediate/              # SQLite DB (gitignored)
└── static/ + templates/       # Web 前端 (Flask + Jinja2)

event_bus/                     # Node.js 事件总线 (Mygo 新建)
└── server.js                  # 主总线 (:3030) + 5 子总线 (:3031-:3035)
```

### 流水线架构

```
论文 PDF 对 (N 篇，按年份升序)
  │
  ├─ PDF 提取 (pdfplumber → PyMuPDF fallback)
  ├─ QA 配对 (Flash: 全文 QP+MS → 匹配题目与答案)
  │
  ├─ 逐 QA 处理 ────────────────────────────────────
  │   传统路径 (use_emergent=False):
  │     Phase 1 (首篇): Flash summary + topic → 直接入库
  │     Phase 2 (后续): summary → 检索 → 答题 → MS 评分 → 入库
  │   涌现聚类路径 (use_emergent=True, Mygo-003):
  │     所有论文统一处理: encode → EmergentClusterer.assign_qa() → 入库
  │     无 Phase 1/2 分叉, 自适应阈值, 胚胎 KP 可见+衰减
  │
  ├─ 后处理 ────────────────────────────────────────
  │   Topic Merge → Distillation → Review → points.txt
  │   Knowledge Graph (聚类 → KP 节点 → 4 种边 → 融合)
  │   KP 结构精炼 (auto_split / auto_merge, 行为驱动)
  │   Offline Analysis (动词规律 + 难度评估 + 依赖发现)
  │   Pipeline Diagnostics (闭环 + 跨论文一致性)
  │
  └─ 自进化循环 ────────────────────────────────────
      Fragment 迁移: 行为数据驱动的 Topic 边界重塑
      Topic 演化: 分裂 / 合并 / 消解
      自动 KP 生成: 稳定 Topic → Flash 命名 + 解释
      学生反馈闭环: 困惑 → 难度调整
```

### 事件总线架构 (Mygo-000)

```
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────┐  ┌─────┐
│ pipeline │  │ knowledge │  │ analysis │  │ chat │  │ web │
└────┬─────┘  └─────┬─────┘  └────┬─────┘  └──┬───┘  └──┬──┘
     │              │              │           │          │
     └──────────────┴──────────────┴───────────┴──────────┘
                            │
                   Node.js 事件总线
             主总线 (:3030) + 5 子总线 (:3031-:3035)
             28 种 KP 生命周期事件, 显式 manifest 合约
```

### 知识点系统 (Mygo-001/002)

```
VectorEncoder (可插拔接口)
  ├─ EmbeddingEncoder (当前: sentence-transformer, 384-dim)
  └─ SaeEncoder (预留: 稀疏自编码器, 独立仓库训练)

EmergentClusterer
  ├─ 涌现聚类: 无预设 K, 无固定阈值
  ├─ 每 KP 独立自适阈值 (median - MAD)
  ├─ 胚胎 KP 始终可见 + 衰减机制
  └─ 三信号质心调整 (help_feedback / post_process / user_input)

KpSystem (统一入口)
  ├─ ingest_qa(): QA → KP 分配
  ├─ apply_feedback_signal(): 帮助反馈 → 向量调整
  └─ on_paper_completed(): 衰减检查 + 质量晋升 + 事件发布
```

### 行为驱动的自组织知识图谱 (Phase 1-4)

```
阶段 A (1-2 份试卷): 胚胎期
  MS 得分点提取 → 冷启动 Topic 分组 → 行为数据积累

阶段 B (3-4 份试卷): 塑形期
  Fragment 迁移 → Topic 边界重塑 → 稳定 Topic 生成 KP

阶段 C (5+ 份试卷): 固形期
  Topic 分裂/合并/消解 → KP 混入检索池 → 持续演化
```

### LLM 驱动的三层向量空间 (Phase 5)

```
Layer 1: LLM 结构化分类器
  新 QA → LLM 判断与所有 KP 的相关性 [0,1]
  → QA 初始向量 = 高分 KP 向量的加权平均
  → 解决: 新 QA 首次检索就精准命中相关 Topic

Layer 2: 双通道检索
  通道 A (语义): embedding top-30, 高召回
  通道 B (结构): Topic 归属 + 行为图漫步, 高精度
  融合排序 → 检索不依赖单一 embedding 信号

Layer 3: 向量空间自组织
  Fragment 中心性: 核心 Fragment 主导向量调整
  图中心性质心: Eigenvector Centrality + Weiszfeld 几何中位数
  三层级联: Fragment 调整 → KP 质心 → Topic 质心
  → 异常值不影响主聚类
```

### 聊天助手 (5-agent)

```
用户问题
  → Agent 1: Query Analyst — 关键词提取 + 问题分类 + 动词识别
  → 混合检索: QARetriever + Dynamic_Topics stable KP + KP Cache
  → Agent 3: Answer Generator — 类型自适应 + [KB]/[General] 来源标记
  → Agent 4: Critic — 审核循环 (max 2 轮)
  → Agent 5: Follow-up Suggester — 依赖感知的建议
  → 学生记忆记录 + 对话历史持久化
```

### 数据库表 (35 张)

| 分组 | 表 |
|------|----|
| 核心存储 (6) | `qa_pairs`, `question_feedback`, `topic_links`, `exam_sessions`, `api_call_log`, `chat_history` |
| 行为驱动知识图谱 (4) | `ms_fragments`, `fragment_help_map`, `fragment_membership`, `dynamic_topics` |
| 向量空间基础设施 (4) | `fragment_centrality`, `kp_vectors`, `qa_kp_scores`, `topic_vectors` |
| 分析产出 (7) | `topic_dependencies`, `command_verb_patterns`, `topic_difficulty`, `knowledge_points`, `kp_edges`, `qa_kp_membership`, `exam_trends` |
| 学生模型 (4) | `student_memory`, `student_knowledge_state`, `confusion_events`, `student_trajectory` |
| 自适应与进化 (6) | `paper_signatures`, `dimension_baselines`, `diversity_signals`, `calibration_checks`, `correction_rules`, `evolution_history` |
| 运维与缓存 (4) | `analysis_checkpoints`, `schema_version`, `distillation_cache`, `api_call_log` |

### 关键设计决策

- **Mark Scheme 为唯一 Ground Truth**：MS 答案原文拆分为得分点 (MS_Fragments)，不可修改
- **涌现聚类 (Mygo-001)**：KP 由 QA 向量自然聚集形成，每 KP 独立自适应阈值，胚胎 KP 始终可见+衰减
- **可插拔编码器 (Mygo-001)**：`VectorEncoder` 抽象接口，embedding 当前默认，SAE 预留
- **事件驱动解耦 (Mygo-000)**：5 模块间通过 Node.js 事件总线通信，显式 manifest 合约
- **统一流水线 (Mygo-003)**：`use_emergent=True` 消除 Phase 1/2 分叉，所有论文相同处理路径
- **BaseStore 消除样板 (A2)**：`_write()` / `_read_all()` 替代 49 处手动写锁
- **Schema 按域拆分 (A1)**：33 表 DDL 拆为 6 个域文件，便于维护
- **Topic 是活体**：会生长（新 Fragment 加入）、分裂、合并、消解
- **KP 是 Topic 的投影**：Topic 稳定后才由 Flash 生成命名和解释
- **行为数据驱动聚类**：Fragment 是否属于同一概念由其帮助答题的行为决定
- **双通道检索**：embedding 通道（召回）+ Topic 结构通道（精度）+ 行为图漫步
- **Wilson 区间 Beta 权重**：精确下置信界估计，小样本更保守
- **双语 prompt**：`%` 格式化 + `detect_content_lang()` 自动中英文切换
- **线程安全**：`threading.RLock` + WAL 模式 + per-thread 连接池

### 已知限制

- **仅支持文本层 PDF**：不具备 OCR 能力。扫描件、内嵌图片文字、图表均无法处理。扫描版 PDF 需用 Tesseract 等工具预处理
- **内容准确性需人工审核**：LLM 缺乏精确领域知识，输出应视为强草稿
- **Pitfalls 为推论**：系统无真实学生错题数据，陷阱来自 Phase 2 失分推断
- **行为数据冷启动**：前 2 份试卷期间 Fragment 迁移和 Topic 演化受限，知识结构质量随数据积累提升
- **Embedding 仅辅助**：仅用于冷启动临时分组，不作为主要聚类信号

## 开发

### 提交规范

参见 `CLAUDE.md`（gitignored，本地开发参考）。简要规则：

| 前缀 | 用途 |
|------|------|
| `Mygo - NNN` | 架构重构（跨模块结构调整） |
| `WSD-NNN` | 小修复、清理、bug 修正 |

### SAE 项目

稀疏自编码器（SAE）在独立仓库 `../sae-project/` 中训练。通过 `VectorEncoder` 接口可插拔集成，详见 `AgentChatRoom/plan.md`。

## License

本项目基于 **GNU Affero General Public License v3.0 (AGPL-3.0)** 许可。

**核心要求**：任何修改或部署为本软件的组织和个人，必须公开其修改后的源代码。

详见 [LICENSE](LICENSE) 文件。