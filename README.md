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
| Embedding | sentence-transformers (all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2) |
| 数据库 | SQLite WAL 模式，版本化迁移框架 |
| PDF 提取 | pdfplumber (主) + PyMuPDF (fallback)，仅文本层，不支持 OCR/图片 |
| Web | Flask + Jinja2 |

### 目录结构

```
exam_analyzer/
├── main.py                  # CLI 入口
├── app.py                   # Flask Web 入口 (薄封装, 44 行)
├── src/                     # 核心库 (50+ 模块, 三层架构)
│   ├── connection_manager.py # 数据库连接管理: 唯一 conn 持有者 + WAL + 迁移 + transaction()
│   ├── query_builder.py     # 轻量 SQL 构建器: %-格式化, 单表 CRUD, raw() 逃逸阀
│   ├── knowledge_base.py    # QADatabase Facade: 组装 ConnMgr → QueryBuilder → 8 Domain Store
│   ├── retriever.py         # QARetriever: embedding 相似度搜索 + 双通道检索
│   ├── schema.py            # 31 表 DDL + 索引 + 版本化迁移
│   ├── constants.py         # 所有可调参数
│   ├── stores/              # 8 个领域 Store (类型安全的数据访问层)
│   │   ├── qa_store.py      # qa_pairs
│   │   ├── topic_store.py   # dynamic_topics + topic_links + difficulty
│   │   ├── kp_store.py      # knowledge_points + edges + membership
│   │   ├── fragment_store.py# ms_fragments + help_map + centrality
│   │   ├── chat_store.py    # chat_history
│   │   ├── student_store.py # student memory/state/confusions/trajectory
│   │   ├── analysis_store.py# api_call_log + feedback + cache + checkpoints + deps
│   │   └── vector_store.py  # kp_vectors + qa_kp_scores + topic_vectors
│   ├── web/                 # Flask Blueprint Web 层 (P4 完成)
│   │   ├── state.py         # 全局共享状态 + 线程安全访问器
│   │   ├── app_factory.py   # create_app() 工厂
│   │   └── routes_*.py      # 4 Blueprint, 21 路由
│   ├── offline/             # 离线分析包
│   │   ├── verbs.py         # 命令动词提取
│   │   ├── difficulty.py    # 难度评估 + _evaluate_signal()
│   │   ├── dependencies.py  # 主题依赖发现
│   │   └── report.py        # 报告生成 + run_offline_analysis()
│   ├── diagnostics/         # 后处理诊断包
│   │   ├── pitfalls.py      # 陷阱发现 + 考试趋势
│   │   ├── cross_paper.py   # 论文签名 + 基线 + 异常检测
│   │   ├── student.py       # 学生反馈闭环
│   │   ├── migration.py     # Fragment 迁移 + Topic 统计
│   │   └── cascade.py       # Topic 分裂/合并 + 向量级联
│   ├── prompts/             # Prompt 模板 + Pipeline prompt 封装
│   ├── chat/                # 对话上下文 + 5-agent 聊天管线
│   ├── pipeline.py          # Phase 1/2 编排器 + 数据处理辅助函数
│   ├── knowledge_graph.py   # QA 聚类 → KP 生成 → 边发现 → 融合
│   ├── adversarial_refiner.py  # KP 拆分/合并
│   ├── distiller.py         # 蒸馏器 (Flash-based topic distillation)
│   ├── evolution.py         # 自进化循环
│   ├── topic_merger.py      # 余弦相似度 + Flash 审核主题合并
│   └── [其他模块]           # deepseek_client, embedding_cluster, models, utils 等
├── eval/                    # 质量评估
│   └── feedback_agent.py    # 6 维度聊天质量评估
├── tests/                   # 自动化测试 (40+ 单元测试 + Mock 基础设施)
│   ├── conftest.py          # MockFlashClient + mock_db + mock_flash + mock_embedding
│   ├── test_pure_functions.py  # 纯函数测试
│   ├── test_store_crud.py   # Store CRUD 测试
│   ├── test_retrieval.py    # 检索测试
│   └── test_*.py            # mock 流水线 + 错误处理 + 单元测试
├── input/                   # 试卷 PDF (gitignored)
├── intermediate/            # SQLite DB + 处理状态 (gitignored)
├── point/                   # 知识点输出 (gitignored)
└── templates/               # Flask HTML 模板
```

### 流水线架构

```
论文 PDF 对 (N 篇，按年份升序)
  │
  ├─ PDF 提取 (pdfplumber → PyMuPDF fallback)
  ├─ QA 配对 (Flash: 全文 QP+MS → 匹配题目与答案)
  │
  ├─ 逐 QA 处理 ────────────────────────────────────
  │   Phase 1 (首篇): Flash summary + topic → 直接入库
  │   Phase 2 (后续): summary → 检索 + KP 参考 → 答题 → MS 评分 + 失分分类 → 入库
  │   MS 得分点提取: Flash 拆分 MS 答案 → MS_Fragments
  │   行为数据采集: 记录 Fragment 帮助了哪些题目 → Fragment_Help_Map
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

- **Mark Scheme 为唯一 Ground Truth**：MS 答案原文拆分为得分点 (MS_Fragments)，不可修改，所有知识追溯至此
- **行为数据驱动聚类**：两个得分点是否属于同一概念，由它们是否互相帮助答题决定，替代 embedding 相似度聚类
- **Topic 是活体**：会生长（新 Fragment 加入）、分裂（行为分化）、合并（行为趋同）、消解（成员流失）
- **KP 是 Topic 的投影**：Topic 稳定后才由 Flash 生成命名和解释，Topic 变化后重新生成
- **质量由实测驱动**：KP 的有效性由 Phase 2 答题得分验证，非 Flash 自我评价
- **对抗精炼已退役**：Flash 审查 Flash 被行为驱动的质量度量替代
- **LLM 作为结构化分类器**：新 QA 由 LLM 判断与所有 KP 的相关性（1 次调用/QA，分类非生成）
- **双通道检索**：embedding 通道（召回）+ Topic 结构通道（精度）+ 行为图漫步
- **Fragment 中心性 + 图中心性质心**：核心 Fragment 主导向量调整，孤立点被忽略
- **三层级联向量调整**：Fragment → KP → Topic 级联自组织
- **Wilson 区间 Beta 权重**：精确下置信界估计，小样本更保守
- **增量蒸馏缓存**：MD5 指纹比对，仅重蒸变化的 Topic
- **双语 prompt**：`detect_content_lang()` 自动中英文切换
- **`%` 格式化 prompt**：禁止 `.format()`——MS 文本含 `{` `}` 会导致崩溃
- **线程安全 + 动态并发**：`threading.Lock` + `PIPELINE_MAX_WORKERS` 环境变量，默认 8/16

### 已知限制

- **仅支持文本层 PDF**：不具备 OCR 能力。扫描件、内嵌图片文字、图表均无法处理。扫描版 PDF 需用 Tesseract 等工具预处理
- **内容准确性需人工审核**：LLM 缺乏精确领域知识，输出应视为强草稿
- **Pitfalls 为推论**：系统无真实学生错题数据，陷阱来自 Phase 2 失分推断
- **行为数据冷启动**：前 2 份试卷期间 Fragment 迁移和 Topic 演化受限，知识结构质量随数据积累提升
- **Embedding 仅辅助**：仅用于冷启动临时分组，不作为主要聚类信号

## License

本项目基于 **GNU Affero General Public License v3.0 (AGPL-3.0)** 许可。

**核心要求**：任何修改或部署为本软件的组织和个人，必须公开其修改后的源代码。

详见 [LICENSE](LICENSE) 文件。