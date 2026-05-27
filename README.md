# Exam Knowledge Point Analyzer

从配对试卷 PDF（Question Paper + Mark Scheme）中提取知识点，构建知识图谱，提供对话式 AI 学习助手。

## 介绍

本工具自动解析试卷 PDF，配对题目与答案，利用 LLM 提取和蒸馏知识点，生成结构化的复习资料。内置的对话助手支持多轮问答、诊断测验和学习路径推荐。

核心流程：PDF 提取 → QA 配对 → 知识库积累 → 知识点蒸馏 → 知识图谱构建 → 对抗性精炼 → 离线分析。

## 环境配置

- Python 3.10+
- DeepSeek API 访问权限

```bash
pip install -r requirements.txt
```

在 `exam_analyzer/config.json` 中配置 API，或通过环境变量 `DEEPSEEK_API_URL` / `DEEPSEEK_API_KEY` 设置：

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

- **subject**: 科目编号（如 `0478`）
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

在网页中配置 API → 点击「开始分析」即可自动运行全流程。页面实时显示进度、日志和输出结果。分析完成后自动加载知识库，可立即开始对话问答。

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
| LLM | DeepSeek Flash (提取/评分) + Pro (复杂推理，可选) |
| Embedding | sentence-transformers (all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2) |
| 数据库 | SQLite WAL 模式 |
| PDF 提取 | pdfplumber (主) + PyMuPDF (fallback)，仅提取文本层，不支持 OCR/图片 |
| Web | Flask + Jinja2 |
| 聚类 | 余弦相似图连通分量 (HDBSCAN 备选) |

### 目录结构

```
exam_analyzer/
├── main.py                  # CLI 入口
├── app.py                   # Flask Web 服务 + 5-agent 聊天管线
├── src/                     # 核心库 (14 模块)
│   ├── pipeline.py          # 核心流水线编排
│   ├── knowledge_base.py    # SQLite 数据库 + 向量检索
│   ├── knowledge_graph.py   # QA 聚类 → KP 节点 → 4种边 → 融合
│   ├── offline_analyzer.py  # 离线分析：动词、难度、依赖
│   ├── adversarial_refiner.py  # 对抗性精炼：挑战者/辩护者辩论
│   ├── pipeline_diagnostics.py # 闭环改进 + 跨论文一致性
│   ├── question_generator.py   # 题目模板提取 + 参数变异生成
│   ├── deepseek_client.py   # DeepSeek API 客户端 (Flash + Pro)
│   ├── embedding_cluster.py # 模型管理 + 语言检测 + 公共常量
│   ├── pdf_extractor.py     # PDF 文本提取 (pdfplumber + PyMuPDF)
│   ├── file_pairer.py       # 文件名配对
│   ├── models.py            # 数据类定义
│   └── logger.py            # 日志 (RotatingFileHandler)
├── eval/                    # 质量评估
│   └── feedback_agent.py    # 6 维度聊天质量评估
├── tests/                   # 自动化测试 (gitignored)
├── input/                   # 试卷 PDF (gitignored)
├── intermediate/            # SQLite DB + 处理状态 (gitignored)
├── point/                   # 输出 (gitignored)
├── logs/                    # 运行日志 (gitignored)
└── templates/               # Flask HTML 模板
```

### 流水线阶段

```
论文 PDF 对 (N 篇，按年份升序)
  │
  ├─ PDF 提取 (pdfplumber → PyMuPDF fallback)
  ├─ QA 配对 (Flash: 全文 QP+Answer → 匹配题目与答案)
  │
  ├─ 逐 QA 处理 ──────────────────────────
  │   Phase 1 (首篇): Flash summary + topic → 直接入库
  │   Phase 2 (后续): summary → 检索 top-4 → Flash 答题 → Flash 评分 + 失分分类 → 入库
  │
  ├─ 后处理 ──────────────────────────────
  │   Topic Merge → Distillation → Review → points.txt
  │   → Knowledge Graph (聚类 → KP 节点 → 4 种边 → 融合)
  │   → Adversarial Refinement (挑战者/辩护者验证每个 KP)
  │   → Offline Analysis (动词规律 + 难度评估 + 依赖发现)
  │   → Pipeline Diagnostics (闭环改进 + 跨论文一致性)
```

### 聊天助手 (5-agent)

```
用户问题
  → Agent 1: Query Analyst — 关键词提取 + 问题分类 + 动词识别
  → 混合检索: QARetriever + KP Cache + 分析上下文(难度/动词/依赖/学生状态)
  → Agent 3: Answer Generator — 类型自适应 + [KB]/[General] 来源标记 + quiz + path_hint
  → Agent 4: Critic — 审核循环 (max 2 轮)
  → Agent 5: Follow-up Suggester — 依赖感知的建议
  → 学生记忆记录 + 对话历史持久化
```

### 数据流

```
PDF → QA pairs → KB (SQLite) → retrieval → Flash answering → grading
                                                    ↓
                                            topic_links (跨 topic 引用)
                                                    ↓
                                            points.txt (蒸馏输出)
                                                    ↓
                                    离线分析 → 知识图谱 → 对抗精炼 → 闭环
```

### 数据库表 (23 张)

| 分组 | 表 |
|------|----|
| 核心存储 (6) | `qa_pairs`, `question_feedback`, `topic_links`, `exam_sessions`, `api_call_log`, `chat_history` |
| 分析产出 (7) | `topic_dependencies`, `command_verb_patterns`, `topic_difficulty`, `knowledge_points`, `kp_edges`, `qa_kp_membership`, `exam_trends` |
| 学生模型 (4) | `student_memory`, `student_knowledge_state`, `confusion_events`, `student_trajectory` |
| 自适应 (5) | `paper_signatures`, `dimension_baselines`, `diversity_signals`, `calibration_checks`, `correction_rules` |
| 运维 (1) | `analysis_checkpoints` |

### 关键设计决策

- **Mark Scheme 为 ground truth**：不以题目文本提取知识点，而是将 QA 原样存储，在蒸馏阶段批量提取共性概念
- **双语 prompt**：`detect_content_lang()` 自动中英文切换
- **Beta 分布权重**：`weight = (success+1)/(total_attempts+2)`，低权重 topic 标记 `[needs review]`
- **动态 K 检索**：`threshold=0.5, max_cap=15, min_k=3`，聊天用 `max_cap=5`
- **线程安全**：`threading.Lock` 保护 DB 写，double-checked lock 初始化，所有 ThreadPoolExecutor 上限 8 或 16
- **非致命错误**：PDF 提取失败、Flash 调用失败、离线分析失败均不阻塞主流程
- **崩溃恢复**：`processed.json` 标记已完成论文，`analysis_checkpoints` 记录后处理进度
- **`%` 格式化 prompt**：禁止 `.format()`——Mark Scheme 文本含 `{` `}` 会导致崩溃

### 注意事项

1. **首次运行需下载 embedding 模型** (~80MB all-MiniLM-L6-v2, ~120MB paraphrase-multilingual)。国内建议设置 `HF_ENDPOINT=https://hf-mirror.com`
2. **论文按年份升序处理**：最早论文先进 KB，提供更好的检索基础
3. **Phase 2 必须记录 `record_attempt(False)`**：未使用的 QA 也需降低 Beta weight，否则权重失真
4. **DB 重建**：如遇 schema 不匹配，删除 `intermediate/` 重跑

### 已知限制

- **无法处理图片型 PDF**：系统依赖 `pdfplumber` + `PyMuPDF` 提取**文本层**，不具备 OCR 能力。
  - 扫描件 PDF（整页为图片）会提取到空白内容，导致分析失败
  - 内嵌图片中的文字（如流程图标注、截图公式）会被完全忽略
  - 图表、示意图等视觉信息无法被提取和理解
  - 如果你的 PDF 是扫描版，需先用 OCR 工具（如 Adobe Acrobat、Tesseract）将其转为可搜索 PDF
- **内容准确性需人工审核**：LLM 缺乏精确领域知识，输出应视为强草稿
- **Pitfalls 为推论而非经验**：系统无真实学生错题数据，pitfall 来自模型推断
- **难度评估无 ground truth**：基于 Flash 行为信号（miss_rate），与学生真实感知可能有偏差
- **Embedding 模型语言锁定**：首篇论文的语言决定检索模型，跨语言场景精度下降
- **Topic 名称多样性**：不同论文的 Mark Scheme 用词不同——这是有意的，学生需要识别变体命名

## License

本项目基于 **GNU Affero General Public License v3.0 (AGPL-3.0)** 许可。

**核心要求**：任何使用、修改或部署本软件的组织和个人，必须将其修改后的源代码公开。

- 如果你只是在本地运行——无需公开任何东西
- 如果你修改了代码并部署为公开服务（包括网站、API）——必须公开你的修改
- 如果你将本软件或衍生作品分发给他人——必须同时提供完整源代码
- 本条目不阻止商业使用，但要求商业用户同样履行开源义务

详见 [LICENSE](LICENSE) 文件。