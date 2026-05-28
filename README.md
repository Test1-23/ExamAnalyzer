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
| LLM | DeepSeek Flash（全流程，纯 Flash 无 Pro 依赖） |
| Embedding | sentence-transformers (all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2) |
| 数据库 | SQLite WAL 模式，版本化迁移框架 |
| PDF 提取 | pdfplumber (主) + PyMuPDF (fallback)，仅文本层，不支持 OCR/图片 |
| Web | Flask + Jinja2 |

### 目录结构

```
exam_analyzer/
├── main.py                  # CLI 入口
├── app.py                   # Flask Web 服务 + 5-agent 聊天管线
├── src/                     # 核心库 (14 模块)
│   ├── pipeline.py          # 流水线编排: Phase 1/2 + 后处理 + 进化循环
│   ├── knowledge_base.py    # SQLite 数据库 (31表) + 向量检索 + Beta权重
│   ├── knowledge_graph.py   # QA 聚类 → KP 节点 → 4种边 → 融合
│   ├── offline_analyzer.py  # 命令动词/难度/依赖分析
│   ├── adversarial_refiner.py  # KP 拆分/合并 (对抗精炼已退役)
│   ├── pipeline_diagnostics.py # Fragment迁移 + Topic演化 + 学生反馈
│   ├── question_generator.py   # 模板提取 + 参数变异生成
│   ├── deepseek_client.py   # DeepSeek API (Flash + 重试 + JSON提取)
│   ├── embedding_cluster.py # 模型管理 + 语言检测
│   ├── pdf_extractor.py     # PDF 文本提取 (pdfplumber → PyMuPDF)
│   ├── file_pairer.py       # 文件名配对
│   ├── models.py            # 数据类定义
│   └── logger.py            # 日志 (RotatingFileHandler)
├── eval/                    # 质量评估
│   └── feedback_agent.py    # 6 维度聊天质量评估
├── tests/                   # 自动化测试
├── input/                   # 试卷 PDF (gitignored)
├── intermediate/            # SQLite DB + 处理状态 (gitignored)
├── point/                   # 输出 (gitignored)
├── logs/                    # 运行日志 (gitignored)
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

### 数据库表 (31 张)

| 分组 | 表 |
|------|----|
| 核心存储 (6) | `qa_pairs`, `question_feedback`, `topic_links`, `exam_sessions`, `api_call_log`, `chat_history` |
| 行为驱动知识图谱 (4) | `ms_fragments`, `fragment_help_map`, `fragment_membership`, `dynamic_topics` |
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
