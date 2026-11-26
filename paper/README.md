# Paper

This folder holds the research paper for the knowledge-point segmentation work.

## Contents

| File | Description |
|---|---|
| `paper-two-migrations.pdf` | Paper, PDF (7 pages) |
| `paper-two-migrations.docx` | Paper, editable Word source |
| `figures/` | Figures referenced by the paper |

## The paper

**知识点切分中的两次迁移**
**——从几何代理指标到直接指标，从隐藏层到投影注意力**

*Two Migrations in Knowledge-Point Segmentation: From Geometric Proxy Metrics to Direct Metrics, From Hidden Layers to Projected Attention*

It reports how the segmentation of mark-scheme text into knowledge points was
reworked twice: first the evaluation metric (four geometric proxies — d′, silhouette,
within-cluster coherence, fragment self-sufficiency — were abandoned once shown to be
uncorrelated with the target, in favour of human-rated purity, coverage and single-topic
rate), and second the technical route (embedding clustering → encoder hidden layers →
Qwen3-32B attention, where projecting multi-head attention through the output matrix
$W_o$ raised question-to-answer alignment from 0.313 to 0.722, and 0.760 at 30-question
scale). The final segmentation method reaches purity × coverage × single-topic = 0.926.

## Figures

| Figure | Content |
|---|---|
| `fig1_proxy_metrics_fail.png` | d′ negatively correlated with segmentation quality; cone-shaped geometry |
| `fig3_projection.png` | Projection lifts alignment 0.313 → 0.722 |
| `fig4_ranking.png` | Method ranking |
| `fig5_scale_validation.png` | 30-question scale validation |

## Data

The corpus, attention matrices and per-experiment records live outside this folder
(they are local-only working artefacts). The paper states the corpus composition and
filtering rules in its §2.
