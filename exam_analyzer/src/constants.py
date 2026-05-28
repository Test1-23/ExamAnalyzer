"""Centralized tunable constants for ExamAnalyzer.

All magic numbers and tunable parameters live here so they can be
adjusted in one place.  Every constant supports environment-variable
overrides (prefixed ``EXAM_``).

Importing modules should use::

    from .constants import RETRIEVAL_THRESHOLD, CENTRALITY_HIGH, ...

rather than defining their own copies.
"""

import os

_STR = os.environ.get
_INT = lambda k, d: int(_STR(k, str(d)))
_FLT = lambda k, d: float(_STR(k, str(d)))

# ============================================================
# DeepSeek API
# ============================================================
FLASH_MODEL = _STR("EXAM_FLASH_MODEL", "deepseek-v4-flash")
PRO_MODEL = _STR("EXAM_PRO_MODEL", "deepseek-v4-pro")
DEFAULT_MAX_RETRIES = _INT("EXAM_MAX_RETRIES", 3)

# ============================================================
# Embedding
# ============================================================
CJK_DETECTION_THRESHOLD = _FLT("EXAM_CJK_THRESHOLD", 0.05)
EMBEDDING_BATCH_SIZE = _INT("EXAM_EMBEDDING_BATCH", 64)

# ============================================================
# SQLite / Database
# ============================================================
SQLITE_PARAM_CHUNK = _INT("EXAM_SQLITE_CHUNK", 900)

# ============================================================
# Dual-channel Retrieval
# ============================================================
CHANNEL_A_RECALL = _INT("EXAM_CHANNEL_A_RECALL", 30)
BEHAVIOR_CHUNK = _INT("EXAM_BEHAVIOR_CHUNK", 400)
WEIGHT_EMBEDDING = _FLT("EXAM_WEIGHT_EMBEDDING", 0.35)
WEIGHT_TOPIC = _FLT("EXAM_WEIGHT_TOPIC", 0.35)
WEIGHT_BEHAVIOR = _FLT("EXAM_WEIGHT_BEHAVIOR", 0.20)
WEIGHT_KEYWORD = _FLT("EXAM_WEIGHT_KEYWORD", 0.10)

RETRIEVAL_THRESHOLD = _FLT("EXAM_RETRIEVAL_THRESHOLD", 0.5)
RETRIEVAL_MIN_K = _INT("EXAM_RETRIEVAL_MIN_K", 3)
RETRIEVAL_MAX_CAP = _INT("EXAM_RETRIEVAL_MAX_CAP", 15)

# ============================================================
# QA Vector Placement (centrality thresholds)
# ============================================================
CENTRALITY_HIGH = _FLT("EXAM_CENTRALITY_HIGH", 0.8)
CENTRALITY_MID = _FLT("EXAM_CENTRALITY_MID", 0.5)
CENTRALITY_LOW = _FLT("EXAM_CENTRALITY_LOW", 0.2)

# ============================================================
# Topic Merge
# ============================================================
TOPIC_MERGE_COS_THRESHOLD = _FLT("EXAM_TOPIC_MERGE_COS", 0.85)
TOPIC_MERGE_AMBIGUOUS_THRESHOLD = _FLT("EXAM_TOPIC_MERGE_AMBIGUOUS", 0.30)

# ============================================================
# Fragment Help Level
# ============================================================
HELP_DIRECT_THRESHOLD = _FLT("EXAM_HELP_DIRECT", 0.7)
HELP_UNDERSTANDING_THRESHOLD = _FLT("EXAM_HELP_UNDERSTANDING", 0.3)

# ============================================================
# Missed Line Analysis
# ============================================================
MISSED_FILTER_THRESHOLD = _FLT("EXAM_MISSED_FILTER", 0.60)
MISSED_CLUSTER_THRESHOLD = _FLT("EXAM_MISSED_CLUSTER", 0.80)

# ============================================================
# Anomaly Detection
# ============================================================
ANOMALY_ZSCORE_SINGLE = _FLT("EXAM_ANOMALY_SINGLE", 3.0)
ANOMALY_ZSCORE_SYSTEMIC = _FLT("EXAM_ANOMALY_SYSTEMIC", 2.0)
SYSTEMIC_DIMENSION_COUNT = _INT("EXAM_SYSTEMIC_DIMS", 3)

# ============================================================
# Difficulty Classification
# ============================================================
DIFFICULTY_HARD_THRESHOLD = _FLT("EXAM_DIFFICULTY_HARD", 2.0)
DIFFICULTY_EASY_THRESHOLD = _FLT("EXAM_DIFFICULTY_EASY", 1.3)

# ============================================================
# Adversarial Refinement
# ============================================================
MAX_ROUNDS = _INT("EXAM_ADVERSARIAL_ROUNDS", 5)
PASS_THRESHOLD = _INT("EXAM_ADVERSARIAL_PASS", 2)

# ============================================================
# Knowledge Graph Edge Fusion
# ============================================================
EDGE_FUSION_RETRIEVAL_W = _FLT("EXAM_EDGE_FUSION_RW", 0.4)
EDGE_FUSION_SEMANTIC_W = _FLT("EXAM_EDGE_FUSION_SW", 0.3)
EDGE_FUSION_SEQUENTIAL_W = _FLT("EXAM_EDGE_FUSION_SQW", 0.15)
EDGE_FUSION_LEARNING_PATH_W = _FLT("EXAM_EDGE_FUSION_LPW", 0.15)
EDGE_TRANSITION_DIVISOR = _INT("EXAM_EDGE_TRANSITION_DIV", 10)
