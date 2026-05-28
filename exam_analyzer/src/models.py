from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# PDF extraction models
# ============================================================

@dataclass
class ExtractedPage:
    """One page of an extracted PDF."""
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)


@dataclass
class ExtractedPDF:
    """Full PDF extraction result."""
    pages: List[ExtractedPage]
    full_text: str
    used_fallback: bool = False


@dataclass
class ExtractedPair:
    """A paired QP and MS with extracted content."""
    display_name: str
    qp: ExtractedPDF
    ms: ExtractedPDF


@dataclass
class QAPair:
    """A single question matched with its answer (Stage 2 output)."""
    question_number: str  # "2(a)", "3(b)(ii)" — original numbering
    question_text: str
    answer_text: str
    parent_question: str = ""  # "2" — parent question number


# ============================================================
# Parameter specs — replace bloated method signatures in QADatabase
# ============================================================

@dataclass
class KPSpec:
    """Parameters for QADatabase.upsert_kp (was 14 positional args)."""
    kp_id: str
    name: str = ""
    description: str = ""
    cluster_id: Optional[int] = None
    centroid_vector: Optional[bytes] = None
    core_concept: str = ""
    core_detail: str = ""
    variations: str = ""
    scoring_pattern: str = ""
    typical_marks: Optional[float] = None
    cohesion: Optional[float] = None
    evidence_count: int = 0
    quality: str = "draft"
    challenge_history: str = ""


@dataclass
class VerbPatternSpec:
    """Parameters for QADatabase.upsert_verb_pattern (was 11 positional args)."""
    verb: str
    sample_count: int = 0
    avg_answer_length: Optional[float] = None
    median_answer_length: Optional[float] = None
    bullet_ratio: Optional[float] = None
    avg_bullet_count: Optional[float] = None
    avg_miss_rate: Optional[float] = None
    common_missed_patterns: str = ""
    pattern_summary: str = ""
    topic_specific_patterns: str = ""
    verb_family: str = ""


@dataclass
class KpEdgeSpec:
    """Parameters for QADatabase.upsert_kp_edge (was 9 positional args)."""
    source_kp: str
    target_kp: str
    edge_type: str
    retrieval_weight: float = 0
    semantic_weight: float = 0
    sequential_weight: float = 0
    learning_path_weight: float = 0
    combined_strength: float = 0
    confidence: str = "low"


@dataclass
class DependencySpec:
    """Parameters for QADatabase.insert_dependency (was 9 positional args)."""
    prerequisite: str
    dependent: str
    evidence_score: int = 0
    evidence_reason: str = ""
    relationship_type: str = "prerequisite"
    topic_link_count: int = 0
    embedding_cos: Optional[float] = None
    confidence: str = "low"
    validated_by: str = "flash"
