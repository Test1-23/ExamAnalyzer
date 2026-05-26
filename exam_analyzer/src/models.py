from dataclasses import dataclass, field
from typing import List


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
