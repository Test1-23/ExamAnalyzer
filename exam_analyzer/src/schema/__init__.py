"""Database schema DDL — split by domain.

Re-exports the same SCHEMA_DDL / SCHEMA_INDEXES / SCHEMA_MIGRATIONS
that ``schema.py`` previously exported, preserving backward compatibility.
"""

from ._tables_core import CORE_TABLES
from ._tables_kp import KP_TABLES
from ._tables_topic import TOPIC_TABLES
from ._tables_fragment import FRAGMENT_TABLES
from ._tables_student import STUDENT_TABLES
from ._tables_analysis import ANALYSIS_TABLES
from ._indexes import SCHEMA_INDEXES
from ._migrations import SCHEMA_MIGRATIONS

SCHEMA_DDL: list[tuple[str, str]] = (
    CORE_TABLES + KP_TABLES + TOPIC_TABLES
    + FRAGMENT_TABLES + STUDENT_TABLES + ANALYSIS_TABLES
)

__all__ = ["SCHEMA_DDL", "SCHEMA_INDEXES", "SCHEMA_MIGRATIONS"]
