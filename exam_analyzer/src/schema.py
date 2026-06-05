"""Database schema DDL — backward-compatible re-export.

All table definitions now live in ``exam_analyzer/src/schema/``
(split by domain: _tables_core, _tables_kp, _tables_topic,
_tables_fragment, _tables_student, _tables_analysis).

This file exists for backward compatibility — all imports of
``from .schema import SCHEMA_DDL, SCHEMA_INDEXES, SCHEMA_MIGRATIONS``
continue to work unchanged.
"""

from .schema._tables_core import CORE_TABLES
from .schema._tables_kp import KP_TABLES
from .schema._tables_topic import TOPIC_TABLES
from .schema._tables_fragment import FRAGMENT_TABLES
from .schema._tables_student import STUDENT_TABLES
from .schema._tables_analysis import ANALYSIS_TABLES
from .schema._indexes import SCHEMA_INDEXES
from .schema._migrations import SCHEMA_MIGRATIONS

SCHEMA_DDL: list[tuple[str, str]] = (
    CORE_TABLES + KP_TABLES + TOPIC_TABLES
    + FRAGMENT_TABLES + STUDENT_TABLES + ANALYSIS_TABLES
)

__all__ = ["SCHEMA_DDL", "SCHEMA_INDEXES", "SCHEMA_MIGRATIONS"]
