"""Schema indexes — created after table initialization."""

SCHEMA_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_qa_dedup ON qa_pairs(question_text, answer_text)",
    "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_dep_prereq ON topic_dependencies(prerequisite)",
    "CREATE INDEX IF NOT EXISTS idx_dep_dep ON topic_dependencies(dependent)",
]
