"""Schema migrations — applied in version order by ConnectionMgr._run_migrations.

Each entry: (version, description, [ddl_statement, ...])
"""

SCHEMA_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "qa_pairs metadata columns", [
        "ALTER TABLE qa_pairs ADD COLUMN session_id INTEGER REFERENCES exam_sessions(id)",
        "ALTER TABLE qa_pairs ADD COLUMN is_representative BOOLEAN DEFAULT 0",
        "ALTER TABLE qa_pairs ADD COLUMN is_cross_topic BOOLEAN DEFAULT 0",
        "ALTER TABLE qa_pairs ADD COLUMN difficulty_estimate TEXT DEFAULT ''",
        "ALTER TABLE qa_pairs ADD COLUMN command_verb TEXT DEFAULT ''",
        "ALTER TABLE qa_pairs ADD COLUMN command_verb_secondary TEXT DEFAULT ''",
        "ALTER TABLE qa_pairs ADD COLUMN command_verb_inferred BOOLEAN DEFAULT 0",
        "ALTER TABLE qa_pairs ADD COLUMN last_failure_reason TEXT DEFAULT ''",
    ]),
    (2, "question_feedback metadata", [
        "ALTER TABLE question_feedback ADD COLUMN miss_categories TEXT DEFAULT ''",
    ]),
    (3, "Phase 5: fragment_help_map help_level + vector infrastructure", [
        "ALTER TABLE fragment_help_map ADD COLUMN help_level TEXT DEFAULT ''",
    ]),
    (4, "SAE integration: feature_activations + encoder_version columns", [
        "ALTER TABLE kp_vectors ADD COLUMN feature_activations BLOB",
        "ALTER TABLE qa_kp_scores ADD COLUMN encoder_version TEXT DEFAULT 'embedding'",
    ]),
    (5, "knowledge-space layout: vector_positions + fragment norm_key + naming flags", [
        "CREATE TABLE IF NOT EXISTS vector_positions ("
        "object_type TEXT NOT NULL, "
        "object_id TEXT NOT NULL, "
        "position BLOB NOT NULL, "
        "anchor BLOB NOT NULL, "
        "encoder_version TEXT DEFAULT '', "
        "epoch INTEGER DEFAULT 0, "
        "updated_at TEXT DEFAULT (datetime('now')), "
        "PRIMARY KEY (object_type, object_id))",
        "CREATE INDEX IF NOT EXISTS idx_vector_positions_type "
        "ON vector_positions(object_type)",
        "ALTER TABLE ms_fragments ADD COLUMN norm_key TEXT DEFAULT ''",
        "ALTER TABLE knowledge_points ADD COLUMN needs_naming INTEGER DEFAULT 0",
        "ALTER TABLE knowledge_points ADD COLUMN encoder_version TEXT DEFAULT ''",
    ]),
    (6, "attention-based fragment extraction: source + score + span", [
        "ALTER TABLE ms_fragments ADD COLUMN source TEXT DEFAULT 'llm'",
        "ALTER TABLE ms_fragments ADD COLUMN attention_score REAL DEFAULT 0",
        "ALTER TABLE ms_fragments ADD COLUMN token_span TEXT DEFAULT ''",
    ]),
]
