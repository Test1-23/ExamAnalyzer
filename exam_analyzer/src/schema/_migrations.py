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
]
