"""Database schema DDL — extracted from QADatabase._init_tables.

Each entry is a (table_name, create_sql) pair.  QADatabase._init_tables
iterates over SCHEMA_DDL and executes each statement.
"""

SCHEMA_DDL: list[tuple[str, str]] = [
    ("qa_pairs", """
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            knowledge_summary TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            paper TEXT NOT NULL DEFAULT '',
            question_number TEXT NOT NULL DEFAULT '',
            parent_question TEXT NOT NULL DEFAULT '',
            success_count INTEGER DEFAULT 0,
            total_attempts INTEGER DEFAULT 0,
            last_failure_reason TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("api_call_log", """
        CREATE TABLE IF NOT EXISTS api_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL,
            model TEXT NOT NULL,
            paper TEXT DEFAULT '',
            question_number TEXT DEFAULT '',
            latency_ms INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1,
            output_size INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("question_feedback", """
        CREATE TABLE IF NOT EXISTS question_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qa_id INTEGER REFERENCES qa_pairs(id),
            retrieval_count INTEGER DEFAULT 0,
            used_qa_count INTEGER DEFAULT 0,
            step0_topic TEXT DEFAULT '',
            round2_topic TEXT DEFAULT '',
            topic_match INTEGER DEFAULT 0,
            covered_count INTEGER DEFAULT 0,
            missed_count INTEGER DEFAULT 0,
            missed_text TEXT DEFAULT '',
            coverage_ratio REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("topic_links", """
        CREATE TABLE IF NOT EXISTS topic_links (
            src_topic TEXT NOT NULL,
            dst_topic TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            PRIMARY KEY (src_topic, dst_topic)
        );
    """),
    ("exam_sessions", """
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code TEXT NOT NULL,
            season TEXT NOT NULL,
            year INTEGER NOT NULL,
            display_name TEXT NOT NULL,
            UNIQUE(subject_code, season, year, display_name)
        );
    """),
    ("student_memory", """
        CREATE TABLE IF NOT EXISTS student_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_at TEXT DEFAULT (datetime('now')),
            last_recalled_at TEXT
        );
    """),
    ("student_knowledge_state", """
        CREATE TABLE IF NOT EXISTS student_knowledge_state (
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            state TEXT NOT NULL,
            evidence_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (student_id, topic)
        );
    """),
    ("confusion_events", """
        CREATE TABLE IF NOT EXISTS confusion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            trigger_question TEXT,
            confusion_type TEXT,
            resolved BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("chat_history", """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("topic_dependencies", """
        CREATE TABLE IF NOT EXISTS topic_dependencies (
            prerequisite TEXT NOT NULL,
            dependent TEXT NOT NULL,
            evidence_score INTEGER,
            evidence_reason TEXT,
            relationship_type TEXT DEFAULT 'prerequisite',
            topic_link_count INTEGER DEFAULT 0,
            embedding_cos REAL,
            confidence TEXT DEFAULT 'low',
            validated_at TEXT,
            first_seen_at TEXT DEFAULT (datetime('now')),
            last_validated_at TEXT,
            validated_by TEXT DEFAULT 'flash',
            PRIMARY KEY (prerequisite, dependent)
        );
    """),
    ("command_verb_patterns", """
        CREATE TABLE IF NOT EXISTS command_verb_patterns (
            verb TEXT PRIMARY KEY,
            sample_count INTEGER DEFAULT 0,
            avg_answer_length REAL,
            median_answer_length REAL,
            bullet_ratio REAL,
            avg_bullet_count REAL,
            avg_miss_rate REAL,
            common_missed_patterns TEXT,
            pattern_summary TEXT,
            topic_specific_patterns TEXT,
            verb_family TEXT,
            generated_at TEXT DEFAULT (datetime('now')),
            last_updated TEXT
        );
    """),
    ("topic_difficulty", """
        CREATE TABLE IF NOT EXISTS topic_difficulty (
            topic TEXT PRIMARY KEY,
            qa_count INTEGER DEFAULT 0,
            basic_count INTEGER DEFAULT 0,
            intermediate_count INTEGER DEFAULT 0,
            advanced_count INTEGER DEFAULT 0,
            mode_difficulty TEXT,
            avg_miss_rate REAL,
            difficulty_spread BOOLEAN DEFAULT 0,
            assessed_at TEXT,
            assessment_method TEXT DEFAULT 'hybrid'
        );
    """),
    ("analysis_checkpoints", """
        CREATE TABLE IF NOT EXISTS analysis_checkpoints (
            task_name TEXT PRIMARY KEY,
            qa_count_at_run INTEGER,
            completed_at TEXT,
            status TEXT DEFAULT 'pending'
        );
    """),
    ("knowledge_points", """
        CREATE TABLE IF NOT EXISTS knowledge_points (
            id TEXT PRIMARY KEY,
            cluster_id INTEGER,
            name TEXT,
            description TEXT,
            centroid_vector BLOB,
            core_concept TEXT,
            core_detail TEXT,
            variations TEXT,
            scoring_pattern TEXT,
            typical_marks REAL,
            cohesion REAL,
            evidence_count INTEGER DEFAULT 0,
            quality TEXT DEFAULT 'draft',
            challenge_history TEXT,
            first_seen_at TEXT DEFAULT (datetime('now')),
            last_validated_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("kp_edges", """
        CREATE TABLE IF NOT EXISTS kp_edges (
            source_kp TEXT REFERENCES knowledge_points(id),
            target_kp TEXT REFERENCES knowledge_points(id),
            edge_type TEXT,
            retrieval_weight REAL,
            semantic_weight REAL,
            sequential_weight REAL,
            learning_path_weight REAL,
            combined_strength REAL,
            confidence TEXT DEFAULT 'low',
            PRIMARY KEY (source_kp, target_kp, edge_type)
        );
    """),
    ("qa_kp_membership", """
        CREATE TABLE IF NOT EXISTS qa_kp_membership (
            qa_id INTEGER REFERENCES qa_pairs(id),
            kp_id TEXT REFERENCES knowledge_points(id),
            membership_strength REAL,
            is_representative BOOLEAN DEFAULT 0,
            PRIMARY KEY (qa_id, kp_id)
        );
    """),
    ("student_trajectory", """
        CREATE TABLE IF NOT EXISTS student_trajectory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            kp_id TEXT REFERENCES knowledge_points(id),
            from_state TEXT,
            to_state TEXT,
            trigger TEXT,
            recorded_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("exam_trends", """
        CREATE TABLE IF NOT EXISTS exam_trends (
            kp_id TEXT REFERENCES knowledge_points(id),
            year INTEGER,
            season TEXT,
            occurrence_count INTEGER DEFAULT 0,
            avg_difficulty TEXT,
            trend_summary TEXT,
            PRIMARY KEY (kp_id, year, season)
        );
    """),
    ("paper_signatures", """
        CREATE TABLE IF NOT EXISTS paper_signatures (
            display_name TEXT PRIMARY KEY,
            qa_count INTEGER,
            topic_count INTEGER,
            verb_dist TEXT,
            difficulty_dist TEXT,
            avg_miss_rate REAL,
            avg_answer_length REAL,
            topic_purity_avg REAL,
            anomaly_flags TEXT,
            extracted_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("dimension_baselines", """
        CREATE TABLE IF NOT EXISTS dimension_baselines (
            dimension TEXT PRIMARY KEY,
            mean REAL,
            median REAL,
            mad REAL,
            sample_count INTEGER,
            last_updated TEXT
        );
    """),
    ("diversity_signals", """
        CREATE TABLE IF NOT EXISTS diversity_signals (
            display_name TEXT,
            step TEXT,
            signal_name TEXT,
            signal_value REAL,
            normal_range TEXT,
            is_anomaly BOOLEAN DEFAULT 0,
            recorded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (display_name, step, signal_name)
        );
    """),
    ("calibration_checks", """
        CREATE TABLE IF NOT EXISTS calibration_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            check_type TEXT,
            step TEXT,
            qa_id INTEGER,
            system_output TEXT,
            system_confidence REAL,
            check_result TEXT,
            was_calibrated BOOLEAN,
            checked_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("correction_rules", """
        CREATE TABLE IF NOT EXISTS correction_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step TEXT,
            anomaly_pattern TEXT,
            correction_action TEXT,
            times_triggered INTEGER DEFAULT 0,
            success_rate REAL,
            last_triggered TEXT
        );
    """),
    ("distillation_cache", """
        CREATE TABLE IF NOT EXISTS distillation_cache (
            topic TEXT PRIMARY KEY,
            qa_count INTEGER NOT NULL DEFAULT 0,
            qa_ids_hash TEXT NOT NULL DEFAULT '',
            distilled_content TEXT NOT NULL DEFAULT '',
            distilled_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("evolution_history", """
        CREATE TABLE IF NOT EXISTS evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kp_id TEXT REFERENCES knowledge_points(id),
            trigger_type TEXT NOT NULL,
            trigger_detail TEXT,
            old_state TEXT,
            new_state TEXT,
            outcome TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("ms_fragments", """
        CREATE TABLE IF NOT EXISTS ms_fragments (
            point_id TEXT PRIMARY KEY,
            qa_id INTEGER REFERENCES qa_pairs(id),
            point_text TEXT NOT NULL,
            marks INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("fragment_help_map", """
        CREATE TABLE IF NOT EXISTS fragment_help_map (
            fragment_id TEXT REFERENCES ms_fragments(point_id),
            helped_qa_id INTEGER REFERENCES qa_pairs(id),
            help_effect REAL DEFAULT 0,
            help_level TEXT DEFAULT '',
            PRIMARY KEY (fragment_id, helped_qa_id)
        );
    """),
    ("fragment_membership", """
        CREATE TABLE IF NOT EXISTS fragment_membership (
            fragment_id TEXT PRIMARY KEY REFERENCES ms_fragments(point_id),
            topic_id TEXT NOT NULL,
            loyalty REAL DEFAULT 0.5,
            joined_at TEXT DEFAULT (datetime('now')),
            previous_topic_id TEXT
        );
    """),
    ("dynamic_topics", """
        CREATE TABLE IF NOT EXISTS dynamic_topics (
            topic_id TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            mass INTEGER DEFAULT 0,
            cohesion REAL DEFAULT 0,
            stability REAL DEFAULT 0,
            behavioral_profile TEXT DEFAULT '{}',
            quality TEXT DEFAULT 'embryonic',
            parent_topic TEXT,
            child_topics TEXT DEFAULT '[]',
            merged_from TEXT DEFAULT '[]',
            kp_concept TEXT DEFAULT '',
            kp_detail TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            last_evolved_at TEXT
        );
    """),
    ("fragment_centrality", """
        CREATE TABLE IF NOT EXISTS fragment_centrality (
            fragment_id TEXT PRIMARY KEY REFERENCES ms_fragments(point_id),
            verification_count INTEGER DEFAULT 0,
            avg_help_score REAL DEFAULT 0,
            topic_coherence REAL DEFAULT 0,
            variance REAL DEFAULT 0,
            centrality_score REAL DEFAULT 0.2,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("kp_vectors", """
        CREATE TABLE IF NOT EXISTS kp_vectors (
            kp_id TEXT PRIMARY KEY REFERENCES knowledge_points(id),
            vector BLOB NOT NULL,
            adjustment_count INTEGER DEFAULT 0,
            stability REAL DEFAULT 1.0,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """),
    ("qa_kp_scores", """
        CREATE TABLE IF NOT EXISTS qa_kp_scores (
            qa_id INTEGER REFERENCES qa_pairs(id),
            kp_id TEXT REFERENCES knowledge_points(id),
            relevance_score REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (qa_id, kp_id)
        );
    """),
    ("topic_vectors", """
        CREATE TABLE IF NOT EXISTS topic_vectors (
            topic_id TEXT PRIMARY KEY REFERENCES dynamic_topics(topic_id),
            vector BLOB NOT NULL,
            member_kp_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """),
]

# Indexes created after table initialization
SCHEMA_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_qa_dedup ON qa_pairs(question_text, answer_text)",
    "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_dep_prereq ON topic_dependencies(prerequisite)",
    "CREATE INDEX IF NOT EXISTS idx_dep_dep ON topic_dependencies(dependent)",
]

# Schema migrations — applied in version order
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
]
