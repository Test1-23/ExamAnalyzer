"""Analysis, diagnostics, and calibration tables."""

ANALYSIS_TABLES: list[tuple[str, str]] = [
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
]
