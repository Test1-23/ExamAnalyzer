"""Core business tables: qa_pairs, exam_sessions, chat_history, analysis_checkpoints."""

CORE_TABLES: list[tuple[str, str]] = [
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
    ("analysis_checkpoints", """
        CREATE TABLE IF NOT EXISTS analysis_checkpoints (
            task_name TEXT PRIMARY KEY,
            qa_count_at_run INTEGER,
            completed_at TEXT,
            status TEXT DEFAULT 'pending'
        );
    """),
]
