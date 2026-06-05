"""Student-related tables."""

STUDENT_TABLES: list[tuple[str, str]] = [
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
]
