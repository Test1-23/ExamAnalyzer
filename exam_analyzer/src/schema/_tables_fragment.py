"""Fragment / scoring-point related tables."""

FRAGMENT_TABLES: list[tuple[str, str]] = [
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
]
