"""Topic-related tables."""

TOPIC_TABLES: list[tuple[str, str]] = [
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
    ("topic_links", """
        CREATE TABLE IF NOT EXISTS topic_links (
            src_topic TEXT NOT NULL,
            dst_topic TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            PRIMARY KEY (src_topic, dst_topic)
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
    ("topic_vectors", """
        CREATE TABLE IF NOT EXISTS topic_vectors (
            topic_id TEXT PRIMARY KEY REFERENCES dynamic_topics(topic_id),
            vector BLOB NOT NULL,
            member_kp_count INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """),
]
