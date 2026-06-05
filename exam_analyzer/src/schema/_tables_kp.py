"""Knowledge-point related tables."""

KP_TABLES: list[tuple[str, str]] = [
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
    ("kp_vectors", """
        CREATE TABLE IF NOT EXISTS kp_vectors (
            kp_id TEXT PRIMARY KEY REFERENCES knowledge_points(id),
            vector BLOB NOT NULL,
            feature_activations BLOB,
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
            encoder_version TEXT DEFAULT 'embedding',
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (qa_id, kp_id)
        );
    """),
]
