"""SQLite QA knowledge base + embedding retrieval.

QADatabase: stores question-answer pairs.
QARetriever: embedding-based similarity search over the QA database.
"""

import sqlite3
import os
import threading
import numpy as np
from typing import List, Optional
from collections import defaultdict

from .embedding_cluster import _get_model, MODEL_MAP, _detect_language
from .logger import get_logger

_log = get_logger()

# ---- Tunable constants ----
SQLITE_PARAM_CHUNK = 900     # SQLite max bound parameters = 999
CHANNEL_A_RECALL = 30        # dual-channel: embedding recall size
BEHAVIOR_CHUNK = 400         # dual-channel: behavior score chunking
WEIGHT_EMBEDDING = 0.35      # dual-channel: semantic weight
WEIGHT_TOPIC = 0.35          # dual-channel: topic affiliation weight
WEIGHT_BEHAVIOR = 0.20       # dual-channel: behavior history weight
WEIGHT_KEYWORD = 0.10        # dual-channel: keyword match weight


# ============================================================
# Helpers
# ============================================================

def make_topic_id(topic: str) -> str:
    """Sanitize a topic name into a stable ID: ``topic_<sanitized_name>``."""
    sanitized = topic.replace(" ", "_").replace("/", "_")
    return f"topic_{sanitized}"


# ============================================================
# QADatabase
# ============================================================

class QADatabase:
    """Stores and retrieves question-answer pairs with embedding support."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._write_lock:
                if self._conn is None:
                    os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
                    self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.row_factory = sqlite3.Row
                    self._init_tables()
        return self._conn

    def _init_tables(self):
        self.conn.executescript("""
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

            CREATE TABLE IF NOT EXISTS topic_links (
                src_topic TEXT NOT NULL,
                dst_topic TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (src_topic, dst_topic)
            );

            CREATE TABLE IF NOT EXISTS exam_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_code TEXT NOT NULL,
                season TEXT NOT NULL,
                year INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                UNIQUE(subject_code, season, year, display_name)
            );

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

            CREATE TABLE IF NOT EXISTS student_knowledge_state (
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                state TEXT NOT NULL,
                evidence_count INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (student_id, topic)
            );

            CREATE TABLE IF NOT EXISTS confusion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                trigger_question TEXT,
                confusion_type TEXT,
                resolved BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );

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

            CREATE TABLE IF NOT EXISTS analysis_checkpoints (
                task_name TEXT PRIMARY KEY,
                qa_count_at_run INTEGER,
                completed_at TEXT,
                status TEXT DEFAULT 'pending'
            );

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

            CREATE TABLE IF NOT EXISTS qa_kp_membership (
                qa_id INTEGER REFERENCES qa_pairs(id),
                kp_id TEXT REFERENCES knowledge_points(id),
                membership_strength REAL,
                is_representative BOOLEAN DEFAULT 0,
                PRIMARY KEY (qa_id, kp_id)
            );

            CREATE TABLE IF NOT EXISTS student_trajectory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                kp_id TEXT REFERENCES knowledge_points(id),
                from_state TEXT,
                to_state TEXT,
                trigger TEXT,
                recorded_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS exam_trends (
                kp_id TEXT REFERENCES knowledge_points(id),
                year INTEGER,
                season TEXT,
                occurrence_count INTEGER DEFAULT 0,
                avg_difficulty TEXT,
                trend_summary TEXT,
                PRIMARY KEY (kp_id, year, season)
            );

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

            CREATE TABLE IF NOT EXISTS dimension_baselines (
                dimension TEXT PRIMARY KEY,
                mean REAL,
                median REAL,
                mad REAL,
                sample_count INTEGER,
                last_updated TEXT
            );

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

            CREATE TABLE IF NOT EXISTS correction_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step TEXT,
                anomaly_pattern TEXT,
                correction_action TEXT,
                times_triggered INTEGER DEFAULT 0,
                success_rate REAL,
                last_triggered TEXT
            );

            CREATE TABLE IF NOT EXISTS distillation_cache (
                topic TEXT PRIMARY KEY,
                qa_count INTEGER NOT NULL DEFAULT 0,
                qa_ids_hash TEXT NOT NULL DEFAULT '',
                distilled_content TEXT NOT NULL DEFAULT '',
                distilled_at TEXT DEFAULT (datetime('now'))
            );

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

            -- Phase 1: MS Fragment extraction + behavior data
            CREATE TABLE IF NOT EXISTS ms_fragments (
                point_id TEXT PRIMARY KEY,
                qa_id INTEGER REFERENCES qa_pairs(id),
                point_text TEXT NOT NULL,
                marks INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fragment_help_map (
                fragment_id TEXT REFERENCES ms_fragments(point_id),
                helped_qa_id INTEGER REFERENCES qa_pairs(id),
                help_effect REAL DEFAULT 0,
                help_level TEXT DEFAULT '',
                PRIMARY KEY (fragment_id, helped_qa_id)
            );

            CREATE TABLE IF NOT EXISTS fragment_membership (
                fragment_id TEXT PRIMARY KEY REFERENCES ms_fragments(point_id),
                topic_id TEXT NOT NULL,
                loyalty REAL DEFAULT 0.5,
                joined_at TEXT DEFAULT (datetime('now')),
                previous_topic_id TEXT
            );

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

            -- Phase 5: LLM-driven vector space infrastructure
            CREATE TABLE IF NOT EXISTS fragment_centrality (
                fragment_id TEXT PRIMARY KEY REFERENCES ms_fragments(point_id),
                verification_count INTEGER DEFAULT 0,
                avg_help_score REAL DEFAULT 0,
                topic_coherence REAL DEFAULT 0,
                variance REAL DEFAULT 0,
                centrality_score REAL DEFAULT 0.2,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS kp_vectors (
                kp_id TEXT PRIMARY KEY REFERENCES knowledge_points(id),
                vector BLOB NOT NULL,
                adjustment_count INTEGER DEFAULT 0,
                stability REAL DEFAULT 1.0,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS qa_kp_scores (
                qa_id INTEGER REFERENCES qa_pairs(id),
                kp_id TEXT REFERENCES knowledge_points(id),
                relevance_score REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (qa_id, kp_id)
            );

            CREATE TABLE IF NOT EXISTS topic_vectors (
                topic_id TEXT PRIMARY KEY REFERENCES dynamic_topics(topic_id),
                vector BLOB NOT NULL,
                member_kp_count INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            );
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_dedup ON qa_pairs(question_text, answer_text)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dep_prereq ON topic_dependencies(prerequisite)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dep_dep ON topic_dependencies(dependent)"
        )
        self.conn.commit()
        self._run_migrations()

    def _run_migrations(self):
        """Apply pending schema migrations in version order."""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, "
            "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
        )
        current = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) as v FROM schema_version"
        ).fetchone()["v"]

        migrations = [
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

        for version, description, statements in migrations:
            if version <= current:
                continue
            for stmt in statements:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            self.conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            _log.info(f"DB migration v{version}: {description}")
        self.conn.commit()

    def update_qa_topic(self, qa_id: int, topic: str):
        with self._write_lock:
            self.conn.execute("UPDATE qa_pairs SET topic=? WHERE id=?", (topic, qa_id))
            self.conn.commit()

    def rename_topic(self, new_topic: str, old_topic: str) -> int:
        with self._write_lock:
            rows = self.conn.execute(
                "UPDATE qa_pairs SET topic=? WHERE topic=?", (new_topic, old_topic))
            self.conn.commit()
            return rows.rowcount

    def insert(self, question_text: str, answer_text: str,
               topic: str = "", paper: str = "",
               question_number: str = "",
               parent_question: str = "",
               knowledge_summary: str = "") -> int:
        with self._write_lock:
            return self._insert_locked(question_text, answer_text, topic,
                                       paper, question_number, parent_question, knowledge_summary)

    def _insert_locked(self, question_text, answer_text, topic, paper,
                       question_number, parent_question, knowledge_summary):
        existing = self.conn.execute(
            "SELECT id FROM qa_pairs WHERE question_text = ? AND answer_text = ? LIMIT 1",
            (question_text, answer_text),
        ).fetchone()
        if existing:
            return existing["id"]
        cur = self.conn.execute(
            """INSERT INTO qa_pairs
               (question_text, answer_text, knowledge_summary, topic, paper, question_number, parent_question)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (question_text, answer_text, knowledge_summary, topic, paper, question_number, parent_question),
        )
        self.conn.commit()
        return cur.lastrowid

    def get(self, qa_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM qa_pairs WHERE id=?", (qa_id,)).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM qa_pairs ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        row_map = {}
        CHUNK = SQLITE_PARAM_CHUNK  # SQLite max bound parameters = 999
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT * FROM qa_pairs WHERE id IN ({placeholders})", chunk
            ).fetchall()
            for r in rows:
                row_map[r["id"]] = dict(r)
        return [row_map[i] for i in ids if i in row_map]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM qa_pairs").fetchone()
        return row["cnt"] if row else 0

    def record_attempt(self, qa_id: int, success: bool, reason: str = ""):
        with self._write_lock:
            self._record_attempt_locked(qa_id, success, reason)

    def _record_attempt_locked(self, qa_id, success, reason=""):
        if success:
            self.conn.execute(
                "UPDATE qa_pairs SET success_count=success_count+1, total_attempts=total_attempts+1, "
                "last_failure_reason='' WHERE id=?",
                (qa_id,),
            )
        else:
            self.conn.execute(
                "UPDATE qa_pairs SET total_attempts=total_attempts+1, "
                "last_failure_reason=? WHERE id=?",
                (reason, qa_id),
            )
        self.conn.commit()

    def get_topic_groups(self) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for qa in self.get_all():
            topic = qa.get("topic", "") or "(uncategorized)"
            groups[topic].append(qa)
        return dict(groups)

    def log_api_call(self, stage: str, model: str, paper: str = "",
                     question_number: str = "", latency_ms: int = 0,
                     success: bool = True, output_size: int = 0):
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO api_call_log (stage, model, paper, question_number,
                   latency_ms, success, output_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (stage, model, paper, question_number, latency_ms, int(success), output_size),
            )
            self.conn.commit()

    def log_question_feedback(self, qa_id: int, retrieval_count: int = 0,
                              used_qa_count: int = 0, step0_topic: str = "",
                              round2_topic: str = "", covered_count: int = 0,
                              missed_count: int = 0, missed_text: str = "",
                              miss_categories: str = ""):
        with self._write_lock:
            total = covered_count + missed_count
            ratio = (covered_count / total) if total > 0 else 0.0
            match = 1 if (step0_topic and round2_topic
                          and step0_topic.lower() == round2_topic.lower()) else 0
            self.conn.execute(
                """INSERT INTO question_feedback
                   (qa_id, retrieval_count, used_qa_count, step0_topic, round2_topic,
                    topic_match, covered_count, missed_count, missed_text, coverage_ratio, miss_categories)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (qa_id, retrieval_count, used_qa_count, step0_topic, round2_topic,
                 match, covered_count, missed_count, missed_text, ratio, miss_categories),
            )
            self.conn.commit()

    def get_missed_by_topic(self, topic: str) -> list[str]:
        """Return all missed_text entries for a given topic, non-empty only."""
        rows = self.conn.execute(
            """SELECT f.missed_text FROM question_feedback f
               JOIN qa_pairs q ON f.qa_id = q.id
               WHERE q.topic = ? AND f.missed_text != ''""",
            (topic,),
        ).fetchall()
        missed = []
        for r in rows:
            for line in r["missed_text"].split("\n"):
                line = line.strip()
                if line:
                    missed.append(line)
        return missed

    def get_all_weights(self) -> dict[int, dict]:
        """Compute Beta(1,1) posterior weights with Wilson score lower bound.

        Posterior: Beta(s+1, t-s+1) where s=success_count, t=total_attempts.
        Uses Wilson score interval for the lower bound — far more accurate than
        the Normal (Wald) approximation for small t and extreme proportions.
        """
        rows = self.conn.execute("SELECT id, success_count, total_attempts FROM qa_pairs").fetchall()
        result = {}
        for r in rows:
            s, t = r["success_count"], r["total_attempts"]
            mean = (s + 1) / (t + 2)
            if t > 0:
                # Wilson score lower bound (90% one-sided, z=1.282)
                # Applied to Beta(s+1, t-s+1) posterior parameters
                a, b = s + 1, t - s + 1
                n_post = a + b  # = t + 2
                p_hat = a / n_post
                z = 1.282
                z2 = z * z
                denom = 1.0 + z2 / n_post
                center = (p_hat + z2 / (2.0 * n_post)) / denom
                margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n_post)) / n_post) ** 0.5 / denom
                lb = max(0.0, center - margin)
            else:
                lb = 0.0
            result[r["id"]] = {"mean": round(mean, 3), "lower_bound": round(lb, 3), "total": t}
        return result

    # ============================================================
    # Distillation cache — enables incremental distillation
    # ============================================================

    def get_distillation_cache(self) -> dict[str, str]:
        """Return {topic: distilled_content} for all cached topics."""
        rows = self.conn.execute(
            "SELECT topic, distilled_content FROM distillation_cache"
        ).fetchall()
        return {r["topic"]: r["distilled_content"] for r in rows}

    def get_cached_topic_state(self, topic: str) -> dict | None:
        """Return {qa_count, qa_ids_hash} for a cached topic, or None."""
        row = self.conn.execute(
            "SELECT qa_count, qa_ids_hash FROM distillation_cache WHERE topic=?",
            (topic,),
        ).fetchone()
        return {"qa_count": row["qa_count"], "qa_ids_hash": row["qa_ids_hash"]} if row else None

    def upsert_distillation_cache(self, topic: str, qa_count: int,
                                   qa_ids_hash: str, content: str):
        """Store or update cached distillation for a topic."""
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO distillation_cache
                   (topic, qa_count, qa_ids_hash, distilled_content, distilled_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (topic, qa_count, qa_ids_hash, content),
            )
            self.conn.commit()

    def invalidate_distillation_cache(self, topic: str):
        """Remove a topic from the distillation cache (force re-distill)."""
        with self._write_lock:
            self.conn.execute("DELETE FROM distillation_cache WHERE topic=?", (topic,))
            self.conn.commit()

    # ============================================================
    # Evolution history — tracks KP self-improvement events
    # ============================================================

    def record_evolution(self, kp_id: str, trigger_type: str,
                         trigger_detail: str = "", old_state: str = "",
                         new_state: str = "", outcome: str = "pending"):
        """Record an evolution event for a KP."""
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO evolution_history
                   (kp_id, trigger_type, trigger_detail, old_state, new_state, outcome)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kp_id, trigger_type, trigger_detail, old_state, new_state, outcome),
            )
            self.conn.commit()

    def get_pending_evolutions(self, kp_id: str = None) -> list[dict]:
        """Get pending evolution events, optionally filtered by KP."""
        if kp_id:
            rows = self.conn.execute(
                "SELECT * FROM evolution_history WHERE kp_id=? AND outcome='pending' "
                "ORDER BY created_at", (kp_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM evolution_history WHERE outcome='pending' "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    # ============================================================
    # Phase 1: MS Fragments + Dynamic Topics
    # ============================================================

    def insert_fragment(self, point_id: str, qa_id: int, point_text: str,
                        marks: int = 1) -> str:
        """Insert a single MS scoring point fragment. Returns point_id."""
        with self._write_lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks)
                   VALUES (?, ?, ?, ?)""",
                (point_id, qa_id, point_text, marks),
            )
            self.conn.commit()
        return point_id

    def insert_fragments_batch(self, fragments: list[dict]) -> int:
        """Insert multiple fragments for one QA. Each: {point_id, point_text, marks}.
        Returns count inserted."""
        with self._write_lock:
            count = 0
            for f in fragments:
                self.conn.execute(
                    """INSERT OR IGNORE INTO ms_fragments (point_id, qa_id, point_text, marks)
                       VALUES (?, ?, ?, ?)""",
                    (f["point_id"], f["qa_id"], f["point_text"], f.get("marks", 1)),
                )
                count += 1
            self.conn.commit()
        return count

    def set_fragment_membership(self, fragment_id: str, topic_id: str,
                                 loyalty: float = 0.5):
        """Assign a fragment to a topic with initial loyalty."""
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO fragment_membership
                   (fragment_id, topic_id, loyalty, joined_at, previous_topic_id)
                   VALUES (?, ?, ?, datetime('now'),
                    (SELECT topic_id FROM fragment_membership WHERE fragment_id=?))""",
                (fragment_id, topic_id, loyalty, fragment_id),
            )
            self.conn.commit()

    def record_fragment_help_batch(self, fragment_ids: list[str],
                                    helped_qa_id: int, help_effect: float = 0.0):
        """Record that multiple fragments helped answer a question."""
        with self._write_lock:
            for fid in fragment_ids:
                self.conn.execute(
                    """INSERT OR REPLACE INTO fragment_help_map
                       (fragment_id, helped_qa_id, help_effect)
                       VALUES (?, ?, ?)""",
                    (fid, helped_qa_id, help_effect),
                )
            self.conn.commit()

    def upsert_dynamic_topic(self, topic_id: str, name: str = "",
                               quality: str = "embryonic"):
        """Create or update a dynamic topic."""
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO dynamic_topics
                   (topic_id, name, quality, last_evolved_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (topic_id, name, quality),
            )
            self.conn.commit()

    def update_topic_stats(self, topic_id: str, mass: int, cohesion: float,
                            stability: float):
        """Update mass, cohesion, stability for a topic."""
        with self._write_lock:
            self.conn.execute(
                """UPDATE dynamic_topics SET mass=?, cohesion=?, stability=?,
                   last_evolved_at=datetime('now') WHERE topic_id=?""",
                (mass, cohesion, stability, topic_id),
            )
            self.conn.commit()

    def set_topic_kp(self, topic_id: str, kp_concept: str, kp_detail: str):
        """Set the KP text for a stable topic."""
        with self._write_lock:
            self.conn.execute(
                """UPDATE dynamic_topics SET kp_concept=?, kp_detail=?,
                   quality='stable', last_evolved_at=datetime('now')
                   WHERE topic_id=?""",
                (kp_concept, kp_detail, topic_id),
            )
            self.conn.commit()

    def get_stable_topics(self) -> list[dict]:
        """Return all topics with quality='stable' and their KP text."""
        rows = self.conn.execute(
            "SELECT topic_id, name, kp_concept, kp_detail, mass, cohesion, stability "
            "FROM dynamic_topics WHERE quality='stable'"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_topic_fragments(self, topic_id: str) -> list[str]:
        """Return fragment IDs belonging to a topic."""
        rows = self.conn.execute(
            "SELECT fragment_id FROM fragment_membership WHERE topic_id=?",
            (topic_id,),
        ).fetchall()
        return [r["fragment_id"] for r in rows]

    def get_fragment_help_count(self, fragment_id: str) -> int:
        """Return how many questions a fragment has helped."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM fragment_help_map WHERE fragment_id=?",
            (fragment_id,),
        ).fetchone()
        return row["cnt"] if row else 0

    def get_topic_helped_questions(self, topic_id: str) -> set:
        """Return the set of QA IDs that this topic's fragments helped."""
        rows = self.conn.execute(
            """SELECT DISTINCT fhm.helped_qa_id
               FROM fragment_help_map fhm
               JOIN fragment_membership fm ON fhm.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""",
            (topic_id,),
        ).fetchall()
        return {r["helped_qa_id"] for r in rows}

    # ============================================================
    # Phase 5: Fragment centrality + vector infrastructure
    # ============================================================

    def upsert_fragment_centrality(self, fragment_id: str, centrality_score: float,
                                    avg_help_score: float = 0.0, topic_coherence: float = 0.0,
                                    variance: float = 0.0):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO fragment_centrality
                   (fragment_id, verification_count, avg_help_score, topic_coherence,
                    variance, centrality_score, updated_at)
                   VALUES (?, COALESCE((SELECT verification_count FROM fragment_centrality
                    WHERE fragment_id=?), 0) + 1, ?, ?, ?, ?, datetime('now'))""",
                (fragment_id, fragment_id, avg_help_score, topic_coherence, variance, centrality_score),
            )
            self.conn.commit()

    def get_fragment_centrality(self, fragment_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM fragment_centrality WHERE fragment_id=?", (fragment_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_topic_fragment_centralities(self, topic_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT fc.* FROM fragment_centrality fc
               JOIN fragment_membership fm ON fc.fragment_id = fm.fragment_id
               WHERE fm.topic_id = ?""", (topic_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_kp_vector(self, kp_id: str, vector: np.ndarray):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO kp_vectors
                   (kp_id, vector, adjustment_count, updated_at)
                   VALUES (?, ?, COALESCE((SELECT adjustment_count FROM kp_vectors
                    WHERE kp_id=?), 0) + 1, datetime('now'))""",
                (kp_id, vector.tobytes(), kp_id),
            )
            self.conn.commit()

    def get_kp_vector(self, kp_id: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT vector FROM kp_vectors WHERE kp_id=?", (kp_id,)
        ).fetchone()
        return np.frombuffer(row["vector"]) if row else None

    def get_all_kp_vectors(self) -> dict[str, np.ndarray]:
        rows = self.conn.execute("SELECT kp_id, vector FROM kp_vectors").fetchall()
        return {r["kp_id"]: np.frombuffer(r["vector"]) for r in rows}

    def upsert_qa_kp_score(self, qa_id: int, kp_id: str, relevance_score: float):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO qa_kp_scores (qa_id, kp_id, relevance_score)
                   VALUES (?, ?, ?)""",
                (qa_id, kp_id, relevance_score),
            )
            self.conn.commit()

    def get_qa_kp_scores(self, qa_id: int) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT kp_id, relevance_score FROM qa_kp_scores WHERE qa_id=?", (qa_id,)
        ).fetchall()
        return {r["kp_id"]: r["relevance_score"] for r in rows}

    def upsert_topic_vector(self, topic_id: str, vector: np.ndarray, member_count: int = 0):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO topic_vectors
                   (topic_id, vector, member_kp_count, updated_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (topic_id, vector.tobytes(), member_count),
            )
            self.conn.commit()

    def get_topic_vector(self, topic_id: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT vector FROM topic_vectors WHERE topic_id=?", (topic_id,)
        ).fetchone()
        return np.frombuffer(row["vector"]) if row else None

    def record_fragment_help_with_level(self, fragment_id: str, helped_qa_id: int,
                                         help_effect: float = 0.0, help_level: str = ""):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO fragment_help_map
                   (fragment_id, helped_qa_id, help_effect, help_level)
                   VALUES (?, ?, ?, ?)""",
                (fragment_id, helped_qa_id, help_effect, help_level),
            )
            self.conn.commit()

    def upsert_topic_link(self, src_topic: str, dst_topic: str, count: int = 1):
        """Persist a cross-topic QA reference for See also generation."""
        if not src_topic or not dst_topic or src_topic == dst_topic:
            return
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO topic_links (src_topic, dst_topic, count)
                   VALUES (?, ?, ?)
                   ON CONFLICT(src_topic, dst_topic) DO UPDATE SET count = count + ?""",
                (src_topic, dst_topic, count, count),
            )
            self.conn.commit()

    def get_topic_links(self) -> dict:
        """Load all accumulated cross-topic links as {(src, dst): count}."""
        rows = self.conn.execute(
            "SELECT src_topic, dst_topic, count FROM topic_links"
        ).fetchall()
        return {(r["src_topic"], r["dst_topic"]): r["count"] for r in rows}

    # ---- Chat history ----

    def save_chat_message(self, session_id: str, role: str, content: str, sources: str = ""):
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO chat_history (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
                (session_id, role, content, sources),
            )
            self.conn.commit()

    def get_chat_history(self, session_id: str, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, sources FROM chat_history WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"], "sources": r["sources"]} for r in rows]

    def clear_chat_history(self, session_id: str):
        with self._write_lock:
            self.conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            self.conn.commit()

    # ---- Student memory ----

    def save_student_memory(self, student_id: str, memory_type: str, topic: str, content: str):
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO student_memory (student_id, memory_type, topic, content)
                   VALUES (?, ?, ?, ?)""",
                (student_id, memory_type, topic, content),
            )
            self.conn.commit()

    def get_student_memories(self, student_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT memory_type, topic, content, confidence, created_at
               FROM student_memory WHERE student_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (student_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_student_confusions(self, student_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT topic, trigger_question, confusion_type, resolved FROM confusion_events "
            "WHERE student_id = ? ORDER BY created_at DESC",
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_confusion(self, student_id: str, topic: str, trigger: str, ctype: str):
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO confusion_events (student_id, topic, trigger_question, confusion_type)
                   VALUES (?, ?, ?, ?)""",
                (student_id, topic, trigger, ctype),
            )
            self.conn.commit()

    def upsert_knowledge_state(self, student_id: str, topic: str, state: str):
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO student_knowledge_state (student_id, topic, state, evidence_count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(student_id, topic) DO UPDATE SET
                   state = excluded.state,
                   evidence_count = evidence_count + 1,
                   updated_at = datetime('now')""",
                (student_id, topic, state),
            )
            self.conn.commit()

    def get_knowledge_state(self, student_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT topic, state FROM student_knowledge_state WHERE student_id = ?",
            (student_id,),
        ).fetchall()
        return {r["topic"]: r["state"] for r in rows}

    # ---- Exam stats ----

    def get_exam_stats(self, topic: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT s.year, s.season, COUNT(*) as cnt
               FROM qa_pairs q
               JOIN exam_sessions s ON q.session_id = s.id
               WHERE q.topic = ?
               GROUP BY s.year, s.season
               ORDER BY s.year DESC, s.season""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Topic dependencies ----

    def insert_dependency(self, prerequisite: str, dependent: str,
                          evidence_score: int = 0, evidence_reason: str = "",
                          relationship_type: str = "prerequisite",
                          topic_link_count: int = 0, embedding_cos: float = None,
                          confidence: str = "low", validated_by: str = "flash"):
        with self._write_lock:
            existing = self.conn.execute(
                "SELECT first_seen_at FROM topic_dependencies WHERE prerequisite = ? AND dependent = ?",
                (prerequisite, dependent),
            ).fetchone()
            if existing:
                # Update without touching first_seen_at
                self.conn.execute(
                    """UPDATE topic_dependencies SET
                       evidence_score = ?, evidence_reason = ?,
                       relationship_type = ?, topic_link_count = ?,
                       embedding_cos = ?, confidence = ?,
                       validated_at = datetime('now'),
                       last_validated_at = datetime('now'),
                       validated_by = ?
                       WHERE prerequisite = ? AND dependent = ?""",
                    (evidence_score, evidence_reason, relationship_type,
                     topic_link_count, embedding_cos, confidence, validated_by,
                     prerequisite, dependent),
                )
            else:
                self.conn.execute(
                    """INSERT INTO topic_dependencies
                       (prerequisite, dependent, evidence_score, evidence_reason,
                        relationship_type, topic_link_count, embedding_cos,
                        confidence, validated_at, first_seen_at, last_validated_at, validated_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'), ?)""",
                    (prerequisite, dependent, evidence_score, evidence_reason,
                     relationship_type, topic_link_count, embedding_cos,
                     confidence, validated_by),
                )
            self.conn.commit()

    def get_dependencies(self, confidence: str = None) -> list[dict]:
        if confidence:
            rows = self.conn.execute(
                "SELECT * FROM topic_dependencies WHERE confidence = ? ORDER BY prerequisite",
                (confidence,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM topic_dependencies ORDER BY prerequisite"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_dependency_graph(self) -> dict:
        """Return {topic: {prerequisites: [...], dependents: [...]}} for all topics."""
        rows = self.conn.execute(
            "SELECT prerequisite, dependent, relationship_type, confidence FROM topic_dependencies"
        ).fetchall()
        graph: dict[str, dict] = {}
        for r in rows:
            pre, dep = r["prerequisite"], r["dependent"]
            for t in (pre, dep):
                if t not in graph:
                    graph[t] = {"prerequisites": [], "dependents": []}
            graph[dep]["prerequisites"].append({
                "topic": pre, "type": r["relationship_type"], "confidence": r["confidence"]
            })
            graph[pre]["dependents"].append({
                "topic": dep, "type": r["relationship_type"], "confidence": r["confidence"]
            })
        return graph

    def get_direct_prerequisites(self, topic: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT prerequisite, evidence_score, confidence, relationship_type
               FROM topic_dependencies WHERE dependent = ?""",
            (topic,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_transitive_prerequisites(self, topic: str, max_depth: int = 5) -> list[str]:
        """BFS to find all transitive prerequisites of a topic."""
        seen = set()
        frontier = [topic]
        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            for t in frontier:
                rows = self.conn.execute(
                    "SELECT prerequisite FROM topic_dependencies WHERE dependent = ?",
                    (t,),
                ).fetchall()
                for r in rows:
                    pre = r["prerequisite"]
                    if pre not in seen:
                        seen.add(pre)
                        next_frontier.append(pre)
            frontier = next_frontier
        return list(seen)

    # ---- Command verb patterns ----

    def upsert_verb_pattern(self, verb: str, sample_count: int = 0,
                            avg_answer_length: float = None,
                            median_answer_length: float = None,
                            bullet_ratio: float = None,
                            avg_bullet_count: float = None,
                            avg_miss_rate: float = None,
                            common_missed_patterns: str = "",
                            pattern_summary: str = "",
                            topic_specific_patterns: str = "",
                            verb_family: str = ""):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO command_verb_patterns
                   (verb, sample_count, avg_answer_length, median_answer_length,
                    bullet_ratio, avg_bullet_count, avg_miss_rate,
                    common_missed_patterns, pattern_summary,
                    topic_specific_patterns, verb_family, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (verb, sample_count, avg_answer_length, median_answer_length,
                 bullet_ratio, avg_bullet_count, avg_miss_rate,
                 common_missed_patterns, pattern_summary,
                 topic_specific_patterns, verb_family),
            )
            self.conn.commit()

    def get_verb_patterns(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM command_verb_patterns ORDER BY sample_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_verb_for_qa(self, qa_id: int) -> dict:
        row = self.conn.execute(
            "SELECT command_verb, command_verb_secondary, command_verb_inferred FROM qa_pairs WHERE id = ?",
            (qa_id,),
        ).fetchone()
        return dict(row) if row else {}

    # ---- Topic difficulty ----

    def upsert_topic_difficulty(self, topic: str, qa_count: int = 0,
                                basic_count: int = 0, intermediate_count: int = 0,
                                advanced_count: int = 0, mode_difficulty: str = "",
                                avg_miss_rate: float = None,
                                difficulty_spread: bool = False,
                                assessment_method: str = "hybrid"):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO topic_difficulty
                   (topic, qa_count, basic_count, intermediate_count, advanced_count,
                    mode_difficulty, avg_miss_rate, difficulty_spread,
                    assessed_at, assessment_method)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)""",
                (topic, qa_count, basic_count, intermediate_count, advanced_count,
                 mode_difficulty, avg_miss_rate, int(difficulty_spread),
                 assessment_method),
            )
            self.conn.commit()

    def get_topic_difficulty(self, topic: str = None) -> list[dict]:
        if topic:
            rows = self.conn.execute(
                "SELECT * FROM topic_difficulty WHERE topic = ?", (topic,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM topic_difficulty ORDER BY mode_difficulty, topic"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_qa_difficulty(self, qa_id: int) -> str:
        row = self.conn.execute(
            "SELECT difficulty_estimate FROM qa_pairs WHERE id = ?", (qa_id,)
        ).fetchone()
        return row["difficulty_estimate"] if row else ""

    def get_effective_miss_rate(self, qa_id: int) -> float:
        """Return effective miss rate using only knowledge_gap + insufficient_detail misses.
        Excludes misinterpretation and retrieval_quality from the difficulty signal.
        Returns None if no feedback data available."""
        row = self.conn.execute(
            "SELECT covered_count, missed_count, miss_categories FROM question_feedback WHERE qa_id = ?",
            (qa_id,),
        ).fetchone()
        if not row:
            return None
        total = row["covered_count"] + row["missed_count"]
        if total == 0:
            return None
        cats_json = row["miss_categories"] or ""
        if cats_json:
            try:
                import json
                cats = json.loads(cats_json)
                effective = cats.get("knowledge_gap", 0) + cats.get("insufficient_detail", 0) * 0.5
                return effective / total
            except (json.JSONDecodeError, TypeError):
                pass
        return row["missed_count"] / total

    # ---- Analysis checkpoints ----

    def checkpoint(self, task_name: str, qa_count: int, status: str = "completed"):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO analysis_checkpoints
                   (task_name, qa_count_at_run, completed_at, status)
                   VALUES (?, ?, datetime('now'), ?)""",
                (task_name, qa_count, status),
            )
            self.conn.commit()

    def get_checkpoint(self, task_name: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM analysis_checkpoints WHERE task_name = ?", (task_name,)
        ).fetchone()
        return dict(row) if row else {}

    def clear_checkpoint(self, task_name: str):
        with self._write_lock:
            self.conn.execute(
                "DELETE FROM analysis_checkpoints WHERE task_name = ?", (task_name,)
            )
            self.conn.commit()

    # ---- Topic vectors helper (for dependency candidate generation) ----

    def get_topic_answer_texts(self) -> dict[str, str]:
        """Return {topic: concatenated_answer_texts} for dependency embedding.
        Uses answer_text (ground truth), not knowledge_summary (Flash-generated)."""
        rows = self.conn.execute(
            "SELECT topic, answer_text FROM qa_pairs WHERE topic != '' AND topic != '(uncategorized)'"
        ).fetchall()
        texts: dict[str, list[str]] = {}
        for r in rows:
            texts.setdefault(r["topic"], []).append(r["answer_text"])
        return {t: " ".join(parts)[:2000] for t, parts in texts.items()}

    # ---- Knowledge Points (KP graph) ----

    def upsert_kp(self, kp_id: str, name: str = "", description: str = "",
                  cluster_id: int = None, centroid_vector: bytes = None,
                  core_concept: str = "", core_detail: str = "",
                  variations: str = "", scoring_pattern: str = "",
                  typical_marks: float = None, cohesion: float = None,
                  evidence_count: int = 0, quality: str = "draft",
                  challenge_history: str = ""):
        with self._write_lock:
            existing = self.conn.execute(
                "SELECT id FROM knowledge_points WHERE id = ?", (kp_id,)
            ).fetchone()
            if existing:
                self.conn.execute(
                    """UPDATE knowledge_points SET name=?, description=?,
                       cluster_id=?, centroid_vector=?, core_concept=?,
                       core_detail=?, variations=?, scoring_pattern=?,
                       typical_marks=?, cohesion=?, evidence_count=?,
                       quality=?, challenge_history=?,
                       last_validated_at=datetime('now')
                       WHERE id=?""",
                    (name, description, cluster_id, centroid_vector,
                     core_concept, core_detail, variations, scoring_pattern,
                     typical_marks, cohesion, evidence_count, quality,
                     challenge_history, kp_id),
                )
            else:
                self.conn.execute(
                    """INSERT INTO knowledge_points
                       (id, name, description, cluster_id, centroid_vector,
                        core_concept, core_detail, variations, scoring_pattern,
                        typical_marks, cohesion, evidence_count, quality,
                        challenge_history)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (kp_id, name, description, cluster_id, centroid_vector,
                     core_concept, core_detail, variations, scoring_pattern,
                     typical_marks, cohesion, evidence_count, quality,
                     challenge_history),
                )
            self.conn.commit()

    def get_all_kps(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, description, core_concept, core_detail, "
            "cohesion, evidence_count, quality FROM knowledge_points "
            "ORDER BY evidence_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_kp_by_id(self, kp_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM knowledge_points WHERE id = ?", (kp_id,)
        ).fetchone()
        return dict(row) if row else {}

    def get_kp_representative_qas(self, kp_id: str, limit: int = 3) -> list[dict]:
        rows = self.conn.execute(
            """SELECT q.* FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ? AND m.is_representative = 1
               LIMIT ?""",
            (kp_id, limit),
        ).fetchall()
        if not rows:
            rows = self.conn.execute(
                """SELECT q.* FROM qa_pairs q
                   JOIN qa_kp_membership m ON q.id = m.qa_id
                   WHERE m.kp_id = ?
                   ORDER BY m.membership_strength DESC LIMIT ?""",
                (kp_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- KP Edges ----

    def upsert_kp_edge(self, source_kp: str, target_kp: str, edge_type: str,
                       retrieval_weight: float = 0, semantic_weight: float = 0,
                       sequential_weight: float = 0, learning_path_weight: float = 0,
                       combined_strength: float = 0, confidence: str = "low"):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO kp_edges
                   (source_kp, target_kp, edge_type, retrieval_weight,
                    semantic_weight, sequential_weight, learning_path_weight,
                    combined_strength, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source_kp, target_kp, edge_type, retrieval_weight,
                 semantic_weight, sequential_weight, learning_path_weight,
                 combined_strength, confidence),
            )
            self.conn.commit()

    def get_kp_edges(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self.conn.execute(
                "SELECT * FROM kp_edges WHERE source_kp = ? OR target_kp = ?",
                (kp_id, kp_id),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM kp_edges").fetchall()
        return [dict(r) for r in rows]

    def get_kp_graph(self) -> dict:
        """Return {kp_id: {prerequisites: [...], dependents: [...]}}."""
        rows = self.conn.execute(
            "SELECT source_kp, target_kp, edge_type, confidence FROM kp_edges"
        ).fetchall()
        graph: dict[str, dict] = {}
        for r in rows:
            s, t = r["source_kp"], r["target_kp"]
            for k in (s, t):
                if k not in graph:
                    graph[k] = {"prerequisites": [], "dependents": []}
            if r["edge_type"] in ("prerequisite", "corequisite"):
                graph[t]["prerequisites"].append(
                    {"kp": s, "type": r["edge_type"], "confidence": r["confidence"]}
                )
                graph[s]["dependents"].append(
                    {"kp": t, "type": r["edge_type"], "confidence": r["confidence"]}
                )
        return graph

    # ---- QA-KP Membership ----

    def set_qa_kp_membership(self, qa_id: int, kp_id: str,
                             membership_strength: float = 1.0,
                             is_representative: bool = False):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO qa_kp_membership
                   (qa_id, kp_id, membership_strength, is_representative)
                   VALUES (?, ?, ?, ?)""",
                (qa_id, kp_id, membership_strength, int(is_representative)),
            )
            self.conn.commit()

    def get_kp_qas(self, kp_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT q.*, m.membership_strength, m.is_representative
               FROM qa_pairs q
               JOIN qa_kp_membership m ON q.id = m.qa_id
               WHERE m.kp_id = ?""",
            (kp_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Student Trajectory ----

    def record_trajectory(self, student_id: str, kp_id: str,
                          from_state: str, to_state: str, trigger: str = ""):
        with self._write_lock:
            self.conn.execute(
                """INSERT INTO student_trajectory
                   (student_id, kp_id, from_state, to_state, trigger)
                   VALUES (?, ?, ?, ?, ?)""",
                (student_id, kp_id, from_state, to_state, trigger),
            )
            self.conn.commit()

    def get_student_trajectory(self, student_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM student_trajectory
               WHERE student_id = ? ORDER BY recorded_at DESC LIMIT ?""",
            (student_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Exam Trends ----

    def upsert_exam_trend(self, kp_id: str, year: int, season: str,
                          occurrence_count: int = 0, avg_difficulty: str = "",
                          trend_summary: str = ""):
        with self._write_lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO exam_trends
                   (kp_id, year, season, occurrence_count, avg_difficulty, trend_summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kp_id, year, season, occurrence_count, avg_difficulty, trend_summary),
            )
            self.conn.commit()

    def get_exam_trends(self, kp_id: str = None) -> list[dict]:
        if kp_id:
            rows = self.conn.execute(
                "SELECT * FROM exam_trends WHERE kp_id = ? ORDER BY year, season",
                (kp_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM exam_trends ORDER BY kp_id, year, season"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        with self._write_lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# ============================================================
# QARetriever
# ============================================================

def log_schema_status(db, debug_cb=None):
    """Diagnostic: log DB schema state for test verification."""
    def _d(msg):
        if debug_cb:
            debug_cb(f"[DB] {msg}")
    try:
        tables = [r["name"] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        new_tables = ["topic_dependencies", "command_verb_patterns", "topic_difficulty",
                      "analysis_checkpoints", "knowledge_points", "kp_edges",
                      "qa_kp_membership", "student_trajectory", "exam_trends"]
        present = [t for t in new_tables if t in tables]
        missing = [t for t in new_tables if t not in tables]
        _d(f"Tables: {len(tables)} total, {len(present)} new tables present"
           + (f", MISSING: {missing}" if missing else ""))

        qa_cols = [c["name"] for c in db.conn.execute("PRAGMA table_info(qa_pairs)").fetchall()]
        new_qa_cols = ["command_verb", "command_verb_secondary", "command_verb_inferred",
                       "difficulty_estimate", "is_representative", "is_cross_topic", "session_id"]
        qa_present = [c for c in new_qa_cols if c in qa_cols]
        qa_missing = [c for c in new_qa_cols if c not in qa_cols]
        _d(f"qa_pairs: {len(qa_cols)} columns"
           + (f", MISSING: {qa_missing}" if qa_missing else " (all expected present)"))

        fb_cols = [c["name"] for c in db.conn.execute("PRAGMA table_info(question_feedback)").fetchall()]
        _d(f"question_feedback has miss_categories: {'miss_categories' in fb_cols}")
    except Exception as e:
        _d(f"Schema status check failed: {e}")


class QARetriever:
    """Retrieve similar past questions from the QA database via embedding similarity."""

    def __init__(self, db: QADatabase):
        self._db = db
        self._embeddings: Optional[np.ndarray] = None
        self._id_map: dict[int, int] = {}
        self._embed_model_name: Optional[str] = None

    def _ensure_embeddings(self):
        qas = self._db.get_all()
        if not qas:
            self._embeddings = np.empty((0, 384))
            self._id_map = {}
            self._embed_model_name = None
            return
        # Use raw QA text for corpus embeddings (ground truth, no Flash distortion).
        # knowledge_summary is only used as fallback if QA text is unavailable.
        texts = [
            (qa["question_text"] + " " + qa["answer_text"])
            if (qa.get("question_text") or qa.get("answer_text"))
            else (qa.get("knowledge_summary", ""))
            for qa in qas
        ]
        lang = _detect_language(texts)
        self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        self._embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        self._id_map = {qa["id"]: i for i, qa in enumerate(qas)}

    def search(self, query: str, threshold: float = 0.5,
               min_k: int = 3, max_cap: int = 15) -> List[dict]:
        if self._embeddings is None or len(self._embeddings) == 0:
            self._ensure_embeddings()
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Use the SAME model as the corpus to keep vectors compatible
        if self._embed_model_name is None:
            lang = _detect_language([query])
            self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, query_vec)

        n = min(len(scores), max_cap)
        if len(scores) <= n:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, n)[:n]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        id_list = list(self._id_map.keys())
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold or len(results) < min_k:
                qa = self._db.get(id_list[idx])
                if qa:
                    qa["_score"] = score
                    results.append(qa)

        _log.debug(f"Retrieval: qlen={len(query)}, results={len(results)}, threshold={threshold}")
        return results

    def search_dual_channel(self, query: str, threshold: float = 0.5,
                             min_k: int = 3, max_cap: int = 15,
                             query_topic: str = "", query_kp_scores: dict = None) -> List[dict]:
        """Layer 2 dual-channel retrieval: embedding + structure + behavior.

        Channel A (semantic): embedding top-30
        Channel B (structure): topic affiliation + behavioral graph walk (walk-1)
        Mixed ranking: 0.35×embedding + 0.35×topic_match + 0.20×behavior + 0.10×keyword
        """
        # Ensure embeddings are ready
        if self._embeddings is None or len(self._embeddings) == 0:
            self._ensure_embeddings()
        if self._embeddings is None or len(self._embeddings) == 0:
            return self.search(query, threshold, min_k, max_cap)

        # Channel A: embedding top-30 (high recall)
        if self._embed_model_name is None:
            lang = _detect_language([query])
            self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, query_vec)

        channel_a_size = min(len(scores), 30)
        top_a = np.argpartition(-scores, channel_a_size - 1)[:channel_a_size]
        top_a = top_a[np.argsort(-scores[top_a])]

        id_list = list(self._id_map.keys())
        channel_a_ids = {id_list[idx] for idx in top_a}
        channel_a_qas = [self._db.get(id_list[idx]) for idx in top_a]
        channel_a_qas = [q for q in channel_a_qas if q]

        # Channel B: structure — topic affiliation + graph walk
        channel_b_ids = set()
        keyword_query = set(query.lower().split())

        # B1: Topic affiliation — QAs in the same or adjacent topics
        if query_topic:
            topic_rows = self._db.conn.execute(
                "SELECT id, topic FROM qa_pairs WHERE topic=? AND topic!='' LIMIT 20",
                (query_topic,)
            ).fetchall()
            channel_b_ids.update(r["id"] for r in topic_rows)

            # Adjacent topics via topic_links
            adj_rows = self._db.conn.execute(
                "SELECT DISTINCT dst_topic FROM topic_links WHERE src_topic=? UNION "
                "SELECT DISTINCT src_topic FROM topic_links WHERE dst_topic=?",
                (query_topic, query_topic)
            ).fetchall()
            for adj in adj_rows[:5]:
                adj_rows2 = self._db.conn.execute(
                    "SELECT id FROM qa_pairs WHERE topic=? LIMIT 10",
                    (adj["dst_topic"],)
                ).fetchall()
                channel_b_ids.update(r["id"] for r in adj_rows2)

        # B2: Graph walk — QAs whose fragments helped same questions
        walk_source = list(channel_a_ids)[:15] if channel_a_ids else []  # single conversion, reused
        if walk_source:
            placeholders_a = ",".join("?" * len(walk_source))
            walk_rows = self._db.conn.execute(
                f"SELECT DISTINCT f2.qa_id FROM ("
                f"SELECT DISTINCT fhm1.helped_qa_id FROM fragment_help_map fhm1 "
                f"JOIN ms_fragments mf1 ON fhm1.fragment_id = mf1.point_id "
                f"WHERE mf1.qa_id IN ({placeholders_a})"
                f") shared_helps "
                f"JOIN fragment_help_map fhm2 ON shared_helps.helped_qa_id = fhm2.helped_qa_id "
                f"JOIN ms_fragments f2 ON fhm2.fragment_id = f2.point_id "
                f"LIMIT 30",
                walk_source
            ).fetchall()
            channel_b_ids.update(r["qa_id"] for r in walk_rows)

        # Remove Channel A overlap
        channel_b_ids -= channel_a_ids

        # Build candidate pool
        candidates = []
        for idx in top_a:
            qa_id = id_list[idx]
            qa = self._db.get(qa_id)
            if qa:
                qa["_score"] = float(scores[idx])
                qa["_channel"] = "embedding"
                candidates.append(qa)

        for qa_id in channel_b_ids:
            qa = self._db.get(qa_id)
            if qa:
                qa["_score"] = 0.0
                qa["_channel"] = "structure"
                candidates.append(qa)

        # Pre-load topic adjacency for fast lookup (avoid O(N) DB queries)
        candidate_topics = {qa.get("topic", "") for qa in candidates if qa.get("topic")}
        adjacency_map = {}
        if query_topic and candidate_topics:
            for ct in candidate_topics:
                adj_row = self._db.conn.execute(
                    "SELECT COUNT(*) as cnt FROM topic_links "
                    "WHERE (src_topic=? AND dst_topic=?) OR (src_topic=? AND dst_topic=?)",
                    (query_topic, ct, ct, query_topic)
                ).fetchone()
                if adj_row and adj_row["cnt"] > 0:
                    adjacency_map[ct] = True

        # Pre-load helped question set from Channel A (reuse walk_source, one query)
        helped_qa_set = set()
        if walk_source:
            bh_all_rows = self._db.conn.execute(
                f"SELECT DISTINCT helped_qa_id FROM fragment_help_map fhm2 "
                f"JOIN ms_fragments mf2 ON fhm2.fragment_id = mf2.point_id "
                f"WHERE mf2.qa_id IN ({','.join('?' * len(walk_source))})",
                walk_source
            ).fetchall()
            helped_qa_set = {r["helped_qa_id"] for r in bh_all_rows}

        # Behavior scores: chunked batch to avoid SQLite 999-param limit
        candidate_qa_ids = [qa["id"] for qa in candidates]
        behavior_scores = {}
        if helped_qa_set and candidate_qa_ids:
            helped_list = list(helped_qa_set)
            CHUNK = BEHAVIOR_CHUNK   # leave room for candidate_qa_ids chunk
            for i in range(0, len(candidate_qa_ids), CHUNK):
                c_chunk = candidate_qa_ids[i:i + CHUNK]
                for j in range(0, len(helped_list), CHUNK):
                    h_chunk = helped_list[j:j + CHUNK]
                    bh_batch_rows = self._db.conn.execute(
                        f"SELECT mf.qa_id, COUNT(*) as cnt FROM fragment_help_map fhm "
                        f"JOIN ms_fragments mf ON fhm.fragment_id = mf.point_id "
                        f"WHERE mf.qa_id IN ({','.join('?' * len(c_chunk))})"
                        f" AND fhm.helped_qa_id IN ({','.join('?' * len(h_chunk))})"
                        f" GROUP BY mf.qa_id",
                        c_chunk + h_chunk
                    ).fetchall()
                    for r in bh_batch_rows:
                        behavior_scores[r["qa_id"]] = min(1.0, r["cnt"] / 10.0)

        # Mixed ranking (pre-loaded data, zero DB queries)
        for qa in candidates:
            emb_score = qa.get("_score", 0.0)
            qa_topic = qa.get("topic", "")

            topic_score = 0.0
            if query_topic and qa_topic:
                if qa_topic == query_topic:
                    topic_score = 1.0
                elif qa_topic in adjacency_map:
                    topic_score = 0.5

            behavior_score = behavior_scores.get(qa["id"], 0.0)

            qa_text = (qa.get("question_text", "") + " " + qa.get("answer_text", "")).lower()
            qa_keywords = set(qa_text.split())
            kw_jaccard = len(keyword_query & qa_keywords) / max(len(keyword_query | qa_keywords), 1)

            composite = (WEIGHT_EMBEDDING * emb_score + WEIGHT_TOPIC * topic_score
                         + WEIGHT_BEHAVIOR * behavior_score + WEIGHT_KEYWORD * kw_jaccard)
            qa["_score"] = composite

        # Sort by composite score and return top-k
        candidates.sort(key=lambda q: q["_score"], reverse=True)
        results = []
        for qa in candidates:
            if qa["_score"] >= threshold or len(results) < min_k:
                results.append(qa)

        _log.debug(f"Dual-channel retrieval: qlen={len(query)}, "
                   f"chA={len(channel_a_ids)}, chB={len(channel_b_ids)}, results={len(results)}")
        return results[:max_cap]

    def add_qa(self, qa_id: int, summary_text: str):
        """Add a new QA vector to the index. Uses raw QA text for embedding
        (consistent with _ensure_embeddings), falls back to summary_text."""
        if qa_id in self._id_map:
            return
        # Fetch QA for raw text; fall back to summary_text if DB unavailable
        qa = self._db.get(qa_id)
        if qa and (qa.get("question_text") or qa.get("answer_text")):
            encode_text = qa["question_text"] + " " + qa["answer_text"]
        else:
            encode_text = summary_text or ""
        # Use the same model as the corpus to keep vectors compatible
        if self._embed_model_name is None:
            lang = _detect_language([encode_text])
            self._embed_model_name = MODEL_MAP[lang]
        model = _get_model(self._embed_model_name)
        new_vec = model.encode([encode_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = new_vec.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, new_vec])
        self._id_map[qa_id] = len(self._embeddings) - 1

    def rebuild(self):
        self._embeddings = None
        self._id_map = {}
        self._embed_model_name = None
        self._ensure_embeddings()

    def count(self) -> int:
        """Return the number of QAs in the underlying database."""
        return self._db.count()

    def clear_chat_history(self, session_id: str):
        """Proxy to clear chat history in underlying DB."""
        self._db.clear_chat_history(session_id)
