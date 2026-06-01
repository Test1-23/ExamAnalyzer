import json
import os
import statistics
import numpy as np
from collections import defaultdict

from ..deepseek_client import call_flash, create_client
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, TOPIC_EMBED_MODEL, detect_content_lang
from ..models import KPSpec
from ..constants import (
    SQLITE_PARAM_CHUNK, TOPIC_MERGE_COS_THRESHOLD, TOPIC_MERGE_AMBIGUOUS_THRESHOLD,
)
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()

# Phase 3: Topic evolution — split, merge, dissolve
# ═══════════════════════════════════════════════════════════════

def _detect_topic_splits(db: QADatabase, debug_cb=None) -> int:
    """Detect topics with diverging behavioral subgroups and split them."""
    topics = db.conn.execute(
        "SELECT topic_id, mass FROM dynamic_topics WHERE mass >= 6 AND quality != 'dissolved'"
    ).fetchall()
    if not topics:
        return 0

    splits = 0
    for t in topics:
        topic_id = t["topic_id"]
        frags = db.topic.get_fragments(topic_id)
        if len(frags) < 6:
            continue

        # Batch-load all fragment help data for this topic (chunked for SQLite 999-param limit)
        from .constants import SQLITE_PARAM_CHUNK
        frag_helps = {}
        for i in range(0, len(frags), SQLITE_PARAM_CHUNK):
            chunk = frags[i:i + SQLITE_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            help_rows = db.conn.execute(
                f"SELECT fragment_id, helped_qa_id FROM fragment_help_map "
                f"WHERE fragment_id IN ({placeholders})",
                chunk
            ).fetchall()
            for r in help_rows:
                frag_helps.setdefault(r["fragment_id"], set()).add(r["helped_qa_id"])

        # Build pairwise behavioral similarity matrix
        n = len(frags)
        sim_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                fi_set = frag_helps.get(frags[i], set())
                fj_set = frag_helps.get(frags[j], set())
                if fi_set and fj_set:
                    sim = len(fi_set & fj_set) / len(fi_set | fj_set)
                else:
                    sim = 0.0
                sim_matrix[(i, j)] = sim

        # Find two subgroups with high internal + low cross similarity
        # Greedy: pick first pair with low cross-sim, build groups around them
        best_i, best_j = -1, -1
        best_diff = 0.0
        for (i, j), cross in sim_matrix.items():
            # Compute avg intra vs cross for this candidate split
            intra_i = sum(sim_matrix.get((min(i, k), max(i, k)), 0)
                         for k in range(n) if k != i and k != j)
            intra_j = sum(sim_matrix.get((min(j, k), max(j, k)), 0)
                         for k in range(n) if k != i and k != j)
            diff = max(0.0, (intra_i + intra_j) / max(n - 2, 1)) - cross
            if diff > best_diff:
                best_diff = diff
                best_i, best_j = i, j

        if best_diff < 0.3:
            continue

        # Build subgroups
        s1 = {frags[best_i]}
        s2 = {frags[best_j]}
        for k in range(n):
            if k == best_i or k == best_j:
                continue
            sim_to_s1 = sum(sim_matrix.get((min(k, x), max(k, x)), 0)
                           for x in [best_i] + [idx for idx, f in enumerate(frags) if f in s1])
            sim_to_s2 = sum(sim_matrix.get((min(k, x), max(k, x)), 0)
                           for x in [best_j] + [idx for idx, f in enumerate(frags) if f in s2])
            if sim_to_s1 >= sim_to_s2:
                s1.add(frags[k])
            else:
                s2.add(frags[k])

        if len(s1) < 3 or len(s2) < 3:
            continue

        # Check internal cohesion of each subgroup
        s1_sims = [sim_matrix.get((min(i, j), max(i, j)), 0)
                   for i in range(n) for j in range(i + 1, n)
                   if frags[i] in s1 and frags[j] in s1]
        s2_sims = [sim_matrix.get((min(i, j), max(i, j)), 0)
                   for i in range(n) for j in range(i + 1, n)
                   if frags[i] in s2 and frags[j] in s2]
        s1_avg = sum(s1_sims) / len(s1_sims) if s1_sims else 0
        s2_avg = sum(s2_sims) / len(s2_sims) if s2_sims else 0

        if s1_avg < 0.4 or s2_avg < 0.4:
            continue

        # Execute split
        new_id_a = f"{topic_id}_a"
        new_id_b = f"{topic_id}_b"
        old_name = db.conn.execute(
            "SELECT name FROM dynamic_topics WHERE topic_id=?", (topic_id,)
        ).fetchone()
        old_name = old_name["name"] if old_name else topic_id

        with db.transaction():
            db.topic.upsert(new_id_a, name=f"{old_name} (A)", quality="embryonic")
            db.topic.upsert(new_id_b, name=f"{old_name} (B)", quality="embryonic")
            for fid in s1:
                db.topic.set_fragment_membership(fid, new_id_a, loyalty=0.5)
            for fid in s2:
                db.topic.set_fragment_membership(fid, new_id_b, loyalty=0.5)

            db.conn.execute(
                """UPDATE dynamic_topics SET quality='dissolved',
                   child_topics=?, last_evolved_at=datetime('now')
                   WHERE topic_id=?""",
                (json.dumps([new_id_a, new_id_b]), topic_id),
            )
        splits += 1
        if debug_cb:
            from ..error_utils import log_info
            log_info(debug_cb, "Split", f"{topic_id} -> [{new_id_a}]({len(s1)}) + "
                     f"[{new_id_b}]({len(s2)})")

    return splits


def _detect_topic_merges(db: QADatabase, debug_cb=None) -> int:
    """Detect topic pairs with high behavioral overlap and merge them."""
    from .queries import DiagnosticQueries
    dq = DiagnosticQueries(db)
    topics = dq.get_active_topics()
    if len(topics) < 2:
        return 0

    topic_ids = [t["topic_id"] for t in topics]

    # Pre-load fragment help data for affinity computation
    frag_helps = dq.load_fragment_helps()
    topic_helps = {tid: db.fragment.get_topic_helped_questions(tid) for tid in topic_ids}

    merges = 0
    merged_set = set()

    for i in range(len(topic_ids)):
        if topic_ids[i] in merged_set:
            continue
        for j in range(i + 1, len(topic_ids)):
            if topic_ids[j] in merged_set:
                continue
            a, b = topic_ids[i], topic_ids[j]

            # Behavioral overlap: helped question sets
            a_helped = db.fragment.get_topic_helped_questions(a)
            b_helped = db.fragment.get_topic_helped_questions(b)
            if not a_helped or not b_helped:
                continue

            overlap = len(a_helped & b_helped)
            union = len(a_helped | b_helped)
            jaccard = overlap / union if union > 0 else 0

            if jaccard < 0.5:
                continue

            # Bidirectional fragment affinity
            a_frags = db.topic.get_fragments(a)
            b_frags = db.topic.get_fragments(b)
            a_aff = sum(_compute_affinity(db, fid, b, frag_helps, topic_helps)
                        for fid in a_frags)
            b_aff = sum(_compute_affinity(db, fid, a, frag_helps, topic_helps)
                        for fid in b_frags)
            avg_a = a_aff / len(a_frags) if a_frags else 0
            avg_b = b_aff / len(b_frags) if b_frags else 0

            if avg_a < 0.4 or avg_b < 0.4:
                continue

            # Execute merge
            new_id = f"{a}_m{b}"
            name_a = db.conn.execute(
                "SELECT name FROM dynamic_topics WHERE topic_id=?", (a,)
            ).fetchone()
            name_b = db.conn.execute(
                "SELECT name FROM dynamic_topics WHERE topic_id=?", (b,)
            ).fetchone()
            merged_name = f"{(name_a['name'] if name_a else a)} + {(name_b['name'] if name_b else b)}"

            with db.transaction():
                db.topic.upsert(new_id, name=merged_name, quality="embryonic")
                for fid in a_frags + b_frags:
                    db.topic.set_fragment_membership(fid, new_id, loyalty=0.5)

                db.conn.execute(
                    """UPDATE dynamic_topics SET quality='dissolved',
                       last_evolved_at=datetime('now') WHERE topic_id IN (?, ?)""",
                    (a, b),
                )
                db.conn.execute(
                    "UPDATE dynamic_topics SET merged_from=? WHERE topic_id=?",
                    (json.dumps([a, b]), new_id),
                )
            merged_set.add(a)
            merged_set.add(b)
            merges += 1
            if debug_cb:
                from ..error_utils import log_info
                log_info(debug_cb, "Merge", f"{a} + {b} -> {new_id} (jaccard={jaccard:.2f})")

    return merges


def _process_dissolved_topics(db: QADatabase, debug_cb=None) -> int:
    """Redistribute orphan fragments from dissolved topics."""
    dissolved = db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality='dissolved'"
    ).fetchall()
    if not dissolved:
        return 0

    all_topics = [r["topic_id"] for r in db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality != 'dissolved'"
    ).fetchall()]

    redistributed = 0
    for row in dissolved:
        topic_id = row["topic_id"]
        frags = db.topic.get_fragments(topic_id)
        if not frags:
            continue

        for fid in frags:
            best_aff = 0.0
            best_topic = None
            for other in all_topics:
                aff = _compute_affinity(db, fid, other)
                if aff > best_aff:
                    best_aff = aff
                    best_topic = other

            if best_topic and best_aff > 0.1:
                db.topic.set_fragment_membership(fid, best_topic, loyalty=0.3)
            else:
                # Orphan: mark with low loyalty to current (dissolved) topic
                db.topic.set_fragment_membership(fid, topic_id, loyalty=0.0)
            redistributed += 1

    if debug_cb and redistributed:
        from ..error_utils import log_info
        log_info(debug_cb, "Dissolved", f"{len(dissolved)} topics, {redistributed} fragments redistributed")
    return redistributed




def _compute_graph_centroid(vectors: list[np.ndarray], weights: list[float],
                             max_iter: int = 20, tol: float = 1e-4) -> np.ndarray:
    """Geometric median weighted by eigenvector-like centrality.
    Uses Weiszfeld algorithm with centrality weights. Robust to outliers."""
    if not vectors:
        return np.zeros(384)
    if len(vectors) == 1:
        return vectors[0].copy()

    total_w = sum(weights) or 1.0
    centroid = sum(w * v for w, v in zip(weights, vectors)) / total_w

    for _ in range(max_iter):
        numerator = np.zeros_like(centroid)
        denominator = 0.0
        for v, w in zip(vectors, weights):
            dist = np.linalg.norm(centroid - v)
            if dist < tol:
                return centroid
            weight = w / dist
            numerator += weight * v
            denominator += weight
        new_centroid = numerator / denominator if denominator > 0 else centroid
        if np.linalg.norm(new_centroid - centroid) < tol:
            return new_centroid
        centroid = new_centroid
    return centroid


def _adjust_vectors_from_feedback(db: QADatabase, debug_cb=None) -> dict:
    """Three-layer cascade: adjust Fragment→KP→Topic vectors from LLM feedback."""
    result = {"fragments_adjusted": 0, "kps_adjusted": 0, "topics_adjusted": 0}

    # Layer 1: Adjust fragment centrality from LLM help data
    # Batch-read per topic via get_topic_fragment_centralities (avoid N+1 SELECTs)
    topics = db.conn.execute(
        "SELECT topic_id FROM dynamic_topics WHERE quality != 'dissolved'"
    ).fetchall()
    centrality_updates = []
    for t in topics:
        cent_rows = db.fragment.get_topic_centralities(t["topic_id"])
        for cent in cent_rows:
            if cent["verification_count"] < 1:
                continue
            cohesion = cent.get("topic_coherence", 0.5)
            # EMA 权重分配: 60% 历史中心度 + 30% 帮助性能 + 10% 主题一致性
            new_centrality = (0.60 * cent["centrality_score"]
                              + 0.30 * cent["avg_help_score"]
                              + 0.10 * cohesion)
            centrality_updates.append((
                cent["fragment_id"],
                round(min(1.0, new_centrality), 3),
                cent["avg_help_score"],
                cohesion,
                cent.get("variance", 0),
            ))
            result["fragments_adjusted"] += 1

    if centrality_updates:
        with db.transaction():
            db.conn.executemany(
                """INSERT OR REPLACE INTO fragment_centrality
                   (fragment_id, verification_count, avg_help_score, topic_coherence,
                    variance, centrality_score, updated_at)
                   VALUES (?, COALESCE((SELECT verification_count FROM fragment_centrality
                    WHERE fragment_id=?), 0) + 1, ?, ?, ?, ?, datetime('now'))""",
                [(fid, fid, ahs, tc, var, cs) for fid, cs, ahs, tc, var in centrality_updates],
            )

    # Layer 2: KP vectors (cascade from member QA embeddings)
    # Two-phase: collect all unique QA texts → batch-encode once → distribute to KPs
    kp_rows = db.conn.execute(
        "SELECT id FROM knowledge_points WHERE quality != 'disputed'"
    ).fetchall()

    # Phase 1: Collect unique QA texts with kp_id → qa_id list mappings
    kp_qa_map = {}          # kp_id → list of qa_ids
    qa_id_to_idx = {}       # qa_id → index in all_texts (dedup)
    all_texts = []
    for kp_row in kp_rows:
        kp_id = kp_row["id"]
        member_rows = db.conn.execute(
            "SELECT qa_id FROM qa_kp_membership WHERE kp_id=?", (kp_id,)
        ).fetchall()
        if not member_rows:
            continue
        qa_ids = [r["qa_id"] for r in member_rows[:5]]
        kp_qa_map[kp_id] = qa_ids
        for qid in qa_ids:
            if qid not in qa_id_to_idx:
                qa = db.get(qid)
                if qa:
                    text = (qa.get("question_text", "") + " " + qa.get("answer_text", ""))[:500]
                    qa_id_to_idx[qid] = len(all_texts)
                    all_texts.append(text)

    # Phase 2: Batch-encode all unique QA texts at once
    if all_texts:
        model = _get_model(TOPIC_EMBED_MODEL)
        all_vecs = model.encode(all_texts, batch_size=64,
                                normalize_embeddings=True, convert_to_numpy=True)

        # Phase 3: Distribute — each KP looks up its member vectors by index
        for kp_id, qa_ids in kp_qa_map.items():
            indices = [qa_id_to_idx[qid] for qid in qa_ids if qid in qa_id_to_idx]
            if indices:
                kp_vecs = all_vecs[indices]
                centroid = _compute_graph_centroid(list(kp_vecs), [1.0] * len(indices))
                db.vector.upsert_kp_vector(kp_id, centroid)
                result["kps_adjusted"] += 1

    # Layer 3: Topic vectors (cascade from KP vectors)
    for t in topics:
        topic_id = t["topic_id"]
        kp_ids = [r["id"] for r in db.conn.execute(
            """SELECT DISTINCT kp.id FROM knowledge_points kp
               JOIN qa_kp_membership qkm ON kp.id = qkm.kp_id
               JOIN qa_pairs q ON qkm.qa_id = q.id
               WHERE q.topic = (SELECT name FROM dynamic_topics WHERE topic_id=?)""",
            (topic_id,)
        ).fetchall()]
        if len(kp_ids) < 2:
            continue
        vecs = []
        weights = []
        for kp_id in kp_ids:
            v = db.vector.get_kp_vector(kp_id)
            if v is not None:
                vecs.append(v)
                weights.append(1.0)
        if vecs:
            centroid = _compute_graph_centroid(vecs, weights)
            db.vector.upsert_topic_vector(topic_id, centroid, len(kp_ids))
            result["topics_adjusted"] += 1

    if debug_cb and sum(result.values()) > 0:
        from ..error_utils import log_info
        log_info(debug_cb, "Vector cascade", f"{result['fragments_adjusted']} fragments, "
                 f"{result['kps_adjusted']} KPs, {result['topics_adjusted']} topics")
    return result
