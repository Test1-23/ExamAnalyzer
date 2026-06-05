import json
import os
import re
from collections import deque

from ..deepseek_client import create_client, call_flash
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from ..models import VerbPatternSpec, DependencySpec
from ..prompt_factory import PromptType, PromptBuilder
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()

# ============================================================
# Task 1: Dependency Discovery
# ============================================================

def discover_dependencies(db: QADatabase, client, debug, progress_cb=None) -> list:
    """Discover prerequisite relationships between topics.

    Returns: [(prerequisite, dependent, score, confidence), ...]
    """
    _log.info("Offline Task 1: Dependency discovery starting")
    debug("Task 1: Discovering topic dependencies...")

    qas = db.get_all()
    if not qas:
        _log.info("Task 1: No QAs, skipping")
        return []

    current_count = len(qas)
    prev = db.analysis.get_checkpoint("dependencies")
    if prev and prev.get("qa_count_at_run") == current_count:
        _log.info("Task 1: Already complete, skipping")
        return []

    if progress_cb:
        progress_cb(0, "Generating dependency candidates...")

    # Phase 0: Candidate generation
    candidates = _phase0_generate_candidates(db, debug)

    if progress_cb:
        progress_cb(40, f"Validating {len(candidates)} candidate pairs...")

    # Phase 1: Batch Flash validation
    validated = _phase1_validate_candidates(db, candidates, client, debug)

    if progress_cb:
        progress_cb(70, "Post-processing dependency graph...")

    # Phase 2: Graph post-processing
    _phase2_postprocess_dependencies(db, validated, debug)

    db.analysis.checkpoint("dependencies", current_count, "completed")

    # Dependency confidence distribution
    conf_counts = db.conn.execute(
        "SELECT confidence, COUNT(*) as cnt FROM topic_dependencies GROUP BY confidence"
    ).fetchall()
    conf_map = {r["confidence"]: r["cnt"] for r in conf_counts}
    if conf_map:
        cstr = ", ".join(f"{k}={v}" for k, v in sorted(conf_map.items()))
        node_count = db.conn.execute(
            "SELECT COUNT(DISTINCT prerequisite) + COUNT(DISTINCT dependent) FROM topic_dependencies"
        ).fetchone()
        # count unique nodes from both columns
        nodes = set()
        for r in db.analysis.get_dependencies():
            nodes.add(r["prerequisite"]); nodes.add(r["dependent"])
        edge_count = sum(conf_map.values())
        from ..error_utils import log_info
        log_info(debug, "Dependencies", f"{cstr} - {len(nodes)} nodes, {edge_count} edges")

    _log.info(f"Task 1: Complete. {len(validated)} dependencies stored")
    return validated


def _phase0_generate_candidates(db, debug):
    """Generate candidate dependency pairs from topic_links and embedding similarity."""
    topic_links = db.topic.get_links()
    topic_texts = db.qa.get_topic_answer_texts()
    topics = list(topic_texts.keys())

    if len(topics) < 2:
        return []

    candidates = set()

    # Source A: topic_links (behavioral evidence from Phase 2)
    for (src, dst), count in topic_links.items():
        if src not in topic_texts or dst not in topic_texts:
            continue
        rev_count = topic_links.get((dst, src), 0)

        if count >= 2 and rev_count >= 2 and abs(count - rev_count) <= max(count, rev_count) * 0.5:
            # Bidirectional ≈ symmetric → corequisite, handled in post-processing
            candidates.add((dst, src, "topic_link_symmetric", min(count, rev_count)))
            candidates.add((src, dst, "topic_link_symmetric", min(count, rev_count)))
        elif count >= 2 and count > rev_count * 3:
            # Strong unidirectional: dependency = reverse of topic_link direction
            candidates.add((dst, src, "topic_link", count))
        elif count >= 1:
            # Weak: only use if embedding also supports
            pass  # handled by Source B below with higher threshold

    # Source B: embedding similarity
    if len(topics) >= 2:
        try:
            model = _get_model(TOPIC_EMBED_MODEL)
            texts = [topic_texts[t] for t in topics]
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            cos_matrix = vecs @ vecs.T
        except Exception as e:
            from ..error_utils import log_exception
            log_exception(debug, "Topic embedding", "", e)
            cos_matrix = None

        if cos_matrix is not None:
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    cos = float(cos_matrix[i][j])
                    if cos < 0.5:
                        continue
                    a, b = topics[i], topics[j]
                    # Check if already in candidates from topic_links
                    already = any(
                        (c[0] == a and c[1] == b) or (c[0] == b and c[1] == a)
                        for c in candidates
                    )
                    if not already:
                        # Both directions are candidates (no directional evidence yet)
                        candidates.add((a, b, "embed_only", round(cos, 3)))
                        candidates.add((b, a, "embed_only", round(cos, 3)))

    from ..error_utils import log_info
    log_info(debug, "Dependency candidates", f"{len(candidates)} pairs "
             f"(topic_links={len(topic_links)}, topics={len(topics)})")
    return list(candidates)


def _phase1_validate_candidates(db, candidates, client, debug):
    """Batch Flash validation of dependency candidates (5 pairs per call). Parallelized."""
    topic_texts = db.qa.get_topic_answer_texts()
    batch_size = 5

    sorted_candidates = sorted(candidates, key=lambda c: (
        0 if c[2] in ("topic_link", "topic_link_symmetric") else 1, -c[3]
    ))
    batches = [sorted_candidates[i:i+batch_size] for i in range(0, len(sorted_candidates), batch_size)]

    def _validate_batch(batch):
        sample_text = ""
        for pre, dep, ev_type, strength in batch:
            sample_text += topic_texts.get(pre, "") + " " + topic_texts.get(dep, "") + " "
        lang = detect_content_lang(sample_text[:2000])

        pairs_block = ""
        if lang == 'en':
            pairs_block += "Evaluate these topic pairs:\n\n"
        else:
            pairs_block += "评估以下 topic 对：\n\n"
        for i, (pre, dep, ev_type, strength) in enumerate(batch):
            pairs_block += (f"Pair {i}: Topic A [{pre}] → Topic B [{dep}]\n"
                            f"  A QAs: {topic_texts.get(pre, '')[:600]}\n"
                            f"  B QAs: {topic_texts.get(dep, '')[:600]}\n\n")

        messages = PromptBuilder.build(PromptType.DEPENDENCY, lang=lang, pairs_block=pairs_block)
        try:
            result, _ = call_flash(client, messages, max_retries=1, debug=debug)
            pairs = result.get("pairs", []) if isinstance(result, dict) else []
        except Exception as e:
            from ..error_utils import log_exception
            log_exception(debug, "Dependency validation", f"batch={b}", e)
            return []

        batch_results = []
        for p in pairs:
            idx = p.get("index", -1)
            if 0 <= idx < len(batch):
                pre, dep, ev_type, strength = batch[idx]
                score = p.get("score", 1)
                confidence = "low"
                if score == 2:
                    if ev_type == "topic_link" and strength >= 4: confidence = "high"
                    elif ev_type == "topic_link" and strength >= 2: confidence = "medium"
                    elif ev_type == "topic_link_symmetric": confidence = "medium"
                if score >= 1:
                    batch_results.append({
                        "prerequisite": pre, "dependent": dep,
                        "score": score, "reason": p.get("reason", ""),
                        "evidence_type": ev_type, "evidence_strength": strength,
                        "confidence": confidence,
                        "relationship_type": "corequisite" if ev_type == "topic_link_symmetric" else "prerequisite",
                    })
        return batch_results

    validated = []
    if batches:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_w = get_worker_limit(len(batches), api_heavy=True)
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(_validate_batch, b): b for b in batches}
            for future in as_completed(futures):
                try:
                    validated.extend(future.result())
                except Exception as e:
                    from ..error_utils import log_exception
                    log_exception(debug, "Dependency validation", f"thread", e)

    from ..error_utils import log_info
    log_info(debug, "Flash validated", f"{len(validated)} dependencies (from {len(candidates)} candidates, {len(batches)} batches parallel)")
    return validated


def _phase2_postprocess_dependencies(db, validated, debug):
    """Store dependencies, apply transitive reduction, detect cycles."""
    # Build adjacency for graph operations
    edges = {}  # (pre, dep) -> metadata
    for v in validated:
        key = (v["prerequisite"], v["dependent"])
        if key in edges:
            # Keep higher confidence
            if v["confidence"] == "high" or edges[key]["confidence"] == "low":
                edges[key] = v
        else:
            edges[key] = v

    # Transitive reduction: remove A→C if A→B→C exists
    topics_set = set()
    for pre, dep in edges:
        topics_set.add(pre)
        topics_set.add(dep)
    topics = list(topics_set)

    # Build adjacency list
    adj = {t: set() for t in topics}
    for pre, dep in edges:
        adj.setdefault(pre, set()).add(dep)

    # BFS from each node to find redundant edges
    redundant = set()
    for a in topics:
        if a not in adj:
            continue
        for b in list(adj.get(a, set())):
            # Check if there's a path a→...→b that doesn't use the direct edge
            visited = {a}
            frontier = deque([a])
            while frontier:
                curr = frontier.popleft()
                for nxt in adj.get(curr, set()):
                    if nxt == b and curr != a:
                        redundant.add((a, b))
                        break
                    if nxt not in visited and nxt != b:
                        visited.add(nxt)
                        frontier.append(nxt)
                if (a, b) in redundant:
                    break

    # Store non-redundant edges
    stored = 0
    for (pre, dep), v in edges.items():
        if (pre, dep) in redundant:
            from ..error_utils import log_info
            log_info(debug, "Transitive reduction", f"removed {pre}->{dep}")
            continue
        db.analysis.insert_dependency(DependencySpec(
            prerequisite=pre, dependent=dep,
            evidence_score=v["score"],
            evidence_reason=v.get("reason", ""),
            relationship_type=v.get("relationship_type", "prerequisite"),
            topic_link_count=v.get("evidence_strength", 0) if v["evidence_type"] != "embed_only" else 0,
            embedding_cos=v.get("evidence_strength", None) if v["evidence_type"] == "embed_only" else None,
            confidence=v["confidence"],
            validated_by="flash",
        ))
        stored += 1

    from ..error_utils import log_info
    log_info(debug, "Dependencies stored", f"{stored} (removed {len(redundant)} transitive)")

    # Detect remaining cycles (co-requisites) — check both new candidates and pre-existing DB edges
    cycles_found = 0
    with db.transaction():
        for pre, dep in list(edges.keys()):
            rev_key = (dep, pre)
            has_reverse = rev_key in edges
            if not has_reverse and (pre, dep) not in redundant:
                # Check DB for pre-existing reverse edge from a previous pipeline run
                db_row = db.conn.execute(
                    "SELECT 1 FROM topic_dependencies WHERE prerequisite=? AND dependent=?",
                    rev_key,
                ).fetchone()
                has_reverse = db_row is not None
            if has_reverse and (pre, dep) not in redundant and rev_key not in redundant:
                # Bidirectional dependency → update both to corequisite
                db.conn.execute(
                    """UPDATE topic_dependencies SET relationship_type = 'corequisite'
                       WHERE prerequisite = ? AND dependent = ?""",
                    (pre, dep),
                )
                cycles_found += 1
        from ..error_utils import log_info
        log_info(debug, "Co-requisite cycles", f"{cycles_found} found")


# ============================================================
# Orchestrator
# ============================================================

