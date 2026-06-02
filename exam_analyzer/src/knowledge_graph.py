"""Knowledge graph: QA clustering → KP node generation → edge discovery.

Replaces the linear distillation pipeline with an evolving graph structure.
Each cluster of semantically similar QAs becomes a Knowledge Point (KP).
KPs are linked by prerequisite/corequisite/related edges discovered from
Phase 2 retrieval behavior, embedding similarity, exam ordering, and student paths.
"""

import os
import re
from collections import deque
import numpy as np

from .deepseek_client import call_flash, create_client
from .knowledge_base import QADatabase
from .embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from .models import KPSpec, KpEdgeSpec
from .constants import (
    EDGE_FUSION_RETRIEVAL_W, EDGE_FUSION_SEMANTIC_W,
    EDGE_FUSION_SEQUENTIAL_W, EDGE_FUSION_LEARNING_PATH_W,
    EDGE_TRANSITION_DIVISOR,
)
from .logger import get_logger
from .utils import get_worker_limit

_log = get_logger()


def _build_qa_to_kp_mapping(clusters, qa_list, kp_ids):
    """Build QA index -> KP ID mapping from clusters. Shared by edge discovery."""
    qa_to_kp = {}
    for kp_id in kp_ids:
        cluster_idx = _parse_kp_cluster_idx(kp_id)
        if cluster_idx is not None and cluster_idx < len(clusters):
            for qa_idx in clusters[cluster_idx]:
                qa_to_kp[qa_list[qa_idx]["id"]] = kp_id
    return qa_to_kp


def _compute_sequential_transitions(qa_list, clusters, kp_ids):
    """Compute KP transition counts from exam question ordering.

    Pure function — zero DB interaction. Accepts QA list, cluster indices,
    and KP IDs; returns transition counts and per-pair paper evidence sets.

    Returns:
        (transitions, paper_kp_pairs) where:
        - transitions: {(kp_a, kp_b): count} — how many papers exhibit A→B
        - paper_kp_pairs: {(kp_a, kp_b): {paper_names}} — which papers
    """
    qa_to_kp = _build_qa_to_kp_mapping(clusters, qa_list, kp_ids)

    # Group QAs by paper, sort by question_number (natural sort)
    paper_order = {}
    for qa in qa_list:
        paper = qa.get("paper", "")
        qn = qa.get("question_number", "")
        if paper and qn:
            paper_order.setdefault(paper, []).append((qn, qa["id"]))

    # Count KP transitions within papers
    transitions = {}
    paper_kp_pairs = {}
    for paper, qas in paper_order.items():
        qas.sort(key=lambda x: _qn_sort_key(x[0]))
        seen_kps = []
        for _, qa_id in qas:
            kp = qa_to_kp.get(qa_id)
            if kp and (not seen_kps or seen_kps[-1] != kp):
                seen_kps.append(kp)
        for i in range(len(seen_kps) - 1):
            pair = (seen_kps[i], seen_kps[i + 1])
            transitions[pair] = transitions.get(pair, 0) + 1
            paper_kp_pairs.setdefault(pair, set()).add(paper)

    return transitions, paper_kp_pairs


def _transition_weight(count: int) -> float:
    """Normalize transition count to [0, 1] weight."""
    return min(count / EDGE_TRANSITION_DIVISOR, 1.0)


def _parse_kp_cluster_idx(kp_id: str) -> int | None:
    """Extract cluster index from KP ID, handling auto-split suffixes like 'kp_0000s0'."""
    m = re.match(r'kp_(\d+)', kp_id)
    return int(m.group(1)) if m else None


def _qn_sort_key(qn: str) -> tuple:
    """Natural sort key for question numbers: "1(a)" -> (1, "a"), "10(a)" -> (10, "a")."""
    m = re.match(r'^(\d+)', qn)
    num = int(m.group(1)) if m else 0
    suffix = qn[m.end():] if m else qn
    return (num, suffix)


def _build_similarity_graph(qa_vectors, threshold=0.70):
    """Build adjacency graph from QA vectors. Edge if cosine >= threshold.
    threshold=0.70: 聚类相似度阈值 — ↑更高→更多更小的簇, ↓更低→更少更大的簇"""
    n = len(qa_vectors)
    if n == 0:
        return {}, []
    cos_matrix = qa_vectors @ qa_vectors.T
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if float(cos_matrix[i][j]) >= threshold:
                adj[i].add(j)
                adj[j].add(i)
    return adj, cos_matrix


def _find_clusters(adj, qa_count):
    """Find connected components in similarity graph. Each component = one cluster."""
    visited = set()
    clusters = []
    noise = []
    for i in range(qa_count):
        if i in visited:
            continue
        # BFS to find connected component
        component = []
        frontier = deque([i])
        visited.add(i)
        while frontier:
            curr = frontier.popleft()
            component.append(curr)
            for nxt in adj.get(curr, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)
        if len(component) >= 2:
            clusters.append(component)
        else:
            noise.append(component[0])
    return clusters, noise


def _compute_centroid(qa_vectors, indices):
    """Compute mean vector of a cluster."""
    if not indices:
        return None
    vecs = qa_vectors[indices]
    centroid = np.mean(vecs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1.0)
    return centroid


def _compute_cohesion(qa_vectors, indices, centroid):
    """Average cosine similarity of cluster members to centroid."""
    if not indices or centroid is None:
        return 0.0
    sims = [float(np.dot(qa_vectors[i], centroid)) for i in indices]
    return sum(sims) / len(sims)


def _cosine_to_centroid(qa_vectors, qa_idx, centroid):
    """Cosine similarity of a single QA to a centroid."""
    if centroid is None:
        return 0.0
    return float(np.dot(qa_vectors[qa_idx], centroid))


def cluster_qas(db: QADatabase, debug=None) -> dict:
    """Group all QAs into clusters using cosine similarity graph.

    Returns: {
        "clusters": [[qa_idx, ...], ...],   # indices into qa_list
        "noise": [qa_idx, ...],             # unclustered QAs
        "centroids": [ndarray, ...],         # centroid per cluster
        "cohesions": [float, ...],           # cohesion per cluster
    }
    """
    if debug:
        debug("Clustering QAs into knowledge points...")

    qas = db.get_all()
    if len(qas) < 2:
        return {"clusters": [], "noise": list(range(len(qas))), "centroids": [],
                "cohesions": [], "qa_vectors": np.empty((0, 384)), "qa_list": qas}

    # Encode all QAs
    texts = [
        qa["question_text"] + " " + qa["answer_text"]
        if (qa.get("question_text") or qa.get("answer_text"))
        else qa.get("knowledge_summary", "")
        for qa in qas
    ]
    if debug:
        debug("Loading KG embedding model (may take 20-60s on first run)...")
    model = _get_model(TOPIC_EMBED_MODEL)
    qa_vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    if debug:
        debug("KG embedding complete")

    # Build similarity graph and find clusters
    adj, cos_matrix = _build_similarity_graph(qa_vectors, threshold=0.70)
    clusters, noise = _find_clusters(adj, len(qas))

    # Compute centroids and cohesion
    centroids = []
    cohesions = []
    for cluster in clusters:
        centroid = _compute_centroid(qa_vectors, cluster)
        centroids.append(centroid)
        cohesions.append(_compute_cohesion(qa_vectors, cluster, centroid))

    if debug:
        sizes = sorted([len(c) for c in clusters], reverse=True) if clusters else []
        size_str = f"sizes={sizes[:10]}" + ("..." if len(sizes) > 10 else "") if sizes else "none"
        coh_vals = [c for c in cohesions if c is not None]
        coh_str = ""
        if coh_vals:
            coh_str = f", cohesion: min={min(coh_vals):.2f}, max={max(coh_vals):.2f}, avg={sum(coh_vals)/len(coh_vals):.2f}"
        noise_pct = len(noise) / (len(clusters) + len(noise)) * 100 if (len(clusters) + len(noise)) > 0 else 0
        from .error_utils import log_info
        log_info(debug, "KG Clusters", f"{len(clusters)} ({size_str}{coh_str})")
        from .error_utils import log_info
        log_info(debug, "KG Noise", f"{len(noise)} QAs ({noise_pct:.0f}% of total)")

    return {
        "clusters": clusters,
        "noise": noise,
        "centroids": centroids,
        "cohesions": cohesions,
        "qa_vectors": qa_vectors,
        "qa_list": qas,
    }


def generate_kps(db: QADatabase, clustering: dict, client, debug=None) -> list[str]:
    """Generate KP nodes from clusters. Flash names each cluster.

    Returns list of kp_ids.
    """
    clusters = clustering["clusters"]
    qa_list = clustering["qa_list"]
    centroids = clustering["centroids"]
    cohesions = clustering["cohesions"]

    if not clusters:
        if debug:
            debug("  No clusters to generate KPs from")
        return []

    kp_ids = []
    batch_size = 5
    batches = [(i, clusters[i:i+batch_size]) for i in range(0, len(clusters), batch_size)]

    def _name_batch(batch_start, batch_clusters):
        batch_results = []
        batch_cohesions = cohesions[batch_start:batch_start + batch_size]
        b = batch_start  # for kp_id calculation below

        sample_text = ""
        for ci, cluster in enumerate(batch_clusters):
            for idx in cluster[:3]:  # top 3 QAs per cluster
                sample_text += qa_list[idx]["question_text"] + " "
        lang = detect_content_lang(sample_text[:2000])

        if lang == 'en':
            sys = (
                "You are a curriculum knowledge organizer. For each group of related exam questions, "
                "give the group a concise topic name and a 1-sentence description of the core concept. "
                "Use standard terminology. Output JSON."
            )
            usr = "Name each question group:\n\n"
            for ci, cluster in enumerate(batch_clusters):
                usr += f"Group {ci} ({len(cluster)} questions, cohesion={batch_cohesions[ci]:.2f}):\n"
                for idx in cluster[:3]:
                    qa = qa_list[idx]
                    usr += f"  Q: {qa['question_text'][:200]}\n  A: {qa['answer_text'][:200]}\n\n"
            usr += (
                'Return: {"groups": [{"index": 0, "name": "Topic Name", '
                '"description": "1-sentence core concept description"}, ...]}'
            )
        else:
            sys = (
                "你是一个课程知识组织专家。为每组相关考题命名并描述核心概念。Output JSON。"
            )
            usr = "为以下题目组命名：\n\n"
            for ci, cluster in enumerate(batch_clusters):
                usr += f"组 {ci} ({len(cluster)} 题, 凝聚力={batch_cohesions[ci]:.2f}):\n"
                for idx in cluster[:3]:
                    qa = qa_list[idx]
                    usr += f"  Q: {qa['question_text'][:200]}\n  A: {qa['answer_text'][:200]}\n\n"
            usr += (
                '返回: {"groups": [{"index": 0, "name": "主题名", '
                '"description": "一句话核心概念描述"}, ...]}'
            )

        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        try:
            result, _ = call_flash(client, messages, max_retries=1, debug=debug)
            groups = result.get("groups", []) if isinstance(result, dict) else []
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "KP naming", "batch", e)
            groups = []

        # Fallback: auto-name clusters that Flash failed to name
        if not groups:
            for gi_in_batch, cluster in enumerate(batch_clusters):
                fallback_topic = qa_list[cluster[0]].get("topic", "Unnamed")
                global_idx = batch_start + gi_in_batch
                groups.append({
                    "index": gi_in_batch,
                    "name": f"{fallback_topic} (auto-{global_idx})",
                    "description": "",
                })

        for g in groups:
            gi = g.get("index", -1)
            if 0 <= gi < len(batch_clusters):
                cluster_idx = b + gi  # global cluster index
                kp_id = f"kp_{cluster_idx:04d}"
                name = g.get("name", f"Unnamed_{cluster_idx}")
                description = g.get("description", "")

                # Find representative QAs (closest to centroid)
                cluster = clusters[cluster_idx]
                centroid = centroids[cluster_idx]
                qa_dists = [
                    (idx, _cosine_to_centroid(clustering["qa_vectors"], idx, centroid))
                    for idx in cluster
                ]
                qa_dists.sort(key=lambda x: -x[1])
                representatives = qa_dists[:3]

                # Store KP
                centroid_bytes = centroid.tobytes() if centroid is not None else None
                db.kp.upsert(KPSpec(
                    kp_id=kp_id, name=name, description=description,
                    cluster_id=cluster_idx, centroid_vector=centroid_bytes,
                    cohesion=cohesions[cluster_idx],
                    evidence_count=len(cluster),
                    quality="draft",
                ))

                # Store QA-KP membership
                for qa_idx, dist in qa_dists:
                    db.kp.set_membership(
                        qa_id=qa_list[qa_idx]["id"],
                        kp_id=kp_id,
                        membership_strength=round(dist, 3),
                        is_representative=(qa_idx == representatives[0][0]),
                    )

                batch_results.append(kp_id)
                if debug:
                    from .error_utils import log_info
                    log_info(debug, "KP named", f"{kp_id}: '{name}' ({len(cluster)} QAs, cohesion={cohesions[cluster_idx]:.2f})")
        return batch_results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_w = get_worker_limit(len(batches), api_heavy=True)
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = {executor.submit(_name_batch, s, c): s for s, c in batches}
        for future in as_completed(futures):
            try:
                kp_ids.extend(future.result())
            except Exception as e:
                from .error_utils import log_exception
                log_exception(debug, "KP naming", "batch_thread", e)

    if debug:
        total_clusters = len(clusters)
        from .error_utils import log_info
        log_info(debug, "KG KPs generated", f"{len(kp_ids)} (cluster coverage: {len(kp_ids)}/{total_clusters})")
        batch_count = (len(clusters) + batch_size - 1) // batch_size if clusters else 0
        from .error_utils import log_info
        log_info(debug, "KG Flash naming", f"{batch_count} batches")

    return kp_ids


def _compute_retrieval_candidates(topic_links: dict, qa_list: list[dict],
                                   qa_to_kp: dict) -> list[tuple]:
    """Compute retrieval edge candidates from topic link counts.

    Pure function — zero DB interaction. Accepts topic_links dict (from
    db.topic.get_links()), QA list, and QA→KP mapping. Returns list of
    (target_kp, source_kp, count) tuples for topic pairs with count >= 2.
    Direction is reversed: dst_topic→source_kp, src_topic→target_kp.

    Caller is responsible for creating KpEdgeSpec and calling db.kp.upsert_edge().
    """
    candidates = []
    for (src_topic, dst_topic), count in topic_links.items():
        if count < 2:
            continue
        src_kps = set()
        dst_kps = set()
        for qa in qa_list:
            if qa.get("topic") == src_topic and qa["id"] in qa_to_kp:
                src_kps.add(qa_to_kp[qa["id"]])
            if qa.get("topic") == dst_topic and qa["id"] in qa_to_kp:
                dst_kps.add(qa_to_kp[qa["id"]])
        for sk in src_kps:
            for dk in dst_kps:
                if sk != dk:
                    candidates.append((dk, sk, count))  # reversed direction
    return candidates


def _compute_semantic_edges(kp_centroids: dict) -> list[tuple]:
    """Compute semantic edge candidates from KP centroid cosine similarity.

    Pure function — zero DB interaction. Accepts {kp_id: ndarray} dict,
    returns list of (kp_a, kp_b, cos_score, confidence) tuples for pairs
    with cosine >= 0.5.

    Confidence: "medium" if cos >= 0.65, else "low".
    """
    candidates = []
    if len(kp_centroids) < 2:
        return candidates

    kp_ids_list = list(kp_centroids.keys())
    centroids_mat = np.stack([kp_centroids[k] for k in kp_ids_list])
    cos_mat = centroids_mat @ centroids_mat.T
    for i in range(len(kp_ids_list)):
        for j in range(i + 1, len(kp_ids_list)):
            cos = float(cos_mat[i][j])
            if cos >= 0.5:
                candidates.append((
                    kp_ids_list[i], kp_ids_list[j],
                    round(cos, 3),
                    "medium" if cos >= 0.65 else "low",
                ))
    return candidates


def discover_kp_edges(db: QADatabase, clustering: dict, kp_ids: list[str],
                      debug=None) -> int:
    """Discover edges between KPs: retrieval (Phase 2 behavior) + semantic (embedding).

    Returns number of edges created.
    """
    if len(kp_ids) < 2:
        return 0

    if debug:
        debug("Discovering KP edges...")

    qa_list = clustering["qa_list"]
    centroids_list = clustering["centroids"]
    clusters = clustering["clusters"]

    # Build KP centroid map for semantic edges
    kp_centroids = {}
    for kp_id in kp_ids:
        cluster_idx = _parse_kp_cluster_idx(kp_id)
        if (cluster_idx is not None
                and cluster_idx < len(centroids_list)
                and centroids_list[cluster_idx] is not None):
            kp_centroids[kp_id] = centroids_list[cluster_idx]

    # Semantic edges: cosine between KP centroids (computation extracted)
    edge_count = 0
    for a, b, cos, confidence in _compute_semantic_edges(kp_centroids):
        db.kp.upsert_edge(KpEdgeSpec(
            source_kp=a, target_kp=b,
            edge_type="related",
            semantic_weight=cos,
            combined_strength=cos,
            confidence=confidence,
        ))
        edge_count += 1

    # Retrieval edges: from topic_links (Phase 2 behavior)
    topic_links = db.topic.get_links()
    if topic_links:
        qa_to_kp = _build_qa_to_kp_mapping(clusters, qa_list, kp_ids)
        for dk, sk, count in _compute_retrieval_candidates(topic_links, qa_list, qa_to_kp):
            db.kp.upsert_edge(KpEdgeSpec(
                source_kp=dk, target_kp=sk,  # reversed
                edge_type="prerequisite",
                retrieval_weight=count,
                combined_strength=_transition_weight(count),
                confidence="medium" if count >= 4 else "low",
            ))
            edge_count += 1

    if debug:
        from .error_utils import log_info
        log_info(debug, "KP edges", f"{edge_count} discovered")

    return edge_count


def discover_sequential_edges(db: QADatabase, clustering: dict, kp_ids: list[str],
                             debug=None) -> int:
    """Discover edges from exam question ordering (sequential edges).
    If KP A consistently appears before KP B across multiple papers, create a sequential edge."""
    if len(kp_ids) < 2:
        return 0

    qa_list = clustering["qa_list"]
    clusters = clustering["clusters"]

    transitions, paper_kp_pairs = _compute_sequential_transitions(qa_list, clusters, kp_ids)

    # Create edges for consistent transitions (>= 3 different papers)
    edge_count = 0
    for (a, b), count in transitions.items():
        # 时序边: ≥3 场不同考试支持才建立 sequential 边
        num_papers = len(paper_kp_pairs.get((a, b), set()))
        if num_papers >= 3 and a != b:
            db.kp.upsert_edge(KpEdgeSpec(
                source_kp=a, target_kp=b,
                edge_type="sequential",
                sequential_weight=_transition_weight(num_papers),
                combined_strength=_transition_weight(num_papers),
                confidence="high" if num_papers >= 5 else "medium",
            ))
            edge_count += 1

    if debug:
        from .error_utils import log_info
        log_info(debug, "Sequential edges", f"{edge_count} (from {len(transitions)} transitions)")

    return edge_count


def discover_learning_path_edges(db: QADatabase, kp_ids: list[str],
                                 debug=None) -> int:
    """Discover edges from student learning paths (learning_path edges).
    If >= 3 students ask about KP A then KP B in the same session, create an edge."""
    if len(kp_ids) < 2:
        return 0

    # Read student trajectories, find KP transitions within sessions
    rows = db.student.get_all_trajectories()

    if not rows:
        return 0

    transitions = {}
    student_pairs = {}
    current_student = None
    prev_kp = None
    for r in rows:
        sid, kp = r["student_id"], r["kp_id"]
        if sid != current_student:
            current_student = sid
            prev_kp = kp
            continue
        if kp and prev_kp and kp != prev_kp:
            pair = (prev_kp, kp)
            transitions[pair] = transitions.get(pair, 0) + 1
            student_pairs.setdefault(pair, set()).add(sid)
        prev_kp = kp

    edge_count = 0
    for (a, b), count in transitions.items():
        num_students = len(student_pairs.get((a, b), set()))
        if num_students >= 3 and a != b:
            db.kp.upsert_edge(KpEdgeSpec(
                source_kp=a, target_kp=b,
                edge_type="learning_path",
                learning_path_weight=_transition_weight(num_students),
                combined_strength=_transition_weight(num_students),
                confidence="high" if num_students >= 5 else "medium",
            ))
            edge_count += 1

    if debug:
        from .error_utils import log_info
        log_info(debug, "Learning path edges", f"{edge_count} (from {len(transitions)} transitions)")

    return edge_count


def fuse_all_edges(db: QADatabase, kp_ids: list[str], debug=None):
    """Merge multi-signal edges: if an edge has supporting evidence from multiple sources,
    upgrade its confidence and compute combined strength."""
    edges = db.kp.get_edges()
    if not edges:
        return

    # Group by (source, target) regardless of edge_type
    grouped = {}
    for e in edges:
        key = (e["source_kp"], e["target_kp"])
        grouped.setdefault(key, []).append(e)

    for (src, tgt), edge_list in grouped.items():
        rw = max(e.get("retrieval_weight", 0) or 0 for e in edge_list)
        sw = max(e.get("semantic_weight", 0) or 0 for e in edge_list)
        sq = max(e.get("sequential_weight", 0) or 0 for e in edge_list)
        lp = max(e.get("learning_path_weight", 0) or 0 for e in edge_list)

        combined = (rw * EDGE_FUSION_RETRIEVAL_W + sw * EDGE_FUSION_SEMANTIC_W
                    + sq * EDGE_FUSION_SEQUENTIAL_W + lp * EDGE_FUSION_LEARNING_PATH_W)

        # Confidence from fusion rules
        if (rw > 0 and sw > 0) or (rw > 0 and sq > 0):
            confidence = "high"
        elif (sw > 0 and lp > 0) or (sq > 0 and lp > 0):
            confidence = "medium"
        elif rw > 0 or sw > 0:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine dominant edge type
        types = set(e["edge_type"] for e in edge_list)
        if "prerequisite" in types or "corequisite" in types:
            etype = "prerequisite" if "prerequisite" in types else "corequisite"
        elif "sequential" in types:
            etype = "related"
        else:
            etype = "related"

        # Atomic DELETE+INSERT via Store — edge_type is part of the PK, so
        # INSERT OR REPLACE would create duplicates when type changes.
        db.kp.replace_edge(KpEdgeSpec(
            source_kp=src, target_kp=tgt,
            edge_type=etype,
            retrieval_weight=rw, semantic_weight=sw,
            sequential_weight=sq, learning_path_weight=lp,
            combined_strength=round(combined, 3),
            confidence=confidence,
        ))

    if debug:
        from .error_utils import log_info
        log_info(debug, "Edge fusion", f"{len(grouped)} unique pairs from {len(edges)} edges")


def run_knowledge_graph(db, api_url: str, api_key: str,
                        debug=None):
    """Main entry point: cluster QAs, generate KPs, discover edges.
    Called from pipeline after QA processing and topic merge complete.

    This replaces (or augments) the distillation step.
    """
    def _debug(msg):
        if debug:
            debug(f"[KG] {msg}")
        else:
            print(f"[KG] {msg}")

    _debug("Starting knowledge graph construction...")

    if db.count() < 2:
        _debug("Not enough QAs for clustering, skipping")
        return

    client = create_client(api_url, api_key)

    # Step 1: Cluster QAs
    clustering = cluster_qas(db, _debug)

    # Step 2: Generate KP nodes
    kp_ids = generate_kps(db, clustering, client, _debug)

    # Step 3: Discover edges (semantic + retrieval)
    if kp_ids:
        discover_kp_edges(db, clustering, kp_ids, _debug)
        discover_sequential_edges(db, clustering, kp_ids, _debug)
        discover_learning_path_edges(db, kp_ids, _debug)
        fuse_all_edges(db, kp_ids, _debug)

    # Summary: edges by type
    edge_counts = db.kp.get_edge_counts()
    edge_str = ", ".join(f"{r['edge_type']}={r['cnt']}" for r in edge_counts)
    from .error_utils import log_info
    log_info(_debug, "KG Edges", edge_str if edge_str else "none")

    # Post-fusion duplicate check
    dup_rows = db.kp.get_duplicate_edges()
    if dup_rows:
        duplicates = ", ".join(f"{r['source_kp']}<->{r['target_kp']}(x{r['cnt']})" for r in dup_rows)
        from .error_utils import log_info
        log_info(_debug, "KG Post-fusion check", f"duplicates: {duplicates}")
    else:
        _debug("  [KG] Post-fusion duplicate check: clean (0 duplicates)")

    from .error_utils import log_info
    log_info(_debug, "KG Complete", f"{len(kp_ids)} KPs from {len(clustering['clusters'])} clusters")
    db.analysis.checkpoint("knowledge_graph", db.count(), "completed")

    _debug("Knowledge graph construction complete")
