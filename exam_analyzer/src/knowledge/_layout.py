"""Knowledge-space layout: fragments -> anchors -> relaxation -> KP clusters.

This is the DB-facing half of the geometry work (the pure math lives in
_geometry.py). A single call:

1. loads QAs and MS fragments (splitting MS text deterministically when the
   fragment table is empty),
2. encodes anchors in one batch,
3. relaxes knowledge-level co-occurrence edges,
4. clusters fragments into knowledge-point candidates,
5. (optionally) persists positions, KP rows, memberships and the
   dynamic_topics mirror used by the chat layer.

Design reference: AgentChatRoom/plan.md + the layout plan in this session.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._geometry import (
    LayoutConfig,
    build_cross_paper_edges,
    build_edges,
    cluster_positions,
    layout_metrics,
    normalize_rows,
    relax,
)


# ============================================================
# Result container
# ============================================================


@dataclass
class LayoutResult:
    """Outcome of one layout pass (metrics are JSON-serialisable)."""

    metrics: dict = field(default_factory=dict)
    kp_count: int = 0
    fragment_count: int = 0
    qa_count: int = 0
    edge_count: int = 0
    collapsed: bool = False
    persisted: bool = False


# ============================================================
# Small helpers
# ============================================================


def _norm_key(text: str) -> str:
    """Normalised text fingerprint (delegates to _fragments when available)."""
    try:
        from ._fragments import norm_key  # type: ignore
        return norm_key(text)
    except Exception:
        cleaned = re.sub(r"\W+", " ", (text or "").lower()).strip()
        return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16]


def _qa_anchor_text(qa: dict) -> str:
    """Text used for a QA's own embedding anchor."""
    parts = [
        qa.get("question_text") or "",
        qa.get("knowledge_summary") or "",
    ]
    text = " ".join(p for p in parts if p).strip()
    return text or (qa.get("answer_text") or "")[:500]


def _fragment_texts(fragments: list[dict]) -> list[str]:
    return [(f.get("point_text") or "").strip() for f in fragments]


def _fragment_order_key(frag: dict) -> tuple[int, int]:
    """Stable order key: (qa_id, numeric fragment index).

    The DB returns point_id as TEXT, so 'f_1_10' sorts before 'f_1_2'. A
    lexicographic row order changes the relaxation order and therefore the
    resulting clusters, so the numeric suffix must be parsed and compared.
    """
    match = re.search(r"_(\d+)$", str(frag.get("point_id", "")))
    index = int(match.group(1)) if match else 0
    return (int(frag.get("qa_id", 0)), index)


def _sorted_fragments(fragments: list[dict]) -> list[dict]:
    return sorted(fragments, key=_fragment_order_key)


# ============================================================
# Fragment loading / splitting
# ============================================================


def _collect_fragments(db, qas: list[dict], split_missing: bool) -> tuple[list[dict], bool]:
    """Load fragments from the DB, or derive them from MS text on first run.

    Returns (fragments, generated). The generated flag tells the caller the
    rows are not in ms_fragments yet and must be inserted when persisting.
    """
    existing = db.fragment.get_all_for_layout()
    if existing:
        return _sorted_fragments(existing), False
    if not split_missing:
        return [], False

    from ._fragments import split_fragments  # lazy import: pure module

    generated: list[dict] = []
    for qa in qas:
        text = qa.get("answer_text") or ""
        generated.extend(split_fragments(text, int(qa["id"])))
    return generated, True


# ============================================================
# Main layout pass
# ============================================================


def run_layout(
    db,
    encoder,
    config: LayoutConfig | None = None,
    *,
    persist: bool = True,
    split_missing: bool = True,
    encoder_version: str = "",
    progress: Callable[[str], None] | None = None,
) -> LayoutResult:
    """Compute (and optionally persist) one knowledge-space layout pass."""
    cfg = config or LayoutConfig()
    say = progress or (lambda _msg: None)

    qas = db.get_all()
    say(f"loaded {len(qas)} QAs")

    fragments, generated = _collect_fragments(db, qas, split_missing)
    say(f"fragments: {len(fragments)} ({'derived from MS text' if generated else 'from DB'})")

    if not fragments:
        raise RuntimeError(
            "no MS fragments available; run with split_missing=True "
            "or populate ms_fragments"
        )

    frag_texts = _fragment_texts(fragments)
    qa_texts = [_qa_anchor_text(qa) for qa in qas]

    say(f"encoding {len(frag_texts) + len(qa_texts)} texts")
    embeddings = normalize_rows(np.asarray(
        encoder.encode(frag_texts + qa_texts), dtype=np.float64
    ))
    frag_emb = embeddings[: len(fragments)]
    qa_emb = embeddings[len(fragments):]

    # anchors: fragments first, QAs pulled towards their fragments
    by_qa: dict[int, list[int]] = {}
    for idx, frag in enumerate(fragments):
        by_qa.setdefault(int(frag["qa_id"]), []).append(idx)

    anchors = np.zeros((len(fragments) + len(qas), frag_emb.shape[1]), dtype=np.float64)
    anchors[: len(fragments)] = frag_emb

    for qa_pos, qa in enumerate(qas):
        row = len(fragments) + qa_pos
        member_rows = by_qa.get(int(qa["id"]), [])
        if member_rows:
            anchors[row] = 0.7 * frag_emb[member_rows].mean(axis=0) + 0.3 * qa_emb[qa_pos]
        else:
            anchors[row] = qa_emb[qa_pos]
    anchors = normalize_rows(anchors)

    # index + edges
    index: dict[tuple[str, str], int] = {}
    for idx, frag in enumerate(fragments):
        index[("fragment", str(frag["point_id"]))] = idx
    for qa_pos, qa in enumerate(qas):
        index[("qa", str(qa["id"]))] = len(fragments) + qa_pos

    qa_fragments = {
        int(qa_id): [str(fragments[i]["point_id"]) for i in rows]
        for qa_id, rows in by_qa.items()
    }
    helps = db.fragment.get_help_edges()
    paper_of = [str(next((qa.get("paper", "") for qa in qas
                          if int(qa["id"]) == int(f["qa_id"])), "")) for f in fragments]
    # QA-fragment edges are disabled (weight 0): the design experiment shows
    # they cluster by question/paper instead of by knowledge point.  QAs still
    # inherit their fragments' KP through the majority vote below.
    edges = build_edges(
        index, qa_fragments, helps,
        same_qa_fragment_weight=0.1,
        qa_fragment_weight=0.0,
    )
    edges += build_cross_paper_edges(frag_emb, paper_of, topk=3, cos_min=0.90)
    say(f"edges: {len(edges)} (helps={len(helps)}, cross-paper included)")

    positions, diag = relax(anchors, edges, cfg)
    say(f"relaxed: epochs={diag['epochs_run']} collapsed={diag['collapsed']}")

    # cluster fragments only; QAs inherit their fragments' KP
    frag_pos = positions[: len(fragments)]
    qa_pos = positions[len(fragments):]
    labels = cluster_positions(frag_pos, cfg)

    clusters: dict[int, dict] = {}
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        centroid = normalize_rows(frag_pos[rows].mean(axis=0).reshape(1, -1))[0]
        members = [fragments[i] for i in rows]
        rep = max(members, key=lambda f: len(f.get("point_text") or ""))
        # Stable identity: the smallest norm_key in the cluster. Using the
        # representative's key would change when a tie flips, renaming the KP
        # between runs.
        keys = sorted(
            (f.get("norm_key") or _norm_key(f.get("point_text") or ""))
            for f in members
        )
        key = keys[0] if keys else ""
        cohesion = 1.0
        if len(rows) > 1:
            m = normalize_rows(frag_pos[rows])
            sim = m @ m.T
            cohesion = float(sim[~np.eye(len(rows), dtype=bool)].mean())
        clusters[int(label)] = {
            "centroid": centroid,
            "members": members,
            "representative": rep,
            "kp_id": f"kp_{key[:8]}",
            "cohesion": round(cohesion, 4),
            "evidence": len(members),
            "quality": "stable" if len(members) >= 3 else "embryonic",
        }

    # QA assignment by fragment majority
    frag_label = {str(fragments[i]["point_id"]): int(labels[i])
                  for i in range(len(fragments))}
    assignments: dict[int, tuple[str, float]] = {}
    for qa_pos_i, qa in enumerate(qas):
        member_ids = [str(fragments[i]["point_id"])
                      for i in by_qa.get(int(qa["id"]), [])]
        if member_ids:
            votes: dict[int, int] = {}
            for fid in member_ids:
                votes[frag_label[fid]] = votes.get(frag_label[fid], 0) + 1
            label = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            strength = float(np.dot(qa_pos[qa_pos_i], clusters[label]["centroid"]))
        else:
            sims = {lab: float(np.dot(qa_pos[qa_pos_i], c["centroid"]))
                    for lab, c in clusters.items()}
            label = max(sims, key=sims.get)
            strength = sims[label]
        assignments[int(qa["id"])] = (clusters[label]["kp_id"], round(strength, 4))

    metrics = _evaluate(qas, fragments, labels, frag_pos, assignments, diag)
    say("metrics: " + ", ".join(
        f"{k}={v}" for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)))

    result = LayoutResult(
        metrics=metrics,
        kp_count=len(clusters),
        fragment_count=len(fragments),
        qa_count=len(qas),
        edge_count=len(edges),
        collapsed=bool(diag["collapsed"]),
        persisted=False,
    )

    if persist:
        _persist(db, qas, fragments, clusters, labels, assignments,
                 positions, anchors, diag, generated, encoder_version)
        result.persisted = True
        say(f"persisted: {len(clusters)} KPs, {len(qas)} memberships")

    return result


# ============================================================
# Evaluation
# ============================================================


def _evaluate(qas: list[dict], fragments: list[dict], labels: np.ndarray,
              positions: np.ndarray, assignments: dict, diag: dict) -> dict:
    metrics = layout_metrics(positions, labels, anchors=None)
    metrics = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in metrics.items()}
    metrics["collapsed"] = bool(diag["collapsed"])
    metrics["epochs_run"] = diag["epochs_run"]

    qa_paper = {int(qa["id"]): qa.get("paper", "") for qa in qas}
    qa_topic = {int(qa["id"]): qa.get("topic", "") for qa in qas}

    # 1. cross-paper recurrence recall (label-free concept ground truth)
    frag_paper = {str(f["point_id"]): qa_paper.get(int(f["qa_id"]), "")
                  for f in fragments}
    groups: dict[str, list[int]] = {}
    for idx, frag in enumerate(fragments):
        key = frag.get("norm_key") or _norm_key(frag.get("point_text") or "")
        groups.setdefault(key, []).append(idx)

    hits = total = 0
    for idxs in groups.values():
        papers = {frag_paper[str(fragments[i]["point_id"])] for i in idxs}
        if len(papers) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if (frag_paper[str(fragments[idxs[a]]["point_id"])]
                        == frag_paper[str(fragments[idxs[b]]["point_id"])]):
                    continue
                total += 1
                hits += int(labels[idxs[a]] == labels[idxs[b]])
    metrics["cross_paper_recall"] = round(hits / total, 4) if total else 0.0
    metrics["cross_paper_pairs"] = total

    # 2. per-KP single-paper share (guard against paper-clustered layouts)
    shares = []
    for kp_id in {v[0] for v in assignments.values()}:
        papers = [qa_paper[qa_id] for qa_id, (kpid, _s) in assignments.items()
                  if kpid == kp_id]
        if not papers:
            continue
        counts: dict[str, int] = {}
        for p in papers:
            counts[p] = counts.get(p, 0) + 1
        shares.append(max(counts.values()) / len(papers))
    metrics["single_paper_share"] = round(float(np.mean(shares)), 4) if shares else 0.0

    # 3. legacy-topic purity (diagnostic only)
    kp_topics: dict[str, dict[str, int]] = {}
    for qa_id, (kp_id, _s) in assignments.items():
        bucket = kp_topics.setdefault(kp_id, {})
        topic = qa_topic.get(qa_id, "")
        bucket[topic] = bucket.get(topic, 0) + 1
    purities = [max(b.values()) / sum(b.values()) for b in kp_topics.values()
                if sum(b.values())]
    metrics["legacy_topic_purity"] = (
        round(float(np.mean(purities)), 4) if purities else 0.0)

    return metrics


# ============================================================
# Persistence
# ============================================================


def _persist(db, qas, fragments, clusters, labels, assignments, positions,
             anchors, diag, generated, encoder_version) -> None:
    from ..models import KPSpec

    epoch = diag["epochs_run"]
    version = encoder_version or "embedding:v1"

    with db.transaction():
        if generated:
            db.fragment.insert_batch([
                {
                    "point_id": f["point_id"],
                    "qa_id": int(f["qa_id"]),
                    "point_text": f["point_text"],
                    "marks": int(f.get("marks", 1)),
                }
                for f in fragments
            ])

        db.kp.reset_memberships()

        kp_ids: list[str] = []
        for label, cluster in clusters.items():
            kp_ids.append(cluster["kp_id"])
            db.kp.upsert(KPSpec(
                kp_id=cluster["kp_id"],
                name=cluster["kp_id"],
                cluster_id=int(label),
                centroid_vector=np.asarray(
                    cluster["centroid"], dtype=np.float32).tobytes(),
                cohesion=cluster["cohesion"],
                evidence_count=cluster["evidence"],
                quality=cluster["quality"],
                needs_naming=1,
                encoder_version=version,
                core_concept=(cluster["representative"].get("point_text") or "")[:200],
            ))
            db.vector.upsert_kp_vector(
                cluster["kp_id"],
                np.asarray(cluster["centroid"], dtype=np.float64))
            db.topic.upsert(cluster["kp_id"], name=cluster["kp_id"],
                            quality=cluster["quality"])
            db.topic.update_stats(cluster["kp_id"], mass=cluster["evidence"],
                                  cohesion=cluster["cohesion"],
                                  stability=cluster["cohesion"])
            for frag in cluster["members"]:
                db.topic.set_fragment_membership(
                    str(frag["point_id"]), cluster["kp_id"], loyalty=0.5)

        for qa_id, (kp_id, strength) in assignments.items():
            db.kp.set_membership(qa_id, kp_id, strength)

        db.kp.mark_archived(kp_ids)
        db.topic.prune(kp_ids)
        db.vector.prune_kp_vectors(kp_ids)

        rows = []
        for idx, frag in enumerate(fragments):
            rows.append({
                "object_type": "fragment",
                "object_id": str(frag["point_id"]),
                "position": positions[idx],
                "anchor": anchors[idx],
                "encoder_version": version,
                "epoch": epoch,
            })
        for qa_pos, qa in enumerate(qas):
            row = len(fragments) + qa_pos
            rows.append({
                "object_type": "qa",
                "object_id": str(qa["id"]),
                "position": positions[row],
                "anchor": anchors[row],
                "encoder_version": version,
                "epoch": epoch,
            })
        for cluster in clusters.values():
            rows.append({
                "object_type": "kp",
                "object_id": cluster["kp_id"],
                "position": cluster["centroid"],
                "anchor": cluster["centroid"],
                "encoder_version": version,
                "epoch": epoch,
            })
        db.vector.upsert_positions_batch(rows)
