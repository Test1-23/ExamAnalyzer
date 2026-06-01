"""Topic merge: cosine similarity + Flash review for ambiguous pairs."""
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import TOPIC_MERGE_COS_THRESHOLD, TOPIC_MERGE_AMBIGUOUS_THRESHOLD
from .deepseek_client import call_flash
from .embedding_cluster import detect_content_lang, EmbeddingClusterer
from .knowledge_base import QADatabase
from .utils import get_worker_limit


def _escape_pct(text: str) -> str:
    return text.replace("%", "%%")


def _resolve_transitive(mergers: dict) -> dict:
    """Resolve transitive merge chains into flat one-step mappings.

    {"B": "A", "C": "B"} → {"B": "A", "C": "A"} (C->B->A flattened to C->A).
    Returns {old_topic: ultimate_canonical} with cycles detected and logged.
    """
    from .logger import get_logger
    _log = get_logger()

    flat = {}
    for old_topic in mergers:
        visited = {old_topic}
        canonical = mergers[old_topic]
        while canonical in mergers:
            if canonical in visited:
                _log.warning(
                    "Topic merge cycle for '%s' — breaking at '%s'",
                    old_topic, canonical)
                break
            visited.add(canonical)
            canonical = mergers[canonical]
        if canonical != old_topic:
            flat[old_topic] = canonical
    return flat


def merge_similar_topics(db: QADatabase, client, debug) -> None:
    """Merge topics with similar answer content. Batch-encodes all topics once."""
    groups = db.get_topic_groups()
    topics = [(t, qas) for t, qas in groups.items() if t and t != "(uncategorized)"]
    n = len(topics)
    if n < 2:
        return

    topic_names = [t for t, _ in topics]
    all_answers = [" ".join(qa["answer_text"] for qa in qas) for _, qas in topics]

    try:
        clusterer = EmbeddingClusterer(all_answers)
        vecs = clusterer.vectors
        cos_matrix = vecs @ vecs.T
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Topic merge encoding", "", e)
        return

    mergers = {}
    done = set()
    ambiguous = []
    for i in range(n):
        if topic_names[i] in done:
            continue
        for j in range(i + 1, n):
            if topic_names[j] in done:
                continue
            cos = float(cos_matrix[i][j])
            if cos >= TOPIC_MERGE_COS_THRESHOLD:
                cnt_i = len(groups.get(topic_names[i], []))
                cnt_j = len(groups.get(topic_names[j], []))
                if cnt_i >= cnt_j:
                    canonical = topic_names[i]
                    mergers[topic_names[j]] = canonical
                    done.add(topic_names[j])
                else:
                    canonical = topic_names[j]
                    mergers[topic_names[i]] = canonical
                    done.add(topic_names[i])
                from .error_utils import log_info
                log_info(debug, "Topic merge", f"-> '{canonical}' (cos={cos:.2f})")
            elif cos >= TOPIC_MERGE_AMBIGUOUS_THRESHOLD:
                ambiguous.append((topic_names[i], topic_names[j], cos))

    if ambiguous:
        merged_by_flash = _flash_review_merges(ambiguous, db, client, debug)
        mergers.update(merged_by_flash)
        for t in merged_by_flash:
            done.add(t)

    if mergers:
        # Resolve transitive chains (A→B→C) into flat one-step mappings (A→C, B→C)
        flat_mergers = _resolve_transitive(mergers)

        merged_count = 0
        for old_topic, canonical in flat_mergers.items():
            merged_count += db.qa.rename_topic(canonical, old_topic)

        all_links = db.conn.execute(
            "SELECT src_topic, dst_topic, count FROM topic_links"
        ).fetchall()
        merged = {}
        for r in all_links:
            src = flat_mergers.get(r["src_topic"], r["src_topic"])
            dst = flat_mergers.get(r["dst_topic"], r["dst_topic"])
            if src == dst:
                continue
            key = (src, dst)
            merged[key] = merged.get(key, 0) + r["count"]
        with db.transaction():
            db.conn.execute("DELETE FROM topic_links")
            for (src, dst), total in merged.items():
                db.conn.execute(
                    "INSERT INTO topic_links (src_topic, dst_topic, count) VALUES (?, ?, ?)",
                    (src, dst, total),
                )
        from .error_utils import log_info
        log_info(debug, "Topic merge", f"{len(flat_mergers)} groups, {merged_count} QAs affected, "
              f"{len(all_links)} links -> {len(merged)} after merge")


def _flash_review_merges(ambiguous: list, db: QADatabase, client, debug) -> dict:
    """Send ambiguous topic pairs to Flash for merge review."""
    if not ambiguous:
        return {}

    topic_texts = {}
    for t1, t2, _ in ambiguous:
        for t in (t1, t2):
            if t not in topic_texts:
                rows = db.conn.execute("SELECT answer_text FROM qa_pairs WHERE topic=?", (t,)).fetchall()
                topic_texts[t] = " ".join(r["answer_text"] for r in rows) if rows else ""

    sample_text = " ".join(topic_texts.values())[:2000]
    lang = detect_content_lang(sample_text)
    if lang == 'en':
        sys = "Decide whether two topics should be merged. Output JSON."
        usr_tpl = ("Topic A '%s': %s\n\nTopic B '%s': %s\n\n"
                   "Cosine similarity: %.2f\n"
                   "Should these topics be merged? Similar names but different content -> do NOT merge.\n"
                   'If merge: {"merge": true, "canonical": "chosen name"} '
                   'If no merge: {"merge": false, "canonical": ""}')
    else:
        sys = "判断两个主题是否应合并。Output JSON."
        usr_tpl = ("主题A '%s': %s\n\n主题B '%s': %s\n\n"
                   "余弦相似度: %.2f\n"
                   "这两个主题是否应该合并? 注意: 标题相似但考察内容不同则不应合并。\n"
                   '若合并: {"merge": true, "canonical": "保留的主题名"} '
                   '若不合并: {"merge": false, "canonical": ""}')

    mergers = {}

    def _review_one(t1, t2, cos):
        a1 = topic_texts.get(t1, "")
        a2 = topic_texts.get(t2, "")
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr_tpl % (_escape_pct(t1), _escape_pct(a1[:500]), _escape_pct(t2), _escape_pct(a2[:500]), cos)},
        ]
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            if isinstance(result, dict) and result.get("merge"):
                canonical = result.get("canonical", "")
                if not canonical:
                    return None
                if canonical.lower() == t1.lower():
                    return [(t2, canonical)]
                elif canonical.lower() == t2.lower():
                    return [(t1, canonical)]
                else:
                    # Flash returned a new/renamed canonical — merge both into it
                    return [(t1, canonical), (t2, canonical)]
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "Flash merge review", f"t1={t1},t2={t2}", e)
        return None

    with ThreadPoolExecutor(max_workers=get_worker_limit(len(ambiguous), api_heavy=True)) as executor:
        futures = {executor.submit(_review_one, t1, t2, cos): (t1, t2) for t1, t2, cos in ambiguous}
        for future in as_completed(futures):
            mappings = future.result()
            if mappings:
                for t_from, canonical in mappings:
                    if t_from not in mergers or len(canonical) < len(mergers[t_from]):
                        mergers[t_from] = canonical
                    debug(f"  Flash merge: '{t_from}' -> '{canonical}'")

    return mergers
