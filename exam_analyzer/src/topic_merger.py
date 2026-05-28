"""Topic merge: cosine similarity + Flash review for ambiguous pairs."""
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import TOPIC_MERGE_COS_THRESHOLD, TOPIC_MERGE_AMBIGUOUS_THRESHOLD
from .deepseek_client import call_flash
from .embedding_cluster import detect_content_lang
from .knowledge_base import QADatabase
from .utils import get_worker_limit


def _escape_pct(text: str) -> str:
    return text.replace("%", "%%")


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
        from .embedding_cluster import EmbeddingClusterer
        clusterer = EmbeddingClusterer(all_answers)
        vecs = clusterer.vectors
        cos_matrix = vecs @ vecs.T
    except Exception as e:
        debug(f"  Topic merge encoding failed: {e}")
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
                debug(f"  Topic merge: -> '{canonical}' (cos={cos:.2f})")
            elif cos >= TOPIC_MERGE_AMBIGUOUS_THRESHOLD:
                ambiguous.append((topic_names[i], topic_names[j], cos))

    if ambiguous:
        merged_by_flash = _flash_review_merges(ambiguous, db, client, debug)
        mergers.update(merged_by_flash)
        for t in merged_by_flash:
            done.add(t)

    if mergers:
        merged_count = 0
        for old_topic, new_topic in mergers.items():
            merged_count += db.rename_topic(new_topic, old_topic)

        all_links = db.conn.execute(
            "SELECT src_topic, dst_topic, count FROM topic_links"
        ).fetchall()
        merged = {}
        for r in all_links:
            src = mergers.get(r["src_topic"], r["src_topic"])
            dst = mergers.get(r["dst_topic"], r["dst_topic"])
            if src == dst:
                continue
            key = (src, dst)
            merged[key] = merged.get(key, 0) + r["count"]
        db.conn.execute("DELETE FROM topic_links")
        for (src, dst), total in merged.items():
            db.conn.execute(
                "INSERT INTO topic_links (src_topic, dst_topic, count) VALUES (?, ?, ?)",
                (src, dst, total),
            )
        db.conn.commit()
        debug(f"  Topic merge: {len(mergers)} groups, {merged_count} QAs affected, "
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
            {"role": "user", "content": usr_tpl % (t1, _escape_pct(a1[:500]), t2, _escape_pct(a2[:500]), cos)},
        ]
        try:
            result = call_flash(client, messages, max_retries=1, debug_callback=debug)
            if isinstance(result, dict) and result.get("merge"):
                canonical = result.get("canonical", "")
                if not canonical:
                    return None
                return (t2, canonical)
        except Exception as e:
            debug(f"  Flash merge review failed for '{t1}'/'{t2}': {e}")
        return None

    with ThreadPoolExecutor(max_workers=get_worker_limit(len(ambiguous), api_heavy=True)) as executor:
        futures = {executor.submit(_review_one, t1, t2, cos): (t1, t2) for t1, t2, cos in ambiguous}
        for future in as_completed(futures):
            result = future.result()
            if result:
                t2, canonical = result
                if t2 not in mergers or len(canonical) < len(mergers[t2]):
                    mergers[t2] = canonical
                debug(f"  Flash merge: '{t2}' -> '{canonical}'")

    return mergers
