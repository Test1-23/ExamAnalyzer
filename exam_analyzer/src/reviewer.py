"""Post-distillation review: structural + content review via Flash, plus formatting."""
from .deepseek_client import call_flash
from .embedding_cluster import detect_content_lang


def review_distilled(content: str, client, topic_links: dict, topic_related: dict, debug) -> str:
    """Post-distillation review in focused batches.

    Batch A (LLM): structural — duplicates, topic mismatches
    Batch B (LLM): content — scoring examples, calculation steps
    Rules-based: formatting normalization
    Then insert See also / Related from topic_links and topic_related.
    """
    if not content.strip() or len(content) < 500:
        return content

    lang = detect_content_lang(content)

    # ---- Batch A: Structural review ----
    if lang == 'en':
        sys_a = "You are a knowledge base reviewer. Find structural issues. Output JSON."
        usr_a = (f"{content}\n\n"
                 "Fix these structural issues:\n"
                 "1. Duplicate KPs across topics — merge into best topic\n"
                 "2. KP content contradicts its topic name OR topic name is too broad/narrow — fix topic or remove\n"
                 "Preserve everything else as-is.\n"
                 'Return JSON: {"content": "corrected file content"}')
    else:
        sys_a = "你是知识库结构审核专家。查找结构问题。Output JSON。"
        usr_a = (f"{content}\n\n"
                 "修复以下结构问题:\n"
                 "1. 不同topic下的重复KP — 合并到最合适的topic\n"
                 "2. KP内容与topic名矛盾或topic名不匹配 — 修正topic名或移除\n"
                 "保留其他内容不变。\n"
                 '返回 JSON: {"content": "修正后的完整文件"}')

    reviewed = content
    try:
        result, _ = call_flash(client, [{"role": "system", "content": sys_a},
                                      {"role": "user", "content": usr_a}],
                           max_retries=1, debug_callback=debug)
        if isinstance(result, dict) and result.get("content"):
            reviewed = result["content"]
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Review batch A", "", e)

    # ---- Batch B: Content review ----
    if lang == 'en':
        sys_b = "You are a knowledge base reviewer. Fix scoring-related issues. Output JSON."
        usr_b = (f"{reviewed}\n\n"
                 "Fix these content issues:\n"
                 "7. Scoring fields missing concrete example answer sentence — add a sample full-mark answer in quotes\n"
                 "8. Calculation KPs scoring missing step-by-step mark allocation — add mark breakdown per step\n"
                 "Preserve everything else as-is.\n"
                 'Return JSON: {"content": "corrected file content"}')
    else:
        sys_b = "你是知识库内容审核专家。修复评分相关问题。Output JSON。"
        usr_b = (f"{reviewed}\n\n"
                 "修复以下内容问题:\n"
                 "7. Scoring字段缺少具体答案示例 — 补充带引号的完整答案范例\n"
                 "8. 计算类KP的Scoring缺少分步给分 — 补充每步分值\n"
                 "保留其他内容不变。\n"
                 '返回 JSON: {"content": "修正后的完整文件"}')

    try:
        result, _ = call_flash(client, [{"role": "system", "content": sys_b},
                                      {"role": "user", "content": usr_b}],
                           max_retries=1, debug_callback=debug)
        if isinstance(result, dict) and result.get("content"):
            reviewed = result["content"]
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Review batch B", "", e)

    # ---- Rules-based pass: formatting normalization ----
    reviewed = _normalize_formatting(reviewed)

    # ---- Insert See also from accumulated topic_links + topic_related ----
    if topic_links or topic_related:
        reviewed = _insert_see_also(reviewed, topic_links, topic_related, debug)

    return reviewed


def _normalize_formatting(content: str) -> str:
    """Rules-based formatting normalization (no LLM call)."""
    lines = content.split("\n")
    result = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            line = "  " + "- " + stripped[2:]
        line = line.rstrip()
        is_blank = not line.strip()
        if is_blank:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)
    return "\n".join(result)


def _insert_see_also(content: str, topic_links: dict, topic_related: dict, debug) -> str:
    """Insert See also / Related lines based on cross-topic references.

    topic_links (strong): Phase 2 runtime cross-topic QA usage -> "See also: X (ref N)"
    topic_related (medium): retrieval co-occurrence -> "Related: X"
    """
    if not topic_links and not topic_related:
        return content

    annotations: dict[str, list[str]] = {}

    for (src, dst), count in topic_links.items():
        if count >= 2 and src and dst:
            annotations.setdefault(src, []).append(f"See also: {dst} (ref {count})")

    for src, refs in topic_related.items():
        if src not in annotations:
            parts = [f"{rt}" for rt, cnt in refs[:3] if cnt >= 2]
            if parts:
                annotations.setdefault(src, []).append(f"Related: {', '.join(parts)}")

    if not annotations:
        return content

    lines = content.split("\n")
    out: list[str] = []
    seen_sections: set[str] = set()
    topics_by_len = sorted(annotations.keys(), key=len, reverse=True)

    for line in lines:
        stripped = line.strip()
        current_topic = ""
        for topic in topics_by_len:
            if stripped.startswith(topic) and (stripped == topic or stripped[len(topic)] == ' '):
                current_topic = topic
                break
            if stripped.startswith(topic + "  [") or stripped.startswith(topic + " ["):
                current_topic = topic
                break

        if current_topic and current_topic not in seen_sections:
            out.append(line)
            for note in annotations.get(current_topic, []):
                out.append(f"   {note}")
            seen_sections.add(current_topic)
        else:
            out.append(line)

    from .error_utils import log_info
    log_info(debug, "Reviewer", f"See also: {len(seen_sections)} topics annotated "
          f"(links={len(topic_links)}, related={len(topic_related)})")
    return "\n".join(out)
