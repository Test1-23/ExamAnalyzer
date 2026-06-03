"""Adversarial knowledge refinement: challenger/defender debate for KP quality.

Each KP is validated through rounds of adversarial debate:
- Challenger: finds flaws, missing context, over-generalizations
- Defender: addresses challenges, modifies KP if needed
- Converges when challenger returns "PASS" for 2 consecutive rounds or max rounds reached.

Also performs cross-KP consistency checking in batches.
"""

import json
import os

from .deepseek_client import call_flash
from .knowledge_base import QADatabase
from .embedding_cluster import detect_content_lang
from .models import KPSpec, KpEdgeSpec
from .logger import get_logger

_log = get_logger()

from .constants import MAX_ROUNDS, PASS_THRESHOLD
from .prompt_factory import PromptType, PromptBuilder
from .utils import get_worker_limit


def refine_kp(db: QADatabase, kp_id: str, client, debug=None) -> dict:
    """Run adversarial refinement on a single KP. Returns final KP state."""
    kp = db.get_kp_by_id(kp_id)
    if not kp:
        return {}

    qas = db.kp.get_representative_qas(kp_id)
    if not qas:
        return kp

    lang = detect_content_lang(
        (kp.get("core_concept", "") or kp.get("description", "")) +
        " ".join(qa.get("question_text", "") for qa in qas[:2])
    )

    history = []
    prev_issues = ""
    pass_streak = 0
    total_rounds = 0
    any_refinement_done = False

    for round_num in range(1, MAX_ROUNDS + 1):
        total_rounds = round_num

        # Challenger — pre-compute QA text block
        qa_block = ""
        for i, qa in enumerate(qas[:5]):
            qa_block += f"QA{i+1}: Q: {qa['question_text'][:300]}\nA: {qa['answer_text'][:300]}\n\n"
        prev_block = f"Previous challenges (already addressed): {prev_issues}\n" if prev_issues else ""
        if lang != 'en':
            prev_block = f"已处理的挑战: {prev_issues}\n" if prev_issues else ""
        chal_msgs = PromptBuilder.build(PromptType.KP_CHALLENGER,
            kp_name=kp.get("name", ""),
            kp_concept=kp.get("core_concept", "") or kp.get("description", ""),
            kp_detail=kp.get("core_detail", ""),
            qa_texts=qa_block,
            prev_challenges=prev_block,
            lang=lang,
        )
        try:
            chal_result, _ = call_flash(client, chal_msgs, max_retries=1, debug=debug)
            chal_result = chal_result if isinstance(chal_result, dict) else {}
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "KP refine", f"challenger,kp={kp_id},round={round_num}", e)
            break

        if chal_result.get("pass"):
            pass_streak += 1
            history.append({"round": round_num, "role": "challenger", "pass": True})
            if pass_streak >= PASS_THRESHOLD:
                break
            continue

        pass_streak = 0
        issues = chal_result.get("issues", [])
        severity = chal_result.get("severity", "minor")
        history.append({"round": round_num, "role": "challenger", "pass": False, "issues": issues, "severity": severity})

        # Defender — pre-compute QA text block and issues text
        qa_block = ""
        for i, qa in enumerate(qas[:5]):
            qa_block += f"QA{i+1}: Q: {qa['question_text'][:300]}\nA: {qa['answer_text'][:300]}\n\n"
        issues_text = "\n".join(f"- {i}" for i in issues)
        def_msgs = PromptBuilder.build(PromptType.KP_DEFENDER,
            kp_name=kp.get("name", ""),
            kp_concept=kp.get("core_concept", "") or kp.get("description", ""),
            kp_detail=kp.get("core_detail", ""),
            issues_text=issues_text,
            qa_texts=qa_block,
            lang=lang,
        )
        try:
            def_result, _ = call_flash(client, def_msgs, max_retries=1, debug=debug)
            def_result = def_result if isinstance(def_result, dict) else {}
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "KP refine", f"defender,kp={kp_id},round={round_num}", e)
            break

        if def_result.get("revised_concept"):
            kp["core_concept"] = def_result["revised_concept"]
            any_refinement_done = True
        if def_result.get("revised_detail"):
            kp["core_detail"] = def_result["revised_detail"]
            any_refinement_done = True

        history.append({
            "round": round_num, "role": "defender",
            "changes": def_result.get("changes_made", []),
            "dismissed": def_result.get("challenges_dismissed", []),
        })

        prev_issues = "\n".join(
            f"[R{round_num}] {i}" for i in issues
        )

        # Update KP in DB with revised content
        db.kp.upsert(KPSpec(
            kp_id=kp_id,
            name=kp.get("name", ""),
            description=kp.get("description", ""),
            core_concept=kp.get("core_concept", ""),
            core_detail=kp.get("core_detail", ""),
            cohesion=kp.get("cohesion"),
            evidence_count=kp.get("evidence_count", 0),
            quality="accepted",
            challenge_history=json.dumps(history),
        ))

    # Final quality determination
    # pass_streak >= PASS_THRESHOLD means adversarial review completed cleanly (no issues found)
    # — this IS a successful refinement, even if no text was changed.
    if pass_streak >= PASS_THRESHOLD:
        quality = "verified"
    elif not any_refinement_done:
        quality = kp.get("quality", "draft")  # interrupted, keep existing
    elif total_rounds >= MAX_ROUNDS:
        quality = "disputed"  # exhausted rounds without pass streak
    else:
        quality = "accepted"

    db.kp.upsert(KPSpec(
        kp_id=kp_id,
        name=kp.get("name", ""),
        description=kp.get("description", ""),
        core_concept=kp.get("core_concept", ""),
        core_detail=kp.get("core_detail", ""),
        cohesion=kp.get("cohesion"),
        evidence_count=kp.get("evidence_count", 0),
        quality=quality,
        challenge_history=json.dumps(history),
    ))

    if debug:
        from .error_utils import log_info
        log_info(debug, "KP refine", f"{kp_id}: quality={quality}, rounds={total_rounds}, "
                 f"pass_streak={pass_streak}")

    return kp


def cross_kp_consistency(db: QADatabase, kp_ids: list[str], client, debug=None) -> dict:
    """Check consistency across KPs in batches. Returns issues found."""
    if len(kp_ids) < 2:
        return {"issues": []}

    batch_size = 10
    batches = [(kp_ids[i:i+batch_size]) for i in range(0, len(kp_ids), batch_size)]

    def _check_batch(batch):
        kps = [db.get_kp_by_id(kid) for kid in batch]
        kps = [k for k in kps if k]
        if len(kps) < 2:
            return []

        sample_text = " ".join(k.get("name", "") + " " + (k.get("core_concept", "") or k.get("description", "")) for k in kps)
        lang = detect_content_lang(sample_text[:2000])

        kp_texts = ""
        for k in kps:
            kp_texts += f"[{k['id']}] {k['name']}: {k.get('core_concept', '') or k.get('description', '')}\n"

        messages = PromptBuilder.build(PromptType.KP_CONSISTENCY,
            kp_texts=kp_texts, lang=lang)
        try:
            result, _ = call_flash(client, messages, max_retries=1, debug=debug)
            return result.get("issues", []) if isinstance(result, dict) else []
        except Exception as e:
            from .error_utils import log_exception
            log_exception(debug, "KP consistency", "", e)
            return []

    all_issues = []
    if batches:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_w = get_worker_limit(len(batches), api_heavy=True)
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(_check_batch, b): b for b in batches}
            for future in as_completed(futures):
                try:
                    all_issues.extend(future.result())
                except Exception as e:
                    from .error_utils import log_exception
                    log_exception(debug, "KP consistency", "thread", e)

    if debug:
        from .error_utils import log_info
        log_info(debug, "KP consistency", f"{len(all_issues)} issues across {len(kp_ids)} KPs "
                 f"({len(batches)} batches parallel)")

    return {"issues": all_issues}




# ============================================================
# KP auto-split: actually split over-broad KPs into two
# ============================================================

def auto_split_kp(db: QADatabase, kp_id: str, client, debug=None) -> list[str]:
    """Try to split an over-broad KP into two. Returns new KP IDs (empty if no split)."""
    kp = db.get_kp_by_id(kp_id)
    if not kp:
        return []

    qas = db.kp.get_representative_qas(kp_id)
    if len(qas) < 4:
        return []

    # Get all member QAs
    member_rows = db.conn.execute(
        "SELECT qa_id FROM qa_kp_membership WHERE kp_id=?", (kp_id,)
    ).fetchall()
    all_qa_ids = [r["qa_id"] for r in member_rows]
    if len(all_qa_ids) < 4:
        return []

    all_qas = db.qa.get_by_ids(all_qa_ids)

    lang = detect_content_lang(
        (kp.get("core_concept", "") or kp.get("description", "")) +
        " ".join(qa.get("question_text", "") for qa in all_qas[:2])
    )

    # Build prompt to decide if and how to split
    qa_texts = ""
    prompt_qa_ids = []
    for i, qa in enumerate(all_qas[:10]):
        qa_texts += f"[{qa['id']}] {qa['question_text'][:200]}\n  A: {qa['answer_text'][:200]}\n\n"
        prompt_qa_ids.append(qa["id"])

    # Build example IDs using json.dumps for safe injection (handles int/float/str)
    example_ids = all_qa_ids[:2] if len(all_qa_ids) >= 2 else all_qa_ids
    ex_a_json = json.dumps([example_ids[0]])
    ex_b_json = json.dumps([example_ids[-1] if len(example_ids) >= 2 else example_ids[0]])

    messages = PromptBuilder.build(PromptType.KP_SPLIT,
        kp_name=kp.get("name", ""),
        kp_concept=kp.get("core_concept", "") or kp.get("description", ""),
        kp_detail=kp.get("core_detail", ""),
        qa_texts=qa_texts,
        ex_a=ex_a_json,
        ex_b=ex_b_json,
        lang=lang,
    )
    try:
        result, _ = call_flash(client, messages, max_retries=1, debug=debug)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Auto-split KP", f"kp={kp_id}", e)
        return []

    if not isinstance(result, dict) or not result.get("split"):
        return []

    new_ids = []
    for key in ("kp_a", "kp_b"):
        sub = result.get(key, {})
        if not sub or not sub.get("concept"):
            continue
        new_id = f"{kp_id}s{len(new_ids)}"
        sub_qa_ids = [int(i) for i in sub.get("qa_ids", []) if isinstance(i, (int, float))
                      and int(i) in all_qa_ids]  # validate: reject position indices, accept only real DB IDs
        if not sub_qa_ids:
            continue
        db.kp.upsert(KPSpec(kp_id=new_id, name=f"{kp['name']} ({chr(97+len(new_ids))})",
                     description=sub["concept"], core_concept=sub["concept"],
                     core_detail="", cohesion=kp.get("cohesion"),
                     evidence_count=len(sub_qa_ids), quality="draft"))
        with db.transaction():
            for qa_id in sub_qa_ids:
                db.conn.execute(
                    "INSERT OR REPLACE INTO qa_kp_membership (qa_id, kp_id, membership_strength) VALUES (?, ?, 0.8)",
                    (qa_id, new_id),
                )
                # Remove from parent KP to avoid duplicate membership
                db.conn.execute(
                    "DELETE FROM qa_kp_membership WHERE qa_id=? AND kp_id=?",
                    (qa_id, kp_id),
                )
        new_ids.append(new_id)

    if new_ids and debug:
        from .error_utils import log_info
        log_info(debug, "Auto-split", f"KP {kp_id} into {len(new_ids)} KPs: {new_ids}")

    return new_ids


# ============================================================
# KP auto-merge: actually merge KPs flagged as duplicates
# ============================================================

def auto_merge_kps(db: QADatabase, issues: list[dict], debug=None) -> int:
    """Merge KPs flagged by cross_kp_consistency. Returns number merged."""
    merged = 0
    for issue in issues:
        action = issue.get("action", "")
        if action:
            if action != "merge":
                continue
        else:
            # Fallback: keyword match on suggestion text (backward compat)
            suggestion = (issue.get("suggestion", "") or "").lower()
            if "merge" not in suggestion:
                continue

        kp_a = issue.get("kp_a", "")
        kp_b = issue.get("kp_b", "")
        if not kp_a or not kp_b or kp_a == kp_b:
            continue

        kp_a_data = db.get_kp_by_id(kp_a)
        kp_b_data = db.get_kp_by_id(kp_b)
        if not kp_a_data or not kp_b_data:
            continue

        # Merge worse into better: prefer higher evidence_count, then quality.
        # Skip merge if both KPs are equally good — let human decide.
        quality_rank = {"verified": 4, "accepted": 3, "draft": 2, "disputed": 1, None: 0}
        a_score = (kp_a_data.get("evidence_count", 0),
                   quality_rank.get(kp_a_data.get("quality"), 0))
        b_score = (kp_b_data.get("evidence_count", 0),
                   quality_rank.get(kp_b_data.get("quality"), 0))
        if a_score == b_score:
            if debug:
                from .error_utils import log_info
                log_info(debug, "Auto-merge skip", f"{kp_a} vs {kp_b}")
            continue
        if b_score > a_score:
            kp_a, kp_b = kp_b, kp_a

        with db.transaction():
            # Move all QAs from B to A
            db.conn.execute(
                "UPDATE qa_kp_membership SET kp_id=? WHERE kp_id=?",
                (kp_a, kp_b),
            )
            # Re-route edges from B to A, deduplicating conflicts
            edges = db.conn.execute(
                "SELECT source_kp, target_kp, edge_type, retrieval_weight, semantic_weight, "
                "sequential_weight, learning_path_weight, combined_strength, confidence "
                "FROM kp_edges WHERE source_kp=? OR target_kp=?",
                (kp_b, kp_b),
            ).fetchall()
            db.conn.execute("DELETE FROM kp_edges WHERE source_kp=? OR target_kp=?", (kp_b, kp_b))
            for edge in edges:
                new_src = kp_a if edge["source_kp"] == kp_b else edge["source_kp"]
                new_tgt = kp_a if edge["target_kp"] == kp_b else edge["target_kp"]
                if new_src == new_tgt:
                    continue
                db.kp.upsert_edge(KpEdgeSpec(
                    source_kp=new_src, target_kp=new_tgt,
                    edge_type=edge["edge_type"],
                    retrieval_weight=edge["retrieval_weight"],
                    semantic_weight=edge["semantic_weight"],
                    sequential_weight=edge["sequential_weight"],
                    learning_path_weight=edge["learning_path_weight"],
                    combined_strength=edge["combined_strength"],
                    confidence=edge["confidence"],
                ))
            # Delete B
            db.conn.execute("DELETE FROM knowledge_points WHERE id=?", (kp_b,))

        db.analysis.record_evolution(
            kp_id=kp_a,
            trigger_type="auto_merge",
            trigger_detail=f"Merged {kp_b} into {kp_a}: {issue.get('issue', '')}",
            old_state=f"two separate KPs",
            new_state=f"merged into {kp_a}",
            outcome="completed",
        )
        merged += 1

        if debug:
            from .error_utils import log_info
            log_info(debug, "Auto-merge", f"{kp_b} -> {kp_a} ({issue.get('issue', '')})")

    return merged
