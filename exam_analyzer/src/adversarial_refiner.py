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
from .utils import get_worker_limit


def _build_challenger_prompt(kp: dict, qas: list[dict], lang: str, prev_challenges: str = ""):
    """Build prompt for the challenger agent."""
    qa_texts = ""
    for i, qa in enumerate(qas[:5]):
        qa_texts += f"QA{i+1}: Q: {qa['question_text'][:300]}\nA: {qa['answer_text'][:300]}\n\n"

    if lang == 'en':
        sys = "You are a rigorous knowledge reviewer. Find flaws in this knowledge point. Output JSON."
        usr = (
            f"Knowledge Point: [{kp['name']}]\n"
            f"Concept: {kp.get('core_concept', '') or kp.get('description', '')}\n"
            f"Detail: {kp.get('core_detail', '')}\n\n"
            f"Supporting QAs:\n{qa_texts}\n"
            f"{'Previous challenges (already addressed): ' + prev_challenges if prev_challenges else ''}"
            "Find issues:\n"
            "1. Does the concept statement match ALL supporting QAs? (over-generalization?)\n"
            "2. Is any important nuance from the QAs missing in the concept/detail?\n"
            "3. Are there edge cases the KP doesn't cover?\n"
            "4. Could this KP be split into two separate concepts?\n\n"
            "If you find NO issues, return PASS. Otherwise list specific problems.\n"
            'Return JSON: {"pass": true/false, "issues": ["issue 1", "issue 2"], '
            '"severity": "critical|minor|cosmetic"}'
        )
    else:
        sys = "你是一个严谨的知识审核专家。找出知识点的缺陷。Output JSON。"
        usr = (
            f"知识点: [{kp['name']}]\n"
            f"概念: {kp.get('core_concept', '') or kp.get('description', '')}\n"
            f"细节: {kp.get('core_detail', '')}\n\n"
            f"支撑QA:\n{qa_texts}\n"
            f"{'已处理的挑战: ' + prev_challenges if prev_challenges else ''}"
            "找出问题:\n"
            "1. 概念陈述是否与所有支撑QA一致？\n"
            "2. 是否遗漏了QA中的重要细节？\n"
            "3. 是否有边缘情况未覆盖？\n"
            "4. 这个KP是否应拆分为两个独立概念？\n\n"
            "如无问题返回PASS，否则列出具体问题。\n"
            '返回 JSON: {"pass": true/false, "issues": ["问题1"], "severity": "critical|minor|cosmetic"}'
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def _build_defender_prompt(kp: dict, issues: list[str], qas: list[dict], lang: str):
    """Build prompt for the defender agent."""
    qa_texts = ""
    for i, qa in enumerate(qas[:5]):
        qa_texts += f"QA{i+1}: Q: {qa['question_text'][:300]}\nA: {qa['answer_text'][:300]}\n\n"

    issues_text = "\n".join(f"- {i}" for i in issues)

    if lang == 'en':
        sys = "You are a knowledge curator. Defend or revise this knowledge point against challenges. Output JSON."
        usr = (
            f"Knowledge Point: [{kp['name']}]\n"
            f"Current concept: {kp.get('core_concept', '') or kp.get('description', '')}\n"
            f"Current detail: {kp.get('core_detail', '')}\n\n"
            f"Challenges:\n{issues_text}\n\n"
            f"Supporting QAs:\n{qa_texts}\n"
            "For each challenge: either revise the KP to address it, or explain why it's invalid.\n"
            "Return the revised concept and detail.\n"
            'Return JSON: {"revised_concept": "...", "revised_detail": "...", '
            '"changes_made": ["change 1"], "challenges_dismissed": ["dismissed 1"]}'
        )
    else:
        sys = "你是一个知识策展人。针对挑战为知识点辩护或修订。Output JSON。"
        usr = (
            f"知识点: [{kp['name']}]\n"
            f"当前概念: {kp.get('core_concept', '') or kp.get('description', '')}\n"
            f"当前细节: {kp.get('core_detail', '')}\n\n"
            f"挑战:\n{issues_text}\n\n"
            f"支撑QA:\n{qa_texts}\n"
            "对每个挑战: 修订KP或解释为何无效。返回修订后的概念和细节。\n"
            '返回 JSON: {"revised_concept": "...", "revised_detail": "...", '
            '"changes_made": ["修改1"], "challenges_dismissed": ["驳回1"]}'
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


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

        # Challenger
        chal_msgs = _build_challenger_prompt(kp, qas, lang, prev_issues)
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

        # Defender
        def_msgs = _build_defender_prompt(kp, issues, qas, lang)
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

        if lang == 'en':
            sys = "Check these knowledge points for cross-KP consistency issues. Output JSON."
            usr = f"{kp_texts}\nCheck: duplicates, contradictions, merges, dependency direction.\n" \
                  'Return: {"issues": [{"kp_a":"id1","kp_b":"id2","issue":"...","suggestion":"...","action":"merge|split|no_change"}]}'
        else:
            sys = "检查这些知识点之间的跨KP一致性问题。Output JSON。"
            usr = f"{kp_texts}\n检查: 重复/矛盾/应合并/依赖方向不一致\n" \
                  '返回: {"issues": [{"kp_a":"id1","kp_b":"id2","issue":"...","suggestion":"...","action":"merge|split|no_change"}]}'

        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
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

    if lang == 'en':
        sys = "Decide whether this KP should be split into two. Output JSON."
        usr = (
            f"KP: [{kp['name']}] — {kp.get('core_concept', '') or kp.get('description', '')}\n\n"
            f"Member QAs:\n{qa_texts}\n"
            "Should this KP be split? If yes, provide two sub-concepts and assign each QA to one.\n"
            f'Return: {{"split": true/false, "kp_a": {{"concept": "...", "qa_ids": {ex_a_json}}}, '
            f'"kp_b": {{"concept": "...", "qa_ids": {ex_b_json}}}}}'
        )
    else:
        sys = "判断此KP是否应拆分为两个。Output JSON。"
        usr = (
            f"KP: [{kp['name']}] — {kp.get('core_concept', '') or kp.get('description', '')}\n\n"
            f"成员QA:\n{qa_texts}\n"
            "此KP是否应拆分？若是，提供两个子概念并分配QA。\n"
            f'返回: {{"split": true/false, "kp_a": {{"concept": "...", "qa_ids": {ex_a_json}}}, '
            f'"kp_b": {{"concept": "...", "qa_ids": {ex_b_json}}}}}'
        )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
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
