"""Knowledge distillation: Flash-driven extraction of KPs from topic-grouped QAs.

The Distiller class replaces the monolithic _distill_points function from pipeline.py.
Closures that captured 7 shared variables are now methods sharing state via ``self``.
"""
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .constants import MISSED_FILTER_THRESHOLD, MISSED_CLUSTER_THRESHOLD
from .deepseek_client import call_flash
from .embedding_cluster import (
    detect_content_lang, _get_model, MODEL_MAP, _detect_language, cluster_by_cosine,
)
from .knowledge_base import QADatabase
from .utils import get_worker_limit


def _escape_pct(text: str) -> str:
    """Double ``%`` signs so user text is safe for ``%``-format interpolation."""
    return text.replace("%", "%%")


def _build_missed_ref(db: QADatabase, topic: str, qas: list[dict], debug) -> str:
    """Build a reference text of recurring difficulty patterns from Phase 2 missed_points.

    Only returns patterns that pass MS similarity filter and appear >= 2 times.
    """
    raw_missed = db.analysis.get_missed_by_topic(topic)
    if len(raw_missed) < 3:
        return ""

    all_answers = [qa["answer_text"] for qa in qas if qa.get("answer_text")]
    if not all_answers:
        return ""

    try:
        model = _get_model(MODEL_MAP[_detect_language(raw_missed + all_answers)])
        missed_vecs = model.encode(raw_missed, normalize_embeddings=True, convert_to_numpy=True)
        answer_vecs = model.encode(all_answers, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Missed embedding", f"topic={topic}", e)
        return ""

    filtered = []
    for i, line in enumerate(raw_missed):
        best_cos = max(float(np.dot(missed_vecs[i], av)) for av in answer_vecs)
        if best_cos >= MISSED_FILTER_THRESHOLD:
            filtered.append(line)

    if len(filtered) < 2:
        return ""

    try:
        fvecs = model.encode(filtered, normalize_embeddings=True, convert_to_numpy=True)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug, "Missed cluster encoding", f"topic={topic}", e)
        return ""
    groups = cluster_by_cosine(fvecs, MISSED_CLUSTER_THRESHOLD, min_group_size=2)
    patterns = [filtered[g[0]] for g in groups]

    if not patterns:
        return ""

    ref = ("\nReference: the following difficulty patterns were observed when answering "
           "similar questions. Use them to inform pitfalls ONLY IF they align with "
           "markscheme scoring criteria:\n")
    for p in patterns[:5]:
        ref += f"- {p}\n"
    from .error_utils import log_info
    log_info(debug, "Missed patterns", f"'{topic}': {len(patterns)} from {len(filtered)}/{len(raw_missed)} lines")
    return ref


class Distiller:
    """Orchestrates knowledge distillation from QA groups to structured text output.

    Replaces the monolithic ``_distill_points`` function.  Shared state (db, client,
    debug, prompts) lives on the instance so that methods submitted to
    ThreadPoolExecutor stay thin.
    """

    def __init__(self, db: QADatabase, client, debug) -> None:
        self._db = db
        self._client = client
        self._debug = debug
        # Set by _build_prompts
        self._lang = "en"
        self._dist_sys = ""
        self._dist_usr = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Main entry point: distill all topics and return formatted text."""
        groups = self._db.get_topic_groups()
        if not groups:
            return ""

        weights = self._db.get_all_weights()
        topic_meta = self._compute_topic_meta(groups, weights)
        self._build_prompts(groups)

        cache = self._db.get_distillation_cache()
        topic_items, cached_results, skipped_count = self._prepare_topic_items(
            groups, weights, cache, topic_meta)

        if skipped_count:
            from .error_utils import log_info
            log_info(self._debug, "Incremental", f"reusing {skipped_count} cached, distilling {len(topic_items)} topics")

        if not topic_items and not cached_results:
            return ""

        # Split: small topics (<= 3 QAs) batched, large topics individual
        SMALL_TOPIC_THRESHOLD = 3
        BATCH_SIZE = 5
        small_topics = [t for t in topic_items if t[1] <= SMALL_TOPIC_THRESHOLD]
        large_topics = [t for t in topic_items if t[1] > SMALL_TOPIC_THRESHOLD]

        batches = []
        for i in range(0, len(small_topics), BATCH_SIZE):
            batches.append(small_topics[i:i + BATCH_SIZE])

        if batches:
            from .error_utils import log_info
            log_info(self._debug, "Batching", f"{len(small_topics)} small topics into {len(batches)} batches, "
                        f"{len(large_topics)} large topics individually")

        # Execute: batch + individual tasks via thread pool
        results = {}
        batched_topics = set()

        if batches or large_topics:
            with ThreadPoolExecutor(
                max_workers=get_worker_limit(len(batches) + len(large_topics), api_heavy=True)
            ) as executor:
                future_map = {}

                for batch in batches:
                    f = executor.submit(self._distill_batch, batch)
                    future_map[f] = ("batch", [t[0] for t in batch])

                for item in large_topics:
                    f = executor.submit(self._distill_one, *item)
                    future_map[f] = ("single", [item[0]])

                for future in as_completed(future_map):
                    try:
                        mode, topics_in_task = future_map[future]
                        result_val = future.result()
                        if mode == "batch":
                            for topic, text in result_val.items():
                                if text:
                                    results[topic] = text
                                    batched_topics.add(topic)
                        else:
                            topic, text = result_val
                            if text:
                                results[topic] = text
                    except Exception as e:
                        from .error_utils import log_exception
                        _, topics_in_task = future_map.get(future, ("unknown", []))
                        log_exception(self._debug, "Distillation thread", f"topics={topics_in_task}", e)

            # Fallback: small topics missed by batch get individual distillation
            missed_small = [t for t in small_topics if t[0] not in results]
            if missed_small:
                from .error_utils import log_info
                log_info(self._debug, "Batch missed", f"{len(missed_small)} topics, running individual distillation")
                for item in missed_small:
                    try:
                        topic, text = self._distill_one(*item)
                        if text:
                            results[topic] = text
                    except Exception as e:
                        from .error_utils import log_exception
                        log_exception(self._debug, "Fallback distillation", f"topic={item[0]}", e)

        # Assemble output in original topic order
        all_lines = []
        for topic in groups:
            if topic in results:
                all_lines.append(results[topic])
            elif topic in cached_results:
                all_lines.append(cached_results[topic])

        return "\n\n".join(all_lines) + "\n"

    # ------------------------------------------------------------------
    # Phase 1: Topic metadata
    # ------------------------------------------------------------------

    def _compute_topic_meta(self, groups: dict, weights: dict) -> dict:
        """Compute per-topic quality metrics (no side effects)."""
        topic_meta = {}
        for topic, qas in groups.items():
            if not topic or topic == "(uncategorized)":
                continue
            qa_means = [weights.get(qa["id"], {}).get("mean", 0.5) for qa in qas]
            best_lb = max((weights.get(qa["id"], {}).get("lower_bound", 0) for qa in qas), default=0)
            best_mean = max(qa_means, default=0.5)
            variance = statistics.variance(qa_means) if len(qa_means) >= 2 else 0.0
            quality_flag = ""
            if best_mean < 0.4 and variance < 0.02 and len(qa_means) >= 2:
                quality_flag = " [low quality - review suggested]"
            topic_meta[topic] = {
                "lb": best_lb, "mean": best_mean, "n_qa": len(qas),
                "variance": variance, "quality_flag": quality_flag,
            }
        return topic_meta

    # ------------------------------------------------------------------
    # Phase 2: Bilingual prompts
    # ------------------------------------------------------------------

    def _build_prompts(self, groups: dict) -> None:
        """Detect language and build distill prompts (sets instance attrs)."""
        all_text = " ".join(qa["question_text"] for g in groups.values() for qa in g)[:3000]
        self._lang = detect_content_lang(all_text)

        if self._lang == 'en':
            self._dist_sys = (
                "You are a knowledge distillation expert. Extract general knowledge points "
                "from exam answers. Output JSON. For each KP include: concept, detail, "
                "pitfall (common exam mistake, not extra knowledge), "
                "scoring (what earns marks + a sample full-mark answer phrase), "
                "confidence (\"high\" if from multiple/high-weight QAs, \"low\" if from single/low-weight QA). "
                "Prioritize high-weight QAs. "
                'Example: {"knowledge_points": ['
                '{"concept":"Binary addition proceeds column by column from LSB to MSB",'
                '"detail":"Per-bit rules: 0+0=0, 0+1=1, 1+1=0 carry 1",'
                '"pitfall":"Forgetting to add carry bits to the next column",'
                '"scoring":"Show working carries and final result. E.g. \'1+1=0 carry 1 to next column\'",'
                '"confidence":"high"}]}'
            )
            self._dist_usr = (
                "Topic: %s\n\n"
                "The following %d questions cover this topic:\n\n%s\n"
                "Task: distill general knowledge points. Format each as:\n"
                "  - concept: 1-sentence statement\n"
                "  - detail: key specifics or worked example\n"
                "  - pitfall: common exam mistake (not extra facts)\n"
                "  - scoring: what earns marks + sample answer phrase\n"
                "  - confidence: high|low\n\n"
                "Remove question-specific details (values, names). Keep common principles.\n"
                "Use consistent terminology: if source QAs use different names for the same concept, "
                "pick the most frequent term and use it throughout all KPs in this topic.\n"
                'Return JSON: {"knowledge_points": [{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}'
            )
        else:
            self._dist_sys = (
                "你是一个知识蒸馏专家。从题目的答案中提取通用知识点。Output JSON。"
                "每个知识点包含: concept(概念), detail(细节/例子), "
                "pitfall(常见考试错误，非额外知识点), "
                "scoring(得分要点 + 满分答案示例措辞), "
                "confidence(high=来自多个/高权重QA, low=来自单个/低权重QA)。"
                "优先从高权重QA提取。"
                '示例: {"knowledge_points": ['
                '{"concept":"二进制加法从LSB到MSB逐列进行",'
                '"detail":"逐位规则: 0+0=0, 0+1=1, 1+1=0进位1",'
                '"pitfall":"忘记将进位加到下一列",'
                '"scoring":"展示进位过程和最终结果。如 \'1+1=0 进位1至下一列\'",'
                '"confidence":"high"}]}'
            )
            self._dist_usr = (
                "主题: %s\n\n"
                "以下 %d 道题目涉及此主题:\n\n%s\n"
                "任务: 蒸馏出通用知识点。格式:\n"
                "  - concept: 1句话概念陈述\n"
                "  - detail: 关键细节或计算示例\n"
                "  - pitfall: 常见考试错误(非额外知识点)\n"
                "  - scoring: 得分要点 + 满分答案示例措辞\n"
                "  - confidence: high|low\n\n"
                "去除题目特定细节（数值、名称），保留共性技术原理。\n"
                "术语一致: 若多个题目的答案使用不同名称指代同一概念，"
                "选择出现最多的术语，在本主题的所有 KP 中统一使用。\n"
                '返回 JSON: {"knowledge_points": [{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}'
            )

    # ------------------------------------------------------------------
    # Phase 3: Topic item preparation
    # ------------------------------------------------------------------

    def _prepare_topic_items(self, groups: dict, weights: dict, cache: dict,
                              topic_meta: dict) -> tuple[list, dict, int]:
        """Build topic_items list with cache check and QA text assembly.

        Returns (topic_items, cached_results, skipped_count).
        Each topic_item is (topic, n_qa, qa_texts, marker, qas, qa_ids_hash).
        """
        topic_items = []
        cached_results = {}
        skipped_count = 0

        for topic, qas in groups.items():
            if not topic or topic == "(uncategorized)":
                continue
            meta = topic_meta.get(topic)
            if not meta:
                continue

            # Determine marker
            if meta["lb"] < 0.25:
                if meta["n_qa"] > 1 and meta["lb"] < 0.15:
                    continue
                marker = "  [needs review]"
            elif meta["lb"] >= 0.5:
                marker = "  [core]"
            else:
                marker = ""
            marker += meta.get("quality_flag", "")

            # Cache fingerprint
            qa_ids_sorted = sorted(qa["id"] for qa in qas)
            qa_ids_hash = hashlib.md5(",".join(map(str, qa_ids_sorted)).encode()).hexdigest()
            cached_state = self._db.get_cached_topic_state(topic)

            if (cached_state
                    and cached_state["qa_count"] == len(qas)
                    and cached_state["qa_ids_hash"] == qa_ids_hash
                    and topic in cache):
                cached_results[topic] = cache[topic]
                skipped_count += 1
                continue

            # Build QA text block sorted by Beta weight descending
            qas_sorted = sorted(
                qas,
                key=lambda qa: weights.get(qa["id"], {}).get("mean", 0.5),
                reverse=True,
            )
            qa_texts = ""
            for i, qa in enumerate(qas_sorted):
                w = weights.get(qa["id"], {})
                qa_texts += (f"Q{i+1} [{qa['paper']}] (weight={w.get('mean',0.5):.2f}): "
                             f"{qa['question_text']}\nA: {qa['answer_text']}\n\n")

            # Append missed-pattern reference
            missed_ref = _build_missed_ref(self._db, topic, qas, self._debug)
            if missed_ref:
                qa_texts += missed_ref

            # Low-weight topic hint
            if meta["n_qa"] <= 2 and meta["mean"] < 0.5:
                if self._lang == 'en':
                    qa_texts += ("\nNote: this topic has limited QA data. Only extract "
                                 "knowledge points clearly and directly supported by the "
                                 "answers above. Prefer fewer high-confidence KPs.\n")
                else:
                    qa_texts += ("\n注意: 此主题的题目数据有限。请仅提取答案中明确直接支持的"
                                 "知识点。宁可输出少量高置信度的知识点，也不要输出多个低置信度的。\n")

            topic_items.append((topic, len(qas), qa_texts, marker, qas, qa_ids_hash))

        return topic_items, cached_results, skipped_count

    # ------------------------------------------------------------------
    # Shared KP formatting (eliminates duplication between batch/single)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_kp_text(kps: list, topic_name: str, marker: str) -> str:
        """Format a list of KP dicts/strings into text output lines.

        Shared by ``_distill_one`` and ``_distill_batch`` — previously duplicated.
        """
        parts = [f"{topic_name}{marker}"]
        for i, kp in enumerate(kps, 1):
            if isinstance(kp, str):
                parts.append(f"{i}. {kp}")
            else:
                conf = kp.get('confidence', 'high')
                prefix = "(review) " if conf == 'low' else ""
                parts.append(f"{i}. {prefix}{kp.get('concept') or kp.get('detail', str(kp))}")
                if kp.get('detail'):
                    parts.append(f"   Detail: {kp['detail']}")
                if kp.get('pitfall'):
                    parts.append(f"   Pitfall: {kp['pitfall']}")
                if kp.get('scoring'):
                    parts.append(f"   Scoring: {kp['scoring']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Phase 5: API-bound distillation workers (submitted to thread pool)
    # ------------------------------------------------------------------

    def _distill_one(self, topic: str, n_qa: int, qa_texts: str,
                     marker: str, qas, qa_ids_hash: str) -> tuple[str, str]:
        """Distill a single (large) topic to KPs via Flash API.

        Returns (topic, content_string).
        """
        messages = [
            {"role": "system", "content": self._dist_sys},
            {"role": "user", "content": self._dist_usr % (_escape_pct(topic), n_qa, _escape_pct(qa_texts))},
        ]
        try:
            result = call_flash(self._client, messages, max_retries=1, debug_callback=self._debug)
            kps = result.get("knowledge_points", [])
        except Exception as e:
            from .error_utils import log_exception
            log_exception(self._debug, "Distillation", f"topic={topic}", e)
            kps = []

        if not kps:
            cache = self._db.get_distillation_cache()
            if topic in cache:
                from .error_utils import log_info
                log_info(self._debug, "Distillation fallback", f"'{topic}', reusing cached content")
                return (topic, cache[topic])
            return (topic, "")

        content = self._format_kp_text(kps, topic, marker)
        self._db.upsert_distillation_cache(topic, n_qa, qa_ids_hash, content)
        return (topic, content)

    def _distill_batch(self, batch_topics: list) -> dict[str, str]:
        """Distill multiple small topics in a single Flash API call.

        Returns {topic_name: content_string}.
        """
        sections = []
        for topic, n_qa, qa_texts, marker, qas, qa_ids_hash in batch_topics:
            sections.append(f"=== Topic: {topic} ===\n{qa_texts}")
        combined = "\n".join(sections)

        if self._lang == 'en':
            batch_sys = (
                "You are a knowledge distillation expert. Extract knowledge points "
                "from MULTIPLE topics in one pass. Output JSON with a 'topics' array. "
                'Example: {"topics": [{"topic": "TopicA", "knowledge_points": ['
                '{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}]}'
            )
            batch_usr = (
                "Distill knowledge points for ALL topics below. "
                "For each topic, return 1-3 knowledge points. "
                "Remove question-specific details, keep common principles.\n\n%s\n\n"
                'Return JSON: {"topics": [{"topic": "exact topic name", '
                '"knowledge_points": [...]}, ...]}'
            )
        else:
            batch_sys = (
                "你是一个知识蒸馏专家。一次性从多个主题中提取知识点。Output JSON。"
                '格式: {"topics": [{"topic": "主题名", "knowledge_points": ['
                '{"concept":"...","detail":"...","pitfall":"...","scoring":"...","confidence":"high"}]}]}'
            )
            batch_usr = (
                "为以下所有主题蒸馏知识点。每个主题输出1-3个知识点。"
                "去除题目特定细节，保留共性技术原理。\n\n%s\n\n"
                '返回 JSON: {"topics": [{"topic": "确切的主题名", '
                '"knowledge_points": [...]}, ...]}'
            )

        messages = [
            {"role": "system", "content": batch_sys},
            {"role": "user", "content": batch_usr % _escape_pct(combined)},
        ]
        try:
            result = call_flash(self._client, messages, max_retries=1, debug_callback=self._debug)
            flash_result = result.get("topics", [])
        except Exception as e:
            from .error_utils import log_exception
            log_exception(self._debug, "Batch distillation", "", e)
            flash_result = []

        batch_results: dict[str, str] = {}
        for td in flash_result:
            t_name = td.get("topic", "")
            kps = td.get("knowledge_points", [])
            if not t_name or not kps:
                continue

            # Match returned topic name to batch item
            marker = ""
            qa_ids_hash = ""
            n_qa = 0
            for bt_topic, bt_n_qa, _, bt_marker, _, bt_hash in batch_topics:
                if bt_topic.strip().lower() == t_name.strip().lower():
                    marker = bt_marker
                    n_qa = bt_n_qa
                    qa_ids_hash = bt_hash
                    break
            else:
                from .error_utils import log_info
                log_info(self._debug, "Batch name mismatch", f"model returned '{t_name}'")

            content = self._format_kp_text(kps, t_name, marker)
            self._db.upsert_distillation_cache(t_name, n_qa, qa_ids_hash, content)
            batch_results[t_name] = content

        # Log missed topics
        returned_topics = {td.get("topic", "").strip().lower() for td in flash_result}
        for bt_topic, _, _, _, _, _ in batch_topics:
            if bt_topic.strip().lower() not in returned_topics:
                from .error_utils import log_info
                log_info(self._debug, "Batch missed topic", f"'{bt_topic}', falling back to individual")

        return batch_results
