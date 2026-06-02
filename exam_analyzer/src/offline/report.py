import json
import os
import re
from collections import deque

from ..deepseek_client import create_client, call_flash
from ..knowledge_base import QADatabase
from ..embedding_cluster import _get_model, detect_content_lang, TOPIC_EMBED_MODEL
from ..models import VerbPatternSpec, DependencySpec
from ..prompt_factory import VERB_PATTERN_SUMMARY, DIFFICULTY_RATE, DEPENDENCY_VALIDATE
from ..logger import get_logger
from ..utils import get_worker_limit

_log = get_logger()


from .verbs import analyze_command_verbs
from .difficulty import assess_difficulty
from .dependencies import discover_dependencies

def _write_verb_report(db: QADatabase, output_dir: str, subject_code: str):
    """Write human-readable command verb analysis report to point/{subject}_verb_patterns.txt"""
    patterns = db.get_verb_patterns()
    if not patterns:
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{subject_code}_verb_patterns.txt")
    lines = [f"Command Verb Answer Patterns — {subject_code}", "=" * 60, ""]

    for p in patterns:
        verb = p["verb"]
        family = p.get("verb_family", "") or ""
        n = p["sample_count"]
        lines.append(f"{verb}  [{family}]  (n={n})")
        lines.append("-" * 40)
        if p.get("avg_answer_length"):
            lines.append(f"  Avg answer length: {p['avg_answer_length']:.0f} chars  "
                         f"(median: {p['median_answer_length']:.0f})")
        if p.get("bullet_ratio") is not None:
            lines.append(f"  Bullet/list usage: {p['bullet_ratio']*100:.0f}%  "
                         f"(avg {p['avg_bullet_count']:.1f} bullets)")
        if p.get("avg_miss_rate") is not None:
            lines.append(f"  Avg miss rate (AI): {p['avg_miss_rate']*100:.0f}%")
        if p.get("pattern_summary"):
            lines.append(f"  Pattern: {p['pattern_summary']}")
        if p.get("topic_specific_patterns"):
            try:
                tsp = json.loads(p["topic_specific_patterns"])
                for topic, note in tsp.items():
                    lines.append(f"    [{topic}] {note}")
            except Exception as e:
                from ..error_utils import log_exception
                log_exception(_log, "Report JSON parse", "verb_pattern", e, level="warning")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def run_offline_analysis(db, api_url: str, api_key: str,
                         progress_callback=None, debug=None,
                         output_dir: str = None):
    """Run all three offline analysis tasks in order.

    Called from pipeline.run_pipeline() after main processing completes.
    output_dir: directory for analysis report files (defaults to point/ adjacent to db_path)
    """
    def _debug(msg):
        if debug:
            debug(f"[Offline] {msg}")
        else:
            print(f"[Offline] {msg}")

    def _progress(pct, status):
        if progress_callback:
            progress_callback(pct, f"[Analysis] {status}")

    _debug("Starting offline analysis pipeline")

    if db.count() == 0:
        _debug("No QAs in database, skipping offline analysis")
        return

    # Derive subject code from db_path
    subject_code = "unknown"
    m = re.search(r'(\d+)_knowledge\.db', db.db_path)
    if m:
        subject_code = m.group(1)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(db.db_path) or ".", "..", "point")
        output_dir = os.path.normpath(output_dir)

    client = create_client(api_url, api_key)

    # Task 2: Command verb analysis (independent, runs first)
    _progress(0, "Extracting command verbs...")
    verb_data = analyze_command_verbs(db, client, _debug, _progress)

    # Task 2b: Write verb report
    report_path = _write_verb_report(db, output_dir, subject_code)
    if report_path:
        _debug(f"Verb pattern report: {report_path}")

    # Task 3: Difficulty assessment (depends on Task 2 for verb_length_percentile)
    _progress(33, "Assessing difficulty...")
    difficulty_data = assess_difficulty(db, client, verb_data, _debug, _progress)

    # Task 1: Dependency discovery (depends on Task 3 for cross_topic signal, Task 2 for verbs)
    _progress(66, "Discovering dependencies...")
    discover_dependencies(db, client, _debug, _progress)

    _progress(100, "Offline analysis complete")

    _debug("Offline analysis pipeline finished")
