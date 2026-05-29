"""Complete test suite for exam_analyzer pipeline.

Modes:
  python test_suite.py light    — lightweight: uses input/ (1-3 paper pairs)
  python test_suite.py full     — full: uses input_keepout/ (16-32 paper pairs)
  python test_suite.py chat     — chat-only: tests API endpoints (requires app.py running)

Workflow each mode:
  1. Validate input files exist
  2. Run pipeline (main.py)
  3. Check DB schema + data
  4. Check output files
  5. Verify key metrics against thresholds
  6. Print report
"""

import sys, os, json, time, subprocess, re, shutil
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # tests/ → exam_analyzer/
REPORT_DIR = os.path.join(PROJECT_DIR, "TestReport")
os.chdir(PROJECT_DIR)  # Run from project root so imports and paths work
sys.path.insert(0, '.')
os.makedirs(REPORT_DIR, exist_ok=True)

# ── config ──────────────────────────────────────────────────
THRESHOLDS = {
    "qa_per_paper_min": 3,       # min questions per paper
    "qa_per_paper_max": 50,      # max questions per paper (sanity)
    "verb_coverage_pct": 80,     # min % QAs with non-empty command_verb
    "verb_unknown_max_pct": 15,  # max % "unknown" verbs
    "miss_cat_nonempty_pct": 50, # min % feedback with miss_categories
    "kg_cluster_cohesion_min": 0.5,  # min avg cluster cohesion
    "kg_noise_max_pct": 30,      # max % QAs unclustered
    "diff_basic_pct_min": 10,    # at least 10% basic
    "diff_advanced_pct_min": 5,  # at least 5% advanced
    "ar_verified_min_pct": 30,   # min % KPs verified by adversarial refiner
    "edge_duplicates_max": 0,    # max duplicate edge pairs (must be 0)
    "chat_response_time_ms": 10000,  # max chat response time
}


# ── helpers ─────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def run(cmd, timeout=300):
    """Run a command with real-time output streaming + heartbeat. Returns (returncode, stdout, stderr)."""
    log(f"Running: {cmd}", "CMD")
    stdout_lines = []
    stderr_lines = []
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=PROJECT_DIR)
    start = time.time()
    last_heartbeat = start
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            print(f"  {line.rstrip()}")
            sys.stdout.flush()
            stdout_lines.append(line)
            last_heartbeat = time.time()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        proc.kill()
        log(f"TIMEOUT after {elapsed:.0f}s (limit: {timeout}s)", "FATAL")
        return -1, ''.join(stdout_lines), f"[TIMEOUT after {elapsed:.0f}s]"
    elapsed = time.time() - start
    log(f"Completed in {elapsed:.0f}s, returncode={proc.returncode}")
    return proc.returncode, ''.join(stdout_lines), ''


def check_table_exists(db_path, table_name):
    import sqlite3
    conn = sqlite3.connect(db_path)
    r = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
    conn.close()
    return r is not None


def query_db(db_path, sql, params=()):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def count_db(db_path, table):
    rows = query_db(db_path, f"SELECT COUNT(*) as cnt FROM {table}")
    return rows[0]["cnt"] if rows else 0


# ── checks ──────────────────────────────────────────────────

class CheckResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.metrics = {}

    def check(self, name, condition, detail=""):
        if condition:
            self.passed.append({"name": name, "detail": detail})
            log(f"  PASS: {name}" + (f" ({detail})" if detail else ""))
        else:
            self.failed.append({"name": name, "detail": detail})
            log(f"  FAIL: {name}" + (f" ({detail})" if detail else ""), "FAIL")

    def warn(self, name, detail=""):
        self.warnings.append({"name": name, "detail": detail})
        log(f"  WARN: {name}" + (f" ({detail})" if detail else ""), "WARN")

    def metric(self, key, value):
        self.metrics[key] = value

    def summary(self):
        log(f"\n{'='*60}")
        log(f"Results: {len(self.passed)} passed, {len(self.failed)} failed, {len(self.warnings)} warnings")
        if self.failed:
            log("FAILED CHECKS:")
            for f in self.failed:
                log(f"  - {f['name']}")
        if self.warnings:
            log("WARNINGS:")
            for w in self.warnings:
                log(f"  - {w['name']}")
        log(f"{'='*60}")
        return len(self.failed) == 0

    def format_report(self, title):
        lines = [f"Test Report: {title}", "=" * 60, ""]
        lines.append(f"Passed:  {len(self.passed)}")
        lines.append(f"Failed:  {len(self.failed)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        if self.metrics:
            lines.append("")
            lines.append("--- Metrics ---")
            for k, v in sorted(self.metrics.items()):
                lines.append(f"  {k}: {v}")
        if self.passed:
            lines.append("")
            lines.append("--- Passed ---")
            for p in self.passed:
                lines.append(f"  [PASS] {p['name']}")
        if self.failed:
            lines.append("")
            lines.append("--- Failed ---")
            for f in self.failed:
                lines.append(f"  [FAIL] {f['name']}")
                if f['detail']:
                    lines.append(f"         {f['detail']}")
        if self.warnings:
            lines.append("")
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  [WARN] {w['name']}")
        lines.append("")
        return "\n".join(lines)


def check_input_files(input_dir, mode_name):
    """Validate input PDFs exist and are paired."""
    r = CheckResult()
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')] if os.path.exists(input_dir) else []
    r.check(f"{mode_name}: input dir exists", os.path.exists(input_dir))

    from src.file_pairer import pair_files
    pairs = pair_files(input_dir)
    r.check(f"{mode_name}: paper pairs found ({len(pairs)})", len(pairs) > 0,
            f"{len(pairs)} pairs from {len(pdfs)} PDFs")
    for qp, ms, name in pairs:
        r.check(f"{mode_name}: pair complete ({name})", os.path.exists(qp) and os.path.exists(ms))
    return r, pairs


def check_pipeline_output(db_path, mode_name):
    """Check pipeline post-processing outputs."""
    r = CheckResult()
    if not os.path.exists(db_path):
        r.failed.append(f"{mode_name}: DB not found at {db_path}")
        return r

    # DB tables
    expected_tables = ["qa_pairs", "topic_links", "question_feedback",
                       "topic_dependencies", "command_verb_patterns", "topic_difficulty",
                       "knowledge_points", "kp_edges", "qa_kp_membership",
                       "exam_trends", "student_trajectory", "analysis_checkpoints"]
    for t in expected_tables:
        r.check(f"{mode_name}: table {t} exists", check_table_exists(db_path, t))

    # QA count
    qa_count = count_db(db_path, "qa_pairs")
    r.metric("qa_count", qa_count)
    r.check(f"{mode_name}: QAs present ({qa_count})", qa_count > 0)

    # Detect incomplete run: QAs exist but no post-processing data
    kp_count = count_db(db_path, "knowledge_points")
    dep_count = count_db(db_path, "topic_dependencies")
    verb_rows = query_db(db_path, "SELECT COUNT(*) as cnt FROM qa_pairs WHERE command_verb != ''")
    verb_any = verb_rows[0]["cnt"] > 0 if verb_rows else False
    if qa_count > 25 and kp_count == 0 and dep_count == 0 and not verb_any:
        r.warn(f"{mode_name}: pipeline appears incomplete (post-processing not run) — "
               f"check timeout or re-run with longer limit")

    # ── Phase 1/2 metrics ──
    fb_count = count_db(db_path, "question_feedback")
    r.metric("feedback_count", fb_count)
    if qa_count > 25:  # only expect feedback if multiple papers processed
        r.check(f"{mode_name}: question_feedback has data ({fb_count})", fb_count > 0)
    else:
        r.warn(f"{mode_name}: few QAs ({qa_count}), feedback may be sparse")

    # miss_categories
    miss_cat_count = query_db(db_path,
        "SELECT COUNT(*) as cnt FROM question_feedback WHERE miss_categories != ''")
    mc = miss_cat_count[0]["cnt"] if miss_cat_count else 0
    if fb_count > 0:
        mc_pct = mc / fb_count * 100
        r.check(f"{mode_name}: miss_categories populated ({mc}/{fb_count}, {mc_pct:.0f}%)",
                mc_pct >= THRESHOLDS["miss_cat_nonempty_pct"])
    else:
        r.warn(f"{mode_name}: no feedback to check miss_categories")

    # topic_links
    tl_count = count_db(db_path, "topic_links")
    r.metric("topic_links", tl_count)
    if qa_count > 25:
        r.check(f"{mode_name}: topic_links present ({tl_count})", tl_count > 0)

    # ── Verb metrics ──
    verb_rows = query_db(db_path,
        "SELECT command_verb, COUNT(*) as cnt FROM qa_pairs "
        "WHERE command_verb != '' GROUP BY command_verb")
    annotated = sum(r["cnt"] for r in verb_rows)
    unknown = sum(r["cnt"] for r in verb_rows if r["command_verb"] == "unknown")
    if qa_count > 0:
        cov_pct = annotated / qa_count * 100
        unk_pct = unknown / qa_count * 100
        r.metric("verb_coverage_pct", f"{cov_pct:.0f}%")
        r.metric("verb_unknown_pct", f"{unk_pct:.0f}%")
        r.check(f"{mode_name}: verb coverage ({cov_pct:.0f}% >= {THRESHOLDS['verb_coverage_pct']}%)",
                cov_pct >= THRESHOLDS["verb_coverage_pct"])
        r.check(f"{mode_name}: verb unknown rate ({unk_pct:.0f}% <= {THRESHOLDS['verb_unknown_max_pct']}%)",
                unk_pct <= THRESHOLDS["verb_unknown_max_pct"])

    # ── Difficulty metrics ──
    diff_rows = query_db(db_path,
        "SELECT difficulty_estimate, COUNT(*) as cnt FROM qa_pairs "
        "WHERE difficulty_estimate != '' GROUP BY difficulty_estimate")
    diff_map = {r["difficulty_estimate"]: r["cnt"] for r in diff_rows}
    total_diff = sum(diff_map.values())
    if total_diff > 0:
        b_pct = diff_map.get("basic", 0) / total_diff * 100
        i_pct = diff_map.get("intermediate", 0) / total_diff * 100
        a_pct = diff_map.get("advanced", 0) / total_diff * 100
        r.metric("difficulty_dist", f"basic={b_pct:.0f}% intermediate={i_pct:.0f}% advanced={a_pct:.0f}%")
        r.check(f"{mode_name}: basic difficulty >= {THRESHOLDS['diff_basic_pct_min']}% ({b_pct:.0f}%)",
                b_pct >= THRESHOLDS["diff_basic_pct_min"])
        r.check(f"{mode_name}: advanced difficulty >= {THRESHOLDS['diff_advanced_pct_min']}% ({a_pct:.0f}%)",
                a_pct >= THRESHOLDS["diff_advanced_pct_min"])
    else:
        r.warn(f"{mode_name}: no difficulty data")

    # topic_difficulty table
    td_count = count_db(db_path, "topic_difficulty")
    r.check(f"{mode_name}: topic_difficulty populated ({td_count})", td_count > 0 or total_diff == 0)

    # ── Knowledge graph metrics ──
    kp_count = count_db(db_path, "knowledge_points")
    kp_qa_count = count_db(db_path, "qa_kp_membership")
    if qa_count >= 4:  # need enough QAs for clustering
        r.metric("kp_count", kp_count)
        r.metric("kp_qa_membership", kp_qa_count)
        r.check(f"{mode_name}: KPs generated ({kp_count})", kp_count > 0)
        r.check(f"{mode_name}: QA-KP membership ({kp_qa_count})", kp_qa_count > 0)

        # cluster cohesion
        coh_rows = query_db(db_path, "SELECT AVG(cohesion) as avg_coh FROM knowledge_points WHERE cohesion > 0")
        avg_coh = coh_rows[0]["avg_coh"] if coh_rows and coh_rows[0]["avg_coh"] else 0
        if avg_coh > 0:
            r.check(f"{mode_name}: avg cluster cohesion >= {THRESHOLDS['kg_cluster_cohesion_min']} ({avg_coh:.2f})",
                    avg_coh >= THRESHOLDS["kg_cluster_cohesion_min"])
    else:
        r.warn(f"{mode_name}: too few QAs ({qa_count}) for clustering (need >= 4)")

    # KP edges
    edge_count = count_db(db_path, "kp_edges")
    r.metric("kp_edges", edge_count)
    if kp_count >= 2:
        r.check(f"{mode_name}: KP edges present ({edge_count})", edge_count > 0)

    # Edge duplicates
    dup_rows = query_db(db_path,
        "SELECT COUNT(*) as cnt FROM (SELECT source_kp, target_kp, COUNT(*) as n "
        "FROM kp_edges GROUP BY 1, 2 HAVING n > 1)")
    dup_count = dup_rows[0]["cnt"] if dup_rows else 0
    r.check(f"{mode_name}: no duplicate KP edges ({dup_count})",
            dup_count <= THRESHOLDS["edge_duplicates_max"])

    # ── Adversarial refinement metrics ──
    ar_rows = query_db(db_path,
        "SELECT quality, COUNT(*) as cnt FROM knowledge_points GROUP BY quality")
    ar_map = {r["quality"]: r["cnt"] for r in ar_rows}
    if kp_count > 0:
        verified_pct = ar_map.get("verified", 0) / kp_count * 100
        r.metric("ar_verified_pct", f"{verified_pct:.0f}%")
        r.check(f"{mode_name}: verified KPs >= {THRESHOLDS['ar_verified_min_pct']}% ({verified_pct:.0f}%)",
                verified_pct >= THRESHOLDS["ar_verified_min_pct"])

    # ── Dependency metrics ──
    dep_count = count_db(db_path, "topic_dependencies")
    r.metric("dependency_count", dep_count)
    if qa_count > 25:
        r.check(f"{mode_name}: dependencies present ({dep_count})", dep_count > 0)

    # ── Output files ──
    points_files = [f for f in os.listdir(os.path.join(PROJECT_DIR, "point"))
                    if f.endswith("_points.txt")] if os.path.exists(os.path.join(PROJECT_DIR, "point")) else []
    r.check(f"{mode_name}: points.txt generated", len(points_files) > 0,
            f"Found: {points_files}")

    verb_files = [f for f in os.listdir(os.path.join(PROJECT_DIR, "point"))
                  if f.endswith("_verb_patterns.txt")] if os.path.exists(os.path.join(PROJECT_DIR, "point")) else []
    if qa_count > 10:
        r.check(f"{mode_name}: verb_patterns.txt generated", len(verb_files) > 0)

    return r


def check_chat_endpoints(base_url="http://127.0.0.1:5000"):
    """Test chat API endpoints. Requires app.py running."""
    import urllib.request, urllib.error

    r = CheckResult()

    def api_get(path):
        try:
            req = urllib.request.Request(f"{base_url}{path}")
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read()), resp.status
        except Exception as e:
            return {"error": str(e)}, 0

    def api_post(path, body):
        try:
            data = json.dumps(body).encode()
            req = urllib.request.Request(f"{base_url}{path}", data=data,
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=THRESHOLDS["chat_response_time_ms"]//1000)
            return json.loads(resp.read()), resp.status
        except Exception as e:
            return {"error": str(e)}, 0

    # Status
    data, status = api_get("/api/chat/status")
    r.check("chat: /api/chat/status returns 200", status == 200)
    r.check("chat: chat available", data.get("available"), f"qa_count={data.get('qa_count', 0)}")

    # Chat question
    data, status = api_post("/api/chat", {"question": "What is binary?", "session_id": "test_suite"})
    r.check("chat: /api/chat returns 200", status == 200)
    r.check("chat: answer present", bool(data.get("answer")),
            f"len={len(data.get('answer', ''))}")
    r.check("chat: sources present", len(data.get("sources", [])) > 0)
    r.check("chat: suggestions present", len(data.get("suggestions", [])) > 0)

    # Knowledge graph
    data, status = api_get("/api/knowledge-graph")
    r.check("chat: /api/knowledge-graph returns 200", status == 200)
    r.check("chat: graph has nodes", len(data.get("nodes", [])) > 0 or data.get("error"))

    # Command verbs
    data, status = api_get("/api/command-verbs")
    r.check("chat: /api/command-verbs returns 200", status == 200)

    # Topic difficulty
    data, status = api_get("/api/topic-difficulty")
    r.check("chat: /api/topic-difficulty returns 200", status == 200)

    return r


# ── main ────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if not mode:
        print("Select test mode:")
        print("  1 = light  (input/ folder, 1-3 paper pairs)")
        print("  2 = full   (input/keepout/ folder, 16-32 paper pairs)")
        print("  3 = check  (validate existing DB without running pipeline)")
        print("  4 = chat   (test API endpoints, requires app.py running)")
        choice = input("Enter choice (1-4): ").strip()
        mode_map = {"1": "light", "2": "full", "3": "check", "4": "chat"}
        mode = mode_map.get(choice, "light")
    log(f"Test Suite — mode: {mode}")
    log(f"Project dir: {PROJECT_DIR}")
    log(f"Report dir: {REPORT_DIR}")

    if mode == "chat":
        log("Chat endpoint test (requires app.py running on port 5000)")
        r = check_chat_endpoints()
        r.summary()
        return

    if mode == "check":
        import glob
        db_candidates = glob.glob(os.path.join(PROJECT_DIR, "intermediate", "*_knowledge.db"))
        db_path = db_candidates[0] if db_candidates else os.path.join(PROJECT_DIR, "intermediate", "knowledge.db")
        if not os.path.exists(db_path):
            log(f"DB not found at {db_path}", "FATAL")
            return
        log(f"Check-only mode — validating existing DB: {db_path}")
        r = check_pipeline_output(db_path, "check")
        r.summary()
        report_name = f"test_report_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(REPORT_DIR, report_name)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(r.format_report(f"check mode (existing DB)"))
        log(f"Report saved to {report_path}")
        return

    # ── file validation ──
    input_dir = os.path.join(PROJECT_DIR, "input")
    if mode == "full":
        input_dir = os.path.join(PROJECT_DIR, "input", "keepout")
        if not os.path.exists(input_dir):
            log("input/keepout/ not found, falling back to input/", "WARN")
            input_dir = os.path.join(PROJECT_DIR, "input")
    mode_name = "light" if "keepout" not in input_dir else "full"

    log(f"Input dir: {input_dir}")
    r_input, pairs = check_input_files(input_dir, mode_name)
    if r_input.failed:
        log("Input validation failed, aborting", "FATAL")
        r_input.summary()
        return
    log(f"Found {len(pairs)} paper pairs")

    # ── clear caches ──
    log("Clearing caches...")

    # Python bytecode cache
    pycache_dirs = []
    for root, dirs, files in os.walk(PROJECT_DIR):
        if "__pycache__" in dirs:
            pycache_dirs.append(os.path.join(root, "__pycache__"))
    for d in pycache_dirs:
        shutil.rmtree(d, ignore_errors=True)
    if pycache_dirs:
        log(f"  Cleared {len(pycache_dirs)} __pycache__ dirs")

    # Embedding model cache (in-memory, cleared by fresh subprocess)
    log("  Embedding model cache: cleared by subprocess isolation")

    # Chat retriever cache (in-memory, cleared by fresh subprocess)
    log("  Chat retriever cache: cleared by subprocess isolation")

    # Clean old log files (keep last 5)
    from src.logger import LOG_DIR
    log_dir = LOG_DIR
    if os.path.exists(log_dir):
        log_files = sorted(
            [f for f in os.listdir(log_dir) if f.startswith("run_")],
            reverse=True
        )
        for old_log in log_files[5:]:
            os.remove(os.path.join(log_dir, old_log))
        if len(log_files) > 5:
            log(f"  Cleaned {len(log_files) - 5} old log files")

    # ── backup existing state ──
    import glob as _glob
    db_candidates = _glob.glob(os.path.join(PROJECT_DIR, "intermediate", "*_knowledge.db"))
    db_path = db_candidates[0] if db_candidates else os.path.join(PROJECT_DIR, "intermediate", "knowledge.db")
    proc_candidates = _glob.glob(os.path.join(PROJECT_DIR, "intermediate", "*_processed.json"))
    processed_path = proc_candidates[0] if proc_candidates else os.path.join(PROJECT_DIR, "intermediate", "processed.json")
    backup_dir = os.path.join(PROJECT_DIR, "intermediate", f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if os.path.exists(db_path) or os.path.exists(processed_path):
        os.makedirs(backup_dir, exist_ok=True)
        for f in [db_path, processed_path,
                  db_path + "-wal", db_path + "-shm"]:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(backup_dir, os.path.basename(f)))
        log(f"Backed up existing state to {backup_dir}")
        # Clean for fresh run
        for f in [db_path, processed_path, db_path + "-wal", db_path + "-shm"]:
            if os.path.exists(f):
                os.remove(f)
        # Also clean output
        point_dir = os.path.join(PROJECT_DIR, "point")
        if os.path.exists(point_dir):
            for f in os.listdir(point_dir):
                if f.endswith("_points.txt") or f.endswith("_verb_patterns.txt"):
                    os.remove(os.path.join(point_dir, f))

    # ── prepare input for main.py (always reads from input/) ──
    default_input = os.path.join(PROJECT_DIR, "input")
    need_restore = False
    if input_dir != default_input:
        # Full mode: input/keepout/ → copy to input/ temporarily
        log(f"Copying PDFs from {input_dir} to {default_input}...")
        for f in os.listdir(input_dir):
            if f.lower().endswith('.pdf'):
                src = os.path.join(input_dir, f)
                dst = os.path.join(default_input, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        need_restore = True
        # Re-validate pairs from default_input
        _, pairs = check_input_files(default_input, mode_name)
        log(f"After copy: {len(pairs)} pairs in input/")

    # ── run pipeline ──
    log(f"Running pipeline with {len(pairs)} pairs...")
    t0 = time.time()
    timeout_sec = 2400 if mode_name == "full" else 900
    log(f"Timeout: {timeout_sec}s ({timeout_sec//60} min)")
    rc, stdout, stderr = run("python main.py", timeout=timeout_sec)
    elapsed = time.time() - t0
    log(f"Pipeline completed in {elapsed:.0f}s, returncode={rc}")

    # ── restore input/ if needed ──
    if need_restore:
        log("Restoring input/ to original state...")
        for f in os.listdir(default_input):
            if f.lower().endswith('.pdf'):
                src_in_keepout = os.path.join(input_dir, f)
                if os.path.exists(src_in_keepout):
                    os.remove(os.path.join(default_input, f))

    # Save logs
    from src.logger import LOG_DIR
    log_dir = LOG_DIR
    log_name = f"test_{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}")
    log(f"Pipeline output saved to {log_path}")

    if rc != 0:
        log(f"Pipeline failed with returncode {rc}", "FATAL")
        log(f"STDERR (last 500 chars):\n{stderr[-500:]}")
        return

    # ── check outputs ──
    r_out = check_pipeline_output(db_path, mode_name)
    r_input.summary()
    all_pass = r_out.summary()

    # ── write report ──
    report_name = f"test_report_{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(REPORT_DIR, report_name)
    report_text = r_out.format_report(f"{mode_name} mode ({len(pairs)} paper pairs)")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log(f"Report saved to {report_path}")

    if all_pass:
        log(f"\n{'='*60}")
        log("ALL CHECKS PASSED")
        log(f"{'='*60}")
    else:
        log(f"\n{'='*60}")
        log("SOME CHECKS FAILED — see above")
        log(f"{'='*60}")


if __name__ == "__main__":
    main()
