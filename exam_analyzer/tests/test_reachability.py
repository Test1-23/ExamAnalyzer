"""Stage 0: Reachability test — uses existing DB, no new PDFs needed.
Run: python test_reachability.py (from exam_analyzer/ directory)
Verifies every code path completes without crashing."""
import sys, os, json

# tests/ → exam_analyzer/ (project root)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
sys.path.insert(0, '.')

from src.knowledge_base import QADatabase, log_schema_status

import glob as _glob
db_candidates = _glob.glob(os.path.join('intermediate', '*_knowledge.db'))
db_path = db_candidates[0] if db_candidates else os.path.join('intermediate', 'knowledge.db')
if not os.path.exists(db_path):
    print(f'ERROR: DB not found at {db_path}')
    print('Run the pipeline first (python main.py) or check the path.')
    sys.exit(1)

db = QADatabase(db_path)
print(f'QAs: {db.count()}')
log_schema_status(db, lambda m: print(m))

cfg_path = 'config.json'
if not os.path.exists(cfg_path):
    print(f'ERROR: config.json not found at {cfg_path}')
    sys.exit(1)
with open(cfg_path) as f:
    cfg = json.load(f)
from src.deepseek_client import create_client
client = create_client(cfg['api_url'], cfg['api_key'])

# ---- Test 1: Knowledge Graph ----
print('\n=== Test 1: Knowledge Graph ===')
from src.knowledge_graph import cluster_qas, generate_kps, discover_kp_edges
cl = cluster_qas(db, lambda m: print(m))
if cl['clusters']:
    kp_ids = generate_kps(db, cl, client, lambda m: print(m))
    if kp_ids:
        discover_kp_edges(db, cl, kp_ids, lambda m: print(m))
        from src.knowledge_graph import discover_sequential_edges, discover_learning_path_edges, fuse_all_edges
        discover_sequential_edges(db, cl, kp_ids, lambda m: print(m))
        discover_learning_path_edges(db, kp_ids, lambda m: print(m))
        fuse_all_edges(db, kp_ids, lambda m: print(m))
else:
    print('No clusters formed (may be normal with few QAs)')

# ---- Test 2: Adversarial Refinement ----
print('\n=== Test 2: Adversarial Refinement ===')
from src.adversarial_refiner import refine_kp, cross_kp_consistency
kps = db.get_all_kps()
if kps:
    refine_kp(db, kps[0]['id'], client, lambda m: print(m))
    if len(kps) >= 2:
        cross_kp_consistency(db, [k['id'] for k in kps[:5]], client, lambda m: print(m))
else:
    print('No KPs to refine (may be normal)')

# ---- Test 3: Offline Analysis ----
print('\n=== Test 3: Offline Analysis ===')
from src.offline_analyzer import analyze_command_verbs, assess_difficulty, discover_dependencies
verb_data = analyze_command_verbs(db, client, lambda m: print(m))
assess_difficulty(db, client, verb_data if verb_data else {}, lambda m: print(m))
discover_dependencies(db, client, lambda m: print(m))

# ---- Test 4: Closed Loop ----
print('\n=== Test 4: Closed Loop ===')
kps = db.get_all_kps()
from src.pipeline_diagnostics import auto_discover_pitfalls, compute_exam_trends
for kp in kps[:3]:
    auto_discover_pitfalls(db, kp['id'], lambda m: print(m))
compute_exam_trends(db, lambda m: print(m))

# ---- Test 5: Question Generator ----
print('\n=== Test 5: Question Generator ===')
from src.question_generator import extract_template, generate_variation, generate_answer
kps = db.get_all_kps()
if kps:
    tmpl = extract_template(db, kps[0]['id'], client, lambda m: print(m))
    if tmpl:
        q = generate_variation(tmpl, 'intermediate')
        ans = generate_answer(q, kps[0]['id'], db, client, lambda m: print(m))
        print(f'Generated: {q[:100]}')
        answer_text = ans.get("answer", "")
        print(f'Answer: {answer_text[:100].encode("ascii", errors="replace").decode("ascii")}')
        print(f'Validated: {ans.get("validated")}')
    else:
        print('No template extracted (may be normal)')

db.close()
print('\n=== ALL REACHABILITY CHECKS PASSED ===')
