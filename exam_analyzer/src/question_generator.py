"""Question generator: extract templates from QAs, vary parameters, generate + validate answers.

Provides unlimited adaptive practice questions for students.
"""

import random

from .deepseek_client import call_flash, create_client
from .knowledge_base import QADatabase
from .embedding_cluster import detect_content_lang
from .logger import get_logger

_log = get_logger()


def extract_template(db: QADatabase, kp_id: str, client, debug_cb=None) -> dict:
    """Extract a question template from a KP's representative QA."""
    qas = db.kp.get_representative_qas(kp_id, limit=1)
    if not qas:
        return {}

    qa = qas[0]
    lang = detect_content_lang(qa["question_text"])

    if lang == 'en':
        sys = "Extract the question structure as a reusable template. Output JSON."
        usr = (
            f"Question: {qa['question_text']}\n"
            f"Answer: {qa['answer_text'][:300]}\n\n"
            "Extract:\n"
            "1. command_verb: the action (state, explain, calculate, convert, ...)\n"
            "2. input_type: what kind of input/data (e.g. 'binary_number', 'sql_query')\n"
            "3. output_type: what kind of answer (e.g. 'hexadecimal', 'table_definition')\n"
            "4. params: list of variable parameters with {name, type, example_value, constraints}\n"
            "5. template_text: the question with parameters replaced by {param_name}\n"
            'Return: {"verb": "convert", "input_type": "binary_number", '
            '"output_type": "hexadecimal", '
            '"params": [{"name": "binary_value", "type": "8bit_binary", '
            '"example": "10110011", "constraints": ["length=8", "not_all_zeros"]}], '
            '"template": "Convert the binary number {binary_value} to hexadecimal."}'
        )
    else:
        sys = "提取题目结构为可复用模板。Output JSON。"
        usr = (
            f"题目: {qa['question_text']}\n"
            f"答案: {qa['answer_text'][:300]}\n\n"
            "提取: command_verb, input_type, output_type, params(参数列表), template_text\n"
            '返回 JSON: {"verb": "...", "input_type": "...", "output_type": "...", '
            '"params": [...], "template": "..."}'
        )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug_cb, "Template extraction", f"kp={kp_id}", e)
        return {}


def generate_variation(template: dict, difficulty: str = "basic") -> str:
    """Generate a question variation by replacing template parameters.

    For real use, param substitution would be done by Flash with domain knowledge.
    This is a simplified version that replaces params with example variations.
    """
    if not template or "template" not in template:
        return ""

    text = template["template"]
    params = template.get("params", [])

    # Simple param substitution (Flash-based generation is better but costs API calls)
    for p in params:
        name = p.get("name", "")
        placeholder = "{" + name + "}"
        if placeholder in text:
            replacement = _generate_param_value(p, difficulty)
            text = text.replace(placeholder, replacement)

    return text


def _generate_param_value(param: dict, difficulty: str) -> str:
    """Generate a plausible value for a parameter based on difficulty."""
    ptype = param.get("type", "")
    example = param.get("example", "0")
    constraints = param.get("constraints", [])

    if "binary" in ptype.lower():
        if difficulty == "basic":
            # Simple 4-bit binary
            val = random.randint(1, 15)
            return format(val, '04b')
        elif difficulty == "intermediate":
            val = random.randint(16, 255)
            return format(val, '08b')
        else:
            val = random.randint(256, 65535)
            return format(val, '016b')

    if "hex" in ptype.lower() or "hexadecimal" in ptype.lower():
        val = random.randint(1, 255)
        return format(val, '02X')

    if "decimal" in ptype.lower() or "denary" in ptype.lower():
        if difficulty == "basic":
            return str(random.randint(1, 100))
        else:
            return str(random.randint(100, 10000))

    # Default: return example value (Flash may return int, ensure str)
    return str(example)


def generate_answer(question: str, kp_id: str, db: QADatabase, client, debug_cb=None) -> dict:
    """Generate and validate an answer for a generated question."""
    qas = db.kp.get_representative_qas(kp_id, limit=3)
    if not qas:
        return {"answer": "", "validated": False}

    lang = detect_content_lang(question)
    qa_texts = ""
    for i, qa in enumerate(qas, 1):
        qa_texts += f"Q{i}: {qa['question_text']}\nA: {qa['answer_text']}\n\n"

    if lang == 'en':
        sys = "Answer this exam question using the reference QAs as style guide. Output JSON."
        usr = (
            f"Reference QAs:\n{qa_texts}\n"
            f"Question: {question}\n\n"
            f"Answer in the style of the references. Be precise and complete.\n"
            'Return JSON: {"answer": "your answer"}'
        )
    else:
        sys = "用参考QA的风格回答这道考题。Output JSON。"
        usr = (
            f"参考QA:\n{qa_texts}\n"
            f"题目: {question}\n\n"
            "按参考风格回答，精准完整。\n"
            '返回 JSON: {"answer": "你的回答"}'
        )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    try:
        result = call_flash(client, messages, max_retries=1, debug_callback=debug_cb)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug_cb, "Answer generation", "", e)
        return {"answer": "", "validated": False}

    # Validate: compare generated answer to a representative markscheme using scoring pattern
    ref_qa = qas[0]
    validate_sys = "Compare student answer with model answer. Output JSON."
    validate_usr = (
        f"Question: {question}\n"
        f"Student Answer: {answer}\n"
        f"Model Answer: {ref_qa['answer_text']}\n\n"
        "List covered and missed points.\n"
        'Return JSON: {"covered_points": [...], "missed_points": [...], "pass": true/false}'
    )
    try:
        val_result = call_flash(client, [
            {"role": "system", "content": validate_sys},
            {"role": "user", "content": validate_usr},
        ], max_retries=1, debug_callback=debug_cb)
        val_result = val_result if isinstance(val_result, dict) else {}
    except Exception as e:
        from .error_utils import log_exception
        log_exception(debug_cb, "Answer validation", "", e)
        val_result = {}

    covered = len(val_result.get("covered_points", []))
    missed = len(val_result.get("missed_points", []))
    total = covered + missed
    score = covered / total if total > 0 else 0

    return {
        "answer": answer,
        "validated": score >= 0.7,
        "score": round(score, 2),
        "covered": covered,
        "missed": missed,
    }


def generate_questions(db_path: str, kp_id: str, count: int = 3,
                       difficulty: str = "intermediate",
                       api_url: str = "", api_key: str = "",
                       debug_callback=None) -> list[dict]:
    """Generate practice questions for a KP.

    Returns list of {stem, answer, difficulty, validated, score}.
    """
    db = QADatabase(db_path)
    kp = db.get_kp_by_id(kp_id)
    if not kp:
        db.close()
        return []

    client = create_client(api_url, api_key)
    template = extract_template(db, kp_id, client, debug_callback)

    questions = []
    for _ in range(count):
        stem = generate_variation(template, difficulty) if template else ""
        if not stem:
            stem = f"Explain the concept of {kp.get('name', 'this topic')}."

        ans_data = generate_answer(stem, kp_id, db, client, debug_callback)
        questions.append({
            "stem": stem,
            "answer": ans_data.get("answer", ""),
            "difficulty": difficulty,
            "kp_id": kp_id,
            "kp_name": kp.get("name", ""),
            "validated": ans_data.get("validated", False),
            "score": ans_data.get("score", 0),
        })

    db.close()
    return questions

