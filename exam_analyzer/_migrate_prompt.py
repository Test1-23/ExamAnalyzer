"""One-shot migration script: replace last inline prompt in pipeline.py."""
import sys

with open("src/pipeline.py", "r", encoding="utf-8") as f:
    text = f.read()

# Find _build_answer_prompt boundaries
start = text.find("def _build_answer_prompt(question_text: str, similar_qas: list[dict]) -> list:")
if start < 0:
    print("ERROR: _build_answer_prompt not found")
    sys.exit(1)

# Find next function definition after this one
next_def = text.find("\ndef ", start + 10)
if next_def < 0:
    next_def = text.find("\n# --", start + 10)
if next_def < 0:
    print("ERROR: cannot find end of _build_answer_prompt")
    sys.exit(1)

old = text[start:next_def]

new = '''def _build_answer_prompt(question_text: str, similar_qas: list[dict]) -> list:
    """Build Answer prompt via PromptBuilder (template: _ANSWER_TMPL)."""
    if similar_qas:
        qa_block = ""
        for i, qa in enumerate(similar_qas, 1):
            qa_block += (f"--- Past Q{i} (similarity: {qa.get('_score',0):.2f}) ---\n"
                         f"Q: {qa['question_text']}\nA: {qa['answer_text']}\n\n")
    else:
        qa_block = "(Knowledge base is empty, no past Q&As)\n\n"

    return PromptBuilder.build(
        PromptType.ANSWER, qa_block=qa_block, question_text=question_text,
        lang=detect_content_lang(question_text))'''

if old in text:
    text = text.replace(old, new)
    with open("src/pipeline.py", "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("_build_answer_prompt: REPLACED")
else:
    print(f"NOT FOUND. old chunk size: {len(old)}")
    print(f"First 80 chars: {old[:80]!r}")
    sys.exit(1)
