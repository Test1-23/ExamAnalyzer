"""Pure function tests for adversarial_refiner prompts via PromptBuilder — zero DB, zero network."""

from src.prompt_factory import PromptType, PromptBuilder


_SAMPLE_KP = {
    "name": "Binary Addition",
    "core_concept": "Adding binary numbers column-by-column",
    "core_detail": "0+0=0, 0+1=1, 1+0=1, 1+1=0 carry 1",
}

_QA_BLOCK = ("QA1: Q: Add 101 + 011\nA: 1000\n\n"
             "QA2: Q: What is 1+1 in binary?\nA: 10 (0 carry 1)\n\n")

_ISSUES = ["Over-generalization: concept only covers unsigned addition"]


def _make_challenger(kp=None, qa_block=_QA_BLOCK, lang="en", prev=""):
    kp = kp or _SAMPLE_KP
    return PromptBuilder.build(PromptType.KP_CHALLENGER,
        kp_name=kp.get("name", ""),
        kp_concept=kp.get("core_concept", "") or kp.get("description", ""),
        kp_detail=kp.get("core_detail", ""),
        qa_texts=qa_block,
        prev_challenges=prev,
        lang=lang,
    )


def _make_defender(kp=None, qa_block=_QA_BLOCK, issues=None, lang="en"):
    kp = kp or _SAMPLE_KP
    issues_text = "\n".join(f"- {i}" for i in (issues or _ISSUES))
    return PromptBuilder.build(PromptType.KP_DEFENDER,
        kp_name=kp.get("name", ""),
        kp_concept=kp.get("core_concept", "") or kp.get("description", ""),
        kp_detail=kp.get("core_detail", ""),
        issues_text=issues_text,
        qa_texts=qa_block,
        lang=lang,
    )


# ═══════════════════════════════════════════════════════════════
# KP_CHALLENGER
# ═══════════════════════════════════════════════════════════════

class TestBuildChallengerPrompt:
    def test_english_lang(self):
        msgs = _make_challenger()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "rigorous knowledge reviewer" in msgs[0]["content"].lower()
        assert msgs[1]["role"] == "user"
        assert "Knowledge Point:" in msgs[1]["content"]
        assert "Binary Addition" in msgs[1]["content"]

    def test_chinese_lang(self):
        msgs = _make_challenger(lang="zh")
        assert "rigorous knowledge reviewer" not in msgs[0]["content"].lower()
        assert "Output JSON" in msgs[0]["content"]

    def test_prev_challenges_included(self):
        msgs = _make_challenger(prev="Issue 1: needs more detail\n")
        assert "Issue 1: needs more detail" in msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════
# KP_DEFENDER
# ═══════════════════════════════════════════════════════════════

class TestBuildDefenderPrompt:
    def test_english_lang(self):
        msgs = _make_defender()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "knowledge curator" in msgs[0]["content"].lower()
        assert msgs[1]["role"] == "user"
        assert "Challenges:" in msgs[1]["content"]

    def test_chinese_lang(self):
        msgs = _make_defender(lang="zh")
        assert "knowledge curator" not in msgs[0]["content"].lower()
        assert "Output JSON" in msgs[0]["content"]

    def test_issues_listed_in_prompt(self):
        msgs = _make_defender()
        assert "Over-generalization" in msgs[1]["content"]
