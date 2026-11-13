"""Hermetic unit tests for the deterministic MS scoring-point splitter.

Only src.knowledge._fragments is imported: no database, no network, no
embedding model, no conftest fixtures.  Every test is pure in-memory text.
"""

import sys
from pathlib import Path

import pytest

# Make the exam_analyzer package root importable regardless of the launch dir
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.knowledge._fragments import (  # noqa: E402
    DEFAULT_CONFIG,
    FragmentConfig,
    norm_key,
    split_fragments,
)


# ============================================================
# Real MS samples (verbatim from intermediate/9618_knowledge.db)
# ============================================================

SAMPLE_A = (
    "1 mark / \u2022 The value of the operand is an offset value which is added to "
    "another base value to give the address from which the contents are loaded to "
    "the accumulator"
)

SAMPLE_B = (
    "1 mark for correct answer / Type of compression: lossy / Justification: There "
    "will be fewer samples per second, so data will be permanently lost // There "
    "will be fewer samples per second, so the original sound cannot be re-created"
)

SAMPLE_C = (
    "1 mark for correct threat 3 / 1 mark corresponding description / 1 mark for "
    "correct method of prevention / Threat Description Prevention / Malware / Virus "
    "Malicious code that can alter/delete files Anti-virus // Anti-malware // "
    "Firewall / Spyware Records keystrokes which are sent to third party "
    "Anti-spyware // Firewall // Anti-malware / Hacking Gaining unauthorised access "
    "to a computer network / device Authentication // Firewall"
)


# ============================================================
# Helpers
# ============================================================


def _texts(points):
    return [point["point_text"] for point in points]


def _marks(points):
    return [point["marks"] for point in points]


def _assert_contract(points, qa_id):
    """Every returned dict must satisfy the interface the pipeline relies on."""
    for index, point in enumerate(points):
        assert set(point) == {"point_id", "qa_id", "point_text", "marks", "norm_key"}
        assert point["point_id"] == f"f_{qa_id}_{index}"
        assert point["qa_id"] == qa_id

        text = point["point_text"]
        assert isinstance(text, str) and text.strip()
        assert text == text.strip()
        assert "\ue000" not in text and "\ue001" not in text  # no placeholder leak

        assert isinstance(point["marks"], int) and point["marks"] >= 1

        key = point["norm_key"]
        assert isinstance(key, str) and len(key) == 16
        assert key == norm_key(text)


# ============================================================
# Degenerate inputs
# ============================================================


@pytest.mark.parametrize("answer_text", ["", "   ", "\n\n", "\t \r\n", "\u2022 / // \u00b7", "///"])
def test_empty_or_symbol_only_text_yields_no_points(answer_text):
    assert split_fragments(answer_text, 5) == []


# ============================================================
# Real samples
# ============================================================


def test_sample_a_single_offset_value_point():
    points = split_fragments(SAMPLE_A, 9618)

    _assert_contract(points, 9618)
    assert len(points) == 1
    assert points[0]["marks"] == 1
    assert points[0]["point_text"] == (
        "The value of the operand is an offset value which is added to another base "
        "value to give the address from which the contents are loaded to the "
        "accumulator"
    )


def test_sample_b_alternative_justifications_stay_in_one_point():
    points = split_fragments(SAMPLE_B, 9618)

    _assert_contract(points, 9618)
    assert _texts(points) == [
        "Type of compression: lossy",
        "Justification: There will be fewer samples per second, so data will be "
        "permanently lost // There will be fewer samples per second, so the original "
        "sound cannot be re-created",
    ]
    assert _marks(points) == [1, 1]


def test_sample_c_table_rows_and_accumulated_announcements():
    points = split_fragments(SAMPLE_C, 9618)

    _assert_contract(points, 9618)
    assert len(points) == 7

    # The three leading announcements accumulate onto the first surviving fragment
    assert points[0]["point_text"] == "Threat Description Prevention"
    assert points[0]["marks"] == 3
    assert _marks(points)[1:] == [1] * 6

    # "//" alternatives must never be split across points
    malware_row = next(p for p in points if "Anti-virus" in p["point_text"])
    assert "// Anti-malware" in malware_row["point_text"]
    assert "Firewall" in malware_row["point_text"]

    spyware_row = next(p for p in points if p["point_text"].startswith("Spyware"))
    assert "// Firewall // Anti-malware" in spyware_row["point_text"]

    hacking_row = next(
        p for p in points if p["point_text"].startswith("device Authentication")
    )
    assert hacking_row["point_text"].endswith("// Firewall")


# ============================================================
# Marks announcements
# ============================================================


@pytest.mark.parametrize(
    ("answer_text", "expected_marks", "expected_text"),
    [
        ("2 marks\nfoo bar baz", 2, "foo bar baz"),
        ("(3 marks)\nsome long explanation here", 3, "some long explanation here"),
        ("[4]\nshort note here", 4, "short note here"),
        ("short note (3 marks)", 3, "short note"),
    ],
)
def test_marks_announcement_forms(answer_text, expected_marks, expected_text):
    points = split_fragments(answer_text, 1)

    _assert_contract(points, 1)
    assert _marks(points) == [expected_marks]
    assert _texts(points) == [expected_text]


def test_marks_announcements_accumulate_before_content():
    points = split_fragments("1 mark / 2 marks / a table header row", 1)

    _assert_contract(points, 1)
    assert _texts(points) == ["a table header row"]
    assert _marks(points) == [3]


def test_marks_default_to_one_when_nothing_is_announced():
    points = split_fragments("no announcement at all", 1)

    _assert_contract(points, 1)
    assert _marks(points) == [1]


# ============================================================
# Separators
# ============================================================


def test_newline_bullet_dash_and_numbering_are_separators():
    answer = (
        "first point here\n"
        "\u2022 second point here\n"
        "- third point here\n"
        "1. fourth point here\n"
        "2) fifth point here"
    )
    points = split_fragments(answer, 1)

    _assert_contract(points, 1)
    assert _texts(points) == [
        "first point here",
        "second point here",
        "third point here",
        "fourth point here",
        "fifth point here",
    ]


def test_slash_compounds_are_not_split():
    points = split_fragments("answer and/or s/he wins", 1)

    _assert_contract(points, 1)
    assert _texts(points) == ["answer and/or s/he wins"]


def test_plain_slash_separates_points():
    points = split_fragments("lossy/lossless compression", 1)

    assert _texts(points) == ["lossy", "lossless compression"]
    assert _marks(points) == [1, 1]


def test_double_slash_wins_over_single_slash():
    points = split_fragments("alpha // beta / gamma", 1)

    assert _texts(points) == ["alpha // beta", "gamma"]


def test_url_and_windows_path_are_not_split():
    url_text = "see http://example.com/a/b for details"
    url_points = split_fragments(url_text, 1)
    assert _texts(url_points) == [url_text]

    path_text = "open C:\\Users\\alice\\notes.txt now"
    path_points = split_fragments(path_text, 1)
    assert _texts(path_points) == [path_text]


# ============================================================
# Short-fragment folding
# ============================================================


def test_short_fragment_folds_into_predecessor():
    points = split_fragments("abcdef / xy / ghijkl", 1)

    assert _texts(points) == ["abcdef xy", "ghijkl"]
    assert sum(_marks(points)) == 3  # two folded marks plus the surviving point


def test_leading_short_fragment_folds_into_successor():
    points = split_fragments("ab / longer content here", 1)

    assert len(points) == 1
    assert points[0]["point_text"] == "ab longer content here"
    assert sum(_marks(points)) == 2


# ============================================================
# max_points cap
# ============================================================

ANNOUNCED_FOUR = (
    "1 mark / aaaaaaaaa / 2 marks / bbbbbbbbb / 3 marks / ccccccccc / "
    "4 marks / ddddddddd"
)


def test_announced_marks_are_parsed_in_order():
    points = split_fragments(ANNOUNCED_FOUR, 1)

    assert _marks(points) == [1, 2, 3, 4]
    assert _texts(points) == ["aaaaaaaaa", "bbbbbbbbb", "ccccccccc", "ddddddddd"]


def test_tail_is_merged_when_max_points_is_exceeded():
    uncapped = split_fragments(ANNOUNCED_FOUR, 1, max_points=99)
    capped = split_fragments(ANNOUNCED_FOUR, 1, max_points=2)

    _assert_contract(capped, 1)
    assert len(capped) == 2
    assert _texts(capped) == [
        "aaaaaaaaa",
        "bbbbbbbbb / ccccccccc / ddddddddd",
    ]
    assert sum(_marks(capped)) == sum(_marks(uncapped)) == 10


def test_max_points_one_collapses_all_points():
    points = split_fragments("first point here / second point here", 1, max_points=1)

    _assert_contract(points, 1)
    assert len(points) == 1
    assert points[0]["point_text"] == "first point here / second point here"
    assert points[0]["marks"] == 2


@pytest.mark.parametrize("max_points", [0, -1])
def test_max_points_below_one_yields_no_points(max_points):
    assert split_fragments("a real scoring point", 1, max_points=max_points) == []


def test_long_answer_is_bounded_by_default_cap():
    answer = "\n".join(f"scoring point number {i} with some words" for i in range(200))
    points = split_fragments(answer, 9618)

    _assert_contract(points, 9618)
    assert len(points) == 15  # default max_points
    assert sum(_marks(points)) == 200
    assert points[0]["point_text"] == "scoring point number 0 with some words"
    assert points[-1]["point_text"].endswith("scoring point number 199 with some words")


# ============================================================
# Determinism and payload
# ============================================================


def test_output_is_deterministic_and_ordinally_numbered():
    first = split_fragments(SAMPLE_C, 9618)
    second = split_fragments(SAMPLE_C, 9618)

    assert first == second
    assert [point["point_id"] for point in first] == [
        f"f_9618_{index}" for index in range(len(first))
    ]


def test_point_payload_contract():
    points = split_fragments(
        "first scoring point here / second scoring point here", 42
    )

    _assert_contract(points, 42)
    assert _texts(points) == [
        "first scoring point here",
        "second scoring point here",
    ]


# ============================================================
# norm_key
# ============================================================


def test_norm_key_ignores_case_punctuation_and_whitespace():
    assert norm_key("Hello,  WORLD!") == norm_key("hello world")
    assert norm_key("  anti-virus // firewall ") == norm_key("Anti Virus Firewall")


def test_norm_key_differs_for_different_text():
    assert norm_key("lossy compression") != norm_key("lossless compression")


def test_norm_key_is_stable_lowercase_hex():
    key = norm_key("the value of the operand")

    assert len(key) == 16
    assert all(character in "0123456789abcdef" for character in key)
    assert key == norm_key("the value of the operand")
    assert norm_key("") == norm_key("   ")


# ============================================================
# Configuration surface
# ============================================================


def test_config_defaults_match_documented_behaviour():
    config = FragmentConfig()

    assert config.min_fragment_chars == 3
    assert config.default_marks == 1
    assert config.header_max_chars == 48
    assert isinstance(DEFAULT_CONFIG, FragmentConfig)
