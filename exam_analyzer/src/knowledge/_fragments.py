"""Deterministic mark-scheme (MS) scoring-point splitter — pure functions only.

Mark-scheme answers are free text whose formatting differs from paper to paper:
mark values are announced as "1 mark" / "2 marks" / "(3 marks)" / "[4]",
scoring points are separated by newlines, bullets, line-leading dashes or
numbering, and alternatives that are all acceptable for *one* point are glued
with "//".

This module converts that text into a stable list of scoring-point dicts.
Nothing here touches the database, the network or an embedding model — the
functions are referentially transparent, so the same input always yields
byte-identical output (point_id included).

Splitting assumptions
---------------------
1. A marks announcement ("N mark(s)", "(N marks)", "[N]") is metadata, not
   content.  When it occupies a whole segment it is dropped and its value is
   carried forward to the next content segment; consecutive announcements
   accumulate, so "1 mark" + "1 mark" before one table row awards that row
   2 marks.  An announcement found mid-segment still acts as a boundary.
2. "/" separates scoring points, except inside "and/or" / "he/she" / "s/he"
   compounds, inside "//" alternatives and inside URLs or Windows paths
   ("http" / "www." / backslash).
3. "//" is stashed *before* "/" is considered, so an alternative such as
   "Anti-virus // Anti-malware // Firewall" always stays inside one fragment.
4. Structural separators are newlines, bullets ("• · ● ◦ ▪"), line-leading
   dashes and line-leading numbering ("1." / "1)").
5. A fragment shorter than min_fragment_chars is folded into its predecessor
   (the very first one is folded into its successor), and a fragment with no
   alphanumeric character at all is dropped, so stray table column headers and
   orphan punctuation never become points.
6. marks defaults to 1 when nothing is announced.
7. When more than max_points fragments survive, the tail is merged in order
   into the last kept slot; marks totals are preserved by summation.

Known limitations
-----------------
* A run of announcements that precedes a *table header row* (multi-part
  questions that share marks across a threat / description / prevention table)
  lands on the header row instead of the first data row; the total is right but
  row-level attribution can be off by one.
* One announcement may govern several "/"-separated clauses, e.g.
  "1 mark for correct answer / Type of compression: lossy / Justification:
  ...".  Every resulting fragment still defaults to 1 mark, so the summed
  marks over-counts the announced total.  Redistributing marks is the caller's
  job because only the caller knows the question's real mark budget.
* Prose compounds such as "alter/delete" are split because "/" is a
  separator; the left side usually survives as its own (short) point.
* This is a heuristic, not a parser for any particular exam board's layout:
  header_max_chars decides what counts as announcement-only metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class FragmentConfig:
    """Tunable parameters for split_fragments."""

    # Fragments below this length are folded into a neighbour
    min_fragment_chars: int = 3

    # A segment that starts with a marks announcement is treated as metadata
    # only when it is at most this long — wide enough for
    # "1 mark for correct method of prevention", narrow enough to keep real
    # sentences from being swallowed.
    header_max_chars: int = 48

    # Marks awarded when nothing is announced
    default_marks: int = 1

    # Glue used when the tail of a long answer is collapsed to max_points
    tail_join: str = " / "


DEFAULT_CONFIG = FragmentConfig()


# ============================================================
# Patterns
# ============================================================

# Private-use wrapper for stashed substrings (never present in MS text)
_PH_OPEN = "\ue000"
_PH_CLOSE = "\ue001"
_PLACEHOLDER_RE = re.compile(_PH_OPEN + r"(\d+)" + _PH_CLOSE)

#: Alternatives inside a single point — must survive "/" splitting
_ALT_SEP = "//"

#: Protected slash contexts; matched before "/" is ever a separator
_URL_OR_PATH_RE = re.compile(
    r"[^\s]*https?://[^\s]*"
    r"|[^\s]*www\.[^\s]*"
    r"|[^\s]*\\[^\s]*",
    re.IGNORECASE,
)
_SLASH_COMPOUND_RE = re.compile(r"\b(?:and/or|he/she|s/he)\b", re.IGNORECASE)

#: Words that may follow a marks announcement inside announcement-only
#: metadata ("1 mark for correct answer", "1 mark corresponding description")
_HEADER_QUALIFIERS = frozenset({
    "for", "if", "per", "each", "any", "one", "same", "corresponding",
    "award", "awarded", "max", "maximum", "total", "in", "to",
})

#: Longest residual after an announcement that still counts as metadata
_HEADER_RESIDUAL_MAX = 34

#: Marks announcements, most specific form first
_MARKS_BRACKET_RE = re.compile(r"[\(\[]\s*(\d{1,3})\s*marks?\s*[\)\]]", re.IGNORECASE)
_MARKS_NUM_BRACKET_RE = re.compile(r"\[\s*(\d{1,3})\s*\]")
_MARKS_WORD_RE = re.compile(r"(?<![\w/])(\d{1,3})\s*marks?\b", re.IGNORECASE)

#: Structural separators — newline, bullet, slash (protected forms stashed)
_SEP_RE = re.compile(r"[\r\n]+|\s*[•·●◦▪]\s*|\s*/\s*")

#: Line-leading list decoration ("- ", "1. ", "2) ")
_LEAD_MARKER_RE = re.compile(r"^(?:[-–—]+|\d{1,2}[.)])\s+")

_WS_RE = re.compile(r"\s+")


# ============================================================
# Intermediate fragment
# ============================================================


@dataclass
class _Fragment:
    """Mutable scoring point used until the final dicts are built."""

    text: str
    marks: int


# ============================================================
# Normalisation fingerprint
# ============================================================


def norm_key(text: str) -> str:
    """Return a deterministic 16-hex-char fingerprint of text.

    Case (casefold), punctuation and whitespace runs are normalised away, so
    two scoring points that differ only in formatting share a key.  Unicode
    word characters (including CJK) are preserved, and the result depends on
    nothing but the input — no salt, no hash randomisation.
    """
    normalised = _WS_RE.sub(" ", text.casefold()).strip()
    normalised = re.sub(r"[\W_]+", " ", normalised, flags=re.UNICODE)
    normalised = _WS_RE.sub(" ", normalised).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]


# ============================================================
# Protection of slash-sensitive substrings
# ============================================================


def _stash(literal: str, tokens: list[str]) -> str:
    """Append literal to tokens and return its placeholder."""
    tokens.append(literal)
    return f"{_PH_OPEN}{len(tokens) - 1}{_PH_CLOSE}"


def _protect(text: str, tokens: list[str]) -> str:
    """Replace slash-sensitive substrings with placeholders.

    Order matters: URLs / paths are stashed first so their "//" is never
    mistaken for an alternative marker, then slash compounds, then "//".
    """
    # Never let the input collide with the private-use placeholder alphabet
    text = text.replace(_PH_OPEN, " ").replace(_PH_CLOSE, " ")
    text = _URL_OR_PATH_RE.sub(lambda m: _stash(m.group(0), tokens), text)
    text = _SLASH_COMPOUND_RE.sub(lambda m: _stash(m.group(0), tokens), text)
    return text.replace(_ALT_SEP, _stash(_ALT_SEP, tokens))


def _restore(text: str, tokens: list[str]) -> str:
    """Substitute stashed substrings back into text."""
    return _PLACEHOLDER_RE.sub(lambda m: tokens[int(m.group(1))], text)


# ============================================================
# Segmentation
# ============================================================


def _segment(protected: str) -> list[str]:
    """Split protected text into raw content segments at structural borders."""
    segments: list[str] = []
    for chunk in _SEP_RE.split(protected):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = _LEAD_MARKER_RE.sub("", chunk, count=1).strip()
        if chunk:
            segments.append(chunk)
    return segments


def _clamp_marks(value: int, config: FragmentConfig = DEFAULT_CONFIG) -> int:
    """Coerce an announced mark value into a sane positive integer."""
    return max(config.default_marks, min(int(value), 99))


def _marker_spans(
    segment: str, config: FragmentConfig = DEFAULT_CONFIG
) -> list[tuple[int, int, int]]:
    """Return (start, end, marks) for every marks announcement in a segment.

    Bracket forms win over the bare word form so "(2 marks)" is consumed as one
    token instead of leaving the parentheses behind.
    """
    spans: list[tuple[int, int, int]] = []

    def overlapped(start: int, end: int) -> bool:
        return any(start < span_end and span_start < end for span_start, span_end, _ in spans)

    for pattern in (_MARKS_BRACKET_RE, _MARKS_NUM_BRACKET_RE, _MARKS_WORD_RE):
        for match in pattern.finditer(segment):
            if overlapped(match.start(), match.end()):
                continue
            spans.append(
                (match.start(), match.end(), _clamp_marks(int(match.group(1)), config))
            )
    spans.sort()
    return spans


def _header_marks(
    segment: str,
    spans: list[tuple[int, int, int]],
    config: FragmentConfig = DEFAULT_CONFIG,
) -> int | None:
    """Marks announced by an announcement-only segment, else None.

    A segment qualifies when it *starts* with the announcement, carries exactly
    one, and is short enough to be metadata rather than a sentence.  Whatever
    follows the announcement must either be nothing or a short qualifier such as
    "for correct answer" / "corresponding description"; anything else is treated
    as content whose announcement has merely been stripped.
    """
    if len(spans) != 1:
        return None
    start, end, marks = spans[0]
    if segment[:start].strip():
        return None
    if len(segment) > config.header_max_chars:
        return None
    residual = segment[end:].strip()
    if not residual:
        return marks
    if len(residual) > _HEADER_RESIDUAL_MAX:
        return None
    if residual.split(" ", 1)[0].casefold() in _HEADER_QUALIFIERS:
        return marks
    return None


# ============================================================
# Collection, folding, collapsing
# ============================================================


def _collect(
    segments: list[str],
    tokens: list[str],
    config: FragmentConfig = DEFAULT_CONFIG,
) -> tuple[list[_Fragment], int, str]:
    """Turn segments into fragments, returning (fragments, pending, header).

    pending / header are the trailing announcement state that could not be
    attached to any following content.
    """
    fragments: list[_Fragment] = []
    pending = 0
    last_header = ""

    def emit(raw: str, announced: int) -> None:
        text = _WS_RE.sub(" ", _restore(raw, tokens)).strip()
        if not text:
            return
        fragments.append(_Fragment(text=text, marks=announced or config.default_marks))

    for segment in segments:
        spans = _marker_spans(segment, config)
        if not spans:
            emit(segment, pending)
            pending = 0
            continue

        header = _header_marks(segment, spans, config)
        if header is not None:
            pending += header
            last_header = _WS_RE.sub(" ", segment).strip()
            continue

        # Announcements act as boundaries: text before one belongs to the
        # accumulated marks, text after it opens the next point.  A marker that
        # closes the segment annotates the text before it ("... (2 marks)").
        cursor = 0
        for start, end, marks in spans:
            chunk = segment[cursor:start]
            if chunk.strip():
                annotates = not segment[end:].strip()
                emit(chunk, pending or (marks if annotates else 0))
                pending = 0
                if not annotates:
                    pending += marks
            else:
                pending += marks
            cursor = end
        tail = segment[cursor:]
        if tail.strip():
            emit(tail, pending)
            pending = 0

    return fragments, pending, last_header


def _is_noise(fragment: _Fragment) -> bool:
    """True when a fragment holds no alphanumeric content at all."""
    return not any(character.isalnum() for character in fragment.text)


def _fold_short(fragments: list[_Fragment], config: FragmentConfig = DEFAULT_CONFIG) -> None:
    """Fold sub-min_fragment_chars fragments into a neighbour, in place.

    The first fragment has no predecessor, so it is folded forward; marks of a
    folded fragment are added to the host fragment.
    """
    if len(fragments) >= 2 and len(fragments[0].text) < config.min_fragment_chars:
        first = fragments.pop(0)
        fragments[0].text = f"{first.text} {fragments[0].text}".strip()
        fragments[0].marks += first.marks

    folded: list[_Fragment] = []
    for fragment in fragments:
        if folded and len(fragment.text) < config.min_fragment_chars:
            folded[-1].text = f"{folded[-1].text} {fragment.text}".strip()
            folded[-1].marks += fragment.marks
        else:
            folded.append(fragment)
    fragments[:] = folded


def _collapse_tail(
    fragments: list[_Fragment], max_points: int, config: FragmentConfig = DEFAULT_CONFIG
) -> None:
    """Merge the tail beyond max_points into the last kept fragment."""
    if max_points < 1:
        fragments.clear()
        return
    if len(fragments) <= max_points:
        return
    tail = fragments[max_points - 1 :]
    merged = _Fragment(
        text=config.tail_join.join(fragment.text for fragment in tail),
        marks=sum(fragment.marks for fragment in tail),
    )
    fragments[:] = fragments[: max_points - 1] + [merged]


# ============================================================
# Public entry point
# ============================================================


def split_fragments(answer_text: str, qa_id: int, max_points: int = 15) -> list[dict]:
    """Split a mark-scheme answer into deterministic scoring points.

    Args:
        answer_text: Raw MS answer text for one question (may be empty).
        qa_id: QA row id, echoed into every point id and payload.
        max_points: Hard cap on the number of returned fragments; the tail is
            merged in order so marks totals survive the cap.  A value below 1
            yields an empty list.

    Returns:
        A list of {point_id, qa_id, point_text, marks, norm_key} dicts in
        source order, where point_id is "f_<qa_id>_<index>" and norm_key is
        the norm_key() fingerprint of the text.
    """
    if not answer_text or not answer_text.strip():
        return []

    tokens: list[str] = []
    protected = _protect(answer_text, tokens)
    segments = _segment(protected)
    fragments, pending, last_header = _collect(segments, tokens, DEFAULT_CONFIG)

    # An announcement with nothing left to describe either annotates the last
    # fragment or, if there is no content at all, becomes one itself.
    if pending:
        if fragments:
            fragments[-1].marks = max(fragments[-1].marks, pending)
        else:
            fragments.append(_Fragment(text=last_header or f"{pending} mark", marks=pending))

    fragments = [fragment for fragment in fragments if not _is_noise(fragment)]
    _fold_short(fragments, DEFAULT_CONFIG)
    _collapse_tail(fragments, max_points, DEFAULT_CONFIG)

    return [
        {
            "point_id": f"f_{qa_id}_{index}",
            "qa_id": qa_id,
            "point_text": fragment.text,
            "marks": fragment.marks,
            "norm_key": norm_key(fragment.text),
        }
        for index, fragment in enumerate(fragments)
    ]