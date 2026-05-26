import re
from pathlib import Path
from typing import List, Tuple


def parse_exam_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse an exam filename like '{subject}_{year}_qp_{variant}.pdf'.
    Returns (prefix, doc_type, number) where doc_type is 'qp' or 'ms'.
    Uses greedy prefix matching to handle variable underscore-separated fields.
    """
    stem = Path(filename).stem
    pattern = r'^(.+)_(qp|ms)_(.+)$'
    match = re.match(pattern, stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")
    return match.group(1), match.group(2).lower(), match.group(3)


def pair_files(directory: str) -> List[Tuple[str, str, str]]:
    """
    Scan the given directory for paired qp/ms PDF files.
    Returns list of (qp_path, ms_path, display_name).
    """
    from collections import defaultdict

    pairs = defaultdict(dict)
    pdf_dir = Path(directory)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for f in pdf_dir.glob("*.pdf"):
        try:
            prefix, doc_type, number = parse_exam_filename(f.name)
            key = (prefix, number)
            pairs[key][doc_type] = str(f)
        except ValueError:
            continue

    result = []
    for (prefix, number), docs in pairs.items():
        if "qp" in docs and "ms" in docs:
            display = f"{prefix}_{number}"
            result.append((docs["qp"], docs["ms"], display))
        else:
            missing = "ms" if "qp" in docs else "qp"
            print(f"Warning: {prefix}_{number} missing {missing} file")

    return result
