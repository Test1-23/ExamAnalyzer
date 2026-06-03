"""StageCounters — per-stage failure counters for end-of-pipeline summary."""


class StageCounters:
    """Per-stage failure counters for end-of-pipeline summary.

    Incremented at each ``except Exception`` block in run_pipeline().
    Provides ``summarize()`` for a human-readable failure report.
    """

    def __init__(self):
        self.pdf_extraction = 0
        self.qa_pairing = 0
        self.phase1_worker = 0
        self.phase2_worker = 0
        self.answer_round1 = 0
        self.grade_round2 = 0
        self.fragment_extraction = 0
        self.kp_classification = 0
        self.cross_paper_check = 0
        self.post_processing = 0
        self.stage_list = 0
        self.total_retries = 0

    _fields = (
        "pdf_extraction", "qa_pairing", "phase1_worker", "phase2_worker",
        "answer_round1", "grade_round2", "fragment_extraction",
        "kp_classification", "cross_paper_check", "post_processing",
        "stage_list",
    )

    def summarize(self) -> str:
        parts = []
        for field in self._fields:
            v = getattr(self, field)
            if v > 0:
                parts.append(f"{field}={v}")
        if self.total_retries > 0:
            parts.append(f"retries={self.total_retries}")
        return ", ".join(parts) if parts else "none"

    def total_failures(self) -> int:
        return sum(getattr(self, f) for f in self._fields)

    def nonzero(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in self._fields if getattr(self, f) > 0}

    @property
    def health(self) -> dict:
        failures = self.total_failures()
        if failures == 0:
            return {"status": "ok", "failed_stages": [], "failures": {}}
        return {
            "status": "degraded" if failures < 5 else "unhealthy",
            "failed_stages": list(self.nonzero().keys()),
            "failures": self.nonzero(),
            "retries": self.total_retries,
        }
