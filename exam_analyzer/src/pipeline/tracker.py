"""ProgressTracker — minimal progress tracker for pipeline orchestration."""


class ProgressTracker:
    """Minimal progress tracker with step counting for pipeline orchestration.

    Wraps a progress callback to auto-compute percentage from step count.
    ``step()`` increments the counter; ``set_status()`` updates the label only.
    """

    def __init__(self, total: int, progress_cb, log_cb):
        self._total = max(total, 1)
        self._current = 0
        self._progress = progress_cb
        self._log = log_cb

    def step(self, label: str = ""):
        self._current += 1
        pct = min(int(self._current / self._total * 100), 100)
        if label:
            self._progress(pct, label)

    def set_status(self, msg: str):
        pct = min(int(self._current / self._total * 100), 100)
        self._progress(pct, msg)
