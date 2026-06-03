"""PipelineResult — return value of run_pipeline()."""

from dataclasses import dataclass

from .counters import StageCounters


@dataclass(frozen=True)
class PipelineResult:
    """Return value of run_pipeline() — content + counters for programmatic use.

    ``content`` is the formatted analysis text.
    ``counters`` is the StageCounters with per-stage failure/retry data.
    ``health`` is a dict with status, failed_stages, failures, retries.
    """
    content: str
    counters: StageCounters

    @property
    def health(self) -> dict:
        return self.counters.health
