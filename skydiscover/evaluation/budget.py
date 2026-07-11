"""Thread-safe, durable accounting for evaluator executions."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, Optional


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before an evaluator execution that would exceed its budget."""

    def __init__(self, scope: str, limit: int, consumed: int):
        self.scope = scope
        self.limit = limit
        self.consumed = consumed
        super().__init__(
            f"Evaluation budget exhausted for {scope!r}: "
            f"consumed {consumed} of {limit} executions"
        )


class EvaluationBudget:
    """Reserve evaluator executions atomically and persist their counts.

    A reservation is made immediately before an evaluator is invoked. This
    means retries and cascade stages consume separate reservations, while LLM
    responses that fail parsing consume none.
    """

    SCOPES = ("candidate", "initial", "final_test")

    def __init__(
        self,
        *,
        max_candidate_evaluations: Optional[int] = None,
        report_path: Optional[str] = None,
    ) -> None:
        if max_candidate_evaluations is not None:
            if (
                isinstance(max_candidate_evaluations, bool)
                or not isinstance(max_candidate_evaluations, int)
                or max_candidate_evaluations < 0
            ):
                raise ValueError("max_candidate_evaluations must be a non-negative integer")

        self._limits: Dict[str, Optional[int]] = {
            "candidate": max_candidate_evaluations,
            "initial": None,
            "final_test": None,
        }
        self._counts: Dict[str, int] = defaultdict(int)
        self._report_path = Path(report_path).resolve() if report_path else None
        self._lock = RLock()
        self._on_exhausted: Optional[Callable[[], None]] = None
        self._termination_reason: Optional[str] = None

        if self._report_path and self._report_path.exists():
            self._restore()

    @property
    def enabled(self) -> bool:
        return any(limit is not None for limit in self._limits.values())

    @property
    def report_path(self) -> Optional[Path]:
        return self._report_path

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._termination_reason is not None

    @property
    def termination_reason(self) -> Optional[str]:
        with self._lock:
            return self._termination_reason

    def set_exhaustion_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback used to stop scheduling once the cap is reached."""
        should_call = False
        with self._lock:
            self._on_exhausted = callback
            should_call = self._termination_reason is not None
        if should_call:
            callback()

    def reserve(self, scope: str = "candidate") -> None:
        """Atomically reserve one actual evaluator execution."""
        if scope not in self._limits:
            raise ValueError(
                f"Unknown evaluation budget scope {scope!r}; expected one of {self.SCOPES}"
            )

        callback = None
        error = None
        with self._lock:
            limit = self._limits.get(scope)
            consumed = self._counts[scope]

            if limit is not None and consumed >= limit:
                self._mark_exhausted(scope)
                self._persist()
                callback = self._on_exhausted
                error = EvaluationBudgetExceeded(scope, limit, consumed)
            else:
                self._counts[scope] = consumed + 1
                if limit is not None and self._counts[scope] >= limit:
                    self._mark_exhausted(scope)
                    callback = self._on_exhausted
                self._persist()

        if callback:
            callback()
        if error:
            raise error

    def snapshot(self) -> dict:
        """Return the JSON-compatible budget report."""
        with self._lock:
            scopes = {}
            for scope in self.SCOPES:
                limit = self._limits.get(scope)
                consumed = self._counts[scope]
                scopes[scope] = {
                    "limit": limit,
                    "consumed": consumed,
                    "remaining": None if limit is None else max(0, limit - consumed),
                    "exhausted": limit is not None and consumed >= limit,
                }
            return {
                "schema_version": 1,
                "status": "exhausted" if self._termination_reason else "running",
                "termination_reason": self._termination_reason,
                "scopes": scopes,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    def _mark_exhausted(self, scope: str) -> None:
        if self._termination_reason is None:
            self._termination_reason = f"{scope}_evaluation_budget_exhausted"

    def _restore(self) -> None:
        with self._report_path.open() as f:
            report = json.load(f)

        if report.get("schema_version") != 1:
            raise ValueError(f"Unsupported evaluation budget report: {self._report_path}")

        saved_scopes = report.get("scopes", {})
        saved_limit = saved_scopes.get("candidate", {}).get("limit")
        current_limit = self._limits["candidate"]
        if saved_limit != current_limit:
            raise ValueError(
                "Cannot resume with a different max_candidate_evaluations "
                f"({saved_limit!r} in report, {current_limit!r} in config)"
            )

        for scope, values in saved_scopes.items():
            consumed = values.get("consumed", 0)
            if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
                raise ValueError(f"Invalid consumed count for {scope!r} in {self._report_path}")
            self._counts[scope] = consumed

        self._termination_reason = report.get("termination_reason")
        if current_limit is not None and self._counts["candidate"] >= current_limit:
            self._mark_exhausted("candidate")

    def _persist(self) -> None:
        if self._report_path is None:
            return

        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=self._report_path.parent,
            prefix=f".{self._report_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.snapshot(), f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self._report_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
