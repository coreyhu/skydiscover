"""Stable readers for records persisted by a SkyDiscover run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SolutionGeneration:
    run: str
    program_id: str
    iteration: int | None
    generation: int | None
    template_key: str
    system_prompt: str | None
    user_prompt: str | None
    response: str
    score: float | None
    parent_id: str | None
    parent_score: float | None
    operator_label: str | None
    changes: Any
    solution: str | None
    metrics: dict[str, Any]


@dataclass(frozen=True)
class StrategyGeneration:
    run: str
    iteration_dir: str
    code: str
    score: float | None
    is_new_best: bool | None
    is_fallback: bool
    start_db_stats: Any
    end_db_stats: Any
    system_prompt: str | None = None
    user_prompt: str | None = None
    response: str | None = None


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    run_path = Path(run_dir)
    checkpoints = sorted(
        (run_path / "checkpoints").glob("checkpoint_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    return checkpoints[-1] if checkpoints else None


def iter_solution_generations(
    run_dir: str | Path,
) -> Iterator[SolutionGeneration]:
    run_path = Path(run_dir)
    checkpoint = latest_checkpoint(run_path)
    if checkpoint is None:
        return
    programs = _load_programs(checkpoint)
    for program in programs.values():
        metrics = program.get("metrics") or {}
        score = _number(metrics.get("combined_score"))
        parent_score = _parent_score(program, programs)
        parent_info = program.get("parent_info") or (None, None)
        for template_key, prompt in (program.get("prompts") or {}).items():
            for response in prompt.get("responses") or ():
                yield SolutionGeneration(
                    run=run_path.name,
                    program_id=program["id"],
                    iteration=program.get("iteration_found"),
                    generation=program.get("generation"),
                    template_key=template_key,
                    system_prompt=prompt.get("system"),
                    user_prompt=prompt.get("user"),
                    response=response,
                    score=score,
                    parent_id=program.get("parent_id"),
                    parent_score=parent_score,
                    operator_label=parent_info[0] or None,
                    changes=(program.get("metadata") or {}).get("changes"),
                    solution=program.get("solution"),
                    metrics=metrics,
                )


def iter_strategy_generations(
    run_dir: str | Path,
) -> Iterator[StrategyGeneration]:
    run_path = Path(run_dir)
    for iteration_dir in sorted((run_path / "search").glob("iteration_*")):
        metadata_path = iteration_dir / "metadata.json"
        code_path = iteration_dir / "code.py"
        if not metadata_path.is_file() or not code_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        prompts_path = iteration_dir / "prompts.json"
        prompts = json.loads(prompts_path.read_text()) if prompts_path.is_file() else {}
        yield StrategyGeneration(
            run=run_path.name,
            iteration_dir=iteration_dir.name,
            code=code_path.read_text(),
            score=_number(metadata.get("score", metadata.get("combined_score"))),
            is_new_best=metadata.get("is_new_best"),
            is_fallback=metadata.get("is_fallback", False),
            start_db_stats=metadata.get("start_db_stats"),
            end_db_stats=metadata.get("end_db_stats"),
            system_prompt=prompts.get("system_prompt"),
            user_prompt=prompts.get("user_prompt"),
            response=prompts.get("llm_response"),
        )


def _load_programs(checkpoint: Path) -> dict[str, dict[str, Any]]:
    programs = {}
    for path in (checkpoint / "programs").glob("*.json"):
        program = json.loads(path.read_text())
        programs[program["id"]] = program
    return programs


def _parent_score(
    program: dict[str, Any], programs: dict[str, dict[str, Any]]
) -> float | None:
    parent = programs.get(program.get("parent_id") or "")
    if parent:
        score = _number((parent.get("metrics") or {}).get("combined_score"))
        if score is not None:
            return score
    return _number(
        ((program.get("metadata") or {}).get("parent_metrics") or {}).get(
            "combined_score"
        )
    )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
