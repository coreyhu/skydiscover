import asyncio
import json

from skydiscover.export import (
    iter_solution_generations,
    iter_strategy_generations,
    latest_checkpoint,
)
from skydiscover.search.evox.utils.coevolve_logging import (
    log_search_algorithm_generated,
)
from skydiscover.search.utils.discovery_utils import SerializableResult


def _write_program(checkpoint, program):
    programs = checkpoint / "programs"
    programs.mkdir(parents=True, exist_ok=True)
    (programs / f"{program['id']}.json").write_text(json.dumps(program))


def test_solution_generations_include_lineage_and_prompts(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "checkpoint_2"
    _write_program(checkpoint, {"id": "seed", "metrics": {"combined_score": 0.5}})
    _write_program(
        checkpoint,
        {
            "id": "child",
            "parent_id": "seed",
            "parent_info": ["REFINE", "seed"],
            "metrics": {"combined_score": 0.7},
            "prompts": {
                "diff": {
                    "system": "system",
                    "user": "user",
                    "responses": ["response"],
                }
            },
        },
    )

    assert latest_checkpoint(tmp_path) == checkpoint
    records = list(iter_solution_generations(tmp_path))
    assert len(records) == 1
    assert records[0].parent_score == 0.5
    assert records[0].response == "response"


def test_strategy_generations_include_prompt_provenance(tmp_path):
    iteration = tmp_path / "search" / "iteration_1"
    iteration.mkdir(parents=True)
    (iteration / "metadata.json").write_text(
        json.dumps({"combined_score": 0.2, "is_new_best": True})
    )
    (iteration / "code.py").write_text("class Search: pass")
    (iteration / "prompts.json").write_text(
        json.dumps(
            {
                "system_prompt": "system",
                "user_prompt": "user",
                "llm_response": "response",
            }
        )
    )

    records = list(iter_strategy_generations(tmp_path))
    assert len(records) == 1
    assert records[0].score == 0.2
    assert records[0].system_prompt == "system"
    assert records[0].response == "response"


def test_successful_strategy_logging_persists_prompt_provenance(tmp_path):
    result = SerializableResult(
        child_program_dict={"id": "strategy", "solution": "class Search: pass"},
        prompt={"system": "system", "user": "user"},
        llm_response="response",
    )
    asyncio.run(log_search_algorithm_generated(str(tmp_path), result, iteration=2))

    prompts = json.loads((tmp_path / "iteration_2" / "prompts.json").read_text())
    assert prompts == {
        "system_prompt": "system",
        "user_prompt": "user",
        "llm_response": "response",
    }
