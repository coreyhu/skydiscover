"""Resolved configuration snapshots must be complete and reproducible."""

from __future__ import annotations

import json

from skydiscover.config import Config


def test_resolved_snapshot_is_additive_to_the_existing_projection() -> None:
    config = Config.from_dict({"search": {"type": "evox", "switch_interval": 3, "share_llm": True}})

    assert "switch_interval" not in config.to_dict()["search"]
    assert config.to_resolved_dict()["search"]["switch_interval"] == 3


def test_resolved_snapshot_records_the_requested_seed() -> None:
    config = Config.from_dict({"random_seed": 17})

    assert config.random_seed == 17
    assert config.to_resolved_dict()["random_seed"] == 17


def test_evox_snapshot_includes_effective_search_configuration() -> None:
    config = Config.from_dict(
        {
            "max_iterations": 30,
            "language": "python",
            "search": {
                "type": "evox",
                "num_context_programs": 6,
                "output_dir": "outputs/pilot",
                "switch_interval": 3,
                "share_llm": True,
                "database": {
                    "auto_generate_variation_operators": False,
                    "pilot_label": "B2",
                },
            },
            "benchmark": {
                "enabled": True,
                "name": "pilot",
                "instance": 7,
            },
        }
    )

    resolved = config.to_resolved_dict()

    assert resolved["language"] == "python"
    assert resolved["search"] == {
        "type": "evox",
        "database": {
            "db_path": None,
            "log_prompts": True,
            "database_file_path": resolved["search"]["database"]["database_file_path"],
            "evaluation_file": resolved["search"]["database"]["evaluation_file"],
            "config_path": resolved["search"]["database"]["config_path"],
            "auto_generate_variation_operators": False,
            "pilot_label": "B2",
        },
        "num_context_programs": 6,
        "output_dir": "outputs/pilot",
        "switch_interval": 3,
        "share_llm": True,
    }
    assert resolved["benchmark"]["params"] == {"instance": 7}
    json.dumps(resolved, sort_keys=True)


def test_snapshot_excludes_credentials_and_live_clients() -> None:
    config = Config.from_dict(
        {
            "llm": {
                "api_key": "secret",
                "models": [{"name": "model", "api_key": "model-secret"}],
            },
            "monitor": {"summary_api_key": "monitor-secret"},
        }
    )

    rendered = json.dumps(config.to_resolved_dict(), sort_keys=True)

    assert "secret" not in rendered
    assert "api_key" not in rendered
    assert "init_client" not in rendered


def test_resolved_snapshot_round_trips_supported_configuration() -> None:
    original = Config.from_dict(
        {
            "search": {
                "type": "evox",
                "switch_interval": 4,
                "share_llm": True,
                "database": {"auto_generate_variation_operators": False},
            },
        }
    )

    restored = Config.from_dict(original.to_resolved_dict())

    assert restored.search.switch_interval == 4
    assert restored.search.share_llm is True
    assert restored.search.database.auto_generate_variation_operators is False
