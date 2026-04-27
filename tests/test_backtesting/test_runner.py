"""Tests du runner CLI backtesting (Phase 5 étape 4)."""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from backtesting import runner
from backtesting.runner import (
    _parse_date,
    _result_to_dict,
    _theses_in_window,
    build_parser,
    run_portfolio_mode,
    run_walk_forward_mode,
)


# ---------------------------------------------------------------- helpers


def _seed_thesis(
    *,
    asset_id: str = "NVDA",
    created_at: datetime,
    model_version: str = "v_test",
    entry_price: float | None = 100.0,
) -> None:
    import json as _json

    from memory.database import Thesis, session_scope
    with session_scope() as s:
        s.add(Thesis(
            created_at=created_at,
            asset_type="stock", asset_id=asset_id, sector_id="ai_ml",
            score=80.0,
            score_breakdown_json=_json.dumps({"dimensions": {"momentum": 80.0}}),
            recommendation="BUY", horizon_days=10, entry_price=entry_price,
            entry_conditions_json=None,
            triggers_json="[]", risks_json="[]", catalysts_json="[]",
            narrative="…", model_version=model_version,
            weights_snapshot_json="{}",
        ))


# --------------------------------------------------------------- _parse_date


def test_parse_date_iso():
    assert _parse_date("2026-04-01") == datetime(2026, 4, 1)


def test_parse_date_iso_with_time():
    assert _parse_date("2026-04-01T12:30:00") == datetime(2026, 4, 1, 12, 30, 0)


def test_parse_date_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_date("not a date")


# ----------------------------------------------------------- _result_to_dict


def test_result_to_dict_handles_datetime():
    d = {"ts": datetime(2026, 4, 1, 12)}
    out = _result_to_dict(d)
    assert out["ts"] == "2026-04-01T12:00:00"


def test_result_to_dict_handles_dataclass():
    from dataclasses import dataclass

    @dataclass
    class X:
        a: int = 1
        b: str = "x"

    out = _result_to_dict(X())
    assert out == {"a": 1, "b": "x"}


def test_result_to_dict_handles_nested_lists():
    out = _result_to_dict({"l": [datetime(2026, 1, 1), 2]})
    assert out == {"l": ["2026-01-01T00:00:00", 2]}


# ----------------------------------------------------------- _theses_in_window


def test_theses_in_window_filters_by_model_version(tmp_db):
    _seed_thesis(created_at=datetime(2026, 4, 1), model_version="v_a")
    _seed_thesis(created_at=datetime(2026, 4, 1), asset_id="AMD",
                 model_version="v_b")
    out = list(_theses_in_window(
        "v_a", datetime(2026, 1, 1), datetime(2026, 12, 31),
    ))
    assert len(out) == 1
    assert out[0].model_version == "v_a"


def test_theses_in_window_filters_by_date(tmp_db):
    _seed_thesis(created_at=datetime(2026, 1, 1), model_version="v_a")
    _seed_thesis(created_at=datetime(2026, 6, 1), model_version="v_a")
    out = list(_theses_in_window(
        "v_a", datetime(2026, 4, 1), datetime(2026, 12, 31),
    ))
    assert len(out) == 1
    assert out[0].created_at == datetime(2026, 6, 1)


def test_theses_in_window_empty(tmp_db):
    assert list(_theses_in_window(
        "v_a", datetime(2026, 1, 1), datetime(2026, 12, 31),
    )) == []


# ---------------------------------------------------- run_portfolio_mode


def test_run_portfolio_mode_returns_serializable_dict(tmp_db):
    """Sans thèses → résultat plat mais valide."""
    out = run_portfolio_mode(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 5),
    )
    assert out["mode"] == "portfolio"
    # Doit être sérialisable en JSON sans erreur.
    json.dumps(out)


def test_run_portfolio_mode_includes_metrics(tmp_db):
    out = run_portfolio_mode(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 5),
    )
    assert "metrics" in out["result"]


# ----------------------------------------------- run_walk_forward_mode


def test_run_walk_forward_mode_returns_summary(tmp_db):
    """Walk-forward sans thèses → fold sans gagnant fort, mais structure OK."""
    out = run_walk_forward_mode(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        n_folds=2,
        weight_candidates=["v_a", "v_b"],
    )
    assert out["mode"] == "walk_forward"
    assert out["n_folds"] == 2
    assert "summary" in out
    assert "best_per_fold" in out["summary"]


# --------------------------------------------------------------- CLI parser


def test_parser_portfolio_subcommand():
    parser = build_parser()
    args = parser.parse_args([
        "portfolio", "--start", "2026-01-01", "--end", "2026-04-01",
    ])
    assert args.mode == "portfolio"
    assert args.start == datetime(2026, 1, 1)


def test_parser_walk_forward_requires_folds_and_weights():
    parser = build_parser()
    args = parser.parse_args([
        "walk-forward",
        "--start", "2024-01-01", "--end", "2026-01-01",
        "--folds", "3", "--weights", "v1,v2,v3",
    ])
    assert args.folds == 3
    assert args.weights == "v1,v2,v3"


def test_parser_replay_step_days_default():
    parser = build_parser()
    args = parser.parse_args([
        "replay", "--start", "2026-01-01", "--end", "2026-04-01",
    ])
    assert args.step_days == 1


# ------------------------------------------------------------------ main


def test_main_portfolio_writes_to_stdout(tmp_db, capsys):
    """`main` en mode portfolio écrit du JSON valide sur stdout."""
    rc = runner.main([
        "portfolio",
        "--start", "2026-04-01",
        "--end", "2026-04-05",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["mode"] == "portfolio"


def test_main_writes_to_output_file(tmp_db, tmp_path):
    out_file = tmp_path / "report.json"
    rc = runner.main([
        "portfolio",
        "--start", "2026-04-01",
        "--end", "2026-04-05",
        "--output", str(out_file),
    ])
    assert rc == 0
    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    assert parsed["mode"] == "portfolio"


def test_main_walk_forward_empty_weights_raises(tmp_db):
    with pytest.raises(SystemExit):
        runner.main([
            "walk-forward",
            "--start", "2024-01-01", "--end", "2026-01-01",
            "--folds", "2", "--weights", "  ",
        ])


def test_main_with_benchmark_flag(tmp_db, capsys):
    """Le flag --benchmark est parsé en liste."""
    runner.main([
        "portfolio",
        "--start", "2026-04-01", "--end", "2026-04-05",
        "--benchmark", "SPY,QQQ",
    ])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["benchmark_tickers"] == ["SPY", "QQQ"]
