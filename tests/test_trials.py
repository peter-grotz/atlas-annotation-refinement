"""Tests for the trial log.

The log is the artefact that makes structure *n+1* cheaper than structure *n*,
so retrieval has to work on signatures it has never seen exactly and on entries
whose fields are incomplete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_refine.experiments import (
    COMPARISON_FIELDS,
    TrialLog,
    render_index,
    signature_distance,
)

SHELL = {
    "volume_ratio": 1.17,
    "intensity_separation": 0.42,
    "excess_median_depth": 2.4,
    "excess_surface_fraction": 0.86,
    "excess_largest_component": 0.98,
    "subtractive_recall_ceiling": 0.97,
}
DISPLACED = {
    "volume_ratio": 1.00,
    "intensity_separation": -0.22,
    "excess_median_depth": 1.0,
    "excess_surface_fraction": 0.95,
    "excess_largest_component": 0.55,
    "subtractive_recall_ceiling": 0.62,
}


@pytest.fixture
def log(tmp_path: Path) -> TrialLog:
    return TrialLog(tmp_path / "trials.jsonl")


def _seed(log: TrialLog) -> None:
    log.record(
        structure="a", specimens=["s1"], family="contrast_threshold",
        params={"fraction": 0.42}, signature=SHELL, scores={"dice": 0.97},
        outcome="admitted", reason="Excess separates from the structure in intensity.",
    )
    log.record(
        structure="b", specimens=["s2"], family="local_mean_threshold",
        params={"k": 1.02}, signature=SHELL, scores={"dice": 0.63},
        outcome="rejected",
        reason="Inside a thick bright structure the local mean is the structure itself.",
    )
    log.record(
        structure="c", specimens=["s3"], family="contrast_threshold",
        params={"fraction": 0.42}, signature=DISPLACED, scores={"dice": 0.61},
        outcome="rejected",
        reason="Populations overlap; no threshold can separate a displaced label.",
    )


class TestRoundTrip:
    def test_records_and_reads_back(self, log: TrialLog):
        trial = log.record(
            structure="a", specimens=["s1"], family="f", params={"x": 1.0},
            signature=SHELL, scores={"dice": 0.9}, outcome="admitted", reason="because",
        )
        stored = log.all()
        assert len(stored) == 1
        assert stored[0].trial_id == trial.trial_id
        assert stored[0].params == {"x": 1.0}
        assert stored[0].evidence == "in-sample"

    def test_is_append_only(self, log: TrialLog):
        for i in range(3):
            log.record(structure="a", specimens=[f"s{i}"], family="f", params={},
                       signature=SHELL, scores={}, outcome="explored", reason="")
        assert len(log.all()) == 3
        assert len({t.trial_id for t in log.all()}) == 3

    def test_empty_log_is_readable(self, log: TrialLog):
        assert log.all() == []
        assert log.summary()["total"] == 0
        assert log.similar(SHELL) == []

    def test_creates_parent_directories(self, tmp_path: Path):
        nested = TrialLog(tmp_path / "deep" / "deeper" / "trials.jsonl")
        nested.record(structure="a", specimens=[], family="f", params={},
                      signature={}, scores={}, outcome="explored", reason="")
        assert nested.path.exists()

    def test_each_line_is_valid_json(self, log: TrialLog):
        _seed(log)
        for line in log.path.read_text().splitlines():
            json.loads(line)

    def test_non_numeric_signature_values_are_dropped(self, log: TrialLog):
        log.record(
            structure="a", specimens=[], family="f", params={},
            signature={**SHELL, "dominant_error": "over-coverage"},
            scores={}, outcome="explored", reason="",
        )
        assert "dominant_error" not in log.all()[0].signature
        assert "volume_ratio" in log.all()[0].signature


class TestDistance:
    def test_identical_signatures_are_zero_apart(self):
        assert signature_distance(SHELL, SHELL) == pytest.approx(0.0)

    def test_is_symmetric(self):
        assert signature_distance(SHELL, DISPLACED) == pytest.approx(
            signature_distance(DISPLACED, SHELL)
        )

    def test_distinguishes_the_two_regimes(self):
        assert signature_distance(SHELL, DISPLACED) > 0.5

    def test_partial_signatures_use_shared_fields_only(self):
        partial = {"volume_ratio": SHELL["volume_ratio"]}
        assert signature_distance(partial, SHELL) == pytest.approx(0.0)

    def test_no_shared_fields_is_infinite(self):
        assert signature_distance({"unrelated": 1.0}, SHELL) == float("inf")

    def test_empty_signature_is_infinite(self):
        assert signature_distance({}, SHELL) == float("inf")

    def test_comparison_fields_are_all_scaled(self):
        from atlas_refine.experiments.trials import _FIELD_SCALE

        assert set(COMPARISON_FIELDS) == set(_FIELD_SCALE)


class TestRetrieval:
    def test_orders_by_similarity(self, log: TrialLog):
        _seed(log)
        results = log.similar(SHELL)
        assert results[0][0] <= results[-1][0]
        assert results[0][1].signature["volume_ratio"] == pytest.approx(SHELL["volume_ratio"])

    def test_max_distance_excludes_the_other_regime(self, log: TrialLog):
        _seed(log)
        near = log.similar(SHELL, max_distance=0.1)
        assert {t.structure for _, t in near} == {"a", "b"}

    def test_limit_is_respected(self, log: TrialLog):
        _seed(log)
        assert len(log.similar(SHELL, limit=1)) == 1

    def test_trials_without_a_signature_are_skipped(self, log: TrialLog):
        log.record(structure="x", specimens=[], family="f", params={},
                   signature={}, scores={}, outcome="explored", reason="")
        assert log.similar(SHELL) == []


class TestFamiliesTried:
    def test_reports_outcomes_and_reasons_for_the_matching_regime(self, log: TrialLog):
        _seed(log)
        summary = log.families_tried(SHELL, max_distance=0.1)
        assert set(summary) == {"contrast_threshold", "local_mean_threshold"}
        assert summary["local_mean_threshold"]["outcomes"] == ["rejected"]
        assert "local mean is the structure itself" in summary["local_mean_threshold"]["reasons"][0]

    def test_orders_by_best_observed_score(self, log: TrialLog):
        _seed(log)
        assert list(log.families_tried(SHELL, max_distance=0.1))[0] == "contrast_threshold"

    def test_the_same_family_is_judged_per_regime(self, log: TrialLog):
        """A family that succeeds on one signature and fails on another must
        report the failure when queried against the failing one."""
        _seed(log)
        assert log.families_tried(SHELL, max_distance=0.1)["contrast_threshold"]["best_dice"] == 0.97
        displaced = log.families_tried(DISPLACED, max_distance=0.1)
        assert displaced["contrast_threshold"]["outcomes"] == ["rejected"]

    def test_duplicate_reasons_are_not_repeated(self, log: TrialLog):
        for _ in range(3):
            log.record(structure="a", specimens=[], family="f", params={},
                       signature=SHELL, scores={"dice": 0.1}, outcome="rejected",
                       reason="same reason each time")
        assert len(log.families_tried(SHELL)["f"]["reasons"]) == 1


class TestSummaryAndIndex:
    def test_summary_counts(self, log: TrialLog):
        _seed(log)
        summary = log.summary()
        assert summary["total"] == 3
        assert summary["by_outcome"] == {"admitted": 1, "rejected": 2}
        assert summary["by_structure"] == {"a": 1, "b": 1, "c": 1}
        assert summary["families"] == ["contrast_threshold", "local_mean_threshold"]

    def test_index_lists_families_and_dead_ends(self, log: TrialLog):
        _seed(log)
        text = render_index(log)
        assert "contrast_threshold" in text
        assert "Recorded dead ends" in text
        assert "local mean is the structure itself" in text

    def test_index_on_an_empty_log(self, log: TrialLog):
        assert "No trials recorded yet" in render_index(log)

    def test_index_reports_no_dead_ends_when_none_rejected(self, log: TrialLog):
        log.record(structure="a", specimens=[], family="f", params={},
                   signature=SHELL, scores={"dice": 0.9}, outcome="admitted", reason="worked")
        assert "_none recorded_" in render_index(log)
