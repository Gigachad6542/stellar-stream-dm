"""Tests for the profile-screen Pareto diagnostic."""
from scripts.summarize_gd1_profile_diagnostics import pareto_front


def test_pareto_front_removes_dominated_rows():
    rows = [
        {"name": "a", "primary_relative_improvement": 0.5, "morphology_relative_improvement": 0.1},
        {"name": "b", "primary_relative_improvement": 0.4, "morphology_relative_improvement": 0.2},
        {"name": "dominated", "primary_relative_improvement": 0.3, "morphology_relative_improvement": 0.1},
    ]
    names = {row["name"] for row in pareto_front(rows)}
    assert names == {"a", "b"}
