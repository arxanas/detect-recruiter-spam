from typing import List
from recruiterspam.levelsfyi import (
    Level,
    get_available_companies,
    get_level_at_percentile,
    get_leveling,
    get_median_percentile_at_level,
    get_salary_data,
)

import pytest


def test_available_companies() -> None:
    companies = get_available_companies()
    assert "Facebook" in companies
    assert "Google" in companies
    assert "Microsoft" in companies


def test_leveling() -> None:
    leveling = get_leveling("Google")
    assert leveling == [
        Level(titles=["L3", "SWE II"], percent_workforce=20),
        Level(titles=["L4", "SWE III"], percent_workforce=20),
        Level(titles=["L5", "Senior SWE"], percent_workforce=20),
        Level(titles=["L6", "Staff SWE"], percent_workforce=15),
        Level(titles=["L7", "Senior Staff SWE"], percent_workforce=10),
        Level(titles=["L8", "Principal Engineer"], percent_workforce=7),
        Level(titles=["L9", "Distinguished Engineer"], percent_workforce=3),
        Level(titles=["L10", "Google Fellow"], percent_workforce=3),
    ]


def test_level_tc() -> None:
    tc = get_salary_data(company_name="Google", city_name="Seattle")
    print(tc.keys())
    assert 150 <= tc["L3"] <= 200
    assert 200 <= tc["L4"] <= 300
    assert 300 <= tc["L5"] <= 400


@pytest.mark.parametrize(
    ("level_idx", "expected_percentile"),
    [
        (0, 10),
        (1, 30),
        (2, 50),
        (3, 70),
        (4, 90),
    ],
)
def test_get_median_percentile_at_level(level_idx, expected_percentile) -> None:
    levels = [
        Level(titles=["L1"], percent_workforce=20),
        Level(titles=["L2"], percent_workforce=20),
        Level(titles=["L3"], percent_workforce=20),
        Level(titles=["L4"], percent_workforce=20),
        Level(titles=["L5"], percent_workforce=20),
    ]
    assert (
        get_median_percentile_at_level(
            levels=levels,
            level_title=levels[level_idx].titles[0],
        )
        == expected_percentile
    )


@pytest.mark.parametrize(
    ("percentile", "expected_level"),
    [
        (0, 0),
        (10, 0),
        (19, 0),
        (20, 0),
        (21, 1),
        (100, 4),
    ],
)
def test_get_level_at_percentile(percentile: int, expected_level: int) -> None:
    levels = [
        Level(titles=["L1"], percent_workforce=20),
        Level(titles=["L2"], percent_workforce=20),
        Level(titles=["L3"], percent_workforce=20),
        Level(titles=["L4"], percent_workforce=20),
        Level(titles=["L5"], percent_workforce=20),
    ]
    assert (
        get_level_at_percentile(levels=levels, percentile=percentile)
        == levels[expected_level]
    )
