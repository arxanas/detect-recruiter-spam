from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Set

import requests

CompanyName = str


@dataclass
class Level:
    titles: List[str]
    percent_workforce: int

    @property
    def combined_title(self) -> str:
        return "/".join(title for title in self.titles)


def get_available_companies() -> Set[CompanyName]:
    url = "https://www.levels.fyi/js/availableCompanies.json"
    return set(requests.get(url).json())


def get_leveling() -> Dict[CompanyName, List[Level]]:
    url = "https://www.levels.fyi/js/data.json"
    payload = requests.get(url).json()["Software Engineer"]
    return {
        company_name: [
            Level(titles=level["titles"], percent_workforce=level["percentWorkforce"])
            for level in levels
        ]
        for company_name, levels in payload.items()
    }


def get_salary_data(company_name: CompanyName, city_name: str) -> Dict[str, int]:
    url = "https://www.levels.fyi/js/salaryData.json"
    payload = requests.get(url).json()
    entries = [
        i
        for i in payload
        if i["company"].lower() == company_name.lower()
        and city_name.lower() in i["location"].lower()
    ]
    entries_by_level = defaultdict(list)
    for entry in entries:
        entries_by_level[entry["level"]].append(
            int(float(entry["totalyearlycompensation"]))
        )
    return {
        level_name: int(median(level_tc))
        for level_name, level_tc in entries_by_level.items()
    }


def get_median_percentile_at_level(levels: List[Level], level_title: str) -> int:
    current_percentile = 0
    for level in levels:
        if level_title in level.titles:
            current_percentile += int(level.percent_workforce / 2)
            return current_percentile
        else:
            current_percentile += level.percent_workforce
    raise ValueError(f"Could not determine median percentile for {level}: {levels}")


def get_level_at_percentile(levels: List[Level], percentile: int) -> Level:
    current_percentile = 0
    for i in levels:
        current_percentile += i.percent_workforce
        if current_percentile >= percentile:
            return i
    raise ValueError(f"Could not determine level for percentile {percentile}: {levels}")
