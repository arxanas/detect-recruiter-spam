from base64 import b64decode, b64encode
from collections import defaultdict
from dataclasses import dataclass
from hashlib import md5
from statistics import median
from typing import Dict, List, Set
import json
import zlib


from Crypto.Cipher import AES
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


def decode_payload(payload: str) -> object:
    # From https://www.levels.fyi/js/commonUtils.js
    e = "levelstothemoon!!"
    key = md5(e.encode("utf-8")).digest()
    key = b64encode(key)
    key = key[:16]
    ciphertext = b64decode(payload)
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext_bytes = cipher.decrypt(ciphertext)
    plaintext_bytes = zlib.decompress(plaintext_bytes)
    plaintext = plaintext_bytes.decode("utf-8")
    return json.loads(plaintext)


def get_leveling(company_name: str) -> List[Level]:
    url = "https://api.levels.fyi/v1/levels"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36",
    }
    params = {
        "role": "Software Engineer",
        "companies[]": company_name,
    }
    payload = requests.get(url, headers=headers, params=params).json()["payload"]
    payload = decode_payload(payload)

    all_level_info = {
        company_info["company"]: [
            Level(titles=level["titles"], percent_workforce=level["percent_workforce"])
            for level in company_info["levels"]
        ]
        for company_info in payload["companies"]
    }
    return all_level_info[company_name]


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
