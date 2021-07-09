import argparse
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

from recruiterspam.levelsfyi import (
    CompanyName,
    Level,
    get_available_companies,
    get_level_at_percentile,
    get_leveling,
    get_median_percentile_at_level,
    get_salary_data,
)

from .classify import ClassifyResult, classify
from .train import load_model


@dataclass(eq=True, frozen=True)
class _CurrentCompanyInfo:
    current_company: CompanyName
    current_city: str
    current_level: str

    @classmethod
    def from_environ(cls) -> Optional["_CurrentCompanyInfo"]:
        try:
            current_company = os.environ["CURRENT_COMPANY"]
            current_city = os.environ["CURRENT_CITY"]
            current_level = os.environ["CURRENT_LEVEL"]
            return _CurrentCompanyInfo(
                current_company=current_company,
                current_city=current_city,
                current_level=current_level,
            )
        except KeyError:
            return None


def _get_target_company(
    omit_companies: Set[str],
    text: str,
) -> Optional[CompanyName]:
    all_companies = get_available_companies()
    possible_companies = Counter[CompanyName]()
    for company in all_companies:
        possible_companies[company] = text.count(company)

    for omitted_company in omit_companies:
        try:
            del possible_companies[omitted_company]
        except KeyError:
            pass

    most_common = possible_companies.most_common()
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        (company_name, _count) = most_common[0]
        return company_name
    elif "LinkedIn" in possible_companies:
        omit_companies.add("LinkedIn")
        return _get_target_company(omit_companies, text)
    else:
        return None


def _describe_compensation_rating(compensation_ratio: float) -> str:
    if compensation_ratio >= 1:
        return "A"
    elif compensation_ratio >= 0.95:
        return "B"
    elif compensation_ratio >= 0.9:
        return "C"
    elif compensation_ratio >= 0.85:
        return "D"
    else:
        return "F"


def _get_target_level(
    current_company_info: _CurrentCompanyInfo,
    target_company_name: CompanyName,
) -> Optional[Tuple[Level, float]]:
    current_company_levels = get_leveling()[current_company_info.current_company]
    current_company_salary_data = get_salary_data(
        company_name=current_company_info.current_company,
        city_name=current_company_info.current_city,
    )
    current_company_salary = current_company_salary_data[
        current_company_info.current_level
    ]

    target_company_levels = get_leveling()[target_company_name]
    target_company_salary_data = get_salary_data(
        company_name=target_company_name,
        city_name=current_company_info.current_city,
    )

    current_percentile = get_median_percentile_at_level(
        levels=current_company_levels, level_title=current_company_info.current_level
    )
    target_percentile_level = get_level_at_percentile(
        levels=target_company_levels, percentile=current_percentile
    )
    try:
        compensation_ratio = min(
            100.0,
            next(
                target_company_salary_data[title]
                for title in target_percentile_level.titles
                if title in target_company_salary_data
            )
            / current_company_salary,
        )
    except StopIteration:
        compensation_ratio = 1.0

    for level in target_company_levels:
        try:
            if (
                target_company_salary_data[level.titles[0]]
                >= current_company_salary * 1.20
            ):
                return (level, compensation_ratio)
        except KeyError:
            pass
    return None


def _respond_compensation(text: str) -> Optional[str]:
    current_company_info = _CurrentCompanyInfo.from_environ()
    if current_company_info is None:
        logging.info(
            "No current company set in the environment; not comparing compensation"
        )
        return None

    target_company = _get_target_company(
        omit_companies={current_company_info.current_company}, text=text
    )
    logging.info("Target company detected to be: %s", target_company)
    if target_company is None:
        return None

    target_level_result = _get_target_level(
        current_company_info=current_company_info,
        target_company_name=target_company,
    )
    if target_level_result is None:
        return None
    (target_level, compensation_ratio) = target_level_result
    compensation_rating = _describe_compensation_rating(compensation_ratio)

    return f"""I enjoy competitive compensation at my current company, {current_company_info.current_company.upper()}. I've automatically looked up the compensation details for the company you appear to represent, {target_company.upper()}, and it seems like I would have to receive an offer at the below level to be compensated competitively.

Company name: {target_company.upper()}
Compensation rating: {compensation_rating}
Suggested level for competitive compensation: {target_level.combined_title.upper()}

Can you confirm that you can offer the position at the above level?
"""


def respond(text: str, classify_result: ClassifyResult) -> str:
    paragraphs = ["Hello,"]

    prob = round(classify_result.probability * 100)
    if prob >= 99:
        prob_str = ">99"
    else:
        prob_str = str(prob)
    keywords = ", ".join(keyword.upper() for keyword in classify_result.top_keywords)
    paragraphs.append(
        f"""
Thank you for your email about this position. I'm currently employed, but I am always open to the right opportunity if it comes along.
"""
    )
    paragraphs.append(
        """
To help me evaluate whether this is a good opportunity for me, please send over the following information:
* The location of the position.
* Whether your company supports remote work.
* The team which is hiring.
* How my experience relates to the team in question.
* When you are looking to fill the position.
"""
    )

    try:
        compensation_text = _respond_compensation(text)
        if compensation_text is not None:
            paragraphs.append(compensation_text)
        else:
            paragraphs.append(
                """
I enjoy competitive compensation at my current company. To ensure that we're not wasting each other's time with a job offer I can't economically accept, please also include a salary range for the position.
"""
            )
    except Exception as e:
        logging.exception(e)

    paragraphs.append(
        """
The following are the details of why your message was flagged as a recruiting message. If this email was sent in error, you can ignore it.

Probability: {prob_str}%
Top keywords: {keywords}
""".format(
            prob_str=prob_str, keywords=keywords
        )
    )
    paragraphs.append("Best,\nRecruiter Reply Bot")
    return "\n\n".join(paragraph.strip() for paragraph in paragraphs)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model",
    )
    args: argparse.Namespace = parser.parse_args()
    model_path: Path = args.model

    model = load_model(model_path)

    logging.info("Reading text from stdin...")
    text = sys.stdin.read()
    classify_result = classify(model=model, text=text, num_top_keywords=5)
    sys.stdout.write(respond(text, classify_result))


if __name__ == "__main__":
    main()
