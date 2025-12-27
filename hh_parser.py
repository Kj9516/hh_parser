import argparse
import csv
import time
from typing import Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from bs4 import BeautifulSoup


HH_API_URL = "https://api.hh.ru/vacancies"
USER_AGENT = "hh_parser/1.0"
REQUEST_TIMEOUT = 10
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download vacancies from hh.ru into a CSV file."
    )
    parser.add_argument(
        "keywords",
        nargs="+",
        help="Keywords to search for (space separated, combined into one search query).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="vacancies.csv",
        help="Path to the CSV file to write (default: vacancies.csv)",
    )
    parser.add_argument(
        "-a",
        "--area",
        help="Area code to limit search (see hh.ru areas API).",
    )
    parser.add_argument(
        "--specialization",
        help="Specialization code (see hh.ru specializations API).",
    )
    parser.add_argument(
        "--industry",
        help="Industry identifier (see hh.ru industries API).",
    )
    parser.add_argument(
        "--employment",
        help="Employment type (e.g. full, part, project, volunteer, probation).",
    )
    parser.add_argument(
        "--schedule",
        help="Schedule type (e.g. fullDay, shift, flexible, remote, flyInFlyOut).",
    )
    parser.add_argument(
        "--experience",
        help="Experience level (noExperience, between1And3, between3And6, moreThan6).",
    )
    parser.add_argument(
        "--page-count",
        type=int,
        default=1,
        help="How many pages of results to fetch (each page contains per-page vacancies).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=20,
        help="How many vacancies to fetch per page (max 100).",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.2,
        help="Seconds to sleep between vacancy detail requests to be polite to the API.",
    )
    return parser.parse_args()


def build_search_params(args: argparse.Namespace) -> Dict[str, str]:
    params: Dict[str, str] = {
        "text": " ".join(args.keywords),
        "per_page": str(args.per_page),
    }
    optional_filters = {
        "area": args.area,
        "specialization": args.specialization,
        "industry": args.industry,
        "employment": args.employment,
        "schedule": args.schedule,
        "experience": args.experience,
    }
    for key, value in optional_filters.items():
        if value:
            params[key] = value
    return params


def build_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def request_json(
    session: requests.Session, url: str, params: Optional[Dict[str, str]] = None
) -> Dict:
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def format_salary(salary: Optional[Dict]) -> str:
    if not salary:
        return "—"
    currency = salary.get("currency", "").upper()
    salary_from = salary.get("from")
    salary_to = salary.get("to")
    parts: List[str] = []
    if salary_from:
        parts.append(f"от {salary_from}")
    if salary_to:
        parts.append(f"до {salary_to}")
    salary_range = " ".join(parts)
    if salary_range and currency:
        salary_range = f"{salary_range} {currency}"
    return salary_range or "—"


def format_skills(key_skills: Iterable[Dict]) -> str:
    return ", ".join(skill.get("name", "").strip() for skill in key_skills if skill.get("name")) or "—"


def format_contacts(contacts: Optional[Dict]) -> str:
    if not contacts:
        return "—"

    names: List[str] = []
    if contacts.get("name"):
        names.append(contacts["name"].strip())
    else:
        firstname = contacts.get("firstname", "").strip()
        lastname = contacts.get("lastname", "").strip()
        middlename = contacts.get("middlename", "").strip()
        full_name = " ".join(part for part in [firstname, middlename, lastname] if part)
        if full_name:
            names.append(full_name)

    phones = []
    for phone in contacts.get("phones", []) or []:
        country = phone.get("country")
        city = phone.get("city")
        number = phone.get("number")
        if country and city and number:
            phones.append(f"+{country} ({city}) {number}")
        elif number:
            phones.append(number)

    email = contacts.get("email")
    contact_parts = names + phones + ([email] if email else [])
    return "; ".join(part for part in contact_parts if part) or "—"


def format_contact_person(contacts: Optional[Dict]) -> str:
    if not contacts:
        return "—"

    if contacts.get("name"):
        name = contacts["name"].strip()
        return name or "—"

    firstname = contacts.get("firstname", "").strip()
    lastname = contacts.get("lastname", "").strip()
    middlename = contacts.get("middlename", "").strip()
    full_name = " ".join(part for part in [firstname, middlename, lastname] if part)
    return full_name or "—"


def parse_description(html: str) -> str:
    if not html:
        return "—"
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n", strip=True)


def fetch_vacancies(args: argparse.Namespace) -> List[Dict[str, str]]:
    base_params = build_search_params(args)
    vacancies: List[Dict[str, str]] = []
    session = build_session()

    for page in range(args.page_count):
        params = dict(base_params, page=str(page))
        try:
            search_data = request_json(session, HH_API_URL, params=params)
        except (requests.RequestException, ValueError) as exc:
            print(f"Ошибка запроса страницы {page}: {exc}")
            break

        total_pages = search_data.get("pages")
        if total_pages is not None and page >= total_pages:
            break
        for item in search_data.get("items", []):
            vacancy_id = item.get("id")
            if not vacancy_id:
                continue
            try:
                detail_data = request_json(session, f"{HH_API_URL}/{vacancy_id}")
            except (requests.RequestException, ValueError) as exc:
                print(f"Ошибка загрузки вакансии {vacancy_id}: {exc}")
                continue
            contacts = detail_data.get("contacts") or {}
            employer = detail_data.get("employer") or {}
            vacancies.append(
                {
                    "title": detail_data.get("name", "—"),
                    "company": employer.get("name", "—"),
                    "inn": employer.get("inn", "—"),
                    "salary": format_salary(detail_data.get("salary")),
                    "skills": format_skills(detail_data.get("key_skills", [])),
                    "link": detail_data.get("alternate_url", ""),
                    "contact_person": format_contact_person(contacts),
                    "contacts": format_contacts(contacts),
                    "description": parse_description(detail_data.get("description", "")),
                }
            )
            time.sleep(args.throttle)
    return vacancies


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "title",
        "company",
        "inn",
        "salary",
        "skills",
        "link",
        "contact_person",
        "contacts",
        "description",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    vacancies = fetch_vacancies(args)
    write_csv(args.output, vacancies)
    print(f"Saved {len(vacancies)} vacancies to {args.output}")


if __name__ == "__main__":
    main()
