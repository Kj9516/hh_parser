#!/usr/bin/env python3
import argparse
import csv
import logging
import random
import sys
import time
import html as html_lib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


HH_API_URL = "https://api.hh.ru/vacancies"

# Россия в HH API — area=113
RUSSIA_AREA_ID = "113"

USER_AGENT = "hh_parser/1.1 (+https://hh.ru/)"
REQUEST_TIMEOUT = 15

# Retry / backoff
RETRY_TOTAL = 6
RETRY_BACKOFF = 0.7  # urllib3 will apply exponential-ish backoff: backoff_factor * (2**(n-1))
STATUS_FORCELIST = (429, 500, 502, 503, 504)

# Rate limiting / anti-ban
DEFAULT_MIN_INTERVAL = 0.35  # минимальный интервал между любыми запросами (сек)
DEFAULT_DETAIL_THROTTLE = 0.25  # доп. пауза между детальными запросами
DEFAULT_SEARCH_THROTTLE = 0.2  # пауза между поисковыми запросами
MAX_429_COOLDOWN = 60  # верхняя граница сна, если HH вернет огромный Retry-After


@dataclass(frozen=True)
class VacancyRow:
    query: str
    title: str
    company: str
    inn: str
    salary: str
    skills: str
    link: str
    contact_person: str
    contacts: str
    owner: str
    description: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download hh.ru vacancies into a CSV file (Russia only).")

    # Вариант 1: старый режим (позиционные keywords) — оставим как запасной
    p.add_argument(
        "keywords",
        nargs="*",
        help="Keywords to search (space separated) if --queries-file is not provided.",
    )

    # Вариант 2: файл с запросами
    p.add_argument(
        "-f",
        "--queries-file",
        help="Path to a file with multiple search queries. "
             "Each line may contain one or several queries separated by commas.",
    )

    p.add_argument("-o", "--output", default="vacancies.csv", help="CSV output path (default: vacancies.csv)")

    # Фильтры (оставим полезные)
    p.add_argument("--specialization", help="Specialization code (see hh.ru specializations API).")
    p.add_argument("--industry", help="Industry identifier (see hh.ru industries API).")
    p.add_argument("--employment", help="Employment type (full, part, project, volunteer, probation).")
    p.add_argument("--schedule", help="Schedule type (fullDay, shift, flexible, remote, flyInFlyOut).")
    p.add_argument("--experience", help="Experience (noExperience, between1And3, between3And6, moreThan6).")

    p.add_argument("--page-count", type=int, default=2, help="Max pages to fetch per query (default: 2)")
    p.add_argument("--per-page", type=int, default=50, help="Vacancies per page (1..100, default: 50)")

    # Анти-бан / лимиты
    p.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL,
                   help="Minimum seconds between ANY API requests (default: 0.35)")
    p.add_argument("--search-throttle", type=float, default=DEFAULT_SEARCH_THROTTLE,
                   help="Extra sleep after each search-page request (default: 0.2)")
    p.add_argument("--detail-throttle", type=float, default=DEFAULT_DETAIL_THROTTLE,
                   help="Extra sleep after each vacancy-detail request (default: 0.25)")
    p.add_argument("--max-details", type=int, default=0,
                   help="If >0, limit number of vacancy detail fetches per query (safety fuse).")

    # verbose
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output (progress logging).")

    args = p.parse_args()

    if args.per_page < 1 or args.per_page > 100:
        p.error("--per-page must be between 1 and 100")
    if args.page_count < 1:
        p.error("--page-count must be >= 1")

    if not args.queries_file and not args.keywords:
        p.error("Provide keywords or --queries-file")

    return args


def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_queries_from_file(path: str) -> List[str]:
    """
    File format:
    - Each non-empty line may contain one or multiple queries separated by commas.
    Examples:
      финдир, финансовый директор, управленческий учет
      CFO remote
    """
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            for q in parts:
                if q:
                    queries.append(q)

    # dedupe preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def build_session() -> requests.Session:
    session = requests.Session()

    retry_strategy = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=list(STATUS_FORCELIST),
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    return session


class RateLimiter:
    """
    Simple global rate limiter: ensures a minimum interval between any two requests,
    plus optional extra throttles handled outside.
    """
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, min_interval)
        self._last_ts = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        delta = now - self._last_ts
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_ts = time.monotonic()


def safe_get_json(
    session: requests.Session,
    limiter: RateLimiter,
    url: str,
    params: Optional[Dict[str, str]] = None,
    *,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Performs GET with:
    - global rate limiting
    - robust handling for 429 with Retry-After
    - raising for non-2xx after retries
    """
    # We will do a small manual loop to handle 429 (even after urllib3 retries),
    # and to provide more controlled cooldowns.
    attempts = RETRY_TOTAL + 2  # small cushion beyond adapter retries
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        limiter.wait()
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            # exponential-ish sleep with jitter
            sleep_s = min(8.0, (RETRY_BACKOFF * (2 ** (attempt - 1)))) + random.uniform(0, 0.25)
            logger.info(f"Request error: {e}. Sleep {sleep_s:.2f}s (attempt {attempt}/{attempts})")
            time.sleep(sleep_s)
            continue

        # Rate limit handling
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            cooldown = None
            if ra:
                try:
                    cooldown = float(ra)
                except ValueError:
                    cooldown = None
            if cooldown is None:
                cooldown = min(8.0, (RETRY_BACKOFF * (2 ** (attempt - 1)))) + random.uniform(0.5, 1.0)
            cooldown = min(float(cooldown), float(MAX_429_COOLDOWN))
            logger.warning(f"429 Too Many Requests. Cooling down for {cooldown:.1f}s (attempt {attempt}/{attempts})")
            time.sleep(cooldown)
            continue

        # Forbidden / ban-ish
        if resp.status_code in (401, 403):
            # Don't hammer; likely blocked or needs auth.
            msg = f"HTTP {resp.status_code} for {url}. Possibly blocked/rate-limited harder."
            raise RuntimeError(msg)

        # Other error codes
        if resp.status_code < 200 or resp.status_code >= 300:
            # Try to provide body snippet for debugging
            snippet = (resp.text or "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {snippet}")
            sleep_s = min(8.0, (RETRY_BACKOFF * (2 ** (attempt - 1)))) + random.uniform(0, 0.25)
            logger.info(f"Bad status {resp.status_code}. Sleep {sleep_s:.2f}s (attempt {attempt}/{attempts})")
            time.sleep(sleep_s)
            continue

        # Parse JSON
        try:
            return resp.json()
        except ValueError as e:
            last_exc = e
            sleep_s = min(4.0, (RETRY_BACKOFF * (2 ** (attempt - 1)))) + random.uniform(0, 0.25)
            logger.info(f"JSON decode error: {e}. Sleep {sleep_s:.2f}s (attempt {attempt}/{attempts})")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch JSON after {attempts} attempts: {last_exc}")


def build_search_params(args: argparse.Namespace, query: str, page: int) -> Dict[str, str]:
    params: Dict[str, str] = {
        "text": query,
        "area": RUSSIA_AREA_ID,         # (2) Только Россия
        "page": str(page),
        "per_page": str(args.per_page),
    }

    optional = {
        "specialization": args.specialization,
        "industry": args.industry,
        "employment": args.employment,
        "schedule": args.schedule,
        "experience": args.experience,
    }
    for k, v in optional.items():
        if v:
            params[k] = str(v)
    return params


def parse_description(html: str) -> str:
    if not html:
        return "—"

    # 1️⃣ Явно декодируем HTML entities
    html = html_lib.unescape(html)

    # 2️⃣ Используем lxml (или html.parser, но после unescape)
    soup = BeautifulSoup(html, "lxml")

    text = soup.get_text("\n", strip=True)
    return text if text else "—"

def safe_text(val: Any) -> str:
    if not val:
        return "—"
    if isinstance(val, str):
        return val.strip()
    return str(val).strip() or "—"



def format_salary(salary: Optional[Dict[str, Any]]) -> str:
    if not salary:
        return "—"
    currency = str(salary.get("currency", "") or "").upper()
    salary_from = salary.get("from")
    salary_to = salary.get("to")

    parts: List[str] = []
    if salary_from is not None:
        parts.append(f"от {salary_from}")
    if salary_to is not None:
        parts.append(f"до {salary_to}")

    salary_range = " ".join(parts).strip()
    if salary_range and currency:
        return f"{salary_range} {currency}"
    return salary_range or "—"


def format_skills(key_skills: Iterable[Dict[str, Any]]) -> str:
    names: List[str] = []
    for skill in key_skills:
        name = (skill.get("name") or "").strip()
        if name:
            names.append(name)
    return ", ".join(names) if names else "—"


def extract_person_name(contacts: Optional[Dict[str, Any]]) -> str:
    if not contacts:
        return "—"
    if contacts.get("name"):
        name = str(contacts.get("name") or "").strip()
        return name or "—"

    firstname = str(contacts.get("firstname") or "").strip()
    lastname = str(contacts.get("lastname") or "").strip()
    middlename = str(contacts.get("middlename") or "").strip()
    full = " ".join([p for p in [firstname, middlename, lastname] if p])
    return full or "—"


def format_contacts(contacts: Optional[Dict[str, Any]]) -> str:
    if not contacts:
        return "—"

    parts: List[str] = []
    person = extract_person_name(contacts)
    if person != "—":
        parts.append(person)

    # phones
    phones: List[str] = []
    for ph in (contacts.get("phones") or []):
        if not isinstance(ph, dict):
            continue
        country = ph.get("country")
        city = ph.get("city")
        number = ph.get("number")
        if number:
            if country and city:
                phones.append(f"+{country} ({city}) {number}")
            else:
                phones.append(str(number))
    if phones:
        parts.extend(phones)

    email = (contacts.get("email") or "")
    email = str(email).strip()
    if email:
        parts.append(email)

    return "; ".join([p for p in parts if p]) or "—"


def extract_owner(detail_data: Dict[str, Any]) -> str:
    """
    (4) Данные владельца/ответственного.
    В HH vacancy details может быть 'contacts' (публичные контакты),
    и иногда может быть 'manager' / 'department' / etc. — зависит от политики работодателя.
    Мы пытаемся вытащить максимально корректно и безопасно.
    """
    # Primary: public contacts person
    contacts = detail_data.get("contacts")
    name = extract_person_name(contacts if isinstance(contacts, dict) else None)
    if name != "—":
        return name

    # Secondary: sometimes 'manager' exists
    manager = detail_data.get("manager")
    if isinstance(manager, dict):
        mname = (manager.get("name") or "").strip()
        if mname:
            return mname

    # Tertiary: employer name as fallback (not a person, but "owner" column won't be empty)
    employer = detail_data.get("employer")
    if isinstance(employer, dict):
        ename = (employer.get("name") or "").strip()
        if ename:
            return ename

    return "—"


def extract_inn(detail_data: Dict[str, Any]) -> str:
    """
    (4) ИНН: наиболее корректный источник — employer.inn в деталях вакансии.
    В списке items ИНН обычно нет.
    """
    employer = detail_data.get("employer")
    if isinstance(employer, dict):
        inn = employer.get("inn")
        if inn is None:
            return "—"
        s = str(inn).strip()
        return s or "—"
    return "—"


def iter_vacancy_rows(
    args: argparse.Namespace,
    queries: List[str],
) -> Iterator[VacancyRow]:
    logger = logging.getLogger("hh")
    session = build_session()
    limiter = RateLimiter(args.min_interval)

    seen_ids: Set[str] = set()

    total_queries = len(queries)
    for qi, query in enumerate(queries, 1):
        logger.info(f"[{qi}/{total_queries}] Query: {query!r}")

        # 1) first page to know pages count
        first_params = build_search_params(args, query=query, page=0)
        search_data = safe_get_json(session, limiter, HH_API_URL, params=first_params, logger=logger)
        time.sleep(max(0.0, args.search_throttle))

        pages = search_data.get("pages")
        if pages is None:
            pages = 1
        try:
            pages_int = int(pages)
        except (ValueError, TypeError):
            pages_int = 1

        max_pages = min(args.page_count, max(1, pages_int))
        logger.info(f"  pages: {pages_int}, fetching up to: {max_pages}")

        fetched_details = 0

        def yield_from_items(items: Any) -> Iterator[VacancyRow]:
            nonlocal fetched_details

            if not isinstance(items, list):
                return

            for item in items:
                if not isinstance(item, dict):
                    continue

                vid = item.get("id")
                if not vid:
                    continue

                vid_str = str(vid)
                if vid_str in seen_ids:
                    continue
                seen_ids.add(vid_str)

                # safety fuse per query
                if args.max_details and fetched_details >= args.max_details:
                    logger.warning(f"  Reached --max-details={args.max_details} for query {query!r}. Stop details.")
                    return

                detail_url = f"{HH_API_URL}/{vid_str}"
                logger.info(f"    detail {vid_str} ...")
                detail = safe_get_json(session, limiter, detail_url, params=None, logger=logger)
                fetched_details += 1

                contacts = detail.get("contacts") if isinstance(detail.get("contacts"), dict) else None
                employer = detail.get("employer") if isinstance(detail.get("employer"), dict) else {}
                title = str(detail.get("name") or "—").strip() or "—"
                company = str(employer.get("name") or "—").strip() or "—"

                yield VacancyRow(
                    query=query,
                    title=title,
                    company=company,
                    inn=extract_inn(detail),
                    salary=format_salary(detail.get("salary") if isinstance(detail.get("salary"), dict) else None),
                    skills=format_skills(detail.get("key_skills") or []),
                    link=str(detail.get("alternate_url") or "").strip(),
                    contact_person=extract_person_name(contacts),
                    contacts=format_contacts(contacts),
                    owner=extract_owner(detail),
                    description=parse_description(str(detail.get("description") or "")),
                )

                time.sleep(max(0.0, args.detail_throttle))

        # page 0
        for row in yield_from_items(search_data.get("items", [])):
            yield row

        # remaining pages
        for page in range(1, max_pages):
            params = build_search_params(args, query=query, page=page)
            logger.info(f"  page {page + 1}/{max_pages} ...")
            data = safe_get_json(session, limiter, HH_API_URL, params=params, logger=logger)
            time.sleep(max(0.0, args.search_throttle))

            for row in yield_from_items(data.get("items", [])):
                yield row

        logger.info(f"  done query {query!r} (unique vacancies so far: {len(seen_ids)})")



def write_csv_stream(path: str, rows: Iterable[VacancyRow], verbose: bool) -> int:
    logger = logging.getLogger("hh")
    fieldnames = [
        "query",
        "title",
        "company",
        "inn",
        "salary",
        "skills",
        "link",
        "contact_person",
        "contacts",
        "owner",
        "description",
    ]

    count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "query": row.query,
                    "title": row.title,
                    "company": row.company,
                    "inn": row.inn,
                    "salary": row.salary,
                    "skills": row.skills,
                    "link": row.link,
                    "contact_person": row.contact_person,
                    "contacts": row.contacts,
                    "owner": row.owner,
                    "description": row.description,
                }
            )
            count += 1
            if verbose and (count % 10 == 0):
                logger.info(f"Saved rows: {count}")
    return count


def collect_queries(args: argparse.Namespace) -> List[str]:
    if args.queries_file:
        queries = read_queries_from_file(args.queries_file)
        if not queries:
            raise ValueError(f"No queries found in file: {args.queries_file}")
        return queries

    # fallback: positional keywords become one query
    q = " ".join([k.strip() for k in (args.keywords or []) if k.strip()])
    if not q:
        raise ValueError("Empty keywords")
    return [q]


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("hh")

    try:
        queries = collect_queries(args)
    except Exception as e:
        logger.error(str(e))
        return 2

    if args.verbose:
        logger.info(f"Russia-only mode enabled: area={RUSSIA_AREA_ID}")
        logger.info(f"Queries: {len(queries)}")
        logger.info(f"Output: {args.output}")
        logger.info(
            f"Rate limits: min_interval={args.min_interval}, "
            f"search_throttle={args.search_throttle}, detail_throttle={args.detail_throttle}, "
            f"max_details={args.max_details or '∞'}"
        )

    try:
        rows_iter = iter_vacancy_rows(args, queries)
        total = write_csv_stream(args.output, rows_iter, args.verbose)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Partial results may be saved.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    print(f"Saved {total} vacancies to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
