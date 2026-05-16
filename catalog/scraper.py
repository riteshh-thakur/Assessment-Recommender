"""
SHL Catalog Scraper
-------------------
Scrapes Individual Test Solutions from:
  https://www.shl.com/solutions/products/product-catalog/

Strategy:
  1. Load catalog page with requests (handles SHL's SSR pages)
  2. Parse assessment rows from the table/grid
  3. For each assessment, follow detail URL and scrape rich description
  4. Save structured JSON to data/catalog.json

Run: python -m catalog.scraper
"""

import json
import time
import logging
import re
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# SHL test type codes and their meanings
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a page and return BeautifulSoup object."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_test_types(type_indicators) -> list[str]:
    """Extract test type codes from indicator elements."""
    types = []
    if not type_indicators:
        return types
    for el in type_indicators:
        text = el.get_text(strip=True).upper()
        if text in TEST_TYPE_MAP:
            types.append(text)
        # Sometimes it's in class or data attributes
        for cls in el.get("class", []):
            for key in TEST_TYPE_MAP:
                if key.lower() in cls.lower():
                    if key not in types:
                        types.append(key)
    return types


def scrape_detail_page(url: str) -> dict:
    """Scrape individual assessment detail page for rich content."""
    result = {
        "description": "",
        "duration": "",
        "languages": [],
        "remote_testing": False,
        "adaptive_irt": False,
        "competencies": [],
        "job_levels": [],
    }

    soup = get_page(url)
    if not soup:
        return result

    # Description — try multiple selectors SHL uses
    desc_selectors = [
        ".product-catalogue-training-calendar__row--description",
        ".product-description",
        "[class*='description']",
        ".field--name-body",
        "article p",
        ".content-block p",
        "main p",
    ]
    for sel in desc_selectors:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 50:
            result["description"] = el.get_text(" ", strip=True)[:1500]
            break

    # If still no description, grab first substantial paragraph
    if not result["description"]:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                result["description"] = text[:1500]
                break

    # Duration
    duration_pattern = re.compile(r'(\d+)\s*(?:min|minute)', re.IGNORECASE)
    page_text = soup.get_text()
    dur_match = duration_pattern.search(page_text)
    if dur_match:
        result["duration"] = f"{dur_match.group(1)} minutes"

    # Remote testing flag
    remote_keywords = ["remote", "online", "unsupervised", "unproctored"]
    page_text_lower = page_text.lower()
    result["remote_testing"] = any(k in page_text_lower for k in remote_keywords)

    # Adaptive/IRT flag
    result["adaptive_irt"] = any(
        k in page_text_lower for k in ["adaptive", "irt", "item response"]
    )

    # Languages — look for language lists
    lang_match = re.findall(
        r'\b(English|French|German|Spanish|Dutch|Portuguese|Italian|Chinese|Japanese|Arabic|Hindi)\b',
        page_text
    )
    result["languages"] = list(dict.fromkeys(lang_match))[:10]  # deduplicate, keep order

    # Job levels
    level_keywords = {
        "graduate": ["graduate", "entry", "fresh"],
        "professional": ["professional", "experienced", "mid"],
        "manager": ["manager", "management", "supervisor"],
        "director": ["director", "senior", "executive", "leadership"],
    }
    for level, keywords in level_keywords.items():
        if any(k in page_text_lower for k in keywords):
            result["job_levels"].append(level)

    return result


def scrape_catalog_page(url: str, start: int = 0) -> tuple[list[dict], bool]:
    """
    Scrape one page of the catalog table.
    Returns (assessments_list, has_next_page).
    SHL catalog uses ?start=N&type=1 pagination for Individual Tests.
    """
    params_url = f"{url}?start={start}&type=1"
    soup = get_page(params_url)
    if not soup:
        return [], False

    assessments = []

    # SHL catalog table — rows have assessment data
    # Try multiple table/row selectors
    rows = soup.select("tr.product-catalogue__row")
    if not rows:
        rows = soup.select("[class*='product-catalogue'] tr")
    if not rows:
        rows = soup.select("table tr")

    logger.info(f"Found {len(rows)} rows at start={start}")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        # Name & URL — usually first column has a link
        name_el = row.select_one("a")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)
        href = name_el.get("href", "")
        if not href:
            continue
        detail_url = href if href.startswith("http") else BASE_URL + href

        # Test type indicators — look for spans/divs with single letter or type class
        type_spans = row.select("span[class*='type'], .product-catalogue__type span, td:nth-child(4) span")
        test_types = parse_test_types(type_spans)

        # Remote & Adaptive checkmarks — SHL uses checkmark icons in specific columns
        remote_col = cols[2] if len(cols) > 2 else None
        adaptive_col = cols[3] if len(cols) > 3 else None

        has_checkmark = lambda el: bool(
            el and (el.select_one(".icon-check, [class*='check'], [class*='tick']") or "✓" in el.get_text())
        )

        remote_testing = has_checkmark(remote_col)
        adaptive_irt = has_checkmark(adaptive_col)

        assessments.append({
            "name": name,
            "url": detail_url,
            "test_types": test_types,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "description": "",
            "duration": "",
            "languages": [],
            "job_levels": [],
        })

    # Check for next page
    next_btn = soup.select_one("[class*='next']:not([disabled]), .pagination__next")
    has_next = bool(next_btn) and len(rows) > 0

    return assessments, has_next


def enrich_with_detail_pages(assessments: list[dict]) -> list[dict]:
    """Visit each assessment's detail page to enrich data."""
    enriched = []
    for i, assessment in enumerate(assessments):
        logger.info(f"Enriching {i+1}/{len(assessments)}: {assessment['name']}")
        detail = scrape_detail_page(assessment["url"])

        # Merge: detail page wins for description/duration, but keep table data for types/remote
        merged = {**assessment}
        if detail["description"]:
            merged["description"] = detail["description"]
        if detail["duration"]:
            merged["duration"] = detail["duration"]
        if detail["languages"]:
            merged["languages"] = detail["languages"]
        if detail["job_levels"]:
            merged["job_levels"] = detail["job_levels"]
        # Detail page remote/adaptive may be more reliable
        if not merged["remote_testing"]:
            merged["remote_testing"] = detail["remote_testing"]
        if not merged["adaptive_irt"]:
            merged["adaptive_irt"] = detail["adaptive_irt"]

        enriched.append(merged)
        time.sleep(0.8)  # polite crawling

    return enriched


def build_search_text(assessment: dict) -> str:
    """
    Build a rich text string for embedding.
    This is what gets vectorized — quality here = quality of retrieval.
    """
    type_labels = [TEST_TYPE_MAP.get(t, t) for t in assessment.get("test_types", [])]
    lines = [
        f"Name: {assessment['name']}",
        f"Test Type: {', '.join(type_labels) if type_labels else 'Unknown'}",
        f"Type Codes: {', '.join(assessment.get('test_types', []))}",
        f"Description: {assessment.get('description', '')}",
        f"Duration: {assessment.get('duration', 'Not specified')}",
        f"Remote Testing: {'Yes' if assessment.get('remote_testing') else 'No'}",
        f"Adaptive/IRT: {'Yes' if assessment.get('adaptive_irt') else 'No'}",
        f"Languages: {', '.join(assessment.get('languages', ['English']))}",
        f"Job Levels: {', '.join(assessment.get('job_levels', []))}",
        f"URL: {assessment['url']}",
    ]
    return "\n".join(lines)


def run_scraper():
    """Main scraper entry point."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting SHL catalog scrape...")
    all_assessments = []
    start = 0
    page = 0

    # Paginate through the catalog
    while True:
        logger.info(f"Scraping page {page+1} (start={start})...")
        assessments, has_next = scrape_catalog_page(CATALOG_URL, start=start)
        all_assessments.extend(assessments)

        if not has_next or not assessments:
            break
        start += len(assessments)
        page += 1
        time.sleep(1)

    logger.info(f"Found {len(all_assessments)} assessments in catalog table.")

    # Enrich with detail pages
    logger.info("Enriching with detail page data...")
    all_assessments = enrich_with_detail_pages(all_assessments)

    # Add search text for embedding
    for a in all_assessments:
        a["search_text"] = build_search_text(a)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_assessments)} assessments to {OUTPUT_PATH}")
    return all_assessments


if __name__ == "__main__":
    run_scraper()
