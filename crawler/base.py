"""
공통 HTTP 유틸리티 — 재시도, rate-limit, User-Agent
"""
import time
import random
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get(url: str, retries: int = 3, delay: float = 1.5, **kwargs) -> requests.Response | None:
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0, 0.5))
            resp = SESSION.get(url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"  [경고] {url[:60]} 시도 {attempt+1}/{retries}: {e}")
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def soup(resp: requests.Response, parser: str = "lxml") -> BeautifulSoup:
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, parser)


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
