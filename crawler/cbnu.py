"""
CBNU 소프트웨어학부 공지사항 크롤러
목록: https://software.cbnu.ac.kr/index.php?mid=sub0401&page={page}
상세: https://software.cbnu.ac.kr/index.php?mid=sub0401&document_srl={srl}

※ 로그인 불필요한 공개 공지사항만 수집
"""
import re
import sys
from urllib.parse import urljoin

sys.path.insert(0, __file__.rsplit("\\", 2)[0])
from crawler.base import get, soup, clean_text

BASE     = "https://software.cbnu.ac.kr"
LIST_URL = BASE + "/index.php?mid=sub0401&page={page}"
SITE     = "cbnu"

# 충북대 전체 학사공지도 포함하려면 아래 URL 추가 가능
EXTRA_BOARDS = {
    "sw_notice":   BASE + "/index.php?mid=sub0401&page={page}",       # SW학부 공지
    "scholarship": BASE + "/index.php?mid=sub0403&page={page}",       # 장학
    "employment":  BASE + "/index.php?mid=sub0402&page={page}",       # 취업/진로
}


# ── 목록 페이지 파싱 ──────────────────────────────────────────────────────────

def parse_list_page(page: int, board_url: str = None) -> list[dict]:
    url = (board_url or LIST_URL).format(page=page)
    resp = get(url)
    if not resp:
        return []

    bs = soup(resp)
    items = []

    # XE/Rhymix CMS 공통 구조
    for row in bs.select("table.bd_lst tbody tr, .board_list tbody tr"):
        a = row.select_one("td.title a, td.subject a, .tit a")
        if not a:
            continue

        href = a.get("href", "")
        if "document_srl=" not in href:
            continue

        full_url = urljoin(BASE, href)
        title = a.get_text(strip=True)

        # 날짜: td.time (XE CMS) → td.date → td.regdate
        date_el = row.select_one("td.time, td.date, td.regdate")
        date_txt = date_el.get_text(strip=True) if date_el else ""

        items.append({
            "source_url": full_url,
            "title_raw": title,
            "list_date_raw": date_txt,
        })

    return items


# ── 상세 페이지 파싱 ──────────────────────────────────────────────────────────

def parse_detail_page(url: str) -> dict | None:
    resp = get(url)
    if not resp:
        return None

    bs = soup(resp)
    result = {"source_url": url, "site": SITE}

    # 제목: XE/Rhymix 게시판 구조 — h1.np_18px 가 본문 제목
    title_raw = ""
    for sel in ("h1.np_18px", ".bd_vw_tit", ".document_title", "h3.title", ".view_title"):
        el = bs.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            if len(t) > 5:
                title_raw = t
                break
    result["title_raw"] = title_raw

    # 날짜 메타
    meta = {}
    date_el = bs.select_one("td.date, td.regdate, .date, .reg_date")
    if date_el:
        meta["작성일"] = date_el.get_text(strip=True)
    for row in bs.select(".bd_vw_info tr, .document_info tr"):
        th = row.select_one("th")
        td = row.select_one("td")
        if th and td:
            meta[th.get_text(strip=True)] = td.get_text(separator=" ", strip=True)
    result["meta"] = meta

    # 본문
    for tag in bs.select("script, style, nav, header, footer, .files_area, .comment_area"):
        tag.decompose()

    body_el = (
        bs.select_one(".bd_vw_con")
        or bs.select_one(".document_content")
        or bs.select_one(".rhymix_content")
        or bs.select_one(".xe_content")
        or bs.select_one(".content")
    )
    result["raw_text"] = clean_text(body_el.get_text(separator="\n")) if body_el else ""

    # 장소 힌트: 본문에서 패턴 추출
    loc_match = re.search(r"장소\s*[:：]\s*([^\n]{2,60})", result["raw_text"])
    if loc_match:
        result["location_hint"] = loc_match.group(1).strip()

    # 기간 힌트
    period_match = re.search(
        r"(?:접수|신청|행사|공모)\s*기간\s*[:：]\s*([^\n]{2,80})", result["raw_text"]
    )
    if period_match:
        result["period_hint"] = period_match.group(1).strip()

    return result


# ── 페이지 수 탐지 ────────────────────────────────────────────────────────────

def get_total_pages(board_url: str = None) -> int:
    url = (board_url or LIST_URL).format(page=1)
    resp = get(url)
    if not resp:
        return 1
    bs = soup(resp)
    last = 1
    for a in bs.select(".pagination a, .paging a, .page_nav a"):
        txt = a.get_text(strip=True)
        if txt.isdigit():
            last = max(last, int(txt))
    return last


# ── 전체 크롤 ─────────────────────────────────────────────────────────────────

def crawl(max_pages: int = 50, existing_urls: set = None,
          boards: dict = None) -> list[dict]:
    existing_urls = existing_urls or set()
    boards = boards or {"sw_notice": LIST_URL}
    results = []

    for board_name, board_url_tpl in boards.items():
        total = min(get_total_pages(board_url_tpl), max_pages)
        print(f"[CBNU/{board_name}] 총 {total} 페이지 크롤 시작")

        for page in range(1, total + 1):
            print(f"  목록 {page}/{total}")
            items = parse_list_page(page, board_url_tpl)
            if not items:
                print(f"  [종료] 페이지 {page} 항목 없음")
                break

            new_count = 0
            for item in items:
                url = item["source_url"]
                if url in existing_urls:
                    continue
                detail = parse_detail_page(url)
                if detail:
                    detail.update({
                        "title_raw": detail.get("title_raw") or item["title_raw"],
                        "list_date_raw": item.get("list_date_raw", ""),
                        "board": board_name,
                    })
                    results.append(detail)
                    existing_urls.add(url)
                    new_count += 1
                    print(f"    ✓ {detail['title_raw'][:50]}")

            if new_count == 0:
                print(f"  [종료] 기존 데이터와 전부 중복")
                break

    print(f"[CBNU] 수집 완료: {len(results)}건")
    return results


if __name__ == "__main__":
    import json, os
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crawled_cbnu.jsonl")
    items = crawl(max_pages=5)
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장: {out} ({len(items)}건)")
