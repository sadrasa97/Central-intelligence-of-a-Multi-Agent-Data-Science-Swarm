"""
================================================================================
Web Search Module for Dr. Data  —  Robust Multi-Backend Edition
================================================================================
Backends tried in order:
  1. duckduckgo_search._text_html  (direct POST, bypasses bing-only lock)
  2. duckduckgo_search._text_lite
  3. duckduckgo_search._text_bing  (default, may rate-limit)
  4. Raw urllib fallback to lite.duckduckgo.com
================================================================================
"""
import os
import re
import json
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Browser-realistic user agents (rotated per retry) ─────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def _get_ua(index: int = 0) -> str:
    env_ua = os.environ.get("WEB_SEARCH_USER_AGENT")
    if env_ua:
        return env_ua
    return _USER_AGENTS[index % len(_USER_AGENTS)]


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ── DDG multi-backend search ───────────────────────────────────────────────────
def _ddg_search_with_backend(
    ddgs_instance,
    query: str,
    backend_fn,
    max_results: int,
    ua_index: int,
) -> List[Dict[str, str]]:
    """Try one DDG backend method; inject headers before call."""
    try:
        # Inject user-agent into the primp client before each attempt
        ddgs_instance.client.headers_update({
            "User-Agent": _get_ua(ua_index),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        })
        raw = backend_fn(query, max_results=max_results)
        return raw or []
    except Exception as e:
        logger.debug(f"DDG backend {backend_fn.__name__} failed: {e}")
        return []


def ddg_search(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """
    Search using DuckDuckGo with automatic backend fallback and UA rotation.
    Returns [{title, url, snippet}].
    """
    query = (query or "").strip()
    if not query:
        return []

    # ── Try duckduckgo_search library ──────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException

        results: List[Dict[str, str]] = []
        seen: set = set()

        # Backend order: html → lite → bing (bypass hardcoded bing-only lock)
        with DDGS(timeout=15) as ddgs:
            backends_to_try = [
                ("html", ddgs._text_html),
                ("lite", ddgs._text_lite),
                ("bing", ddgs._text_bing),
            ]

            for attempt, (backend_name, fn) in enumerate(backends_to_try):
                if attempt > 0:
                    time.sleep(1.5 + attempt)  # backoff between retries

                raw = _ddg_search_with_backend(ddgs, query, fn, max_results, ua_index=attempt)
                if raw:
                    logger.info(f"DDG search OK via '{backend_name}' backend: {len(raw)} results")
                    for r in raw:
                        url = r.get("href") or r.get("url", "")
                        if url and url not in seen:
                            seen.add(url)
                            results.append({
                                "title":   _clean_text(r.get("title", "")),
                                "url":     url.strip(),
                                "snippet": _clean_text(r.get("body") or r.get("snippet", "")),
                            })
                    break
                logger.debug(f"DDG backend '{backend_name}' returned 0 results, trying next...")

        if results:
            return results

    except ImportError:
        logger.warning("duckduckgo_search not installed: pip install duckduckgo-search")
    except Exception as e:
        logger.warning(f"DDG library search failed: {e}")

    # ── Raw urllib fallback to lite.duckduckgo.com ─────────────────────────────
    logger.info("Falling back to raw urllib DDG lite search")
    return _ddg_lite_raw(query, max_results)


def _ddg_lite_raw(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Bypass the library entirely — POST directly to lite.duckduckgo.com.
    Uses full browser headers to avoid 403/rate-limit blocks.
    """
    import urllib.request, urllib.parse, urllib.error
    from html import unescape

    results: List[Dict[str, str]] = []
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()

    for attempt in range(3):
        ua = _get_ua(attempt)
        headers = {
            "User-Agent":      ua,
            "Content-Type":    "application/x-www-form-urlencoded",
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin":          "https://lite.duckduckgo.com",
            "Referer":         "https://lite.duckduckgo.com/",
            "DNT":             "1",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read(200_000).decode("utf-8", errors="ignore")

            # Parse table rows from DDG lite HTML
            # Title + link in <a class="result-link">
            link_pattern = re.compile(
                r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', re.I
            )
            # Snippet in <td class="result-snippet">
            snip_pattern = re.compile(
                r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.I | re.S
            )

            links   = link_pattern.findall(html)
            snippets = snip_pattern.findall(html)

            for i, (href, title) in enumerate(links[:max_results]):
                # DDG lite URLs are redirects — extract real URL
                real_url = href
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    real_url = urllib.parse.unquote(qs["uddg"][0])

                snippet = unescape(re.sub(r"<[^>]+>", " ", snippets[i])).strip() if i < len(snippets) else ""
                results.append({
                    "title":   _clean_text(unescape(title)),
                    "url":     real_url,
                    "snippet": _clean_text(snippet),
                })

            if results:
                logger.info(f"Raw DDG lite fallback: {len(results)} results (attempt {attempt+1})")
                return results

            logger.debug(f"DDG lite raw attempt {attempt+1}: 0 results parsed")
            time.sleep(2)

        except urllib.error.HTTPError as e:
            logger.warning(f"DDG lite raw HTTP {e.code} on attempt {attempt+1} (UA: {ua[:40]})")
            time.sleep(2 + attempt)
        except Exception as e:
            logger.warning(f"DDG lite raw error attempt {attempt+1}: {e}")
            time.sleep(2)

    logger.error("All DDG search methods exhausted — returning empty results")
    return []


# ── Page fetcher ───────────────────────────────────────────────────────────────
def fetch_url_text(
    url: str,
    max_chars: int = 3000,
    timeout_s: int = 12,
) -> str:
    """Fetch page and extract readable text."""
    if not url:
        return ""

    import urllib.request, urllib.error
    from html import unescape

    ua = _get_ua(0)
    req = urllib.request.Request(url, headers={
        "User-Agent":      ua,
        "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT":             "1",
    }, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            html = resp.read(200_000).decode("utf-8", errors="ignore")
    except Exception:
        return ""

    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>",   " ", text,  flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = _clean_text(text)

    return (text[:max_chars].rstrip() + "...") if len(text) > max_chars else text


# ── High-level API ─────────────────────────────────────────────────────────────
def search_web(
    query: str,
    max_results: int = 5,
    fetch_top: bool = True,
    max_chars_per_page: int = 2000,
    user_agent: Optional[str] = None,   # kept for compat, ignored (rotated internally)
) -> Dict[str, object]:
    """
    Search the web and return structured results for agent consumption.
    """
    t0 = time.time()
    results = ddg_search(query=query, max_results=max_results)

    fetched: List[Dict[str, str]] = []
    if fetch_top and results:
        for r in results[:2]:
            url = r.get("url", "")
            if not url:
                continue
            page_text = fetch_url_text(url, max_chars=max_chars_per_page)
            if page_text:
                fetched.append({
                    "title": r.get("title", ""),
                    "url":   url,
                    "text":  page_text,
                })

    return {
        "query":         query,
        "results":       results,
        "fetched_pages": fetched,
        "timing":        {"seconds": round(time.time() - t0, 3)},
    }


def format_search_for_llm(
    search_obj: Dict[str, object],
    max_results: int = 3,
) -> str:
    """Compact string representation for LLM context injection."""
    results = (search_obj.get("results") or [])[:max_results]
    if not results:
        return "[No web search results found]"

    lines = ["=== WEB SEARCH RESULTS ==="]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Summary: {r.get('snippet', '')}"
        )

    fetched = search_obj.get("fetched_pages") or []
    if fetched:
        lines.append("\n=== PAGE EXCERPTS ===")
        for p in fetched:
            lines.append(
                f"Source: {p.get('title', '')} — {p.get('url', '')}\n"
                f"Excerpt: {p.get('text', '')[:600]}"
            )

    t = search_obj.get("timing", {}).get("seconds", "?")
    lines.append(f"\n[Completed in {t}s | {len(results)} results]")
    return "\n".join(lines).strip()


def format_citations(
    search_obj: Dict[str, object],
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """Structured citation list for UI: [{title, url, snippet}]."""
    return [
        {
            "title":   r.get("title", r.get("url", "")),
            "url":     r.get("url", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in (search_obj.get("results") or [])[:max_results]
        if r.get("url")
    ]