#!/usr/bin/env python3
"""Verify that every entry in a .bib file refers to a real paper.

arXiv entries are checked against the arXiv API (ID exists, title and authors match).
Other entries are looked up on Crossref by DOI or title. Any url field must resolve.

Usage: python check_citations.py [ref.bib]
"""
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "thesis-citation-checker/1.0 (mailto:erykhalicki0@gmail.com)"}
ATOM = "{http://www.w3.org/2005/Atom}"
TITLE_THRESHOLD = 0.85


def fetch(url, method="GET", retries=4):
    req = urllib.request.Request(url, headers=UA, method=method)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as err:
            if err.code not in (429, 503) or attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i - 1]
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            j = f.end()
            if body[j] == "{":
                d, k = 1, j + 1
                while d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                fields[f.group(1).lower()] = body[j + 1:k - 1]
            else:
                v = re.match(r'"?([^",]*)', body[j:])
                fields[f.group(1).lower()] = v.group(1) if v else ""
        entries.append({"type": m.group(1).lower(), "key": m.group(2), **fields})
    return [e for e in entries if e["type"] not in ("comment", "string", "preamble")]


def delatex(s):
    s = re.sub(r"\\[`'^\"~=.uvHcdbtr]\s*\{?(\w)\}?", r"\1", s)
    s = re.sub(r"\$[^$]*\$", "", s)
    s = re.sub(r"\\\w+", "", s)
    s = s.replace("{", "").replace("}", "").replace("``", "").replace("''", "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_title(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", delatex(s).lower())).strip()


def search_words(s):
    return [w for w in re.sub(r"[^a-z0-9]", " ", delatex(s).lower()).split() if len(w) > 2]


def title_sim(a, b):
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def surname(name):
    name = delatex(name).strip()
    if "," in name:
        return name.split(",")[0].strip().lower()
    return name.split()[-1].lower() if name.split() else ""


def bib_surnames(author_field):
    names = re.split(r"\s+and\s+", re.sub(r"\s+", " ", author_field))
    return [surname(n) for n in names if n.strip().lower() not in ("others", "et al.", "")]


def arxiv_id(e):
    for field in ("eprint", "url", "journal", "note", "doi"):
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", e.get(field, ""))
        if m and (field == "eprint" or "arxiv" in e.get(field, "").lower()):
            return m.group(1)
    return None


def query_arxiv(ids):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)})
    _, body = fetch(url)
    out = {}
    for entry in ET.fromstring(body).findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id", "")
        m = re.search(r"(\d{4}\.\d{4,5})", eid)
        title = entry.findtext(f"{ATOM}title")
        if not m or title in (None, "Error"):
            continue
        out[m.group(1)] = {
            "title": " ".join(title.split()),
            "authors": [a.findtext(f"{ATOM}name") for a in entry.findall(f"{ATOM}author")],
            "year": entry.findtext(f"{ATOM}published", "")[:4],
        }
    return out


def arxiv_search(query):
    time.sleep(3)
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": 5})
    try:
        root = ET.fromstring(fetch(url)[1])
    except Exception:
        return []
    results = []
    for entry in root.findall(f"{ATOM}entry"):
        m = re.search(r"(\d{4}\.\d{4,5})", entry.findtext(f"{ATOM}id", ""))
        if m:
            results.append((m.group(1), " ".join((entry.findtext(f"{ATOM}title") or "").split())))
    return results


def semantic_scholar_arxiv_id(title):
    url = "https://api.semanticscholar.org/graph/v1/paper/search/match?" + urllib.parse.urlencode(
        {"query": title, "fields": "title,externalIds"})
    try:
        paper = json.loads(fetch(url)[1])["data"][0]
    except Exception:
        return None
    aid = (paper.get("externalIds") or {}).get("ArXiv")
    return aid if aid and title_sim(paper["title"], title) >= TITLE_THRESHOLD else None


def find_arxiv_id(e):
    """Look up an arXiv ID for an entry that doesn't list one, trying several searches in turn."""
    title = delatex(e.get("title", "")).strip()
    if not title:
        return None
    words = search_words(title)
    surnames = bib_surnames(e.get("author", ""))
    phrase = " ".join(re.sub(r"[^a-z0-9]", " ", title.lower()).split())
    queries = [f'ti:"{phrase}"']
    if surnames:
        queries.append(f'ti:"{phrase}" AND au:{surnames[0]}')
        queries.append(" AND ".join([f"ti:{w}" for w in words[:6]] + [f"au:{surnames[0]}"]))
    queries.append(" AND ".join(f"ti:{w}" for w in words))
    for q in queries:
        for aid, found in arxiv_search(q):
            if title_sim(found, title) >= TITLE_THRESHOLD:
                return aid
    return semantic_scholar_arxiv_id(title) or crossref_arxiv_id(title)


def crossref_arxiv_id(title):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": title, "filter": "prefix:10.48550", "rows": 5})
    try:
        items = json.loads(fetch(url)[1])["message"]["items"]
    except Exception:
        return None
    for item in items:
        if item.get("title") and title_sim(item["title"][0], title) >= TITLE_THRESHOLD:
            m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", item.get("DOI", "").lower())
            if m:
                return m.group(1)
    return None


def query_crossref(e):
    if e.get("doi"):
        url = "https://api.crossref.org/works/" + urllib.parse.quote(e["doi"])
        try:
            item = json.loads(fetch(url)[1])["message"]
        except Exception:
            return None
    else:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.bibliographic": delatex(e.get("title", "")), "rows": 5})
        items = json.loads(fetch(url)[1])["message"]["items"]
        items = [i for i in items if i.get("title")]
        if not items:
            return None
        surnames = bib_surnames(e.get("author", ""))

        def score(i):
            authors = [f"{a.get('given', '')} {a.get('family', '')}" for a in i.get("author", [])]
            overlap = sum(author_on_paper(s, authors) for s in surnames) / max(len(surnames), 1)
            return title_sim(i["title"][0], e.get("title", "")) + overlap
        item = max(items, key=score)
    return {
        "title": (item.get("title") or [""])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])],
        "year": str((item.get("issued", {}).get("date-parts") or [[""]])[0][0]),
    }


def author_on_paper(bib_surname, real_authors):
    target = re.sub(r"[^a-z ]", " ", bib_surname.replace("-", " ")).split()
    for a in real_authors:
        tokens = re.sub(r"[^a-z ]", " ", delatex(a).lower().replace("-", " ")).split()
        if tokens[-len(target):] == target or " ".join(target) in " ".join(tokens):
            return True
    return False


def compare(e, real):
    problems, warnings = [], []
    sim = title_sim(e.get("title", ""), real["title"])
    if sim < TITLE_THRESHOLD:
        problems.append(f"title mismatch ({sim:.2f}): real title is '{real['title']}'")
    fake = [s for s in bib_surnames(e.get("author", "")) if not author_on_paper(s, real["authors"])]
    if fake:
        problems.append(f"authors not on paper: {', '.join(fake)} (real: {', '.join(real['authors'])})")
    if e.get("year") and real["year"] and e["year"] != real["year"]:
        warnings.append(f"year {e['year']} vs source {real['year']}")
    return problems, warnings


def check_url(url):
    """Return True if the url resolves, False if it is dead, None if the site blocks bots."""
    for method in ("HEAD", "GET"):
        try:
            return fetch(url, method=method)[0] < 400
        except urllib.error.HTTPError as err:
            if err.code == 404 or err.code == 410:
                return False
            if method == "GET":
                return None if err.code in (401, 403, 429) else False
        except Exception:
            if method == "GET":
                return False
    return False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ref.bib"
    entries = parse_bib(open(path, encoding="utf-8").read())
    ids = {e["key"]: arxiv_id(e) for e in entries}
    found = {}
    for e in entries:
        if not ids[e["key"]]:
            aid = find_arxiv_id(e)
            if aid:
                ids[e["key"]] = found[e["key"]] = aid
    arxiv = query_arxiv(sorted({i for i in ids.values() if i})) if any(ids.values()) else {}

    n_bad = 0
    for e in entries:
        problems, warnings = [], []
        aid = ids[e["key"]]
        if aid:
            real = arxiv.get(aid)
            if not real:
                problems.append(f"arXiv:{aid} does not exist")
        else:
            time.sleep(1)
            real = query_crossref(e)
            if not real:
                problems.append("no match found on Crossref")
        if real:
            p, w = compare(e, real)
            problems += p
            warnings += w
        url_ok = check_url(e["url"]) if e.get("url") else True
        if url_ok is False:
            problems.append(f"url does not resolve: {e['url']}")
        elif url_ok is None:
            warnings.append(f"url blocked automated access, check manually: {e['url']}")
        if e["key"] in found:
            warnings.append(f"no arXiv ID in bib; found by search, consider adding eprint={{{aid}}}")

        source = f"arXiv:{aid}" if aid else "Crossref"
        if e["key"] in found:
            source += ", found by search"
        status = "FAIL" if problems else ("WARN" if warnings else "OK")
        print(f"[{status:4}] {e['key']} ({source})")
        for msg in problems + warnings:
            print(f"         - {msg}")
        n_bad += bool(problems)

    print(f"\n{len(entries) - n_bad}/{len(entries)} entries verified")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
