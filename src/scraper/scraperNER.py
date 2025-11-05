#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LPD Website → CSV/TTL Graph Scraper
Asynchrone Version
"""

import re
import os
import csv
import time
import json
import argparse
import hashlib
import collections
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse, urldefrag
from aiohttp import ClientSession

from bs4 import BeautifulSoup

from ttlExporter import write_ttl


# ------------------ Konfiguration ------------------

UA = "LPD-GraphBot/2.2 (research)"
TIMEOUT = 20
SLEEP = 0.6  # Wird nur noch für nicht-async Teile gebraucht, wie bei Spacy

# Limits / Schwellen
MAX_INTERNAL_LINKS = 20        # pro Seite in 'related_to' übernehmen
MAX_HEADING_MENTIONS = 20      # pro Seite H1..H6 -> mentions
SIM_THRESHOLD = 0.22           # Jaccard-Schwelle für Inhaltsähnlichkeit
MAX_SIM_EDGES_PER_NODE = 8     # pro Seite max. X Similarity-Kanten
HYDRATE_BUDGET_DEFAULT = 60    # zusätzliche Ziel-Seiten für Link-Preview laden

# ---- Domain Dictionary / Seeds (erweiterbar)
ABBREV_APPS = {
    "ePA": "Anwendung;Abkürzung",
    "eHBA": "Anwendung;Abkürzung",
    "KIM": "Anwendung;Abkürzung",
    "TIM": "Anwendung;Abkürzung",
    "DEMIS": "Anwendung;Abkürzung",
    "VSDM": "Anwendung;Abkürzung",
    "NFDM": "Anwendung;Abkürzung",
    "E-Rezept": "Anwendung",
}
ORG_HINTS = [
    "DAI-Labor", "Technische Universität Berlin", "ZE-KI", "ZEKI", "Albatros gGmbH",
    "Senatsverwaltung für Wissenschaft, Gesundheit und Pflege",
    "Der Paritätische Wohlfahrtsverband Berlin", "Gematik",
]
DOC_HINT_WORDS = ["pdf", "Broschüre", "Flyer", "Handout", "Leitfaden", "Checkliste", "Factsheet", "Bericht"]
EVENT_HINT_WORDS = ["Führung Musterausstellung", "Führung Musterwohnung", "Digitaltag", "Veranstalter", "Veranstaltungsort", "Tickets"]

# sehr kleine Stopword-Liste (de/en gemischt) für Tokenisierung
STOP = set("""
der die das ein eine einer eines einem einen und oder aber denn weil als wie bei von mit auf für im in an am zu zur zum des den dem aus durch ohne gegen unter über ist sind war waren wird werden sein
the a an of and or to for in on at by from with without into over under about as be is are was were will would can could should may might not
""".split())

# --- Junk/Navigation-Texte, die nicht als Knoten angelegt werden sollen
UI_JUNK = {
    "skip to content", "zum inhalt springen", "startseite", "home", "zur startseite",
    "mehr lesen", "weiterlesen", "kontakt", "impressum", "datenschutz",
    "cookie einstellungen", "https://www.lebenpflegedigital.de"
}


# ------------------ Utilities ------------------

def clean_url(u: str) -> str:
    return urldefrag(u)[0].split("#")[0]


def same_domain(u: str, domain: str) -> bool:
    try:
        return urlparse(u).netloc.endswith(domain)
    except Exception:
        return False


def normalize_whitespace(s: str) -> str:
    return " ".join((s or "").split())


def clamp_text(text: str, max_chars: int = 900) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


async def get_soup(session, url: str) -> BeautifulSoup:
    try:
        async with session.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT) as r:
            r.raise_for_status()
            text = await r.text()
            return BeautifulSoup(text, "lxml")
    except Exception as e:
        # print(f"Fehler beim Abrufen von {url}: {e}") # Zu viele Ausgaben
        return None


def extract_main_text(soup: BeautifulSoup) -> str:
    if not soup:
        return ""
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "aside"]):
        tag.decompose()
    zones = soup.select("article, main, .entry-content, .post-content, .content, section, #content")
    cand = zones if zones else [soup.body or soup]
    def txt(el): return normalize_whitespace(el.get_text(" ", strip=True))
    t = max((txt(z) for z in cand), key=lambda s: len(s), default="")
    t = re.sub(r"Weiterlesen\s*»?", "", t, flags=re.I)
    return t.strip()


def sent_split(text: str):
    return [s.strip() for s in re.split(r"(?<=[\.\?!])\s+(?=[A-ZÄÖÜ])", text) if s.strip()]


def hash_id(title: str, labels: str) -> str:
    key = (title or "").strip().lower() + "|" + (labels or "").strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def polish_def(d: str) -> str:
    d = d.strip().strip("–-:;,. ")
    if len(d) < 12:
        return ""
    d = re.sub(r"\s{2,}", " ", d)
    if len(d) > 280:
        d = d[:280].rsplit(" ", 1)[0] + " …"
    return d


def meta_description(soup: BeautifulSoup) -> str:
    if not soup:
        return ""
    m = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    return (m.get("content") or "").strip() if m else ""


def first_paragraphs(soup: BeautifulSoup, n: int = 2) -> str:
    if not soup:
        return ""
    ps = [p.get_text(" ", strip=True) for p in soup.select("article p, main p, .entry-content p, .post-content p, .content p, section p")]
    return " ".join(ps[:n]).strip()


def tokenize(text: str):
    text = re.sub(r"[^A-Za-zÄÖÜäöüß0-9]+", " ", (text or "").lower())
    toks = [t for t in text.split() if len(t) > 2 and t not in STOP]
    return set(toks)


# ------------------ Kurzbeschreibung / Snippets ------------------

DEF_PATTERNS = [
    r"(?P<term>\b{t}\b)\s+(?:ist|sind)\s+(?P<def>[^\.!\?;]{10,200})",
    r"(?P<term>\b{t}\b)\s+steht\s+f(?:ü|u)r\s+(?P<def>[^\.!\?;]{5,200})",
    r"(?P<term>\b{t}\b)\s+bedeutet\s+(?P<def>[^\.!\?;]{5,200})",
    r"Als\s+(?P<def>[^\.!\?;]{5,200})\s+bezeichnet\s+man\s+(?P<term>\b{t}\b)",
    r"(?P<term>\b{t}\b)\s*\([^){{}}]{2,80}\)\s*(?:ist|sind|bezeichnet)\s+(?P<def>[^\.!\?;]{5,200})",
]


def pick_short_def(term: str, text: str):
    if not term:
        return "", 0.0
    for pat in DEF_PATTERNS:
        filled = pat.replace("{t}", re.escape(term))
        m = re.compile(filled, flags=re.I).search(text or "")
        if m:
            d = polish_def(m.groupdict().get("def", ""))
            if d:
                return d, 0.95
    colon_pat = re.compile(rf"\b{re.escape(term)}\s*:\s*([^\.!\n\r]{{10,200}})")
    m2 = colon_pat.search(text or "")
    if m2:
        d = polish_def(m2.group(1))
        if d:
            return d, 0.9
    for s in sent_split(text or ""):
        if re.search(rf"\b{re.escape(term)}\b", s, flags=re.I):
            d = polish_def(s)
            if d:
                return d, 0.65
    for s in sent_split(text or ""):
        if len(s.split()) >= 6:
            d = polish_def(s)
            if d:
                return d, 0.5
    return "", 0.0


def build_rich_description(url: str, text: str, soup: BeautifulSoup, title: str,
                           char_floor: int = 420, char_cap: int = 900) -> str:
    if not soup:
        return ""
    pieces = []
    m = meta_description(soup)
    if m:
        pieces.append(m)
    lead = first_paragraphs(soup, n=3)
    if lead:
        pieces.append(lead)
    main_txt = text or extract_main_text(soup)
    if main_txt:
        acc = []
        for s in sent_split(main_txt):
            if len(" ".join(acc)) >= char_floor:
                break
            if len(s.split()) < 5:
                continue
            acc.append(s)
        if acc:
            pieces.append(" ".join(acc))
    seen = set()
    uniq = []
    for p in pieces:
        pp = normalize_whitespace(p)
        if pp and pp not in seen:
            seen.add(pp)
            uniq.append(pp)
    desc = " ".join(uniq)
    if not desc:
        desc = (main_txt or "")[:char_cap]
    desc = normalize_whitespace(desc)
    if len(desc) < 120 and main_txt:
        desc = normalize_whitespace((desc + " " + main_txt)[:char_cap])
    return clamp_text(desc, max_chars=char_cap)


def is_junk_title(t: str) -> bool:
    return not t or normalize_whitespace(t).lower() in UI_JUNK


def section_text_for_heading(soup: BeautifulSoup, heading_text: str, max_chars: int = 700) -> str:
    if not soup:
        return ""
    norm = normalize_whitespace
    target = norm(heading_text)
    h = None
    for el in soup.find_all(re.compile(r"^h[1-6]$")):
        if norm(el.get_text(" ", strip=True)) == target:
            h = el
            break
    if not h:
        return ""
    level = int(h.name[1])
    chunks = []
    for sib in h.next_siblings:
        name = getattr(sib, "name", None)
        if name and re.match(r"^h[1-6]$", name):
            if int(name[1]) <= level:
                break
        if name in {"p", "ul", "ol", "table", "blockquote"}:
            t = normalize_whitespace(sib.get_text(" ", strip=True))
            if t:
                chunks.append(t)
        if len(" ".join(chunks)) > max_chars:
            break
    return clamp_text(" ".join(chunks), max_chars=max_chars)


def snippet_around(term: str, text: str, window_chars: int = 420) -> str:
    if not term or not text:
        return ""
    m = re.search(re.escape(term), text, flags=re.I)
    if not m:
        for s in sent_split(text):
            if term.lower() in s.lower():
                return clamp_text(s, max_chars=480)
        return ""
    start = max(0, m.start() - window_chars)
    end = min(len(text), m.end() + window_chars)
    return clamp_text(text[start:end], max_chars=520)


async def fetch_rich_preview(session, url: str, floor: int = 300, cap: int = 900) -> str:
    soup = await get_soup(session, url)
    if not soup:
        return ""
    text2 = extract_main_text(soup)
    title2 = soup.title.get_text(strip=True) if soup.title else ""
    return build_rich_description(url, text2, soup, title2, char_floor=floor, char_cap=cap)


# ------------------ Typisierung / Labels ------------------

def infer_labels(title, text, url, soup, ents):
    labels = set()
    if any(w.lower() in (title or "").lower() for w in DOC_HINT_WORDS) or ".pdf" in url.lower():
        labels.add("Dokument")
    if any(w.lower() in (text or "").lower() for w in EVENT_HINT_WORDS) or "/veranstaltung" in url.lower() or "/event" in url.lower():
        labels.add("Event")
    if re.search(r"^([A-ZÄÖÜ0-9\-]{2,})\b", (title or "")) and not labels:
        labels.update(ABBREV_APPS.get(title.strip(), "Begriff").split(";"))
    if re.search(r"\bAbkürzung\b|\bsteht für\b|\bAls .* bezeichnet man\b", (text or ""), flags=re.I):
        labels.add("Begriff")
    if any(h.lower() in (text or "").lower() for h in (h.lower() for h in ORG_HINTS)) or "/ueber" in url or "/uber" in url or "/kontakt" in url:
        labels.add("Organisation")
    if re.search(r"\b(Ernst\-Reuter\-Platz|Berlin|Showroom)\b", (text or "")):
        labels.add("Ort")
    if re.search(r"\b(Pflege 4\.0|Smart Home|AAL|Telemedizin|Robotik|Digitalisierung)\b", (text or ""), flags=re.I):
        labels.add("Kategorie")
    if title and title.strip() in ABBREV_APPS:
        for l in ABBREV_APPS[title.strip()].split(";"):
            labels.add(l)
    if ents:
        if any(e[1] == "ORG" for e in ents): labels.add("Organisation")
        if any(e[1] in ("GPE", "LOC") for e in ents): labels.add("Ort")
        if any(e[1] == "EVENT" for e in ents): labels.add("Event")
    if not labels:
        labels.add("Seite")
    return sorted(labels)


def extract_spacy_ents(text):
    try:
        import spacy
        nlp = spacy.load("de_core_news_sm")
        doc = nlp(text or "")
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception:
        return []


# ------------------ Relationserkennung ------------------

REL_PATTERNS = [
    (r"\b(?P<x>[\wÄÖÜäöüß\-\s]{2,60})\s+wird\s+von\s+(?P<y>[\wÄÖÜäöüß\-\s]{2,80})\s+(?:veranstaltet|organisiert)", "organized_by", 0.9),
    (r"\bunter\s+der\s+Leitung\s+von\s+(?P<y>[\wÄÖÜäöüß\.\-\s]{2,80})", "led_by", 0.9),
    (r"\bfindet\b.*\b(?:in|am)\s+(?P<y>Berlin|Ernst\-Reuter\-Platz\s*2|Showroom.*?)\b.*\bstatt", "located_in", 0.85),
    (r"\bwird\s+von\s+(?P<y>[\wÄÖÜäöüß\-\s]{2,80})\s+(?:gefördert|begleitet)", "supported_by", 0.85),
    (r"\bgehört\s+zu\s+(?P<y>[\wÄÖÜäöüß\-\s]{2,80})", "part_of", 0.75),
    (r"\bist\s+Teil\s+von\s+(?P<y>[\wÄÖÜäöüß\-\s]{2,80})", "part_of", 0.75),
    (r"\b(?P<x>[A-ZÄÖÜ0-9\-]{2,20})\s+steht\s+f(?:ü|u)r\s+(?P<y>[^\.!\?;]{2,80})", "abbreviation_of", 0.95),
    (r"\b(?P<x>[\wÄÖÜäöüß\-\s]{2,60})\s+ist\s+ein(?:e|)\s+(?P<y>[\wÄÖÜäöüß\-\s]{2,60})", "is_a", 0.7),
]


def mk_node(title, labels, url, desc_candidates, conf):
    nid = hash_id(title, ";".join(labels))
    return {
        "id": nid, "title": title or "", "labels": ";".join(labels),
        "desc_candidates": desc_candidates or "", "url": url, "confidence": f"{conf:.2f}"
    }


# ------------------ Crawl ------------------
async def crawl(start_url, max_pages, domain):
    seen = set()
    queue = collections.deque()
    pages_data = []

    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        # Initialisiere die Warteschlange
        if not urlparse(start_url).netloc:
            print("Ungültige Start-URL.")
            return pages_data

        base = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"

        # Versuche Sitemap zu laden
        sitemap_urls = []
        for sm in ["/sitemap.xml", "/sitemap_index.xml"]:
            smu = base + sm
            try:
                async with session.get(smu, timeout=10) as r:
                    if r.status == 200 and "<urlset" in await r.text():
                        sitemap_urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", await r.text())
                        break
            except Exception:
                pass

        if sitemap_urls:
            print(f"Gefundene URLs in Sitemap: {len(sitemap_urls)}. Fülle die Warteschlange.")
            for loc in sitemap_urls:
                u = clean_url(loc)
                if same_domain(u, domain) and u not in seen and len(queue) < max_pages:
                    seen.add(u)
                    queue.append(u)

        if not queue and start_url not in seen:
            queue.append(start_url)
            seen.add(start_url)

        print(f"Starte mit {len(queue)} URLs in der Warteschlange...")

        crawled_count = 0
        while queue and crawled_count < max_pages:
            batch = []
            while queue and len(batch) < 10:  # Parallel-Batch von 10 Seiten
                batch.append(queue.popleft())

            if not batch:
                continue

            tasks = [asyncio.create_task(get_soup(session, url)) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (url, soup_result) in enumerate(zip(batch, results)):
                if crawled_count >= max_pages:
                    break

                if isinstance(soup_result, Exception):
                    # Fehlerbehandlung bereits in get_soup, hier nur die Ausgabe
                    continue

                soup = soup_result
                if not soup:
                    continue

                # Neue Links finden und zur Queue hinzufügen
                for a in soup.find_all("a", href=True):
                    href = clean_url(urljoin(url, a["href"]))
                    if href and href not in seen and same_domain(href, domain) and href.startswith("http") and len(seen) < max_pages:
                        seen.add(href)
                        queue.append(href)

                title = soup.title.get_text(strip=True) if soup.title else ""
                text = extract_main_text(soup)
                pages_data.append({"url": url, "title": title, "text": text, "soup": soup})
                crawled_count += 1

                print(f"[{crawled_count:4d}] + {url} (Queue: {len(queue)})")

        return pages_data


# ------------------ Struktur-Extraktion ------------------

def from_similar_entries(soup):
    if not soup:
        return []
    edges = []
    boxes = []
    for h in soup.find_all(re.compile("^h[1-6]$")):
        if "Ähnliche Einträge" in h.get_text():
            boxes.append(h.parent)
    boxes += soup.select(".related, .similar")
    seen = set()
    for box in boxes:
        for a in box.find_all("a", href=True):
            t = a.get_text(" ", strip=True)
            href = clean_url(a["href"])
            if (t, href) in seen:
                continue
            seen.add((t, href))
            edges.append((t, href))
    return edges


def extract_categories_tags(soup):
    if not soup:
        return [], []
    cats, tags = set(), set()
    for a in soup.select("a[rel='category tag'], .cat-links a, .post-categories a"):
        t = a.get_text(" ", strip=True)
        if t:
            cats.add(t)
    for a in soup.select("a[rel='tag'], .tags a, .tagcloud a"):
        t = a.get_text(" ", strip=True)
        if t:
            tags.add(t)
    for a in soup.select(".breadcrumb a, .breadcrumbs a, nav[aria-label='Breadcrumb'] a"):
        t = a.get_text(" ", strip=True)
        if t and not re.match(r"^Start(seite)?$", t, re.I):
            cats.add(t)
    return sorted(cats), sorted(tags)


def extract_headings_terms(soup):
    if not soup:
        return []
    terms = []
    for h in soup.find_all(re.compile("^h[1-6]$")):
        t = h.get_text(" ", strip=True)
        if t and 2 <= len(t) <= 120:
            terms.append(t)
    return terms[:MAX_HEADING_MENTIONS]


def extract_ld_json(soup):
    if not soup:
        return []
    out = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string)
            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                out.extend([d for d in data if isinstance(d, dict)])
        except Exception:
            continue
    return out


def infer_event_structural_edges(soup, page_title, page_url):
    if not soup:
        return []
    edges = []
    for h in soup.find_all(re.compile("^h[1-6]$")):
        txt = h.get_text(" ", strip=True)
        sec = h.find_next_sibling()
        if not sec:
            continue
        if "Veranstalter" in txt:
            a = sec.find("a")
            org = a.get_text(" ", strip=True) if a else sec.get_text(" ", strip=True)
            edges.append({"rel": "organized_by", "dst_title": org, "evidence": "section:Veranstalter", "confidence": 0.9})
        if "Veranstaltungsort" in txt:
            a = sec.find("a")
            loc = a.get_text(" ", strip=True) if a else sec.get_text(" ", strip=True)
            edges.append({"rel": "located_in", "dst_title": loc, "evidence": "section:Veranstaltungsort", "confidence": 0.9})
    return edges


# ------------------ Ähnlichkeitskanten ------------------

def build_similarity_edges(pages, url_to_node_id):
    page_tokens = {}
    inv = collections.defaultdict(set)
    for p in pages:
        toks = tokenize(p["text"])
        page_tokens[p["url"]] = toks
        for t in list(toks)[:200]:
            inv[t].add(p["url"])

    edges = []
    for p in pages:
        u = p["url"]
        cand = set()
        for t in list(page_tokens[u])[:100]:
            cand |= inv[t]
        cand.discard(u)
        scores = []
        for v in cand:
            a, b = page_tokens[u], page_tokens[v]
            if not a or not b:
                continue
            j = len(a & b) / float(len(a | b))
            if j >= SIM_THRESHOLD:
                scores.append((j, v))
        scores.sort(reverse=True)
        for j, v in scores[:MAX_SIM_EDGES_PER_NODE]:
            edges.append({
                "src_id": url_to_node_id[u],
                "dst_id": url_to_node_id.get(v) or url_to_node_id[u],
                "rel": "related_to",
                "evidence": f"content-similarity jaccard={j:.2f}",
                "source_url": u,
                "confidence": f"{min(0.89, 0.5 + j):.2f}"
            })
    return edges


# ------------------ Main ------------------

async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="https://lebenpflegedigital.de")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--out", default="out")
    ap.add_argument("--min-chars", type=int, default=420, help="Minimale Zielgröße der Kurzbeschreibung pro Seite")
    ap.add_argument("--max-chars", type=int, default=900, help="Maximale Länge der Kurzbeschreibung")
    ap.add_argument("--hydrate-budget", type=int, default=HYDRATE_BUDGET_DEFAULT, help="Wie viele interne Linkziele zusätzlich laden")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    domain = urlparse(args.start).netloc or "lebenpflegedigital.de"

    print(f"Starte asynchronen Crawl von {args.start} (max. {args.max_pages} Seiten)...")
    pages = await crawl(args.start, args.max_pages, domain)
    print(f"\nCrawl abgeschlossen. {len(pages)} Seiten gefunden.")
    print("-" * 20)

    nodes = {}
    edges = []
    glossary_rows = []

    title_to_id = {}
    url_to_node_id = {}

    # Sammle URLs für Link-Hydration
    hydrate_urls = []
    hydrated_count = 0
    for p in pages:
        for a in p["soup"].find_all("a", href=True):
            href = clean_url(urljoin(p["url"], a["href"]))
            if href and href not in url_to_node_id and same_domain(href, domain) and href.startswith("http"):
                if hydrated_count < args.hydrate_budget:
                    hydrate_urls.append(href)
                    hydrated_count += 1

    print(f"Lade {len(hydrate_urls)} zusätzliche Link-Vorschauen (async)...")
    previews = {}
    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        tasks = [asyncio.create_task(fetch_rich_preview(session, url, floor=280, cap=args.max_chars)) for url in hydrate_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, result in zip(hydrate_urls, results):
            if isinstance(result, Exception):
                # print(f"Fehler bei Hydration von {url}: {result}") # Zu viele Ausgaben
                previews[url] = ""
            else:
                previews[url] = result
    print("Link-Hydration abgeschlossen.")
    print("-" * 20)

    # Hauptverarbeitung
    print("Verarbeite extrahierte Seiten und Beziehungen...")
    for i, p in enumerate(pages):
        url, title, soup, text = p["url"], p["title"] or "", p["soup"], p["text"]

        # Statusanzeige
        progress = ((i + 1) / len(pages)) * 100
        print(f"[{i+1}/{len(pages)}] {progress:.1f}% - {url}")

        ents = extract_spacy_ents(text)
        labels = infer_labels(title, text, url, soup, ents)
        rich_desc = build_rich_description(url, text, soup, title, char_floor=args.min_chars, char_cap=args.max_chars)
        short, sc = pick_short_def(title, text) if title else ("", 0.0)
        desc = short or rich_desc
        conf = sc if short else (0.72 if rich_desc else 0.6)
        node = mk_node(title or url, labels, url, desc, conf)
        nodes[node["id"]] = node
        title_to_id.setdefault(node["title"], node["id"])
        url_to_node_id[url] = node["id"]
        cats, tags = extract_categories_tags(soup)
        for c in cats:
            if is_junk_title(c): continue
            cctx = snippet_around(c, rich_desc or text, window_chars=420) or (rich_desc[:420] if rich_desc else "")
            cnode = mk_node(c, ["Kategorie"], url, cctx, 0.7 if cctx else 0.6)
            nodes.setdefault(cnode["id"], cnode)
            edges.append({ "src_id": node["id"], "dst_id": cnode["id"], "rel": "part_of", "evidence": "category/breadcrumb", "source_url": url, "confidence": "0.70"})
        for t in tags:
            if is_junk_title(t): continue
            tctx = snippet_around(t, rich_desc or text, window_chars=420)
            tnode = mk_node(t, ["Begriff"], url, tctx, 0.55 if tctx else 0.5)
            nodes.setdefault(tnode["id"], tnode)
            edges.append({ "src_id": node["id"], "dst_id": tnode["id"], "rel": "mentions", "evidence": "tag", "source_url": url, "confidence": "0.55"})
        for term in extract_headings_terms(soup):
            if is_junk_title(term): continue
            ctx = section_text_for_heading(soup, term, max_chars=700)
            if not ctx: ctx = snippet_around(term, rich_desc or text, window_chars=420)
            y_node = mk_node(term, ["Begriff"], url, ctx, 0.6 if ctx else 0.5)
            nodes.setdefault(y_node["id"], y_node)
            edges.append({ "src_id": node["id"], "dst_id": y_node["id"], "rel": "mentions", "evidence": "heading", "source_url": url, "confidence": "0.50"})
        if "Begriff" in labels or "Anwendung" in labels or title in ABBREV_APPS:
            gdef, gsc = pick_short_def(title, text)
            if not gdef:
                gdef = polish_def(meta_description(soup) or first_paragraphs(soup, n=2))
                gsc = 0.6 if gdef else 0.0
            if gdef:
                glossary_rows.append({"term": title, "short_definition": gdef, "confidence": f"{gsc:.2f}", "sources": url})
        for pat, rel, rconf in REL_PATTERNS:
            for m in re.finditer(pat, text or "", flags=re.I):
                x = (m.groupdict().get("x") or title or "").strip()
                y = (m.groupdict().get("y") or "").strip()
                if not x or not y: continue
                y_labels = []
                if rel in ("organized_by", "supported_by", "led_by"): y_labels.append("Organisation")
                if rel in ("located_in", "hosted_at"): y_labels.append("Ort")
                if rel in ("abbreviation_of", "is_a", "part_of"): y_labels.append("Kategorie")
                if not y_labels: y_labels = ["Begriff"]
                y_node = mk_node(y, y_labels, url, "", 0.6)
                nodes.setdefault(y_node["id"], y_node)
                src_id = node["id"]
                if x and x != title:
                    xid = title_to_id.get(x)
                    if not xid:
                        x_node = mk_node(x, ["Begriff"], url, "", 0.6)
                        nodes.setdefault(x_node["id"], x_node)
                        xid = x_node["id"]
                    src_id = xid
                edges.append({ "src_id": src_id, "dst_id": y_node["id"], "rel": rel, "evidence": m.group(0).strip(), "source_url": url, "confidence": f"{rconf:.2f}" })
        for obj in extract_ld_json(soup):
            t = obj.get("@type")
            if isinstance(t, list): t = t[0] if t else None
            if t and str(t).lower() == "event":
                org = obj.get("organizer"); loc = obj.get("location")
                if isinstance(org, dict): org = org.get("name")
                if isinstance(loc, dict): loc = loc.get("name")
                if org:
                    y_node = mk_node(org, ["Organisation"], url, "", 0.85)
                    nodes.setdefault(y_node["id"], y_node)
                    edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": "organized_by", "evidence": "jsonld.organizer", "source_url": url, "confidence": "0.85"})
                if loc:
                    y_node = mk_node(loc, ["Ort"], url, "", 0.85)
                    nodes.setdefault(y_node["id"], y_node)
                    edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": "located_in", "evidence": "jsonld.location", "source_url": url, "confidence": "0.85"})
        for t, href in from_similar_entries(soup):
            y_node = mk_node(t or href, ["Begriff"], href if href.startswith("http") else urljoin(url, href), "", 0.55)
            nodes.setdefault(y_node["id"], y_node)
            edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": "related_to", "evidence": "Ähnliche Einträge", "source_url": url, "confidence": "0.55"})
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.lower().endswith(".pdf"):
                doc_title = a.get_text(" ", strip=True) or href.split("/")[-1]
                doc_url = href if href.startswith("http") else urljoin(url, href)
                dnode = mk_node(doc_title, ["Dokument"], doc_url, "", 0.75)
                nodes.setdefault(dnode["id"], dnode)
                edges.append({"src_id": node["id"], "dst_id": dnode["id"], "rel": "has_document", "evidence": "PDF-Link", "source_url": url, "confidence": "0.75"})
        cnt = 0
        for a in soup.find_all("a", href=True):
            if cnt >= MAX_INTERNAL_LINKS: break
            href = clean_url(urljoin(url, a["href"]))
            if not href or not href.startswith("http"): continue
            if same_domain(href, urlparse(url).netloc):
                t = a.get_text(" ", strip=True) or href
                if is_junk_title(t): continue
                desc = previews.get(href, "")
                y_node = mk_node(t, ["Begriff"], href, desc, 0.6 if desc else 0.5)
                if y_node["id"] not in nodes: nodes[y_node["id"]] = y_node
                edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": "related_to", "evidence": "internal-link", "source_url": url, "confidence": "0.50"})
                cnt += 1
        strongs = set(s.get_text(" ", strip=True) for s in soup.select("strong,b") if s.get_text(strip=True))
        for t in strongs:
            if not (2 <= len(t) <= 120) or is_junk_title(t): continue
            sctx = snippet_around(t, rich_desc or text, window_chars=420)
            y_node = mk_node(t, ["Begriff"], url, sctx, 0.55 if sctx else 0.45)
            nodes.setdefault(y_node["id"], y_node)
            edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": "mentions", "evidence": "bold/strong", "source_url": url, "confidence": "0.45"})
        if "Event" in labels:
            for e in infer_event_structural_edges(soup, title, url):
                y_node = mk_node(e["dst_title"], ["Organisation"] if e["rel"] == "organized_by" else ["Ort"], url, "", e["confidence"])
                nodes.setdefault(y_node["id"], y_node)
                edges.append({"src_id": node["id"], "dst_id": y_node["id"], "rel": e["rel"], "evidence": e["evidence"], "source_url": url, "confidence": f"{e['confidence']:.2f}"})

    edges.extend(build_similarity_edges(pages, url_to_node_id))

    out = args.out
    os.makedirs(out, exist_ok=True)
    nf = ["id", "title", "labels", "desc_candidates", "url", "confidence"]
    with open(f"{out}/nodes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=nf)
        w.writeheader()
        for n in nodes.values():
            w.writerow(n)
    ef = ["src_id", "dst_id", "rel", "evidence", "source_url", "confidence"]
    with open(f"{out}/edges.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ef)
        w.writeheader()
        for e in edges:
            w.writerow(e)
    if glossary_rows:
        gf = ["term", "short_definition", "confidence", "sources"]
        with open(f"{out}/glossary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=gf)
            w.writeheader()
            for r in glossary_rows:
                w.writerow(r)

    print(f"\nAbschluss der Datenverarbeitung.")
    print(f"✔ nodes.csv: {len(nodes)} Knoten")
    print(f"✔ edges.csv: {len(edges)} Kanten")
    print(f"✔ glossary.csv: {len(glossary_rows)} Begriffe")

    write_ttl(f"{out}/nodes.csv", f"{out}/edges.csv", f"{out}/lebenpflege.ttl")


if __name__ == "__main__":
    asyncio.run(main_async())