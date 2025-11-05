import json
import random
import os
import sys
import argparse
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import textwrap
from datetime import datetime


# RAG-/Chat-Pipeline für den Pflege-Concierge:
# - verbindet Neo4j-Wissensgraphen (LPD, MTF, Navel) mit einem lokalen LLM
# - holt passende Knoten/Fakten, verdichtet sie zu Karten und baut daraus einen Prompt
# - generiert damit eine kurze, faktenbasierte Antwort und bewertet sie mit Evaluation
# - unterstützt Smalltalk, Prompt-Augmentation, mehrere KG-Datenbanken und Demo-Modi

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
except NameError:
    pass

from eval.artefacts.evaluation import Evaluation

from neo4j import GraphDatabase
import requests

# --- Globale Konfiguration & Defaults ---
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "mike2504")
os.environ.setdefault("NEO4J_DATA_DB", "kgraphdata")
os.environ.setdefault("NEO4J_DATA_DB_2", "mtfdata")
os.environ.setdefault("NEO4J_DATA_DB_3", "naveldata")
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")
os.environ.setdefault("NEO4J_APOC_INSTALLED", "false")

logging.basicConfig(level=logging.ERROR)
for _name in ("neo4j", "urllib3"):
    logging.getLogger(_name).setLevel(logging.ERROR)

logger = logging.getLogger("rag")
# Zum Aktivieren: logger.setLevel(logging.INFO)

# --- Hilfsfunktionen ---
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default

def _safe_head(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)

def _dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([s for s in seq if s]))

def _lc(s: Optional[str]) -> str:
    return (s or "").lower()

def _extract_product_name(query: str) -> Optional[str]:
    product_keywords = ["gerät", "produkt", "artikel", "device", "technik"]
    query_lower = query.lower()
    specific_products = ["navel", "tante laura", "bosch bewegungsmelder", "caera", "caru care"]
    for product in specific_products:
        if product in query_lower:
            return product.title()

    for keyword in product_keywords:
        match = re.search(f"{keyword}\\s+([\\w\\s\\d.-]*\\b(?:4g|gps|gprs|lpd)?\\b)", query_lower)
        if match:
            name = match.group(1).strip()
            if len(name) > 3:
                return name.title()

    parts = []
    words = re.findall(r'\b[\w-]+\b', query)
    generic_words = {"hat", "das", "gerät", "salind", "gps", "monatliche", "kosten", "was", "ist", "suche", "nenne", "finder", "mir", "was", "zu", "sag", "sage"}
    for word in words:
        if word.lower() not in generic_words and len(word) > 2:
            parts.append(word)

    if not parts:
        return None

    name = " ".join(parts).replace("?", "").replace("!", "").strip()
    return name if len(name) > 3 else None


_ANCHORS = [
    "Leben Pflege Digital", "LebenPflegeDigital",
    "Mein Technik-Finder", "Mein Technikfinder",
    "Caru Care", "Kissen Viktor", "Viktor", "Navel"
]
def _preserve_anchors(text: str) -> str:
    """
    Schützt bekannte Namen + erkannten Produktnamen durch Anführungszeichen
    (nur als Zusatz — verändert die übrigen Tokens nicht).
    """
    if not text:
        return text
    out = text
    # Erkannte Produkt-/Gerätenamen zusätzlich schützen
    pn = _extract_product_name(text)
    anchors = list(_ANCHORS)
    if pn and pn not in anchors:
        anchors.append(pn)

    for a in anchors:
        if not a:
            continue
        # case-insensitive ersetzen, aber Originalschreibung beibehalten
        out = re.sub(rf"(?i)\b{re.escape(a)}\b", lambda m: f"\"{m.group(0)}\"", out)
    return out

# --- Smalltalk-Erweiterung ---
class Smalltalk:
    PATTERNS = [
        ("greeting", re.compile(r"^(hi|hallo|hey|servus|moin|grüß\s*dich|ciao)\b", re.I)),
        ("how_are_you", re.compile(r"wie\s*geht('?|s)?(\s*dir)?", re.I)),
        ("thanks", re.compile(r"\b(danke|dankeschön|thx|merci)\b", re.I)),
        ("bye", re.compile(r"(tsch(ü|u)ss|ciao|bye|auf\s*wiedersehen|bis\s*bald)", re.I)),
        ("who_are_you", re.compile(r"(wer\s*bist\s*du|wie\s*hei(ß|ss)t\s*du)", re.I)),
        ("what_can_you_do", re.compile(r"(was\s*kannst\s*du|hilfe\b|was\s*machst\s*du)", re.I)),
        ("are_you_there", re.compile(r"(bist\s*du\s*da|noch\s*da|hallo\??)$", re.I)),
        ("ok", re.compile(r"^(ok(ay)?|alles\s*klar|passt|super|top|cool)$", re.I)),
        ("joke", re.compile(r"\b(witz|joke)\b", re.I)),
    ]

    RESPONSES = {
        "greeting": ["Hallo! Wie kann ich dir helfen?", "Hi! Schön, dich zu sehen."],
        "how_are_you": ["Mir geht’s gut – danke! Und dir?", "Alles bestens, danke der Nachfrage."],
        "thanks": ["Sehr gern!", "Gern geschehen."],
        "bye": ["Tschüss! Bis bald.", "Auf Wiedersehen!"],
        "who_are_you": ["Ich bin dein Pflege-Concierge-Assistent.", "Ich suche für dich Fakten im Wissensgraphen."],
        "what_can_you_do": ["Ich beantworte Fragen zu Pflegehilfen und Technik.", "Ich helfe dir, passende Infos im Graphen zu finden."],
        "are_you_there": ["Ja, ich bin da.", "Hier!"],
        "ok": ["Super 👍", "Alles klar."],
        "joke": ["Warum können Skelette nicht lügen? – Weil man durch sie hindurchsieht!"],
    }

    @staticmethod
    def detect(question: str) -> Optional[str]:
        q = (question or "").strip()
        for label, pat in Smalltalk.PATTERNS:
            if pat.search(q):
                return label
        return None

    @staticmethod
    def respond(intent: str) -> str:
        choices = Smalltalk.RESPONSES.get(intent, ["Hallo!"])
        return random.choice(choices)

# --- Datenklasse für Faktenkarten ---
@dataclass
class AssertionCard:
    node_id: str
    uri: Optional[str]
    label: str
    types: List[str]
    title: Optional[str]
    snippet: Optional[str]
    facts: List[str]
    sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KGAdapter:
    def __init__(self, uri: str, user: str, password: str,
                 data_db: str = "neo4j", ontology_db: Optional[str] = None,
                 pool_size: int = 5, acquire_timeout: float = 12.0,
                 additional_labels: Optional[List[str]] = None,
                 search_properties: Optional[List[str]] = None):
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=pool_size,
            connection_acquisition_timeout=acquire_timeout,
        )
        self._data_db = data_db
        self._onto_db = ontology_db
        self.allowed_labels = [
            "Begriff", "Event", "Kategorie", "Ort", "Dokument", "Organisation", "Seite", "Thema", "Produkt"
        ]
        if additional_labels:
            self.allowed_labels.extend(additional_labels)
        self.search_properties = search_properties or [
            "title", "titel", "snippet", "beschreibung", "desc_candidates", "hatTitel", "hatBeschreibung"
        ]

    def close(self):
        self._driver.close()

    def ensure_fulltext_index(self, index_name: str = "kgAnyTextIndex") -> None:
        label_union = "|".join(self.allowed_labels)
        props_str = ", ".join(f"n.{p}" for p in self.search_properties)
        if not props_str: return
        with self._driver.session(database=self._data_db) as sess:
            try:
                ex = sess.run(
                    "SHOW INDEXES YIELD name WHERE name = $name RETURN name",
                    {"name": index_name}, timeout=6.0
                ).single()
                if ex:
                    return
            except Exception:
                pass
            try:
                sess.run(
                    f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS "
                    f"FOR (n:{label_union}) ON EACH [{props_str}]",
                    timeout=12.0
                )
            except Exception:
                pass

    def _allowed_label_filter(self, alias: str = "n") -> str:
        return f"any(l IN labels({alias}) WHERE l IN $allowed)"

    def find_direct_hit(self, query: str, use_fulltext: bool = True) -> Optional[Tuple[str, float]]:
        """Hybrid-Suche: erst Fulltext (optional), dann Property-Fallback (robust für LIST<STRING>)."""
        q = (query or "").strip().lower()
        if not q:
            return None
        with self._driver.session(database=self._data_db) as sess:
            # Fulltext zuerst (optional)
            if use_fulltext:
                try:
                    rec = sess.run(
                        """
                        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
                        WHERE any(l IN labels(node) WHERE l IN $allowed)
                        RETURN elementId(node) AS id, score
                        ORDER BY score DESC
                        LIMIT 1
                        """,
                        {"index": "kgAnyTextIndex", "q": q, "allowed": self.allowed_labels}, timeout=6.0
                    ).single()
                    if rec:
                        return rec["id"], float(rec["score"])
                except Exception:
                    pass

            # Fallback: Property-Suche robust
            where_clauses = []
            for p in self.search_properties:
                where_clauses.append(
                    f"""
                    (CASE
                        WHEN n.`{p}` IS NULL THEN false
                        WHEN n.`{p}` IS :: STRING THEN toLower(n.`{p}`) CONTAINS $q
                        WHEN n.`{p}` IS :: LIST<STRING> THEN any(s IN n.`{p}` WHERE toLower(s) CONTAINS $q)
                        ELSE false
                    END)
                    """
                )
            where_str = " OR ".join(where_clauses)

            rec = sess.run(
                f"""
                MATCH (n)
                WHERE {self._allowed_label_filter('n')} AND ({where_str})
                RETURN elementId(n) AS id, 0.95 AS score
                LIMIT 1
                """,
                {"q": q, "allowed": self.allowed_labels}, timeout=6.0
            ).single()
            if rec:
                return rec["id"], float(rec["score"])
        return None

    def search_nodes(self, query: str, top_k: int = 5,
                     index_name: str = "kgAnyTextIndex",
                     use_fulltext: bool = True) -> List[Tuple[str, float]]:
        q = (query or "").strip()
        q_lower = q.lower()
        with self._driver.session(database=self._data_db) as sess:
            if use_fulltext:
                try:
                    res = sess.run(
                        f"""
                        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
                        WITH node, score
                        WHERE {self._allowed_label_filter('node')}
                        RETURN elementId(node) AS id, score
                        ORDER BY score DESC
                        LIMIT $k
                        """,
                        {"index": index_name, "q": q, "k": top_k, "allowed": self.allowed_labels},
                        timeout=8.0,
                    )
                    rows = [(r["id"], float(r["score"])) for r in res]
                    if rows:
                        return rows
                except Exception:
                    pass

            prop_union_list = [f"(CASE WHEN n.`{p}` IS :: LIST<STRING> THEN n.`{p}` WHEN n.`{p}` IS :: STRING THEN [n.`{p}`] ELSE [] END)" for p in self.search_properties]
            prop_union_str = " + ".join(prop_union_list)

            res = sess.run(
                f"""
                MATCH (n)
                WHERE {self._allowed_label_filter('n')}
                WITH n, {prop_union_str} AS props
                UNWIND props AS p
                WITH n, toLower(p) AS s
                WHERE s CONTAINS $q
                RETURN elementId(n) AS id, 1.0 AS score
                LIMIT $k
                """,
                {"q": q_lower, "k": top_k, "allowed": self.allowed_labels},
                timeout=8.0
            )
            return [(r["id"], float(r["score"])) for r in res]

    def build_card(self, node_eid: str, max_neighbors: int = 20) -> AssertionCard:
        with self._driver.session(database=self._data_db) as sess:
            rec = sess.run(
                """
                MATCH (n) WHERE elementId(n) = $id
                WITH n,
                     [(n)-[r]->(m) | {rel:type(r), dir:'out', nbr:m}][..$mx] AS outN,
                     [(m)-[r]->(n) | {rel:type(r), dir:'in',  nbr:m}][..$mx] AS inN
                RETURN labels(n) AS labels,
                       n.uri AS uri,
                       n AS props,
                       outN + inN AS nbrs
                """,
                {"id": node_eid, "mx": int(max_neighbors)},
                timeout=8.0
            ).single()

        if not rec:
            raise ValueError(f"Node {node_eid} not found")

        labels = list(rec["labels"]) if rec["labels"] else []
        uri = rec["uri"]
        node_props = dict(rec["props"])

        title = _safe_head(node_props.get("title") or node_props.get("titel") or node_props.get("hatTitel"))
        snippet = _safe_head(
            node_props.get("snippet")
            or node_props.get("beschreibung")
            or node_props.get("desc_candidates")
            or node_props.get("hatBeschreibung")
        )
        node_source = _safe_head(node_props.get("quelle") or node_props.get("url") or node_props.get("uri"))

        facts: List[str] = []
        sources: List[str] = []

        if node_source and str(node_source).startswith("http"):
            sources.append(str(node_source))


        skip_keys = {"title", "titel", "hatTitel", "snippet",
                     "beschreibung", "hatBeschreibung",
                     "desc_candidates", "uri"}
        for key, val in node_props.items():
            if key in skip_keys:
                continue
            sval = _safe_head(val)
            if sval:
                facts.append(f"{key}: {sval}")

        def pick_title(props: Dict[str, Any]) -> str:
            for k in ("title", "titel", "name", "bezeichnung", "label", "hatTitel"):
                v = props.get(k)
                if v is not None:
                    t = _safe_head(v)
                    if t:
                        return t
            return "(ohne Titel)"

        for item in rec["nbrs"] or []:
            rel = item.get("rel")
            direction = item.get("dir")
            nbr = item.get("nbr")
            if not nbr:
                continue
            nbr_props = dict(nbr)
            nbr_title = pick_title(nbr_props)
            fact = (
                f"{title or '(Knoten)'} —[{rel}]-> {nbr_title}"
                if direction == "out"
                else f"{nbr_title} —[{rel}]-> {title or '(Knoten)'}"
            )
            facts.append(fact)

            for key in ("quelle", "url", "source", "link", "website"):
                sval = _safe_head(nbr_props.get(key))
                if sval and str(sval).startswith("http"):
                    sources.append(str(sval))

        sources = _dedup(sources)

        return AssertionCard(
            node_id=node_eid,
            uri=uri,
            label=":".join(labels),
            types=labels,
            title=title,
            snippet=snippet,
            facts=facts,
            sources=sources,
        )

# --- LLM Client ---
class LLMClient:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, short: bool = False):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.short = short

    def generate(self, prompt: str, temperature: float = 0.45) -> str:
        url = f"{self.base_url}/api/generate"
        options = {"temperature": temperature}
        if self.short:
            options["num_predict"] = 180
        payload = {"model": self.model, "prompt": prompt, "options": options, "stream": False}
        r = requests.post(url, json=payload, timeout=90)
        r.raise_for_status()
        return r.json().get("response", "")


class PromptAugmentor:
    """Erzeugt kompakte Subqueries und optionale LLM-Alternativen."""
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def rewrite_queries(self, question: str) -> List[str]:
        q = (question or "").strip()
        if not q:
            return []
        queries: List[str] = [q]

        name = _extract_product_name(q)
        if name and name.lower() != q.lower():
            queries.append(name)

        def _shorten(s: str, max_words: int = 6) -> str:
            toks = re.findall(r"\w+", s)
            return " ".join(toks[:max_words])

        queries = [_shorten(x) for x in queries if x]

        # Optionale LLM-Expansion
        if self.llm:
            try:
                prompt = (
                    "Erzeuge 1-2 alternative, sehr kurze Suchanfragen (max. 6 Wörter) "
                    "für die folgende Frage. Antworte ausschließlich als JSON-Liste von Strings.\n"
                    f"Frage: {q}"
                )
                raw = self.llm.generate(prompt, temperature=0.2).strip()
                m = re.search(r"\[.*\]", raw, flags=re.S)
                if m:
                    extras = json.loads(m.group(0))
                    for e in extras:
                        if isinstance(e, str) and e.strip():
                            queries.append(_shorten(e))
            except Exception:
                pass

        return _dedup([x for x in queries if x])


class RAGEngine:
    def __init__(self, kg_adapters: List[KGAdapter], llm: Optional[LLMClient] = None,
                 augmentor: Optional[PromptAugmentor] = None,
                 facts_budget_chars: int = 4000,
                 use_fewshots: bool = False,
                 disable_augmentation: bool = False,
                 # Pipeline-Schalter
                 use_kg: bool = True,
                 use_fulltext: bool = True,
                 use_direct_hit: bool = True,
                 use_prompt_augmentation: bool = True):
        self.kg_adapters = kg_adapters
        self.llm = llm
        self.relevance_score_threshold = 0.1
        self.facts_budget_chars = facts_budget_chars
        self.use_fewshots = use_fewshots
        self.disable_augmentation = disable_augmentation
        # Pipeline-Schalter
        self.use_kg = use_kg
        self.use_fulltext = use_fulltext
        self.use_direct_hit = use_direct_hit
        self.use_prompt_augmentation = use_prompt_augmentation and not disable_augmentation
        self.augmentor = None if not self.use_prompt_augmentation else (augmentor or PromptAugmentor(llm))

    @staticmethod
    def _compact_cards_for_prompt(cards: List[AssertionCard], max_facts_per_card: int = 8, budget_chars: int = 4000) -> str:
        """Strukturierter, token-sparsamer Faktenblock mit hartem Budget."""
        chunks: List[str] = []
        for c in cards:
            title = (c.title or "(ohne Titel)")[:160]
            snippet = (c.snippet or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            facts = c.facts[:max_facts_per_card]
            srcs = c.sources[:3]
            block = [
                        f"[Knoten] {title}",
                        f" Info: {snippet}" if snippet else " Info: -",
                    ] + [f" Fakt: {f}" for f in facts] + [f" Quelle: {s}" for s in srcs]
            chunks.append("\n".join(block))
            if len("\n".join(chunks)) >= budget_chars:
                break
        return "\n".join(chunks)

    @staticmethod
    def _sources_from_cards(cards: List[AssertionCard], max_total: int = 5) -> List[str]:
        srcs: List[str] = []
        for c in cards:
            srcs.extend(c.sources)
        return _dedup(srcs)[:max_total]

    def _is_card_relevant_for_query(self, card: AssertionCard, query: str) -> bool:
        product_name = _extract_product_name(query)
        if product_name:
            if _lc(product_name) in _lc(card.title or ""):
                return True


        question_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        card_text_lower = _lc(card.title or "") + " " + \
                          _lc(card.snippet or "") + " " + \
                          " ".join(_lc(f) for f in card.facts)

        relevant_words_found = 0
        for word in question_words:
            if word in card_text_lower:
                relevant_words_found += 1

        if relevant_words_found >= 1 or (len(question_words) > 0 and relevant_words_found / len(question_words) >= 0.3):
            return True

        return False

    def _fewshot_block(self) -> str:
        if not self.use_fewshots:
            return ""
        return (
            "Beispiel:\n"
            "Frage: Was ist System X?\n"
            "Faktenblock (Auszug):\n[Knoten] System X\n Info: Kurze Beschreibung...\n Fakt: System X —[hatFunktion]-> Alarmierung\n Quelle: https://example.org\n"
            "Antwort: System X dient der Alarmierung und Ortung. Quellen: https://example.org\n\n"
        )

    def _build_prompt_with_facts(self, question: str, cards: List[AssertionCard], history: str) -> str:
        facts_block = RAGEngine._compact_cards_for_prompt(cards, budget_chars=self.facts_budget_chars)
        prompt_parts = [
            "Du bist ein freundlicher Pflege-Concierge. Antworte ausformuliert, exakt und knapp.",
            "Regeln:",
            "1) Nutze AUSSCHLIESSLICH den Faktenblock.",
            "2) KEINE Floskeln, KEINE Bulletpoints.",
            "3) Antworte in 1–2 sehr kompakten Sätzen (max. 18 Wörter je Satz).",
            "4) Keine Begrüßung, keine Wiederholung der Frage.",
            "5) Preise nur, wenn explizit gefragt.",
            "6) Fehlt eine Info: Sag das in 1 kurzem Satz.",
            "7) Füge keine Quellenliste ein.",
        ]
        few = self._fewshot_block()
        if few:
            prompt_parts += ["", few]
        if history:
            prompt_parts += ["", f"Bisherige Konversation (Kurz):\n{history[-1200:]}"]
        prompt_parts += ["", f"Frage:\n{question}", "", f"Faktenblock:\n{facts_block}", "", "Antwort:"]
        return "\n".join(prompt_parts)

    def _build_prompt_fallback(self, question: str, history: str) -> str:
        prompt_parts = [
            "Du bist ein strenger Concierge.",
            "Regeln:",
            "1) Wenn im Faktenblock nichts dazu steht, GIB KEINE INHALTLICHE ANTWORT.",
            "2) KEINE Vermutungen, KEINE allgemeinen Erklärungen, KEINE Beispiele.",
            "3) KEINE Links, KEINE Quellenliste.",
            "4) Antworte in 1–2 sehr kompakten Sätzen nur: 'Mir fehlen Fakten zu <Thema>'.",
        ]
        if history:
            prompt_parts += ["", f"Bisherige Konversation (Kurz):\n{history[-1200:]}"]
        prompt_parts += ["", f"Frage:\n{question}", "", "Antwort:"]
        return "\n".join(prompt_parts)

    def _build_prompt_general_knowledge_fallback(self, question: str, history: str, product_name: str) -> str:
        """Neuer Prompt für den Fallback mit allgemeinem Wissen."""
        prompt_parts = [
            "Du bist ein freundlicher Pflege-Concierge.",
            "Regeln:",
            "1) Die folgende Antwort basiert nicht auf den Wissensfakten, die dir vorliegen.",
            f"2) Beziehe dich bei deiner Antwort auf den Namen „{product_name}“ und verwende ihn genau so.",
            "3) Beginne die Antwort mit dem Satz: 'Mir liegen aktuell keine spezifischen Fakten zu dieser Frage vor, aber ich kann Ihnen aus meinem allgemeinen Wissen folgendes dazu sagen:'",
            "4) Antworte im Pflege-/Hilfsmittel-Kontext.",
            "5) Antworte in 1–2 sehr kompakten Sätzen (max. 18 Wörter je Satz).",
            "6) KEINE Floskeln, KEINE Bulletpoints, KEINE Links.",
        ]
        if history:
            prompt_parts += ["", f"Bisherige Konversation (Kurz):\n{history[-1200:]}"]
        prompt_parts += ["", f"Frage:\n{question}", "", "Antwort:"]
        return "\n".join(prompt_parts)

    def _build_prompt_llm_only(self, question: str, history: str) -> str:
        parts = [
            "Du bist ein hilfreicher, prägnanter Assistent.",
            "Antworte fachlich korrekt in 1–2 kompakten Sätzen.",
            "Keine Aufzählungen, keine Links.",
        ]
        if history:
            parts += ["", f"Verlauf (kurz):\n{history[-1200:]}"]
        parts += ["", f"Frage:\n{question}", "", "Antwort:"]
        return "\n".join(parts)

    def _answer_llm_only(self, question: str, history: str):
        prompt = self._build_prompt_llm_only(question, history)
        answer = self.llm.generate(prompt) if self.llm else "(LLM nicht konfiguriert)\n\n" + prompt
        meta = {"cards": [], "prompt": prompt, "sources": [], "evaluation": {}, "subqueries": [question]}
        return answer, meta


    def _search_cards(self, queries: List[str], question_for_relevance: str,
                      top_k: int, max_neighbors: int,
                      product_name_for_direct: Optional[str]) -> Tuple[List[AssertionCard], Dict[str, float], List[Tuple[str, float]]]:
        """
        Führt Suche über alle Adapter & Queries aus, baut Karten und liefert rudimentäre Stats zurück.
        """
        found_cards: List[AssertionCard] = []
        all_node_scores: List[Tuple[str, float]] = []
        direct_hit_used = 0

        # 1) Direkter Treffer auf Produktnamen (optional)
        if self.use_direct_hit and product_name_for_direct:
            for kg_adapter in self.kg_adapters:
                try:
                    dh = kg_adapter.find_direct_hit(product_name_for_direct, use_fulltext=self.use_fulltext)
                except Exception:
                    dh = None
                if dh and dh[1] >= 1.0:
                    try:
                        card = kg_adapter.build_card(dh[0], max_neighbors=max_neighbors)
                        if self._is_card_relevant_for_query(card, question_for_relevance):
                            found_cards = [card]
                            direct_hit_used = 1
                            # kein break hier: wir sammeln trotzdem noch weitere Karten über Suche
                    except Exception:
                        pass

        # 2) Normale Suche über Queries
        for sq in queries:
            for kg_adapter in self.kg_adapters:
                try:
                    hits = kg_adapter.search_nodes(sq, top_k=top_k, use_fulltext=self.use_fulltext)
                    all_node_scores.extend(hits)
                except Exception:
                    continue

        # Deduplizieren & sortieren
        all_node_scores.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        merged: List[Tuple[str, float]] = []
        for node_id, score in all_node_scores:
            if node_id not in seen_ids:
                merged.append((node_id, score))
                seen_ids.add(node_id)

        for node_id, score in merged:
            if len(found_cards) >= top_k:
                break
            for kg_adapter in self.kg_adapters:
                try:
                    card = kg_adapter.build_card(node_id, max_neighbors=max_neighbors)
                    if self._is_card_relevant_for_query(card, question_for_relevance):
                        found_cards.append(card)
                        break
                except ValueError:
                    continue

        # 3) Stats
        unique_entities = len({c.uri for c in found_cards if c.uri})
        top1_score = merged[0][1] if merged else 0.0
        coverage = min(1.0, (len(found_cards) / max(1, top_k)))
        stats = {
            "coverage": coverage,
            "top1_score": top1_score,
            "direct_hit_count": float(direct_hit_used),
            "unique_entities": float(unique_entities),
        }
        return found_cards, stats, merged

    def answer(self, question: str, history: str = "", top_k: int = 5, max_neighbors: int = 12):
        # === Smalltalk-Shortcut (vor RAG) ===
        st_intent = Smalltalk.detect(question)
        if st_intent:
            st_text = Smalltalk.respond(st_intent)
            meta = {
                "mode": "smalltalk",
                "cards": [],
                "prompt": "(smalltalk)",
                "sources": [],
                "evaluation": {},
                "subqueries": [question],
            }
            return st_text, meta

        # Shortcut: LLM-only
        if not self.use_kg:
            return self._answer_llm_only(question, history)

        # Vorverarbeitung: Anker schützen (nur Originalfrage, KEINE Query-Umformulierung)
        retrieval_query = _preserve_anchors(question)

        found_cards: List[AssertionCard] = []
        is_follow_up = "preis" in _lc(question) or "kosten" in _lc(question)
        product_from_history = None
        subqueries: List[str] = []  # nur für Reporting
        prompt: str = ""

        if is_follow_up and self.use_direct_hit:
            # Produkt aus letzter Assistenten-Antwort ziehen
            last_assistant_response = next((line for line in reversed(history.split('\n')) if line.startswith("Assistent:")), None)
            if last_assistant_response:
                product_from_history = _extract_product_name(last_assistant_response)



        if is_follow_up and product_from_history and self.use_direct_hit:
            # Harte Fokussierung bei Folgefragen zu Preis/Kosten
            cards1, stats1, hits1 = self._search_cards(
                queries=[_preserve_anchors(product_from_history)],
                question_for_relevance=product_from_history,
                top_k=top_k,
                max_neighbors=max_neighbors,
                product_name_for_direct=product_from_history
            )
        else:
            cards1, stats1, hits1 = self._search_cards(
                queries=[retrieval_query],
                question_for_relevance=question,
                top_k=top_k,
                max_neighbors=max_neighbors,
                product_name_for_direct=_extract_product_name(question)
            )

        found_cards.extend(cards1)

        # -------------------------------
        # Confidence-Gating (ohne Extra-Klasse)
        # -------------------------------
        def _should_augment(stats: Dict[str, float]) -> bool:
            # Aug nur, wenn Anker schwach oder Coverage niedrig
            if stats.get("direct_hit_count", 0.0) >= 1.0 and stats.get("top1_score", 0.0) >= 0.75:
                return False
            if stats.get("coverage", 0.0) >= 0.60:
                return False
            if stats.get("unique_entities", 0.0) <= 1.0:
                return True
            # Standard: konservativ aktivieren
            return True

        aug_needed = self.use_prompt_augmentation and (self.augmentor is not None) and _should_augment(stats1)

        logger.info({
            "phase": "gating",
            "aug_needed": aug_needed,
            "stats_phase1": stats1
        })

        # ===============================
        # PHASE 2: Optionale Augmentation (zusätzliche Subqueries)
        #  - Originalquery bleibt unberührt (nie ersetzt!)
        #  - Wir fügen nur 1–2 sehr kurze Subqueries hinzu
        # ===============================
        if aug_needed:
            try:
                aug_subqs = (self.augmentor.rewrite_queries(question) or [])
            except Exception:
                aug_subqs = []

            # Filter: keine Dubletten, nicht identisch zur Originalfrage
            aug_subqs = [s for s in aug_subqs if s.strip() and s.strip().lower() != question.strip().lower()]
            # Max. 2 zusätzliche, und Anker schützen
            aug_subqs = [_preserve_anchors(s) for s in aug_subqs[:2]]

            if aug_subqs:
                cards2, stats2, hits2 = self._search_cards(
                    queries=aug_subqs,
                    question_for_relevance=question,
                    top_k=top_k,
                    max_neighbors=max_neighbors,
                    product_name_for_direct=_extract_product_name(question)
                )
                # Merge (per URI deduplizieren)
                seen_uris = {c.uri for c in found_cards if c.uri}
                for c in cards2:
                    if c.uri and c.uri in seen_uris:
                        continue
                    found_cards.append(c)
                    if len(found_cards) >= top_k:
                        break

                # Subqueries für Meta
                subqueries = [retrieval_query] + aug_subqs
            else:
                subqueries = [retrieval_query]
        else:
            subqueries = [retrieval_query]

        # 3) Finalisierung
        # Nochmals per URI deduplizieren (robust)
        seen_uris = set()
        final_cards: List[AssertionCard] = []
        for card in found_cards:
            if card.uri in seen_uris:
                continue
            final_cards.append(card)
            if card.uri:
                seen_uris.add(card.uri)

        is_relevant = bool(final_cards)
        if not is_relevant:
            # Fallback: Allgemeinwissen (deutlich gekennzeichnet)
            topic = _extract_product_name(question) or question.strip()
            prompt = self._build_prompt_general_knowledge_fallback(question, history, topic)
            answer = self.llm.generate(prompt) if self.llm else "(LLM nicht konfiguriert)\n\n" + prompt
            sources: List[str] = []
            evaluation_result: Dict[str, Any] = {}
        else:
            prompt = self._build_prompt_with_facts(question, final_cards, history)
            answer = self.llm.generate(prompt) if self.llm else "(LLM nicht konfiguriert)\n\n" + prompt
            sources = self._sources_from_cards(final_cards)
            evaluator = Evaluation(answer, [c.to_dict() for c in final_cards], question)
            evaluation_result = evaluator.evaluate()
            if evaluation_result.get("consistency", {}).get("is_consistent"):
                # Konsistent → Quellenliste anhängen (falls nicht schon im Text)
                answer = answer.rstrip()
                if sources:
                    answer += "\n\nQuellen:\n- " + "\n- ".join(sources)
            else:
                # Inkonsistent → Fallback-Antwort (streng und kurz)
                fb_prompt = self._build_prompt_fallback(question, history)
                fb_answer = self.llm.generate(fb_prompt) if self.llm else "(LLM nicht konfiguriert)\n\n" + fb_prompt
                answer = fb_answer
                sources = []

        return answer, {
            "cards": [c.to_dict() for c in final_cards],
            "prompt": prompt,
            "sources": sources,
            "evaluation": evaluation_result,
            "subqueries": subqueries,
        }

def run_demo_conversation(rag: "RAGEngine", scenario_csv: str = "all") -> str:
    """
    Spielt 10er-Fragenpakete als Demo-Konversation(en) ab.
    Szenarien: 'lpd', 'mtf', 'general' oder 'all' (Default).
    Mehrere Szenarien kommagetrennt möglich, z. B. "lpd,mtf".
    """
    # Rückwärtskompatible Wrapper-Funktion (rag.answer)
    def _answer_fn(q, hist):
        return rag.answer(q, history=hist)
    return _run_demo_with_answer_fn(_answer_fn, scenario_csv)

def _run_demo_with_answer_fn(answer_fn, scenario_csv: str = "all") -> str:
    """
    Interner Helper, um unterschiedliche Pipelines im Demo-Run austauschbar zu machen.
    """
    demo_sets: Dict[str, List[str]] = {
        "lpd": [
            "Was ist eigentlich LebenPflegeDigital?",
            "Wer steckt denn hinter dem Projekt?",
            "Und richtet sich das Angebot mehr an Pflegekräfte oder an Angehörige?",
            "Kannst du mir mal ein Beispiel nennen, was ich dort finde?",
            "Wenn ich mich über digitale Hilfsmittel informieren will, wo klicke ich am besten hin?",
            "Gibt es auch Veranstaltungen oder Workshops, die dort angekündigt werden?",
            "Und sind die Angebote nur für Berlin oder bundesweit?",
            "Wie unterscheidet sich LPD von der Technikfinder-Seite?",
            "Falls ich mehr Unterstützung will, wie kann ich die Betreiber kontaktieren?",
            "Kannst du die Infos noch mal ganz knapp zusammenfassen?"
        ],
        "mtf": [
            "Was ist das Kissen Viktor?",
            "Und wie funktioniert das konkret im Alltag?",
            "Hat es auch eine Notruf-Funktion?",
            "Klingt spannend – gibt es vergleichbare Produkte?",
            "Wie unterscheidet es sich zum Beispiel von Caru Care?",
            "Gibt es zu diesen Produkten auch Erfahrungsberichte?",
            "Und wie teuer ist das ungefähr?",
            "Müsste ich das selbst kaufen oder wird sowas von der Pflegekasse übernommen?",
            "Okay, und wenn ich es bestellen will – wo finde ich den Anbieter?",
            "Kannst du mir eine Liste mit 2–3 ähnlichen Produkten nennen, die sich für ältere Menschen lohnen würden?"
        ],
        "general": [
            "Was genau ist eigentlich der Navel?",
            "Und kann der auch an Medikamente erinnern?",
            "Wie funktioniert die Dialoggestaltung mit dem Navel?",
            "Aber mal ehrlich: könnte ich den auch als DJ-Assistent benutzen?",
            "Welche Vorteile bringt er im Vergleich zu Caru Care?",
            "Und wie würde er reagieren, wenn jemand stürzt?",
            "Kannst du mir das Ganze in einem Beispiel-Dialog mit einer Pflegekraft zeigen?",
            "Wer hat den Navel eigentlich entwickelt?",
            "Okay, und jetzt sag mal – was kostet ein Döner in Berlin?",
            "Trotzdem spannend, kannst du die Hauptfunktionen des Navel noch mal kurz zusammenfassen?"
        ],
        "strictdialog": [
            "Ich hab von LebenPflegeDigital gehört — was genau ist das kurz?",
            "Und wo finde ich dort genau etwas zum Thema Sturzerkennung, ohne mich zu verlaufen?",
            "Nenn mir aus genau diesem Bereich drei konkrete Lösungen, nur die Namen.",
            "Von den dreien — welche kommt typischerweise auch ohne dauerhaftes WLAN aus?",
            "Gut, dann nimm die zweite — braucht die ein Funk- oder ein WLAN-Gateway?",
            "Wenn ich die bei meiner Mutter einsetze und sie hat kein WLAN — was muss ich zusätzlich besorgen?",
            "Angenommen, wir holen später doch Internet: Wie ändert sich dann die Einrichtung bei genau der?",
            "Kann die auch an Medikamente erinnern und wie stelle ich das bei der konkret ein?",
            "Fasse mir für genau dieses Modell in 2 Sätzen die wichtigsten Vor- und Nachteile für alleinlebende Seniorinnen zusammen.",
            "Notiere mir den exakten Namen von dem, das wir gewählt haben, und gib mir einen Such-String, den ich auf der LPD-Seite eintippen kann."
        ]
    }



    order = ["lpd", "mtf", "general","strictdialog"]
    if scenario_csv is None or scenario_csv.strip().lower() in {"all", "*"}:
        scenarios = order
    else:
        scenarios = [s.strip().lower() for s in scenario_csv.split(",")]
        scenarios = [s for s in scenarios if s in demo_sets] or order  # Fallback auf alle

    lines: List[str] = []

    for scen in scenarios:
        questions = demo_sets[scen]
        history: List[str] = []  # pro Szenario eigener Verlauf
        lines.append("=" * 60)
        lines.append(f" DEMO-KONVERSATION — {scen.upper()} ")
        lines.append("=" * 60 + "\n")

        for q in questions:
            history_str = "\n".join(history)
            answer, meta = answer_fn(q, history_str)
            lines.append(f"Nutzer: {q}")
            for line in (answer or "").strip().splitlines():
                lines.append(f"Assistent: {line}")

            sources = meta.get("sources", []) if isinstance(meta, dict) else []
            if sources:
                lines.append("Quellen:")
                for s in sources:
                    lines.append(f"- {s}")

            lines.append("-" * 60)
            history.append(f"Nutzer: {q}")
            history.append(f"Assistent: {(answer or '').strip()}")

        lines.append("")  # Leerzeile nach jedem Szenario

    return "\n".join(lines)

# --- CLI ---
def print_evaluation(evaluation_result: Dict[str, Any]):
    """
    Hilfsfunktion zum schönen Drucken der Evaluierungsergebnisse.
    """
    if not evaluation_result:
        print("\n=== Evaluierung ===\n")
        print("Keine Evaluierung durchgeführt (keine relevanten Fakten gefunden).")
        return
    consistency = evaluation_result.get("consistency", {})
    relevance = evaluation_result.get("relevance", None)
    print("\n=== Evaluierungsergebnis ===\n")
    print("--- Konsistenz mit Fakten ---")
    if consistency.get("is_consistent"):
        print(f"Status: \033[92m[✓] Konsistent\033[0m")
    else:
        print(f"Status: \033[91m[✗] Inkonsistent (mögliche Halluzination)\033[0m")
    print(f"Score: {consistency.get('score', 0.0):.2f}")
    if consistency.get("unverified_sentences"):
        print("Unverifizierte Sätze:")
        for sentence in consistency["unverified_sentences"]:
            print(f"- {sentence}")
    else:
        print("Alle Sätze konnten mit den Fakten verifiziert werden.")
    print("\n--- Relevanz zur Frage ---")
    if relevance:
        print(f"Relevanz: \033[92m[✓] Relevante Antwort\033[0m")
    else:
        print(f"Relevanz: \033[91m[✗] Irrelevante Antwort\033[0m")


def print_cards(cards: List[Dict[str, Any]]):
    """Hilfsfunktion zum Drucken der Faktenkarten."""
    print("\n=== Gefundene Faktenkarten ===\n")
    if not cards:
        print("Es wurden keine relevanten Faktenkarten gefunden.")
    else:
        for i, card in enumerate(cards):
            title = card.get('title', '(ohne Titel)')
            snippet = card.get('snippet', None)
            facts_count = len(card.get('facts', []))
            print(f"{i+1}. Titel: {title}")
            if snippet:
                print(f"   Ausschnitt: {snippet[:70]}..." if len(snippet) > 70 else f"   Ausschnitt: {snippet}")
            else:
                print("   Ausschnitt: (kein Ausschnitt)")
            print(f"   Fakten: {facts_count}")
            print("-" * 20)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG ↔︎ LLM Adapter (Neo4j RAG) mit Prompt Augmentation und modularen Modi")
    p.add_argument("question", nargs="?", help="Frage, z. B. 'Was ist das DAI-Labor?'.")
    p.add_argument("--demochat", action="store_true", help="Startet einen einfachen Chat-Modus ohne Debugging-Ausgaben.")
    p.add_argument("--democon", action="store_true", help="Startet eine automatische Demo-Konversation mit 10 Fragen.")

    # 🔥 NEU: Matrix-Modus
    p.add_argument("--demomatrix", action="store_true", help="Vergleich: LLM vs. RAG vs. RAG+Aug nebeneinander (schreibt eine TXT).")
    p.add_argument("--matrix-colwidth", type=int, default=80, help="Spaltenbreite je Antwort (Standard: 80).")
    p.add_argument("--matrix-out", type=str, default="", help="Pfad für die TXT-Ausgabe (optional, sonst auto mit Zeitstempel).")

    p.add_argument("--no-llm", action="store_true", help="Nur Karten/Prompt anzeigen")
    p.add_argument("--k", type=int, default=5, help="Top-K Knoten")
    p.add_argument("--nbrs", type=int, default=12, help="Max. Nachbarfakten")
    p.add_argument("--short", action="store_true", help="Kürzere Antwort")
    p.add_argument("--debug", action="store_true", help="Vollständige JSON-Metadaten anzeigen")
    p.add_argument("--eval-only", action="store_true", help="Nur Antwort und Evaluierung anzeigen")

    # Augmentation/Metriken
    p.add_argument("--no-augment", dest="no_augment", action="store_true", help="Prompt Augmentation deaktivieren")
    p.add_argument("--fewshot", action="store_true", help="Ein Mini-Fewshot voranstellen")
    p.add_argument("--facts-budget-chars", type=int, default=4000, help="Max. Zeichen für den Faktenblock")

    # Szenarien
    p.add_argument("--scenario", type=str, default="all", help="lpd, mtf, general, strictdialog (kommasepariert) oder 'all'")

    # Pipeline-Modus für die bisherigen Flows
    p.add_argument("--mode", type=str, default="rag-aug",
                   help="Pipeline-Modus: llm, rag, rag-aug (Standard: rag-aug). Für --democon: mehrere per Komma.")
    return p.parse_args()



def _build_adapters() -> List[KGAdapter]:
    db_name_1 = _env("NEO4J_DATA_DB")
    db_name_2 = _env("NEO4J_DATA_DB_2")
    db_name_3 = _env("NEO4J_DATA_DB_3")
    labels_db2 = ["Produkt"]
    search_props_db2 = ["titel", "hatTitel", "hatBeschreibung", "hatEinsatzzweck"]
    kg_adapter_1 = KGAdapter(
        _env("NEO4J_URI"), _env("NEO4J_USER"), _env("NEO4J_PASSWORD"),
        data_db=db_name_1
    )
    kg_adapter_2 = KGAdapter(
        _env("NEO4J_URI"), _env("NEO4J_USER"), _env("NEO4J_PASSWORD"),
        data_db=db_name_2,
        additional_labels=labels_db2,
        search_properties=search_props_db2
    )
    kg_adapter_3 = KGAdapter(
        _env("NEO4J_URI"), _env("NEO4J_USER"), _env("NEO4J_PASSWORD"),
        data_db=db_name_3
    )
    kg_adapters_list = [kg_adapter_2, kg_adapter_3, kg_adapter_1]
    for adapter in kg_adapters_list:
        adapter.ensure_fulltext_index()
    return kg_adapters_list
def _wrap(s: str, width: int) -> List[str]:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\s+", " ", s)
    return textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False) or [""]

def _render_row(cells: List[str], widths: List[int]) -> List[str]:
    wrapped = [_wrap(c, w) for c, w in zip(cells, widths)]
    h = max(len(x) for x in wrapped)
    lines = []
    for i in range(h):
        parts = []
        for col, w in zip(wrapped, widths):
            parts.append((col[i] if i < len(col) else "").ljust(w))
        lines.append(" │ ".join(parts))
    return lines

def _ts_filename(base: str) -> str:
    if (base or "").strip():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"demo_matrix_{ts}.txt"

def run_demo_matrix(args: argparse.Namespace):
    """
    Vergleich: Spalten = [Frage | LLM | RAG | RAG+Aug]
    Schreibt eine übersichtliche TXT-Datei auf die Platte.
    """
    demo_sets: Dict[str, List[str]] = {
        "lpd": [
            "Was ist eigentlich LebenPflegeDigital?",
            "Wer steckt denn hinter dem Projekt?",
            "Und richtet sich das Angebot mehr an Pflegekräfte oder an Angehörige?",
            "Kannst du mir mal ein Beispiel nennen, was ich dort finde?",
            "Wenn ich mich über digitale Hilfsmittel informieren will, wo klicke ich am besten hin?",
            "Gibt es auch Veranstaltungen oder Workshops, die dort angekündigt werden?",
            "Und sind die Angebote nur für Berlin oder bundesweit?",
            "Wie unterscheidet sich LPD von der Technikfinder-Seite?",
            "Falls ich mehr Unterstützung will, wie kann ich die Betreiber kontaktieren?",
            "Kannst du die Infos noch mal ganz knapp zusammenfassen?"
        ],
        "mtf": [
            "Was ist das Kissen Viktor?",
            "Und wie funktioniert das konkret im Alltag?",
            "Hat es auch eine Notruf-Funktion?",
            "Klingt spannend – gibt es vergleichbare Produkte?",
            "Wie unterscheidet es sich zum Beispiel von Caru Care?",
            "Gibt es zu diesen Produkten auch Erfahrungsberichte?",
            "Und wie teuer ist das ungefähr?",
            "Müsste ich das selbst kaufen oder wird sowas von der Pflegekasse übernommen?",
            "Okay, und wenn ich es bestellen will – wo finde ich den Anbieter?",
            "Kannst du mir eine Liste mit 2–3 ähnlichen Produkten nennen, die sich für ältere Menschen lohnen würden?"
        ],
        "general": [
            "Was genau ist eigentlich der Navel?",
            "Und kann der auch an Medikamente erinnern?",
            "Wie funktioniert die Dialoggestaltung mit dem Navel?",
            "Aber mal ehrlich: könnte ich den auch als DJ-Assistent benutzen?",
            "Welche Vorteile bringt er im Vergleich zu Caru Care?",
            "Und wie würde er reagieren, wenn jemand stürzt?",
            "Kannst du mir das Ganze in einem Beispiel-Dialog mit einer Pflegekraft zeigen?",
            "Wer hat den Navel eigentlich entwickelt?",
            "Okay, und jetzt sag mal – was kostet ein Döner in Berlin?",
            "Trotzdem spannend, kannst du die Hauptfunktionen des Navel noch mal kurz zusammenfassen?"
        ],
        "strictdialog": [
            "Ich hab von LebenPflegeDigital gehört — was genau ist das kurz?",
            "Und wo finde ich dort genau etwas zum Thema Sturzerkennung, ohne mich zu verlaufen?",
            "Nenn mir aus genau diesem Bereich drei konkrete Lösungen, nur die Namen.",
            "Von den dreien — welche kommt typischerweise auch ohne dauerhaftes WLAN aus?",
            "Gut, dann nimm die zweite — braucht die ein Funk- oder ein WLAN-Gateway?",
            "Wenn ich die bei meiner Mutter einsetze und sie hat kein WLAN — was muss ich zusätzlich besorgen?",
            "Angenommen, wir holen später doch Internet: Wie ändert sich dann die Einrichtung bei genau der?",
            "Kann die auch an Medikamente erinnern und wie stelle ich das bei der konkret ein?",
            "Fasse mir für genau dieses Modell in 2 Sätzen die wichtigsten Vor- und Nachteile für alleinlebende Seniorinnen zusammen.",
            "Notiere mir den exakten Namen von dem, das wir gewählt haben, und gib mir einen Such-String, den ich auf der LPD-Seite eintippen kann."
        ]
    }

    order = ["lpd", "mtf", "general", "strictdialog"]
    if args.scenario is None or args.scenario.strip().lower() in {"all", "*"}:
        scenarios = order
    else:
        scenarios = [s.strip().lower() for s in args.scenario.split(",")]
        scenarios = [s for s in scenarios if s in demo_sets] or order

    kg_adapters_list = _build_adapters()
    try:
        llm_client = None if args.no_llm else LLMClient(short=args.short)

        modes: List[Tuple[str, str]] = [
            ("LLM", "llm"),
            ("RAG", "rag"),
            ("RAG+Aug", "rag-aug"),
        ]
        engines: Dict[str, RAGEngine] = {m: _build_engine_for_mode(m, kg_adapters_list, llm_client, args) for _, m in modes}
        histories: Dict[str, List[str]] = {m: [] for _, m in modes}

        colw_q = 40
        colw_a = max(40, int(args.matrix_colwidth))
        widths = [colw_q, colw_a, colw_a, colw_a]
        header = ["Frage", "LLM", "RAG", "RAG+Aug"]

        lines: List[str] = []
        title = f"Vergleichsmatrix  (Spaltenbreite: {colw_a})"
        lines.append(title)
        lines.append("=" * len(title))
        lines += _render_row(header, widths)
        lines.append("-" * (sum(widths) + 3*(len(widths)-1)))

        for scen in scenarios:
            lines.append("")
            scen_title = f"SZ.: {scen.upper()}"
            lines.append(scen_title)
            lines.append("-" * len(scen_title))
            for q in demo_sets[scen]:
                row_ans = {}
                for _, m in modes:
                    hist = "\n".join(histories[m])
                    ans, _meta = engines[m].answer(q, history=hist, top_k=args.k, max_neighbors=args.nbrs)
                    ans = (ans or "").strip()
                    row_ans[m] = ans
                    histories[m].extend([f"Nutzer: {q}", f"Assistent: {ans}"])
                cells = [q, row_ans["llm"], row_ans["rag"], row_ans["rag-aug"]]
                lines += _render_row(cells, widths)
                lines.append("-" * (sum(widths) + 3*(len(widths)-1)))

        out_path = _ts_filename(args.matrix_out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nTXT gespeichert: {out_path}")
        return out_path
    finally:
        for a in kg_adapters_list:
            try:
                a.close()
            except Exception:
                pass

# NEU: Engine-Bau nach Modus
def _build_engine_for_mode(mode: str, kg_adapters_list: List[KGAdapter], llm_client: Optional[LLMClient], args: argparse.Namespace) -> RAGEngine:
    mode = (mode or "rag-aug").lower().strip()
    # Defaults je Modus
    if mode == "llm":
        return RAGEngine(
            kg_adapters=kg_adapters_list,
            llm=llm_client,
            augmentor=None,
            facts_budget_chars=args.facts_budget_chars,
            use_fewshots=args.fewshot,
            disable_augmentation=True,
            use_kg=False, use_fulltext=False, use_direct_hit=False, use_prompt_augmentation=False
        )
    elif mode == "llm-aug":
        return RAGEngine(
            kg_adapters=kg_adapters_list,
            llm=llm_client,
            augmentor=PromptAugmentor(llm_client),
            facts_budget_chars=args.facts_budget_chars,
            use_fewshots=args.fewshot,
            disable_augmentation=False,
            use_kg=False, use_fulltext=False, use_direct_hit=False, use_prompt_augmentation=True
        )
    elif mode == "rag":
        return RAGEngine(
            kg_adapters=kg_adapters_list,
            llm=llm_client,
            augmentor=None,
            facts_budget_chars=args.facts_budget_chars,
            use_fewshots=args.fewshot,
            disable_augmentation=True,
            use_kg=True, use_fulltext=True, use_direct_hit=False, use_prompt_augmentation=False
        )
    else:  # rag-aug (voll)
        return RAGEngine(
            kg_adapters=kg_adapters_list,
            llm=llm_client,
            augmentor=PromptAugmentor(llm_client),
            facts_budget_chars=args.facts_budget_chars,
            use_fewshots=args.fewshot,
            disable_augmentation=False,
            use_kg=True, use_fulltext=True, use_direct_hit=True, use_prompt_augmentation=True
        )

def main_loop(args: argparse.Namespace):
    kg_adapters_list = _build_adapters()
    llm_client = None if args.no_llm else LLMClient(short=args.short)

    rag = _build_engine_for_mode(args.mode, kg_adapters_list, llm_client, args)
    history: List[str] = []

    print("\nWillkommen im Chat-Modus. Tippen Sie Ihre Frage ein.")
    print("Um zu beenden, tippen Sie 'exit' oder 'quit'.\n")

    try:
        while True:
            question = input("\033[92mNutzer:\033[0m ")
            if question.lower() in ("exit", "quit"):
                break

            question = question.encode('utf-8', 'ignore').decode('utf-8')
            history_str = "\n".join(history)
            answer, meta = rag.answer(question, history=history_str, top_k=args.k, max_neighbors=args.nbrs)
            print("\033[91mNavel:\033[0m", answer.strip())
            history.append(f"Nutzer: {question}")
            history.append(f"Assistent: {answer.strip()}")
            if args.debug:
                print("\n=== Metadaten (Debug) ===\n")
                print(json.dumps(meta, ensure_ascii=False, indent=2))
    finally:
        for adapter in kg_adapters_list:
            adapter.close()
        print("\nChat beendet. Auf Wiedersehen!")

def main():
    args = parse_args()
    if args.demochat:
        main_loop(args)
    elif args.democon:
        kg_adapters_list = _build_adapters()
        try:
            llm_client = None if args.no_llm else LLMClient(short=args.short)
            modes = [m.strip() for m in (args.mode or "rag-aug").split(",") if m.strip()]
            for idx, mode in enumerate(modes, start=1):
                rag = _build_engine_for_mode(mode, kg_adapters_list, llm_client, args)
                def _answer_fn(q, hist):
                    return rag.answer(q, history=hist, top_k=args.k, max_neighbors=args.nbrs)
                transcript = _run_demo_with_answer_fn(_answer_fn, args.scenario)
                print("\n" + "="*60)
                print(f" DEMO-KONVERSATION  [{idx}/{len(modes)}]  MODE: {mode.upper()} ")
                print("="*60 + "\n")
                for line in transcript.split("\n"):
                    if line.startswith("Nutzer:"):
                        print(f"\n{line}")
                    elif line.startswith("Assistent:"):
                        print(f"{line}\n" + "-"*60)
                print("\n" + "="*60 + "\n")
        finally:
            for adapter in kg_adapters_list:
                adapter.close()
    elif args.demomatrix:
        run_demo_matrix(args)
    elif args.question is None:
        print("Fehler: Eine Frage muss übergeben werden, wenn --demochat, --democon oder --demomatrix nicht verwendet wird.")
        print("Verwendung: python ihr_skript.py \"Ihre Frage\" oder python ihr_skript.py --demochat")
        sys.exit(1)
    else:
        kg_adapters_list = _build_adapters()
        try:
            llm_client = None if args.no_llm else LLMClient(short=args.short)
            rag = _build_engine_for_mode(args.mode, kg_adapters_list, llm_client, args)
            question = args.question.encode('utf-8', 'ignore').decode('utf-8')
            answer, meta = rag.answer(question, top_k=args.k, max_neighbors=args.nbrs)
            if args.eval_only:
                print("\n=== Antwort ===\n")
                print(answer.strip())
                print_evaluation(meta.get("evaluation", {}))
            elif args.debug:
                print_cards(meta.get('cards', []))
                print("\n=== Antwort ===\n")
                print(answer.strip())
                print("\n=== Metadaten (Debug) ===\n")
                print(json.dumps(meta, ensure_ascii=False, indent=2))
            else:
                print_cards(meta.get('cards', []))
                print("\n=== Antwort ===\n")
                print(answer.strip())
        finally:
            for adapter in kg_adapters_list:
                adapter.close()



if __name__ == "__main__":
    main()
