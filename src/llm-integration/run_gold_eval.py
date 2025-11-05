# Evaluationsskript für KG/RAG-Modi
# Zweck: Gold-Fragen laden, alle Modi (llm, rag, …) abfragen, Metriken (Correctness, Hallu, Coverage, Latenz, Enthaltung) berechnen
# Ausgabe: CSV oder Pretty oder JSONL-Sammeldatei
# Status: wird aktuell verwendet

import json, os, sys, re, argparse, time
from typing import List, Dict, Any, Tuple, Optional


RESET = "\033[0m"; BOLD = "\033[1m"
C_RED = "\033[31m"; C_YEL = "\033[33m"; C_GRN = "\033[32m"; C_CYAN = "\033[36m"; C_GRAY = "\033[90m"
def _supports_color() -> bool: return sys.stdout.isatty()
def _cm(v: float, kind: str, enable: bool) -> str:
    if not enable:
        if kind in ("latency_ms", "abstain", "abst_ok"):
            return f"{int(v)}"
        return f"{v:.3f}"
    if kind in ("correctness","coverage"):
        color = C_GRN if v >= 0.60 else (C_YEL if v >= 0.30 else C_RED)
    elif kind == "hallucination":
        color = C_GRN if v <= 0.20 else (C_YEL if v <= 0.50 else C_RED)
    elif kind == "latency_ms":
        color = C_GRN if v <= 1000 else (C_YEL if v <= 3000 else C_RED)
        return f"{color}{int(v):d}{RESET}"
    elif kind == "abstain":
        color = C_RED if int(v)==1 else C_GRN
        return f"{color}{int(v)}{RESET}"
    elif kind == "abst_ok":
        color = C_GRN if int(v)==1 else C_RED
        return f"{color}{int(v)}{RESET}"
    else:
        color = None
    return f"{color}{v:.3f}{RESET}"

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    if n <= 1: return "" if not s else "…"
    return (s[:n-1] + "…") if len(s) > n else s

# Adapter laden
try:
    import kgadapterv2 as kg
except Exception as e:
    print("Fehler: kgadapterv2 konnte nicht importiert werden.\nBitte lege 'kgadapterv2.py' neben dieses Skript oder setze PYTHONPATH.", file=sys.stderr)
    raise

MODES = ["llm","llm-aug", "rag", "rag-aug"]

# Text-Helfer
def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zäöüß0-9]+", normalize(s))

def normalize_compact(s: str) -> str:
    """Kompakt-Normalisierung (ignoriert Leer- und Sonderzeichen)."""
    return re.sub(r"[^a-z0-9äöüß]", "", (s or "").strip().lower())

# sehr einfache deutsche Suffix-Kappung
_DE_SUFFIXES_LONG = ("lichkeit","igkeit","keiten","heiten","tionen","tionen","ungen","innen")
_DE_SUFFIXES_SHORT = ("tion","heit","keit","ung","chen","lein","nist","ismus","ist","isch","igen","igen","end","ern","ens","en","n","e","er","em","es","s")
def stem_de(w: str) -> str:
    w = w.lower()
    for suf in _DE_SUFFIXES_LONG:
        if w.endswith(suf) and len(w) > len(suf)+2:
            return w[:-len(suf)]
    for suf in _DE_SUFFIXES_SHORT:
        if w.endswith(suf) and len(w) > len(suf)+2:
            return w[:-len(suf)]
    return w

def stem_tokens(tokens: List[str]) -> List[str]:
    return [stem_de(t) for t in tokens]

# Levenshtein für fuzzy Match
def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return dp[lb]

def _sim_lev(a: str, b: str) -> float:
    m = max(1, len(a), len(b))
    return 1.0 - (_levenshtein(a,b) / m)

# Enthaltungs-Patterns (DE/EN)
_ABSTAIN_PATTERNS_RAW = [
    r"\bich\s+bin\s+nicht\s+in\s+der\s+lage\b",
    r"\b(?:ich\s+)?habe\s+keinen?\s+zugriff\s+auf\s+(?:echtzeit|live|aktuelle)\s*daten\b",
    r"\bkeinen?\s+zugriff\s+auf\s+(?:echtzeit|live|aktuelle)\s*daten\b",
    r"\b(?:es\s+ist|das\s+ist)\s+nicht\s+möglich\b.*\b(?:zu\s+(?:sagen|nennen|angeben|liefern|bestätigen))\b",
    r"\bkann\s+(?:ich\s+)?(?:dir|ihnen)?\s*keine?\s+(?:auskunft|angaben?|details?|informationen|infos|daten)\s+(?:geben|machen|liefern|nennen)\b",
    r"\bkann\s+(?:ich\s+)?(?:dir|ihnen)?\s*nicht\s+(?:sagen|beantworten|angeben|nennen|liefern|bestätigen)\b",
    r"\bes\s+tut\s+mir\s+leid[,;:]?\s*aber\b.*\b(?:kein(?:e|en)?\s+zugriff|keine\s+(?:information|informationen|daten|belege)|nicht\s+möglich|kann\s+.*nicht)\b",
    r"\b(?:aus\s+)?datenschutz(?:gründen)?\b.*\b(?:kann|darf)\s+(?:ich\s+)?(?:nicht|keine)\b",
    r"\b(?:privat|vertraulich)\b.*\b(?:nicht\s+(?:verfügbar|öffentlich|geteilt)|kann\s+nicht\s+(?:teilen|offenlegen|preisgeben))\b",
    r"\bnicht\s+öffentlich\s+(?:zugänglich|verfügbar|einsehbar)\b",
    r"\bkeine\s+verlässlichen?\s+(?:quellen|informationen|infos|daten)\b",
    r"\b(?:konnte|kann)\s+keine\s+(?:informationen|infos|daten)\s+finden\b",
    r"\bkeine\s+informationen\s+über\b",
    r"\bmir\s+liegen\s+(?:derzeit|aktuell)?\s*keine\s+(?:spezifischen?|weiteren?|genauen?)?\s+(?:fakten|informationen|infos|daten)(?:\s+\w+){0,5}?\s+vor\b",
    r"\bmir\s+fehlen\s+(?:noch\s+)?(?:konkrete|ausreichende|relevante|hinreichende)?\s*fakten(?:\s+zu\b|\b)",
    r"\b(?:außerhalb|nicht\s+im)\s+(?:thema|scope|zuständigkeitsbereich|bereich)\b",
    r"\b(?:nicht\s+anwendbar|nicht\s+zutreffend|trifft\s+nicht\s+zu)\b",
    r"\bwäre\s+reine\s\spekulation\b",
    r"\b(?:kann|können)\s+die\s+zukunft\s+nicht\s+(?:vorhersagen|prognostizieren)\b",
    r"\bkeine\s+ahnung\b",
    r"\bk\.?\s*a\.?\b",
    r"\bi\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|have|provide|share|disclose|confirm)\b.*\b(?:information|data|answer|details)\b",
    r"\bno\s+access\s+to\s+(?:real[- ]?time|live|current)\s+data\b",
    r"\bi\s+don't\s+have\s+(?:real[- ]?time|live|current)\s+access\b",
    r"\bnot\s+public(?:ly)?\s+available\b",
    r"\bi\s+am\s+not\s+able\s+to\b.*\b(?:provide|share|disclose|answer|confirm)\b",
    r"\b(?:for|due\s+to)\s+privacy\b.*\b(?:cannot|can't|won't)\b",
    r"\bit\s+is\s+not\s+possible\s+to\b.*\b(?:say|provide|answer|confirm)\b",
    r"\bwould\s+be\s+pure\s+speculation\b",
    r"\boutside\s+(?:my|the)\s+(?:scope|domain|knowledge)\b",
    r"\bno\s+idea\b",
    r"\btbd\b",
    r"\btba\b",
    r"\bto\s+be\s+(?:determined|announced)\b",
]

_ABSTAIN_PATTERNS = [re.compile(p, re.UNICODE | re.IGNORECASE) for p in _ABSTAIN_PATTERNS_RAW]

def abstain_match(ans: str) -> Tuple[int, Optional[str]]:
    s = normalize(ans)
    if not s or len(s) < 5:
        return 1, "<empty/too short>"
    for p_raw, p in zip(_ABSTAIN_PATTERNS_RAW, _ABSTAIN_PATTERNS):
        if p.search(s):
            return 1, p_raw
    return 0, None

def abstains(ans: str) -> int:
    m, _ = abstain_match(ans)
    return m

# Korrektheits-Bausteine
def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    A, B = set(a_tokens), set(b_tokens)
    u = len(A|B); i = len(A&B)
    return i/u if u else 0.0

def jaccard_stemmed(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    A = set(stem_tokens(a_tokens))
    B = set(stem_tokens(b_tokens))
    u = len(A|B); i = len(A&B)
    return i/u if u else 0.0

def keyword_recall_exact(answer: str, keywords: List[str]) -> float:
    if not keywords: return 0.0
    ans = normalize(answer)
    hit = 0
    for kw in keywords:
        if not kw: continue
        if normalize(kw) in ans or normalize_compact(kw) in normalize_compact(ans):
            hit += 1
    return hit/len(keywords)

def keyword_recall_soft(answer: str, keywords: List[str]) -> float:
    """Treffer per Soft-Kriterien."""
    if not keywords: return 0.0
    ans_norm = normalize(answer)
    ans_comp = normalize_compact(answer)
    ans_tokens = tokenize(answer)
    ans_tokens_stem = set(stem_tokens(ans_tokens))
    hit = 0
    for kw in keywords:
        if not kw:
            continue
        kw_norm = normalize(kw)
        kw_comp = normalize_compact(kw)
        if kw_norm in ans_norm or (kw_comp and kw_comp in ans_comp):
            hit += 1
            continue
        kw_toks = tokenize(kw)
        kw_stem = set(stem_tokens(kw_toks))
        if kw_stem and len(kw_stem & ans_tokens_stem) / len(kw_stem) >= 0.5:
            hit += 1
            continue
        fuzzy_ok = False
        for kt in kw_toks:
            for at in ans_tokens:
                if _sim_lev(kt, at) >= 0.90:
                    fuzzy_ok = True; break
            if fuzzy_ok: break
        if fuzzy_ok:
            hit += 1
    return hit/len(keywords)

def soft_entity_match(answer: str, gold_answer: str) -> float:
    aa = normalize_compact(answer)
    ga = normalize_compact(gold_answer)
    return 1.0 if ga and ga in aa else 0.0

# Kontext flachziehen
def flatten_text(x: Any, limit_chars: int = 20000) -> str:
    acc: List[str] = []
    def rec(v: Any):
        if len(" ".join(acc)) > limit_chars: return
        if isinstance(v, str): acc.append(v)
        elif isinstance(v, dict):
            for vv in v.values(): rec(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v: rec(vv)
    rec(x); return " ".join(acc)

# Halluzination
def hallucination_rate(ans: str, meta: Dict[str, Any]) -> float:
    ans_tokens = tokenize(ans)
    if not ans_tokens: return 0.0
    ctx_tokens = set(tokenize(flatten_text(meta)))
    if not ctx_tokens: return 1.0
    unsupported = [t for t in ans_tokens if t not in ctx_tokens]
    return len(unsupported)/len(ans_tokens)

# Coverage
def coverage(meta: Dict[str, Any], gold_keywords: List[str]) -> float:
    if not gold_keywords: return 0.0
    ctx = normalize(flatten_text(meta))
    if not ctx: return 0.0
    hits = sum(1 for kw in gold_keywords if normalize(kw) in ctx)
    return hits/len(gold_keywords)

def _is_unanswerable(gold: Dict[str, Any]) -> bool:
    return normalize((gold.get("gold_answer") or "")) == ""

# Score-Komponenten
def _score_components(answer: str, gold: Dict[str, Any]) -> Dict[str, Any]:
    ga = (gold.get("gold_answer") or "").strip()
    kws = gold.get("keywords") or []
    ans_toks = tokenize(answer)
    ga_toks  = tokenize(ga)
    return {
        "em": soft_entity_match(answer, ga) if ga else 0.0,
        "kr_exact": keyword_recall_exact(answer, kws) if kws else 0.0,
        "kr_soft":  keyword_recall_soft(answer, kws) if kws else 0.0,
        "j":  jaccard(ans_toks, ga_toks) if ga else 0.0,
        "js": jaccard_stemmed(ans_toks, ga_toks) if ga else 0.0,
        "short_gold": len(ga_toks) <= 4 if ga else False,
        "has_ga": bool(ga),
        "has_kws": bool(kws),
    }

def score_answer(answer: str, gold: Dict[str, Any]) -> float:
    c = _score_components(answer, gold)
    em, kr_e, kr_s, j, js = c["em"], c["kr_exact"], c["kr_soft"], c["j"], c["js"]
    has_ga, has_kws, short_gold = c["has_ga"], c["has_kws"], c["short_gold"]

    if not has_ga and not has_kws:
        return 0.0

    if has_ga and has_kws:
        if short_gold:
            return 0.55*em + 0.30*kr_s + 0.10*js + 0.05*kr_e
        else:
            return 0.35*em + 0.35*kr_s + 0.20*js + 0.10*kr_e
    elif has_ga:
        return (0.65*em + 0.25*js + 0.10*j) if short_gold else (0.40*em + 0.40*js + 0.20*j)
    else:
        return 0.65*kr_s + 0.35*kr_e

# korrekte Enthaltung
def abstention_correct(answer: str, gold: Dict[str, Any]) -> int:
    exp_abstain = 1 if _is_unanswerable(gold) else 0
    got_abstain = abstains(answer or "")
    return 1 if got_abstain == exp_abstain else 0

# Engines aus Adapter bauen
def build_engines(args) -> Dict[str, "kg.RAGEngine"]:
    kg_adapters = kg._build_adapters()
    llm_client = None if args.no_llm else kg.LLMClient(short=True)
    engines = {m: kg._build_engine_for_mode(m, kg_adapters, llm_client,
                                            argparse.Namespace(facts_budget_chars=4000, fewshot=False, no_augment=False))
               for m in MODES}
    return engines

def load_gold(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# CLI / Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default="100questions.jsonl", help="Pfad zur JSONL Gold-Datei (Default: 100questions.jsonl)")
    ap.add_argument("--no-llm", action="store_true", help="Debug ohne LLM (gibt prompt aus)")
    ap.add_argument("--no-tokens", action="store_true", help="Unterdrückt Token-Zeilen in CSV")
    ap.add_argument("--pretty", action="store_true", help="Kompakte farbige Konsolenansicht statt CSV")
    ap.add_argument("--show-answers", action="store_true", help="Antworttext zwischen den Werten anzeigen")
    ap.add_argument("--ans-width", type=int, default=60, help="Breite der Antwortspalte (Pretty)")
    ap.add_argument("--debug-abstain", action="store_true", help="Zeige Pattern, das zur Enthaltung geführt hat (stderr)")
    ap.add_argument("--debug-correct", action="store_true", help="Zeige EM/KR_ex/KR_soft/J/J_stem-Anteile (stderr)")
    ap.add_argument("--out-jsonl", type=str, default="",
                    help="Pfad zu einer JSONL-Datei: pro (Frage×Modus) eine Zeile mit Antwort & Metriken; am Ende Summary je Modus.")
    ap.add_argument("--append-jsonl", action="store_true",
                    help="JSONL an bestehende Datei anhängen statt überschreiben.")
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        print(f"Fehler: Gold-Datei nicht gefunden: {args.gold}", file=sys.stderr); sys.exit(2)

    gold_items = load_gold(args.gold)
    engines = build_engines(args)

    # CSV-Header
    if not args.pretty:
        if args.show_answers:
            print("question_id,mode,correctness,hallucination_rate,coverage,answer,latency_ms,abstain,abstain_correct")
        else:
            print("question_id,mode,correctness,hallucination_rate,coverage,latency_ms,abstain,abstain_correct")

    pretty_rows = []
    color_on = _supports_color()
    lat_acc: Dict[str, List[float]] = {m: [] for m in MODES}
    abst_acc: Dict[str, List[int]] = {m: [] for m in MODES}
    abstok_acc: Dict[str, List[int]] = {m: [] for m in MODES}

    # Aggregation
    agg = {
        m: {"cnt": 0, "sum_corr": 0.0, "sum_hallu": 0.0, "sum_cov": 0.0,
            "sum_lat": 0.0, "sum_abst": 0, "sum_abstok": 0}
        for m in MODES
    }

    # JSONL öffnen
    out_fh = None
    if args.out_jsonl:
        mode = "a" if args.append_jsonl else "w"
        out_fh = open(args.out_jsonl, mode, encoding="utf-8")

    for item in gold_items:
        qid = item.get("id") or ""
        q = item.get("question") or ""
        keywords = item.get("keywords") or []
        for m in MODES:
            t0 = time.perf_counter()
            ans, meta = engines[m].answer(q, history="", top_k=5, max_neighbors=12)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            abst_flag, abst_pat = abstain_match(ans or "")
            corr = score_answer(ans or "", item)

            if args.debug_correct:
                comp = _score_components(ans or "", item)
                print(f"[CORR] {qid} {m} -> EM={comp['em']:.3f} KR_ex={comp['kr_exact']:.3f} "
                      f"KR_soft={comp['kr_soft']:.3f} J={comp['j']:.3f} J_stem={comp['js']:.3f} "
                      f"short_gold={comp['short_gold']} | score={corr:.3f}", file=sys.stderr)

            hallu = hallucination_rate(ans or "", meta or {})
            cov = coverage(meta or {}, keywords)
            abst_ok = 1 if (abst_flag == (1 if _is_unanswerable(item) else 0)) else 0

            if args.debug_abstain and abst_flag == 1:
                short = _truncate(ans or "", 120)
                print(f"[ABSTAIN] {qid} {m} -> pattern: {abst_pat} | ans: {short}", file=sys.stderr)

            lat_acc[m].append(dt_ms)
            abst_acc[m].append(abst_flag)
            abstok_acc[m].append(abst_ok)

            agg[m]["cnt"] += 1
            agg[m]["sum_corr"]  += float(corr)
            agg[m]["sum_hallu"] += float(hallu)
            agg[m]["sum_cov"]   += float(cov)
            agg[m]["sum_lat"]   += float(dt_ms)
            agg[m]["sum_abst"]  += int(abst_flag)
            agg[m]["sum_abstok"]+= int(abst_ok)

            if out_fh:
                rec = {
                    "type": "row",
                    "id": qid,
                    "mode": m,
                    "question": q,
                    "answer": ans,
                    "metrics": {
                        "correctness": round(corr, 6),
                        "hallucination_rate": round(hallu, 6),
                        "coverage": round(cov, 6),
                        "latency_ms": int(dt_ms),
                        "abstain": int(abst_flag),
                        "abstain_correct": int(abst_ok)
                    }
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.pretty:
                ans_col = _truncate(ans or "", max(10, min(args.ans_width, 120))) if args.show_answers else ""
                pretty_rows.append((qid, m, corr, hallu, cov, ans_col, dt_ms, abst_flag, abst_ok))
            else:
                if args.show_answers:
                    print(f"{qid},{m},{corr:.3f},{hallu:.3f},{cov:.3f},{json.dumps(ans or '', ensure_ascii=False)},{int(dt_ms)},{abst_flag},{abst_ok}")
                else:
                    print(f"{qid},{m},{corr:.3f},{hallu:.3f},{cov:.3f},{int(dt_ms)},{abst_flag},{abst_ok}")
                if not args.no_tokens:
                    ans_tokens = tokenize(ans or "")
                    ctx_tokens = sorted(set(tokenize(flatten_text(meta or {}))))
                    print("Antwort-Tokens: " + json.dumps(ans_tokens, ensure_ascii=False)
                          + " | Kontext-Tokens: " + json.dumps(ctx_tokens, ensure_ascii=False))

    if out_fh:
        summary = {"type": "summary", "per_mode": {}}
        for m, s in agg.items():
            n = s["cnt"] or 1
            summary["per_mode"][m] = {
                "avg_correctness": round(s["sum_corr"]/n, 6),
                "avg_hallucination_rate": round(s["sum_hallu"]/n, 6),
                "avg_coverage": round(s["sum_cov"]/n, 6),
                "avg_latency_ms": round(s["sum_lat"]/n, 3),
                "abstain_rate": round(s["sum_abst"]/n, 6),
                "abstain_ok_rate": round(s["sum_abstok"]/n, 6),
                "n": n
            }
        out_fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        out_fh.close()

    if args.pretty:
        ans_w = max(10, min(args.ans_width, 120))
        print(f"\n{C_CYAN}{BOLD}EVALUATION (Pretty){RESET}")
        if args.show_answers:
            print(C_GRAY + "-" * (90 + ans_w) + RESET)
            print(f"{BOLD}{'Question':<12} {'Mode':<7} {'Corr':>7} {'Hallu':>7} {'Cov':>7} {'Ans':<{ans_w}} {'Lat(ms)':>8} {'Abst':>5} {'AbstOK':>7}{RESET}")
            print(C_GRAY + "-" * (90 + ans_w) + RESET)
        else:
            print(C_GRAY + "-" * 90 + RESET)
            print(f"{BOLD}{'Question':<12} {'Mode':<7} {'Corr':>7} {'Hallu':>7} {'Cov':>7} {'Lat(ms)':>8} {'Abst':>5} {'AbstOK':>7}{RESET}")
            print(C_GRAY + "-" * 90 + RESET)
        for (qid, m, corr, hallu, cov, ans_col, lat, abst, abst_ok) in pretty_rows:
            if args.show_answers:
                print(f"{qid:<12} {m:<7} "
                      f"{_cm(corr,'correctness',color_on):>7} "
                      f"{_cm(hallu,'hallucination',color_on):>7} "
                      f"{_cm(cov,'coverage',color_on):>7} "
                      f"{ans_col:<{ans_w}} "
                      f"{_cm(lat,'latency_ms',color_on):>8} "
                      f"{_cm(abst,'abstain',color_on):>5} "
                      f"{_cm(abst_ok,'abst_ok',color_on):>7}")
            else:
                print(f"{qid:<12} {m:<7} "
                      f"{_cm(corr,'correctness',color_on):>7} "
                      f"{_cm(hallu,'hallucination',color_on):>7} "
                      f"{_cm(cov,'coverage',color_on):>7} "
                      f"{_cm(lat,'latency_ms',color_on):>8} "
                      f"{_cm(abst,'abstain',color_on):>5} "
                      f"{_cm(abst_ok,'abst_ok',color_on):>7}")
        if args.show_answers:
            print(C_GRAY + "-" * (90 + ans_w) + RESET)
        else:
            print(C_GRAY + "-" * 90 + RESET)

        print(f"{BOLD}Averages per mode{RESET}")
        print(f"{BOLD}{'Mode':<7} {'Corr':>7} {'Hallu':>7} {'Cov':>7} {'Lat(ms)':>8} {'Abst%':>6} {'AbstOK%':>8}{RESET}")
        for m in MODES:
            rows = [r for r in pretty_rows if r[1]==m]
            if rows:
                avg_corr = sum(r[2] for r in rows)/len(rows)
                avg_h   = sum(r[3] for r in rows)/len(rows)
                avg_cov = sum(r[4] for r in rows)/len(rows)
                avg_lat = sum(r[6] for r in rows)/len(rows)
                rate_ab = (sum(r[7] for r in rows)/len(rows))*100.0
                rate_ok = (sum(r[8] for r in rows)/len(rows))*100.0
            else:
                avg_corr=avg_h=avg_cov=avg_lat=rate_ab=rate_ok=0.0
            print(f"{m:<7} "
                  f"{_cm(avg_corr,'correctness',color_on):>7} "
                  f"{_cm(avg_h,'hallucination',color_on):>7} "
                  f"{_cm(avg_cov,'coverage',color_on):>7} "
                  f"{_cm(avg_lat,'latency_ms',color_on):>8} "
                  f"{C_GRN if rate_ab==0 else (C_YEL if rate_ab<=20 else C_RED)}{rate_ab:>5.1f}{RESET} "
                  f"{C_GRN if rate_ok>=80 else (C_YEL if rate_ok>=50 else C_RED)}{rate_ok:>7.1f}{RESET}")

if __name__ == "__main__":
    main()
