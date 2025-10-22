#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:36:11 2025

@author: domenico
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess user-submitted text for the AI detector:
- Clean HTML/URLs/emails, normalize quotes/dashes
- (Default) Neutralize time anchors (dates, weekdays, "X days ago")
- Sentence splitting with spaCy's sentencizer (small, no model download)
- Enforce basic length guardrails
- Output JSON to stdout (clean text, sentences, basic stats)

Usage examples:
  python preprocess_user_text.py --in sample.txt
  cat sample.txt | python preprocess_user_text.py
  echo "Your text..." | python preprocess_user_text.py --no-neutralize

Note: The backend already includes equivalent logic; this CLI is for batch jobs or quick tests.
"""

import sys, json, re, argparse
from typing import List, Dict
import spacy

# -----------------------------
# Text cleanup
# -----------------------------
_ASCII_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "—": "-", "–": "-", "‐": "-",
    "…": "...",
})
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w\.\-]+@[\w\.\-]+\.[a-zA-Z]{2,}\b")
RE_HTML = re.compile(r"<[^>]+>")
RE_WS = re.compile(r"[ \t]+")
RE_MULTI_NL = re.compile(r"\n{2,}")
BOUNDARY_QUOTES = ' "\'\t\n\r\f\v“”‘’‛‟«»‹›'

def clean_text(text: str, ascii_only: bool = False) -> str:
    t = (text or "").replace("\xa0", " ")
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = t.translate(_ASCII_MAP)
    if ascii_only:
        t = t.encode("ascii", "ignore").decode("ascii")
    t = RE_WS.sub(" ", t)
    t = RE_MULTI_NL.sub("\n\n", t)
    return t.strip()

# -----------------------------
# Time-anchor neutralization
# -----------------------------
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
WEEKDAYS = r"(?:Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
RE_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
RE_US_DATE  = re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b")
RE_EU_DATE  = re.compile(rf"\b\d{{1,2}}\s+{MONTHS}(?:\s+\d{{4}})?\b")
RE_YEAR     = re.compile(r"\b(19|20)\d{2}\b")
RE_WEEKDAY  = re.compile(rf"\b{WEEKDAYS}\b")
RE_REL      = re.compile(r"\b(today|yesterday|tomorrow|last\s+(week|month|year)|next\s+(week|month|year)|\d+\s+(hours|days|weeks|months|years)\s+ago)\b", re.I)

def neutralize_temporal(text: str) -> str:
    t = text
    t = RE_ISO_DATE.sub("[DATE]", t)
    t = RE_US_DATE.sub("[DATE]", t)
    t = RE_EU_DATE.sub("[DATE]", t)
    t = RE_WEEKDAY.sub("[WEEKDAY]", t)
    t = RE_REL.sub("[REL_TIME]", t)
    t = RE_YEAR.sub("YYYY", t)
    return t

# -----------------------------
# Sentence splitting
# -----------------------------
def _make_sentencizer():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def sentence_split(nlp, text: str) -> List[str]:
    doc = nlp(text)
    out, seen = [], set()
    for s in doc.sents:
        st = s.text.strip(BOUNDARY_QUOTES).strip()
        if len(st) < 15:
            continue
        key = st.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(st)
    return out

def basic_stats(sents: List[str]) -> Dict[str, int]:
    n_sents = len(sents)
    n_words = sum(len(s.split()) for s in sents)
    return {"n_sentences": n_sents, "n_words": n_words}

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="-", help="Input text file (or - for stdin)")
    ap.add_argument("--ascii-only", action="store_true", help="Strip non-ASCII characters")
    ap.add_argument("--no-neutralize", action="store_true", help="Disable date/time neutralization")
    ap.add_argument("--min-sents", type=int, default=10, help="Minimum sentences required")
    ap.add_argument("--max-sents", type=int, default=300, help="Maximum sentences (cap for speed)")
    ap.add_argument("--max-sent-words", type=int, default=80, help="Drop sentences longer than this")
    return ap.parse_args()

def main():
    args = parse_args()
    text = sys.stdin.read() if args.infile == "-" else open(args.infile, "r", encoding="utf-8").read()

    t = clean_text(text, ascii_only=args.ascii_only)
    if not args.no-neutralize:
        t = neutralize_temporal(t)

    nlp = _make_sentencizer()
    sents = sentence_split(nlp, t)
    sents = [s for s in sents if len(s.split()) <= args.max_sent_words]
    if len(sents) > args.max_sents:
        sents = sents[:args.max_sents]

    stats = basic_stats(sents)
    ok = stats["n_sentences"] >= args.min_sents

    out = {
        "ok": bool(ok),
        "reason" if not ok else "message": "too_few_sentences" if not ok else "ok",
        "text_clean": " ".join(sents),
        "sentences": sents,
        "stats": stats,
        "config": {
            "ascii_only": bool(args.ascii_only),
            "neutralize_temporal": not args.no-neutralize,
            "min_sents": args.min_sents,
            "max_sents": args.max_sents,
            "max_sent_words": args.max_sent_words,
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
