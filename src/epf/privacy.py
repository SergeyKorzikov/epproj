"""Privacy layer applied to everything that leaves the numeric stage (evidence packs, quotes).

Rules
  * e-mail addresses, phone numbers, street addresses  -> removed
  * PERSON names                                        -> kept only if on the public-figure allowlist,
                                                           otherwise replaced by a stable pseudonym [P-0xx]
  * messages hitting the `sensitive_abuse` lexicon      -> never exported verbatim (handled upstream)
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
STREET_RE = re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+(\s[A-Z][a-z]+)*\s(street|st|avenue|ave|road|rd|drive|dr|lane|ln|blvd)\b", re.I)


class Masker:
    def __init__(self, allowlist_path: str):
        import spacy

        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        self.allow = set()
        for ln in open(allowlist_path, encoding="utf-8"):
            ln = ln.split("#")[0].strip().lower()
            if ln:
                self.allow.add(ln)
                self.allow.update(t for t in ln.split() if len(t) > 3)  # surname-only mentions
        self.map: dict[str, str] = {}

    def is_public(self, name: str) -> bool:
        n = re.sub(r"[^a-z\s-]", "", name.lower()).strip()
        return bool(n) and (n in self.allow or any(t in self.allow for t in n.split() if len(t) > 3))

    def pseudo(self, name: str) -> str:
        key = name.lower().strip()
        if key not in self.map:
            self.map[key] = f"[P-{len(self.map) + 1:03d}]"
        return self.map[key]

    def mask_name_field(self, field: str) -> str:
        parts = [p.strip() for p in re.split(r"[;,]", re.sub(r"<[^>]*>|\[[^\]]*\]", "", field or "")) if p.strip()]
        return ", ".join(p if self.is_public(p) else self.pseudo(p) for p in parts[:6])

    def mask(self, text: str) -> str:
        text = EMAIL_RE.sub("[email]", text)
        text = PHONE_RE.sub("[tel]", text)
        text = STREET_RE.sub("[адрес]", text)
        doc = self.nlp(text[:20000])
        spans = [(e.start_char, e.end_char, e.text) for e in doc.ents if e.label_ == "PERSON"]
        for s, e, t in reversed(spans):
            if not self.is_public(t):
                text = text[:s] + self.pseudo(t) + text[e:]
        return text
