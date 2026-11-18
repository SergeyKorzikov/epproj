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
        self.roles: dict[str, str] = {}  # normalised name -> role label (set by build_roles)

    def is_public(self, name: str) -> bool:
        n = re.sub(r"[^a-z\s-]", "", name.lower()).strip()
        return bool(n) and (n in self.allow or any(t in self.allow for t in n.split() if len(t) > 3))

    def pseudo(self, name: str) -> str:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z\s-]", " ", name.lower())).strip()
        if key not in self.map:
            role = self.roles.get(key)
            tag = f" · {role}" if role else ""
            self.map[key] = f"[P-{len(self.map) + 1:03d}{tag}]"
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


# ---------------------------------------------------------------- role inference
ROLE_BY_THEME = {
    "th_lifestyle_logistics": "быт/персонал",
    "th_money_deals": "финансы",
    "th_legal_conflict": "право",
    "th_science_patronage": "наука",
    "th_media_reputation": "СМИ/PR",
    "th_politics_geopolitics": "политика",
    "th_philanthropy_status": "искусство/филантропия",
    "th_longevity_biohacking": "медицина/здоровье",
}
ROLE_BY_DOMAIN = [
    (r"\.edu$|\.ac\.|media\.mit|harvard|santafe", "наука"),
    (r"jpmorgan|db\.com|gs\.com|apollo|bank|capital|fund|partners|holdings|ubs|morganstanley", "финансы"),
    (r"law|llp|legal|dershowitz|kirkland|lw\.com", "право"),
    (r"nytimes|wsj|cnbc|bloomberg|reuters|vanityfair|dailymail|nymag|newyorker", "СМИ/PR"),
    (r"jet|aviation|air", "авиация"),
    (r"models|agency", "модельное агентство"),
    (r"gensler|architect|design|studio", "архитектура/дизайн"),
]


def build_roles(je, tags, normalise) -> dict:
    """Role = what *he* writes to this person about (dominant lens of his messages to them),
    overridden by an informative e-mail domain. Requires >=5 messages to label."""
    import pandas as pd

    rows = []
    for idx, rec in je.recipients.fillna("").items():
        for part in re.split(r"[;,]", rec):
            n = normalise(part)
            if n:
                dm = re.search(r"@([\w.-]+)", part)
                rows.append((n, idx, dm.group(1).lower() if dm else ""))
    if not rows:
        return {}
    r = pd.DataFrame(rows, columns=["name", "idx", "dom"])
    cols = [c for c in ROLE_BY_THEME if c in tags]
    t = tags.loc[r.idx, cols].gt(0).astype(int).set_index(r.index)
    r = pd.concat([r, t], axis=1)
    out = {}
    for name, g in r.groupby("name"):
        if len(g) < 5:
            continue
        doms = " ".join(g.dom.unique())
        role = next((lab for pat, lab in ROLE_BY_DOMAIN if doms and re.search(pat, doms)), None)
        if not role:
            share = g[cols].mean()
            if share.max() >= 0.15:
                role = ROLE_BY_THEME[share.idxmax()]
        if role:
            out[name] = role
    return out
