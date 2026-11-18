"""Build 'evidence packs': compact, privacy-masked markdown bundles of his own words, one per lens.

Each pack fits one reading session (~80–100k tokens). Every item carries a stable ID
(source doc + position) so any claim in the final report can be traced to the original file.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .privacy import Masker

PACK_CHARS = 340_000

_OWN = [
    re.compile(r"(?<![A-Za-z'])i(?![A-Za-z'])"),  # lowercase 'i'
    re.compile(r"\b(dont|im|cant|wont|didnt|isnt|doesnt|ive|youre|thats|whats|lets)\b"),
    re.compile(r"\s\.(\s|$)"),  # ' . ' spacing habit
    re.compile(r"(?m)^[a-z]"),  # lowercase sentence starts
]
_JUNK = re.compile(
    r"dear (jeff|jeffrey|mr\.? epstein)|article \d|(washington post|new york times|reuters|bloomberg|associated press)\s*\n"
    r"|\bsuite \d+|\bprincipal\b\s*\n|unsubscribe|view (this|in) (email|browser)|all rights reserved|\bpage \d+ of \d+",
    re.I,
)


def ownness(text: str) -> float:
    """Idiolect score per 100 words. Validated on House Oversight e-mails (>=40 words):
    his median 3.5 vs correspondents 1.35. Used as an authorship check for long texts."""
    w = max(len(text.split()), 1)
    return 100 * sum(len(r.findall(text)) for r in _OWN) / w


def looks_foreign(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    columns = len(lines) > 25 and sum(len(ln) for ln in lines) / len(lines) < 45  # OCR of newspaper columns
    return columns or bool(_JUNK.search(text))


def _context(idx: dict, row) -> str | None:
    """Message he was replying to (earlier message in the same thread)."""
    same = idx.get(row.doc_id)
    if same is None:
        return None
    if str(row.source).startswith("DOJ"):  # DOJ: quoted (earlier) messages sit below -> higher depth
        prev = same[same.quoted_depth == row.quoted_depth + 1]
    else:  # House Oversight sets are chronological
        prev = same[same["order"] == row.order - 1]
    return None if prev.empty else prev.iloc[0].body[:350]


def _select(je: pd.DataFrame, mask: pd.Series, min_words: int, score: pd.Series, budget: int,
            cap: int = 10**9) -> pd.DataFrame:
    """Rank by score, then spread across years so no single year dominates."""
    c = je[mask & (je.body.str.split().str.len() >= min_words)].copy()
    c["_s"] = score[c.index]
    c = c.sort_values("_s", ascending=False)
    c["_rk"] = c.groupby(c.year.fillna(0)).cumcount()
    c = c.sort_values(["_rk", "_s"], ascending=[True, False])
    c["_cum"] = c.body.str.len().clip(upper=cap).add(200).cumsum()
    return c[c._cum <= budget].sort_values("ts", na_position="last")


def write_pack(path: str, title: str, intro: str, items: pd.DataFrame, df_all: pd.DataFrame, masker: Masker,
               max_chars: int | None = None) -> int:
    need = set(items.doc_id)
    sub = df_all[df_all.doc_id.isin(need) & ~df_all.is_epstein]
    idx = {k: g for k, g in sub.groupby("doc_id")}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{intro}\n\nЭлементов: {len(items)}\n\n")
        for r in items.itertuples():
            when = r.ts.strftime("%Y-%m-%d %H:%M") if pd.notna(r.ts) else "без даты"
            to = masker.mask_name_field(r.recipients) or "—"
            f.write(f"### [{r.doc_id}#{r.quoted_depth}] {when} → {to}\n")
            ctx = _context(idx, r)
            if ctx:
                f.write(f"> (до этого ему написали) {masker.mask(ctx).replace(chr(10), ' ')}\n\n")
            body = r.body if not max_chars or len(r.body) <= max_chars else r.body[:max_chars] + " […]"
            f.write(masker.mask(body).strip() + "\n\n")
    return len(items)


def build_all(je, df_all, tags, themes, outdir, allowlist):
    os.makedirs(outdir, exist_ok=True)
    masker = Masker(allowlist)
    from .privacy import build_roles

    norm = lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-z\s-]", " ", re.sub(r"<[^>]*>|\[[^\]]*\]", "", x).lower())).strip()
    masker.roles = build_roles(je, tags, norm)
    print(f"roles inferred for {len(masker.roles)} contacts")
    sensitive = tags.get("th_sensitive_abuse", pd.Series(0, index=tags.index)).gt(0)
    words = je.body.str.split().str.len()
    own = je.body.apply(ownness)
    foreign = je.body.apply(looks_foreign)
    # long texts must pass the authorship check; short ones are attributed by header alone
    trusted = ~foreign & ((words < 120) | (own >= 1.5))
    report = {}

    # 1. long-form: his longest messages — where reasoning and worldview are most visible
    items = _select(je, ~sensitive & trusted & words.between(60, 1500), 60, words.astype(float), PACK_CHARS)
    report["00_longform"] = write_pack(
        f"{outdir}/00_longform.md", "Пакет 00 — его длинные тексты",
        "Самые развёрнутые сообщения: рассуждения, инструкции, мировоззрение.", items, df_all, masker)

    # 2. voice sampler: stratified random sample of ordinary messages (cognitive style baseline)
    rnd = pd.Series(pd.util.hash_pandas_object(je.body, index=False).values % 10_000, index=je.index).astype(float)
    items = _select(je, ~sensitive & trusted & words.between(8, 60), 8, rnd, PACK_CHARS)
    report["01_voice"] = write_pack(
        f"{outdir}/01_voice.md", "Пакет 01 — обычная речь (случайная стратифицированная выборка)",
        "Базовая линия стиля: как он пишет, когда ничего особенного не происходит.", items, df_all, masker)

    # 3. one pack per lens
    for i, (k, v) in enumerate(themes.items(), start=2):
        if v.get("exclude_from_packs"):
            continue
        col = f"th_{k}"
        density = tags[col] / words.clip(lower=10)  # topical focus, not sheer length
        score = density * 100 + tags[col].clip(upper=5) + words.clip(upper=80) / 40
        items = _select(je, tags[col].gt(0) & ~sensitive & trusted & words.le(1500), 6, score, PACK_CHARS, cap=1500)
        report[f"{i:02d}_{k}"] = write_pack(
            f"{outdir}/{i:02d}_{k}.md", f"Пакет {i:02d} — {v['label']}",
            f"Сообщения, где он сам затрагивает тему «{v['label']}». Отобраны по словарю, распределены по годам.",
            items, df_all, masker, max_chars=1500)
    return report
