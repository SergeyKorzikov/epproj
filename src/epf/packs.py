"""Build 'evidence packs': compact, privacy-masked markdown bundles of his own words, one per lens.

Each pack fits one reading session (~80–100k tokens). Every item carries a stable ID
(source doc + position) so any claim in the final report can be traced to the original file.
"""
from __future__ import annotations

import os

import pandas as pd

from .privacy import Masker

PACK_CHARS = 340_000


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


def _select(je: pd.DataFrame, mask: pd.Series, min_words: int, score: pd.Series, budget: int) -> pd.DataFrame:
    """Rank by score, then spread across years so no single year dominates."""
    c = je[mask & (je.body.str.split().str.len() >= min_words)].copy()
    c["_s"] = score[c.index]
    c = c.sort_values("_s", ascending=False)
    c["_rk"] = c.groupby(c.year.fillna(0)).cumcount()
    c = c.sort_values(["_rk", "_s"], ascending=[True, False])
    c["_cum"] = c.body.str.len().add(200).cumsum()
    return c[c._cum <= budget].sort_values("ts", na_position="last")


def write_pack(path: str, title: str, intro: str, items: pd.DataFrame, df_all: pd.DataFrame, masker: Masker) -> int:
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
            f.write(masker.mask(r.body).strip() + "\n\n")
    return len(items)


def build_all(je, df_all, tags, themes, outdir, allowlist):
    os.makedirs(outdir, exist_ok=True)
    masker = Masker(allowlist)
    sensitive = tags.get("th_sensitive_abuse", pd.Series(0, index=tags.index)).gt(0)
    words = je.body.str.split().str.len()
    report = {}

    # 1. long-form: his longest messages — where reasoning and worldview are most visible
    items = _select(je, ~sensitive, 60, words.astype(float), PACK_CHARS)
    report["00_longform"] = write_pack(
        f"{outdir}/00_longform.md", "Пакет 00 — его длинные тексты",
        "Самые развёрнутые сообщения: рассуждения, инструкции, мировоззрение.", items, df_all, masker)

    # 2. voice sampler: stratified random sample of ordinary messages (cognitive style baseline)
    rnd = pd.Series(pd.util.hash_pandas_object(je.body, index=False).values % 10_000, index=je.index).astype(float)
    items = _select(je, ~sensitive & words.between(8, 60), 8, rnd, PACK_CHARS)
    report["01_voice"] = write_pack(
        f"{outdir}/01_voice.md", "Пакет 01 — обычная речь (случайная стратифицированная выборка)",
        "Базовая линия стиля: как он пишет, когда ничего особенного не происходит.", items, df_all, masker)

    # 3. one pack per lens
    for i, (k, v) in enumerate(themes.items(), start=2):
        if v.get("exclude_from_packs"):
            continue
        col = f"th_{k}"
        score = tags[col] * 10 + words.clip(upper=200) / 20
        items = _select(je, tags[col].gt(0) & ~sensitive, 6, score, PACK_CHARS)
        report[f"{i:02d}_{k}"] = write_pack(
            f"{outdir}/{i:02d}_{k}.md", f"Пакет {i:02d} — {v['label']}",
            f"Сообщения, где он сам затрагивает тему «{v['label']}». Отобраны по словарю, распределены по годам.",
            items, df_all, masker)
    return report
