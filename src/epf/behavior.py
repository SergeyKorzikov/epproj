"""Themes, temporal rhythms and the communication system (cybernetic view of his network).

Network outputs are *structural* (how he routed information and access), not a who-knew-whom list.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

from .emailparse import EPSTEIN_RE

# --------------------------------------------------------------------- themes


def load_themes(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    for k, v in cfg.items():
        v["re"] = re.compile(r"\b(" + "|".join(f"(?:{p})" for p in v["patterns"]) + r")\b", re.I)
    return cfg


def tag_themes(df: pd.DataFrame, themes: dict) -> pd.DataFrame:
    text = (df.subject.fillna("") + "\n" + df.body.fillna(""))
    out = pd.DataFrame(index=df.index)
    for k, v in themes.items():
        out[f"th_{k}"] = text.str.count(v["re"])
    return out


def theme_profile(df: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    hit = tags.gt(0)
    t = pd.DataFrame({"share_msgs": hit.mean(), "n_msgs": hit.sum()})
    by_year = hit.groupby(df.year).mean().T
    return t.join(by_year).sort_values("share_msgs", ascending=False)


# --------------------------------------------------------------------- rhythms


def rhythms(je: pd.DataFrame) -> dict:
    d = je[je.ts.notna()].copy()
    d["hour"] = d.ts.dt.hour
    d["dow"] = d.ts.dt.dayofweek
    d["date"] = d.ts.dt.date
    daily = d.groupby("date").size()
    # burstiness B = (σ-μ)/(σ+μ) of inter-message gaps (Goh & Barabási 2008); 0 = Poisson, →1 = bursty
    gaps = d.ts.sort_values().diff().dt.total_seconds().dropna()
    gaps = gaps[gaps > 0]
    B = float((gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean())) if len(gaps) > 10 else None
    return {
        "hour_hist": d.hour.value_counts().sort_index().to_dict(),
        "dow_hist": d.dow.value_counts().sort_index().to_dict(),
        "night_share_0_5": float(d.hour.between(0, 5).mean()),
        "weekend_share": float(d.dow.isin([5, 6]).mean()),
        "active_days": int(daily.size),
        "msgs_per_active_day_median": float(daily.median()),
        "msgs_per_active_day_p95": float(daily.quantile(0.95)),
        "max_day": [str(daily.idxmax()), int(daily.max())] if len(daily) else None,
        "burstiness_B": B,
        "by_year_hour_median": d.groupby(d.ts.dt.year).hour.median().to_dict(),
        "caveat": "timestamps are as printed in the source (sender/archive timezone, often unknown); "
                  "use hour patterns comparatively across years, not as absolute sleep times",
    }


# --------------------------------------------------------------------- network

_ADDR_SPLIT = re.compile(r"[;,]|\band\b")


def _norm_name(s: str) -> str:
    s = re.sub(r"<[^>]*>|\[[^\]]*\]|['\"]", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if "," in s and len(s.split(",")) == 2:  # "Weingarten, Reid" -> "reid weingarten"
        last, first = [x.strip() for x in s.split(",")]
        s = f"{first} {last}"
    return s


def _people(field: str) -> list[str]:
    out = []
    for p in _ADDR_SPLIT.split(field or ""):
        n = _norm_name(p)
        if 2 < len(n) < 40 and not EPSTEIN_RE.search(n) and "redacted" not in n:
            out.append(n)
    return out


def communication_system(df_all: pd.DataFrame) -> dict:
    """Ego-network metrics from his side of the corpus.

    * initiation ratio  — of threads he is in, how often the first message is his (agenda setting)
    * reciprocity       — share of his outgoing contacts who also write to him
    * concentration     — Gini of his outgoing volume across contacts (inner circle vs broadcast)
    * brokerage         — Burt's constraint on his ego network, from co-recipient ties:
                          low constraint = he sits on structural holes between otherwise unlinked people
    * introductions     — explicit 'you should meet / i'll introduce / cc-ing X' acts per year
    """
    import networkx as nx

    je = df_all[df_all.is_epstein]
    inbound = df_all[~df_all.is_epstein & df_all.recipients.fillna("").str.contains(EPSTEIN_RE)]

    out_cnt = Counter(p for r in je.recipients for p in _people(r))
    in_cnt = Counter(_norm_name(s) for s in inbound.sender if s)
    contacts = set(out_cnt)
    recip = len(contacts & set(in_cnt)) / max(len(contacts), 1)

    v = np.sort(np.array(list(out_cnt.values()), dtype=float))
    gini = float(1 - 2 * np.sum(np.cumsum(v) / v.sum()) / len(v) + 1 / len(v)) if len(v) else None

    # threads: doc_id groups messages of one thread in HO data; first message author = initiator
    first = df_all.sort_values(["doc_id", "quoted_depth"], ascending=[True, False]).groupby("doc_id").head(1)
    initiation = float(first.is_epstein.mean())

    # co-recipient graph (alters linked if they appear together on one of his messages or write to each other)
    core = {p for p, _ in out_cnt.most_common(400)}  # Burt constraint is O(n·deg²): restrict to the core
    G = nx.Graph()
    G.add_node("EGO")
    for r, c in zip(je.recipients.fillna(""), je.cc.fillna("")):
        ppl = [p for p in dict.fromkeys(_people(r) + _people(c)) if p in core][:12]
        for p in ppl:
            G.add_edge("EGO", p)
        for i in range(len(ppl)):
            for j in range(i + 1, len(ppl)):
                G.add_edge(ppl[i], ppl[j])
    for s, r in zip(df_all[~df_all.is_epstein].sender.fillna(""), df_all[~df_all.is_epstein].recipients.fillna("")):
        a = _norm_name(s)
        for b in _people(r):
            if a in G and b in G and a != b:
                G.add_edge(a, b)
    constraint = nx.constraint(G, nodes=["EGO"])["EGO"] if G.degree("EGO") > 1 else None
    eff = nx.effective_size(G, nodes=["EGO"])["EGO"] if G.degree("EGO") > 1 else None

    intro_re = re.compile(r"you should meet|(i'?ll|i will|let me) introduce|cc'?ing|meet (him|her)|connect you", re.I)
    intro = je[je.body.str.contains(intro_re)]
    by_year_contacts = {
        int(y): len({p for r in g.recipients for p in _people(r)}) for y, g in je[je.year.notna()].groupby("year")
    }
    return {
        "n_outgoing_contacts": len(contacts),
        "reciprocity": round(recip, 3),
        "outgoing_gini": round(gini, 3) if gini is not None else None,
        "top10_share_of_outgoing": round(sum(c for _, c in out_cnt.most_common(10)) / max(sum(out_cnt.values()), 1), 3),
        "thread_initiation_share": round(initiation, 3),
        "burt_constraint": round(constraint, 4) if constraint is not None else None,
        "effective_size": round(eff, 1) if eff is not None else None,
        "ego_degree": G.degree("EGO"),
        "introductions_total": len(intro),
        "introductions_by_year": intro.year.value_counts().sort_index().to_dict(),
        "unique_contacts_by_year": by_year_contacts,
        "_top_contacts": out_cnt.most_common(60),  # kept local only; not for publication
    }


# --------------------------------------------------------------------- latency


def response_latency(df_all: pd.DataFrame) -> dict:
    """Within threads: how fast he answers others vs. how fast others answer him (minutes)."""
    d = df_all[df_all.ts.notna()].sort_values(["doc_id", "ts"])
    his, theirs = [], []
    for _, g in d.groupby("doc_id"):
        if len(g) < 2:
            continue
        prev = None
        for r in g.itertuples():
            if prev is not None and r.is_epstein != prev.is_epstein:
                dt = (r.ts - prev.ts).total_seconds() / 60
                if 0 < dt < 60 * 24 * 7:
                    (his if r.is_epstein else theirs).append(dt)
            prev = r
    q = lambda x: {k: round(float(np.percentile(x, p)), 1) for k, p in (("p25", 25), ("median", 50), ("p75", 75))} if x else None
    return {"he_replies_min": q(his), "others_reply_min": q(theirs), "n_his": len(his), "n_theirs": len(theirs)}
