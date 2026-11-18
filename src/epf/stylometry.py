"""LIWC-style lexical features for short e-mail texts (English).

Not the licensed LIWC dictionary — an open approximation built from public psycholinguistic
literature (Pennebaker; Tausczik & Pennebaker 2010; hedging/certainty lists from Hyland).
Rates are per 100 words, computed on the pooled text of a group (year, recipient type, …),
because single e-mails are often 3–10 words and per-message rates are noise.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LEX = {
    # pronouns — self-focus vs collective vs other-focus (status & attention markers)
    "i": r"\b(i|me|my|mine|myself|im|i'm|ive|i've|id|i'd|ill|i'll)\b",
    "we": r"\b(we|us|our|ours|ourselves|we're|weve|we've|lets|let's)\b",
    "you": r"\b(you|your|yours|yourself|u|ur|you're|youre|youll|you'll)\b",
    "they": r"\b(they|them|their|theirs|he|him|his|she|her|hers)\b",
    # epistemic stance
    "certainty": r"\b(always|never|definitely|absolutely|certainly|obviously|of course|clearly|sure|exactly|totally|completely|no doubt|must|fact)\b",
    "hedge": r"\b(maybe|perhaps|might|possibly|probably|i think|i guess|seems?|sort of|kind of|not sure|unclear|could be)\b",
    "cause": r"\b(because|cause|since|therefore|thus|hence|so that|reason|why)\b",
    "insight": r"\b(think|know|understand|realize|realise|figure|idea|believe|consider|mean)\b",
    # affect
    "posemo": r"\b(good|great|love|nice|happy|fun|wonderful|excellent|beautiful|best|thanks|thank|enjoy|glad|amazing|cool)\b",
    "negemo": r"\b(bad|hate|angry|sad|awful|terrible|worst|stupid|idiot|annoy\w*|upset|wrong|mess|crap|shit|fuck\w*|damn|hurt)\b",
    "anxiety": r"\b(worr\w*|nervous|afraid|fear\w*|scared|anxious|panic|risk\w*|danger\w*)\b",
    # drives (Pennebaker/LIWC-22 drives)
    "power": r"\b(power\w*|control\w*|boss|command|lead\w*|dominat\w*|important|strong|weak|win\w*|lose|beat|rank|status)\b",
    "achieve": r"\b(success\w*|achiev\w*|goal\w*|effort|accomplish\w*|best|first|better|improv\w*|win\w*)\b",
    "affiliation": r"\b(friend\w*|together|family|partner\w*|team|ally|allies|share|join|love)\b",
    "reward": r"\b(get|got|take|gain\w*|benefit\w*|prize|reward\w*|deal|bonus|profit\w*|opportunit\w*)\b",
    "money": r"\b(money|cash|dollar\w*|pay\w*|paid|fee\w*|million\w*|billion\w*|price\w*|cost\w*|fund\w*)\b",
    # perceptual / bodily
    "body": r"\b(body|bodies|face|skin|hair|weight|muscle|blood|brain|sleep\w*|eat\w*|food|sick|health\w*)\b",
    "sexual_lex": r"\b(sex\w*|naked|nude|breast\w*|kiss\w*|hot)\b",
    # time orientation
    "future": r"\b(will|gonna|going to|tomorrow|next|soon|plan\w*|later)\b",
    "past": r"\b(was|were|had|did|yesterday|ago|last|before|used to)\b",
    # interaction mode
    "question_words": r"\b(what|why|how|when|where|who|which)\b",
    "negation": r"\b(no|not|never|dont|don't|cant|can't|wont|won't|nothing|nobody|none)\b",
    "swear": r"\b(fuck\w*|shit\w*|damn\w*|ass\w*|bitch\w*|crap)\b",
}
LEX_RE = {k: re.compile(v, re.I) for k, v in LEX.items()}
WORD_RE = re.compile(r"[a-zA-Z']+")
IMPERATIVE_START = re.compile(
    r"^\s*(send|call|tell|ask|get|make|have|let|do|don'?t|find|bring|book|check|come|give|go|keep|put|see|set|stop|"
    r"take|try|use|read|write|look|meet|pls|please|move|fix|cancel|confirm|arrange|organi[sz]e)\b",
    re.I,
)


def _features(text: str) -> dict:
    words = WORD_RE.findall(text)
    n = max(len(words), 1)
    f = {k: 100 * len(r.findall(text)) / n for k, r in LEX_RE.items()}
    f["n_words"] = len(words)
    return f


def message_level(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-message structural features (lengths, questions, imperatives, casing)."""
    b = df.body.fillna("")
    out = pd.DataFrame(index=df.index)
    out["n_words"] = b.str.split().str.len()
    out["n_chars"] = b.str.len()
    out["has_question"] = b.str.contains(r"\?")
    out["imperative"] = b.apply(lambda s: bool(IMPERATIVE_START.match(s)))
    out["starts_lower"] = b.str.match(r"^[a-z]")
    out["link_only"] = b.str.fullmatch(r"\s*(https?://\S+\s*)+")
    out["has_link"] = b.str.contains(r"https?://")
    out["ellipsis_dots"] = b.str.count(r"\s\.\s|\.\s\.")  # his ' . ' habit
    out["exclaim"] = b.str.contains("!")
    out["all_lower"] = (b.str.len() > 10) & (b == b.str.lower())
    return out


def grouped_profile(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Pooled LIWC-lite rates per group + structural means."""
    ml = message_level(df)
    rows = []
    for key, g in df.groupby(by):
        if len(g) < 20:
            continue
        f = _features("\n".join(g.body.fillna("")))
        m = ml.loc[g.index]
        f.update(
            group=key, n_msgs=len(g), median_len=float(m.n_words.median()),
            q_share=float(m.has_question.mean()), imperative_share=float(m.imperative.mean()),
            lower_start_share=float(m.starts_lower.mean()), link_only_share=float(m.link_only.mean()),
        )
        rows.append(f)
    return pd.DataFrame(rows).set_index("group").sort_index()


def contrast(je: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    """His rates vs. the pooled rates of his correspondents in the same corpus (a natural baseline)."""
    a = _features("\n".join(je.body.fillna("")))
    b = _features("\n".join(others.body.fillna("")))
    t = pd.DataFrame({"epstein": a, "correspondents": b})
    t["ratio"] = (t.epstein + 1e-3) / (t.correspondents + 1e-3)
    return t.drop(index="n_words").sort_values("ratio", ascending=False)


def distinctive_words(je: pd.DataFrame, others: pd.DataFrame, top: int = 80) -> pd.DataFrame:
    """Log-odds with informative Dirichlet prior (Monroe et al. 2008): words that are 'his'."""
    from collections import Counter

    ca = Counter(w.lower() for t in je.body.fillna("") for w in WORD_RE.findall(t))
    cb = Counter(w.lower() for t in others.body.fillna("") for w in WORD_RE.findall(t))
    prior = ca + cb
    a0, na, nb = sum(prior.values()), sum(ca.values()), sum(cb.values())
    rows = []
    for w, p in prior.items():
        if p < 30:
            continue
        ya, yb = ca[w], cb[w]
        d = np.log((ya + p) / (na + a0 - ya - p)) - np.log((yb + p) / (nb + a0 - yb - p))
        var = 1 / (ya + p) + 1 / (yb + p)
        rows.append((w, d / np.sqrt(var), ya, yb))
    r = pd.DataFrame(rows, columns=["word", "z", "n_je", "n_others"]).sort_values("z")
    return pd.concat([r.tail(top).iloc[::-1].assign(side="his"), r.head(top).assign(side="theirs")])
