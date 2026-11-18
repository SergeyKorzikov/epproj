"""Load public Epstein-file text datasets into one normalized `messages` table.

Sources (all public, already OCR'd by the community, on Hugging Face):
  * kabasshouse/epstein-data   — DOJ release, 1.42M OCR'd documents (documents/*.parquet)
  * notesbymuneeb/epstein-emails — House Oversight estate e-mails, LLM-structured threads
  * to-be/epstein-emails         — House Oversight e-mails, vision-LLM structured
"""
from __future__ import annotations

import glob
import html
import json
import os
import re

import pandas as pd

from .emailparse import Message, clean_body, is_epstein, parse_date, split_document

HF = "https://huggingface.co/datasets"
DOWNLOADS = {
    "emails_muneeb.parquet": f"{HF}/notesbymuneeb/epstein-emails/resolve/main/epstein_email_threads.parquet",
    "emails_tobe.parquet": f"{HF}/to-be/epstein-emails/resolve/main/emails.parquet",
    **{
        f"docs/d{i:02d}.parquet": f"{HF}/kabasshouse/epstein-data/resolve/main/data/documents/documents-{i:05d}-of-00015.parquet"
        for i in range(15)
    },
    "derived_events.parquet": f"{HF}/kabasshouse/epstein-data/resolve/main/data/derived_events/derived_events-00000-of-00001.parquet",
    "financial_transactions.parquet": f"{HF}/kabasshouse/epstein-data/resolve/main/data/financial_transactions/financial_transactions-00000-of-00001.parquet",
}


def download(raw_dir: str) -> None:
    import urllib.request

    for rel, url in DOWNLOADS.items():
        dst = os.path.join(raw_dir, rel)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("downloading", rel)
        urllib.request.urlretrieve(url, dst)


def _row(m: Message) -> dict:
    return {
        "doc_id": m.doc_id, "source": m.source, "order": m.order, "sender": m.sender,
        "recipients": m.recipients, "cc": m.cc, "subject": m.subject, "ts": m.ts,
        "ts_raw": m.ts_raw, "body": m.body, "is_epstein": m.is_epstein,
        "quoted_depth": m.quoted_depth, "body_hash": m.body_hash,
    }


def load_house_muneeb(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for r in df.itertuples():
        for i, x in enumerate(json.loads(r.messages)):
            m = Message(
                doc_id=r.source_file.replace(".txt", ""), source="HO_muneeb", order=i,
                sender=x.get("sender") or "", recipients=", ".join(x.get("recipients") or []), cc="",
                subject=x.get("subject") or "", ts=parse_date(x.get("timestamp")), ts_raw=x.get("timestamp") or "",
                body=clean_body(x.get("body") or ""), is_epstein=is_epstein(x.get("sender")), quoted_depth=i,
            )
            rows.append(_row(m))
    return rows


def _html2text(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>|</div>", "\n", s or "", flags=re.I)
    s = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", s, flags=re.I | re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def load_house_tobe(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for r in df.itertuples():
        m = Message(
            doc_id=r.document_id, source="HO_tobe", order=int(r.message_order), sender=r.from_address or "",
            recipients=r.to_address or "", cc=r.other_recipients or "", subject=r.subject or "",
            ts=parse_date(r.timestamp_raw), ts_raw=r.timestamp_raw or "",
            body=clean_body(_html2text(r.message_html)), is_epstein=is_epstein(r.from_address),
            quoted_depth=int(r.message_order),
        )
        rows.append(_row(m))
    return rows


DOJ_FILTER = r"""
select file_key, dataset, date, full_text from '{path}'
where full_text ilike '%jeevacation%'
   or regexp_matches(full_text, '(?i)from:\s*[^\n]{{0,40}}(jeffrey e\.|jeffrey epstein)')
"""


def load_doj(docs_glob: str, limit_files: int | None = None):
    """Yield message rows from DOJ OCR documents that involve his accounts."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    files = sorted(glob.glob(docs_glob))[:limit_files]
    for f in files:
        rows = []
        for key, ds, date, text in con.execute(DOJ_FILTER.format(path=f)).fetchall():
            for m in split_document(key, text, f"DOJ_{ds}", date):
                rows.append(_row(m))
        print(f"  {os.path.basename(f)}: {len(rows)} message segments")
        yield rows


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Same message appears in many documents (quoted threads, re-productions).
    Keep one copy per (author-flag, body) preferring top-level, dated copies."""
    df = df[df.body.str.len() > 0].copy()
    df["_has_ts"] = df.ts.notna()
    df = df.sort_values(["_has_ts", "quoted_depth"], ascending=[False, True])
    df = df.drop_duplicates(subset=["is_epstein", "body_hash"], keep="first")
    return df.drop(columns="_has_ts").sort_values("ts", na_position="last").reset_index(drop=True)


def near_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Second pass: the same e-mail OCR'd from different productions differs by a few characters.
    Key = first 80 alphanumerics (+ minute timestamp for short messages, which collide by content)."""
    norm = df.body.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str[:80]
    short = df.body.str.split().str.len() < 40
    key = norm.where(~short, norm + "|" + df.ts.dt.strftime("%Y%m%d%H%M").fillna(df.doc_id))
    df = df.assign(_k=key, _has_ts=df.ts.notna()).sort_values(["_has_ts", "quoted_depth"], ascending=[False, True])
    df = df.drop_duplicates(subset=["is_epstein", "_k"]).drop(columns=["_k", "_has_ts"])
    df["body"] = final_clean(df.body)
    df["suspect_mixed"] = df.body.str.split().str.len().gt(300) & df.source.str.startswith("DOJ")
    return df.sort_values("ts", na_position="last").reset_index(drop=True)


_RESIDUE = [
    r"-{2,}\s*Page \d+\s*-{2,}",
    r"(?im)^.*\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}.*$",
    r"(?im)^.*\b(sent|date|subject|to|cc|from)\s*:.*$",
    r"(?i)\[?mailto:[^\]\s]*\]?|<?\s*inline[^>\n]*>?|</?div[^>]*>",
    r"(?i)sorry for (all )?(the )?ty\w*s?\s*\.?",
    r"(?i)\bforwarded message\b",
]


def final_clean(s: pd.Series) -> pd.Series:
    for p in _RESIDUE:
        s = s.str.replace(p, " ", regex=True)
    return s.str.replace(r"[ \t]{2,}", " ", regex=True).str.replace(r"\n{2,}", "\n", regex=True).str.strip()
