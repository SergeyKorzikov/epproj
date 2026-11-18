"""Stage 2b: reconstruct his library — what he bought, linked, and talked about reading.

Sources
  1. Purchase confirmations in the DOJ corpus (Amazon / Kindle / Audible / Barnes & Noble),
     with buyer attribution (addressed to him vs. someone else on his accounts).
  2. Amazon / Goodreads links inside his own messages and messages sent to him.
  3. His own messages that talk about reading (exported as an evidence pack for qualitative reading).

Outputs (in --out):
  library.csv                 one row per (title, source event)
  library_titles.csv          de-duplicated titles with first/last date and evidence count
  packs/14_reading.md         his messages about books/reading
"""
import argparse
import os
import re
import sys
from urllib.parse import unquote

import duckdb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from epf.emailparse import parse_date  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="data/out")
a = ap.parse_args()

FORMATS = r"Kindle Edition|Hardcover|Paperback|Mass Market Paperback|Audio CD|Audible[^\]\n]*|Audiobook|Board book|Library Binding|Leather Bound|NOOK Book|DVD|Blu-ray"
BRACKET = re.compile(rf"^\s*(?:\d+\s+of:\s*)?(?P<t>[^\n\[\]$]{{3,200}}?)\s*\[(?P<f>{FORMATS})\]", re.I | re.M)
SUBJECT = re.compile(r"your (?:amazon\.com|audible|kindle)? ?order of\s+[\"“]?(?P<t>[^\n\"”]{3,200}?)[\"”]?(?:\s+and \d+ more items?)?\s*(?:\.|$)", re.I | re.M)
PLACED = re.compile(r"placed on [^\n]+\n(?P<block>(?:[^\n$]{2,120}\n){1,4})\s*\$\d", re.I)
BUYER_HIM = re.compile(r"(hello|thanks for your order,|dear)\s+jeffrey\s+(e\.?\s+)?epstein|billing address:\s*\n?\s*jeffrey epstein|jeevacation|jeeproject|jeffrey@", re.I)
STOP = re.compile(r"^(order|item|subtotal|total|tax|sold by|shipping|qty|quantity|price|kindle|delivery|your account)\b", re.I)

con = duckdb.connect()
con.execute("SET enable_progress_bar=false")
docs = con.execute(rf"""
    select file_key, dataset, date, full_text from '{a.raw}/docs/*.parquet'
    where regexp_matches(full_text, '(?i)(amazon\.com|kindle|audible|barnes ?& ?noble|nook|abebooks)')
      and regexp_matches(full_text, '(?i)(order|purchase|placed on|auto delivered|thank you for shopping)')
""").fetchall()
print("purchase-like documents:", len(docs))

rows = []
for key, ds, date, text in docs:
    text = re.sub(r"[ \t]+", " ", text or "")
    buyer = "him" if BUYER_HIM.search(text) else "other/unknown"
    bookish_doc = bool(re.search(r"kindle (edition|store|library|ebooks?)|audible|\bbooks?\b|hardcover|paperback|nook book|barnes", text, re.I))
    found = set()
    for m in BRACKET.finditer(text):
        found.add((m.group("t").strip(" .:-"), m.group("f")))
    for m in SUBJECT.finditer(text):
        found.add((m.group("t").strip(" .:-"), ""))
    for m in PLACED.finditer(text):
        t = " ".join(ln.strip() for ln in m.group("block").splitlines() if ln.strip() and not STOP.match(ln.strip()))
        if t:
            found.add((t, ""))
    for t, f in found:
        if 3 <= len(t) <= 200 and not STOP.match(t):
            rows.append(dict(title=t, format=f, source="purchase", buyer=buyer, date=parse_date(date), doc_id=key,
                             is_book=bool(f) or (bookish_doc and len(t.split()) >= 2)))

# photographed books (search/inventory photos classified by the OCR pipeline)
photos = con.execute(rf"""
    select file_key, document_type, full_text from '{a.raw}/docs/*.parquet'
    where lower(document_type) in ('book cover','bookshelf','book page','book spine','library')
""").fetchall()
for key, dt, text in photos:
    lines = [ln.strip() for ln in (text or "").splitlines()
             if len(ln.strip()) > 3 and not re.match(r"^(EFTA|SDNY|CONFIDENTIAL|ISBN|\d)", ln.strip())]
    # keep only title-like lines (short, mostly capitals) — never the page text itself
    heads = [ln for ln in lines[:4] if len(ln.split()) <= 8 and sum(c.isupper() for c in ln) >= 0.6 * sum(c.isalpha() for c in ln)
             and not re.match(r"^(CHAPTER|---|PAGE)", ln, re.I)]
    if heads:
        rows.append(dict(title=" / ".join(heads[:2])[:120], format=dt, source="photo", buyer="", date=None,
                         doc_id=key, is_book=True))

# links in messages
msgs = pd.read_parquet(os.path.join(a.out, "messages_all.parquet"), columns=["doc_id", "ts", "is_epstein", "body"])
LINK = re.compile(r"(?:amazon\.com|amzn\.com)/(?P<slug>[A-Za-z0-9%'\-]{6,160})/(?:dp|gp/product)/(?P<asin>[0-9X]{10})|goodreads\.com/book/show/\d+[.-](?P<gr>[A-Za-z0-9_\-]{4,160})", re.I)
for r in msgs[msgs.body.str.contains(r"amazon\.com/|amzn\.com/|goodreads\.com/book", case=False, regex=True)].itertuples():
    for m in LINK.finditer(r.body):
        slug = m.group("slug") or m.group("gr")
        title = unquote(slug).replace("-", " ").replace("_", " ").strip()
        rows.append(dict(title=title, format="", source="link_by_him" if r.is_epstein else "link_to_him",
                         buyer="", date=r.ts, doc_id=r.doc_id, asin=m.group("asin"), is_book=True))

lib = pd.DataFrame(rows)
lib["norm"] = (lib.title.str.lower().str.replace(r"\(.*?\)|:.*$", "", regex=True)
               .str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
lib = lib[(lib.norm.str.len() >= 3) & lib.is_book.fillna(False).astype(bool)]
PRODUCT = r"\b(\d+ ?(oz|ml|pk|pack|pcs|mm|inch|in)\b|set of|kit|filter|vacuum|purifier|plunger|shears|knife|hat|frames?|battery|slippers|printer|camera|gopro|nikon|belkin|panasonic|wig|jar|splenda|tootsie|toilet|teapot|black \+ decker|uxcell|my passport|scientific|kitty|wishpets|alessi|toefl)"
lib = lib[~(lib.source.eq("purchase") & lib.format.fillna("").eq("") & lib.title.str.contains(PRODUCT, case=False, regex=True))]
lib = lib[~lib.title.str.contains(r"order id|shipment|has shipped|<table|delivery estimate|\bqty\b|^by ", case=False, regex=True)]
lib.to_csv(os.path.join(a.out, "library.csv"), index=False)

titles = (lib.sort_values("date").groupby("norm")
          .agg(title=("title", "first"), formats=("format", lambda x: ", ".join(sorted({f for f in x if f}))),
               sources=("source", lambda x: ", ".join(sorted(set(x)))),
               bought_by_him=("buyer", lambda x: (x == "him").any()),
               n=("doc_id", "nunique"), first=("date", "min"), last=("date", "max"),
               docs=("doc_id", lambda x: " ".join(sorted(set(x))[:5])))
          .sort_values("first"))
titles.to_csv(os.path.join(a.out, "library_titles.csv"))
print(titles.sources.value_counts())
print("titles:", len(titles), "| bought by him:", int(titles.bought_by_him.sum()))

# reading pack: his own words about books
je = msgs[msgs.is_epstein]
READ = r"\b(have you read|did you read|read the book|reading|i read|great book|good book|the book|his book|her book|new book|novel|author|chapter|send me the book|kindle)\b"
rd = je[je.body.str.contains(READ, case=False, regex=True) & je.body.str.split().str.len().between(4, 400)]
rd = rd.drop_duplicates("body").sort_values("ts")
os.makedirs(os.path.join(a.out, "packs"), exist_ok=True)
with open(os.path.join(a.out, "packs", "14_reading.md"), "w", encoding="utf-8") as f:
    f.write(f"# Пакет 14 — он о книгах и чтении\n\nЭлементов: {len(rd)}\n\n")
    for r in rd.itertuples():
        when = r.ts.strftime("%Y-%m-%d") if pd.notna(r.ts) else "без даты"
        f.write(f"### [{r.doc_id}] {when}\n{r.body[:1200]}\n\n")
print("reading pack items:", len(rd))
