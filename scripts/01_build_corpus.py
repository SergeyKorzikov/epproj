"""Stage 1: download public text datasets and build the de-duplicated message corpus.

Outputs (in --out):
  messages_all.parquet   every parsed message (him + others), de-duplicated
  messages_je.parquet    only messages authored by Epstein  ("his voice")
  corpus_stats.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from epf.sources import dedupe, download, load_doj, load_house_muneeb, load_house_tobe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="data/out")
ap.add_argument("--limit-files", type=int, default=None, help="process only N DOJ shards (for a quick test)")
ap.add_argument("--no-download", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if not a.no_download:
    download(a.raw)

rows = []
print("House Oversight (muneeb)…")
rows += load_house_muneeb(os.path.join(a.raw, "emails_muneeb.parquet"))
print("House Oversight (to-be)…")
rows += load_house_tobe(os.path.join(a.raw, "emails_tobe.parquet"))
print("DOJ documents…")
for chunk in load_doj(os.path.join(a.raw, "docs", "*.parquet"), a.limit_files):
    rows += chunk

df = pd.DataFrame(rows)
n_raw = len(df)
df = dedupe(df)
df["year"] = df.ts.dt.year
df.to_parquet(os.path.join(a.out, "messages_all.parquet"))
je = df[df.is_epstein].reset_index(drop=True)
je.to_parquet(os.path.join(a.out, "messages_je.parquet"))

stats = {
    "segments_raw": n_raw,
    "messages_dedup": len(df),
    "je_messages": len(je),
    "je_words": int(je.body.str.split().str.len().sum()),
    "je_dated_share": round(float(je.ts.notna().mean()), 3),
    "je_by_year": {int(k): int(v) for k, v in je.year.value_counts().sort_index().items()},
    "je_by_source": je.source.str.split("_").str[0].value_counts().to_dict(),
}
json.dump(stats, open(os.path.join(a.out, "corpus_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
