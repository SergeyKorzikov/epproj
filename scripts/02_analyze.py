"""Stage 2: quantitative profile (no LLM) + evidence packs for the qualitative stage.

Outputs (in --out):
  metrics.json                rhythms, communication system, latency, corpus facts
  stylometry_by_year.csv      LIWC-lite pooled rates per year
  stylometry_contrast.csv     him vs his correspondents
  distinctive_words.csv       log-odds 'his words' vs 'theirs'
  themes.csv                  share of his messages touching each lens, by year
  packs/*.md                  privacy-masked evidence packs (NOT for publication)
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from epf import behavior, packs, stylometry  # noqa: E402
from epf.sources import near_dedupe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/out")
ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config"))
ap.add_argument("--no-packs", action="store_true")
a = ap.parse_args()

df = near_dedupe(pd.read_parquet(os.path.join(a.out, "messages_all.parquet")))
je = df[df.is_epstein].copy()
others = df[~df.is_epstein]
print(f"his messages: {len(je):,}  others: {len(others):,}")

themes = behavior.load_themes(os.path.join(a.config, "themes.yaml"))
tags = behavior.tag_themes(je, themes)
behavior.theme_profile(je, tags).to_csv(os.path.join(a.out, "themes.csv"))

stylometry.grouped_profile(je, "year").to_csv(os.path.join(a.out, "stylometry_by_year.csv"))
stylometry.contrast(je, others).to_csv(os.path.join(a.out, "stylometry_contrast.csv"))
stylometry.distinctive_words(je, others).to_csv(os.path.join(a.out, "distinctive_words.csv"), index=False)

ml = stylometry.message_level(je)
metrics = {
    "messages": {
        "n": len(je),
        "median_words": float(ml.n_words.median()),
        "share_le_5_words": float((ml.n_words <= 5).mean()),
        "share_questions": float(ml.has_question.mean()),
        "share_imperative_start": float(ml.imperative.mean()),
        "share_all_lowercase": float(ml.all_lower.mean()),
        "share_link_only": float(ml.link_only.mean()),
        "share_exclaim": float(ml.exclaim.mean()),
    },
    "rhythms": behavior.rhythms(je),
    "communication_system": behavior.communication_system(df),
    "latency": behavior.response_latency(df),
    "sensitive_share": float(tags["th_sensitive_abuse"].gt(0).mean()),
}
top = metrics["communication_system"].pop("_top_contacts")
pd.DataFrame(top, columns=["contact", "n"]).to_csv(os.path.join(a.out, "_local_top_contacts.csv"), index=False)
json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in metrics.items() if k != "rhythms"}, indent=1, default=str))

if not a.no_packs:
    rep = packs.build_all(je, df, tags, themes, os.path.join(a.out, "packs"),
                          os.path.join(a.config, "public_figures.txt"))
    print("packs:", rep)
