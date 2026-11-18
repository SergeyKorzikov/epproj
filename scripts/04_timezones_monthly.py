"""Stage 4 checks: time zones of his messages and the monthly test of hypothesis 11.6.

Run from the repo root after scripts/01_build_corpus.py:
    python scripts/04_timezones_monthly.py
Steps (analysis/tz/): tz1 parse offsets -> tz2 compare hourly profiles -> tz3 infer local time
from explicit device offsets -> m1 monthly series + PELT change points -> m2 permutation test
against health-record dates -> m3 time-of-day decline and after-late-night test. ver.py checks
quoted IDs against clean packs and the corpus.
"""
import runpy, sys
steps = ["tz1", "tz2", "tz3", "m1", "m2", "m3"]
for s in steps:
    print(f"=== {s}")
    sys.argv = [s, "data/out/messages_je.parquet"]
    runpy.run_path(f"analysis/tz/{s}.py", run_name="__main__")
