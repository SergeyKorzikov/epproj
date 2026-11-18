"""Compact reader for evidence packs: prints the N most substantive items, one line each.

usage: python peek.py PACK.md [N] [MAXCHARS] [REGEX]
Substance = word count x idiolect score; items starting like a forwarded/foreign text are skipped.
"""
import re
import sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 280
rx = re.compile(sys.argv[4], re.I) if len(sys.argv) > 4 else None

OWN = [re.compile(p) for p in (r"(?<![A-Za-z'])i(?![A-Za-z'])", r"\b(dont|im|cant|wont|didnt|isnt|ive|youre|thats)\b", r"\s\.(\s|$)", r"(?m)^[a-z]")]
items = re.split(r"(?m)^(?=### \[)", open(path, encoding="utf-8").read())[1:]
rows = []
for it in items:
    head, _, body = it.partition("\n")
    body = "\n".join(l for l in body.splitlines() if not l.startswith(">")).strip()
    if rx and not rx.search(body):
        continue
    w = len(body.split())
    if w < 10:
        continue
    own = sum(len(r.findall(body)) for r in OWN) / max(w, 1) * 100
    if own < 1.0:
        continue
    m = re.match(r"### \[([^\]]+)\] (\S+)", head)
    rows.append((min(w, 120) * (1 + own / 5), m.group(1), m.group(2), re.sub(r"\s+", " ", body)))
rows.sort(reverse=True)
for _, i, d, b in sorted(rows[:n], key=lambda r: r[2]):
    print(f"[{i}|{d[:10]}] {b[:mx]}")
print(f"-- {len(rows)} substantive of {len(items)}", file=sys.stderr)
