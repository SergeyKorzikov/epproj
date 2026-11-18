"""Clean evidence pack for full manual reading: his text only, deduped, one item per line.

usage: python clean_pack.py PACK.md OUT.txt [MAXCHARS]
Keeps every item with >=4 words of own text and idiolect score >= 0.5; drops near-duplicates
(same first 120 normalized chars); strips quoted lines, signatures and boilerplate.
"""
import re, sys
src, out = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
OWN = [re.compile(p) for p in (r"(?<![A-Za-z'])i(?![A-Za-z'])", r"\b(dont|im|cant|wont|didnt|isnt|ive|youre|thats)\b", r"\s\.(\s|$)", r"(?m)^[a-z]")]
JUNK = re.compile(r"(?i)(sent from my|sent via|original message|forwarded m|disclaimer|confidential|unsubscribe|this message|privileged|--- page \d+ ---|please consider the environment|jeevacation@|e-mail is intended)")
items = re.split(r"(?m)^(?=### \[)", open(src, encoding="utf-8").read())[1:]
seen, rows = set(), []
for it in items:
    head, _, body = it.partition("\n")
    lines = [l for l in body.splitlines() if not l.startswith(">") and not JUNK.search(l)]
    body = re.sub(r"\s+", " ", " ".join(lines)).strip()
    w = len(body.split())
    if w < 8:
        continue
    own = sum(len(r.findall(body)) for r in OWN) / max(w, 1) * 100
    if own < 0.5 and w > 25:
        continue
    norm = re.sub(r"[^a-z]", "", body.lower())
    keys = {norm[:50], norm[-50:], norm[len(norm)//2:len(norm)//2+50]}
    if keys & seen:
        continue
    seen |= keys
    m = re.match(r"### \[([^\]#]+)[^\]]*\] (\S+)(?: \S+)? → (.*)", head)
    i, d, to = (m.group(1), m.group(2)[:10], m.group(3)[:30]) if m else (head[:20], "", "")
    rows.append(f"[{i}|{d}|{to}] {body[:mx]}")
open(out, "w", encoding="utf-8").write("\n".join(rows))
print(src, len(items), "->", len(rows), "items,", sum(map(len, rows)) // 1000, "k chars")
