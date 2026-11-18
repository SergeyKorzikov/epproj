"""Split raw OCR'd e-mail documents into individual messages and attribute authorship.

The DOJ OCR text comes in several header layouts:
    From: X            |  From:\nX\nTo:\nY\nSent:\n...
plus nested quotes ("On ... wrote:", "-----Original Message-----", forwards).
We split a document into message segments, parse headers, clean bodies.
"""
from __future__ import annotations

import hashlib
import re
import warnings

warnings.filterwarnings("ignore", module="dateutil")
from dataclasses import dataclass, field
from datetime import datetime

from dateutil import parser as dparser

# ---- who is "him" -------------------------------------------------------
EPSTEIN_RE = re.compile(
    r"jeevacation|jeffrey\s*e\.|jeffrey\s+epstein|^\s*je\s*$|^\s*j\s*(<|\[|$)|^\s*jee\b",
    re.I,
)

HEADER_KEYS = ("from", "to", "cc", "bcc", "sent", "date", "subject", "attachments", "importance")
_HDR_LINE = re.compile(r"^\s*(from|to|cc|bcc|sent|date|subject|attachments|importance)\s*:\s*(.*)$", re.I)

BOUNDARY_RE = re.compile(
    r"(^-{2,}\s*original message\s*-{2,}.*$"
    r"|^-{3,}\s*forwarded message\s*-{2,}.*$"
    r"|^begin forwarded message:.*$"
    r"|^on .{5,120}?wr[a-z]{1,4}:\s*$"
    r"|^on .{5,120}?$\n^.{0,80}wr[a-z]{1,4}:\s*$)",
    re.I | re.M,
)

# boilerplate to strip from bodies
BOILER_PATTERNS = [
    r"the information contained in this communication is.*?(all copies thereof,?\s*including all attachments\.?|$)",
    r"this email is confidential and subject to important disclaimers.*?email\.?",
    r"this communication may contain confidential.*?(communication\.|$)",
    r"confidential treatment requested by [^\n]+",
    r"\*{5,}",
    r"={5,}",
    r"sent from my (iphone|ipad|blackberry|samsung)[^\n]*",
    r"please note\s*the information contained.*?$",
    r"\b(EFTA_?\d{6,}|SDNY_[A-Z_]*\d{5,}|JPM-SDNY-\d+|HOUSE_OVERSIGHT_\d+)\b",
]
BOILER_RE = re.compile("|".join(f"(?:{p})" for p in BOILER_PATTERNS), re.I | re.S | re.M)
BATES_RE = re.compile(r"\b(EFTA[_A-Z0-9]*\d{6,}|SDNY_[A-Z_]*\d{5,}|JPM-SDNY-\d+|HOUSE_OVERSIGHT_\d+)\b")

# disclaimer lines survive OCR in fragments; drop a line if it carries >=2 legal-boilerplate cues
_DISC_WORDS = (
    "confidential", "privileged", "addressee", "unauthorized", "disclosure", "copying", "prohibited",
    "unlawful", "notify us", "destroy this", "all copies", "attachments", "inside information",
    "property of", "intended only", "in error", "received this", "communication",
)
_DISC_SOLO = re.compile(r"^(jee|jeffrey epstein|it is the property of|please note|including all attachments\.?|--)$", re.I)


def decode_qp(text: str) -> str:
    """Undo quoted-printable residue that survived OCR ('=nbsp;', '=20', soft breaks, 'co=tained')."""
    t = re.sub(r"([-=_*~#.+])\1{8,}", r"\1\1\1\1\1\1\1\1", text)  # 100k-char rule lines blow up regexes
    t = re.sub(r"[^\n]{3000,}", lambda m: m.group(0)[:3000], t)  # cap pathological lines
    t = re.sub(r"=\r?\n", "", t)
    t = re.sub(r"[=&]nbsp;?", " ", t, flags=re.I)
    t = re.sub(r"=(?:[0-9A-F]{2})", " ", t)
    t = re.sub(r"(?<=\w)=(?=\w)|(?<=\s)=(?=[a-z])|(?<=[a-z])=(?=\s)", "", t)  # 'co=tained' -> 'cotained' (lost char is unrecoverable)
    t = re.sub(r"<[^>\n]{1,40}>", lambda m: m.group(0) if "@" in m.group(0) else " ", t)  # stray html, keep <addr>
    return t


@dataclass
class Message:
    doc_id: str
    source: str
    order: int
    sender: str
    recipients: str
    cc: str
    subject: str
    ts: datetime | None
    ts_raw: str
    body: str
    is_epstein: bool
    quoted_depth: int = 0  # 0 = top message of the document
    extra: dict = field(default_factory=dict)

    @property
    def body_hash(self) -> str:
        norm = re.sub(r"\W+", "", self.body.lower())[:4000]
        return hashlib.md5(norm.encode()).hexdigest()


def is_epstein(sender: str | None) -> bool:
    return bool(sender) and bool(EPSTEIN_RE.search(sender.strip()))


def parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"\(.*?\)", "", raw)
    try:
        d = dparser.parse(raw, fuzzy=True, default=datetime(1900, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return None
    if d.year < 1990 or d.year > 2020:
        return None
    return d.replace(tzinfo=None)


_CUT_RE = re.compile(
    r"(please\s*note|the\s*information\s*contai|return\s*e-?mail\s*or|including\s*all\s*attach|copyright\s*-?\s*all\s*rights"
    r"|<!DOCTYPE\s*plist|confidential,\s*may\s*be|constitute\s*inside|this\s*e-?mail\s*is\s*confidential"
    r"|^[ \t]*on .{5,120}?wr[a-z]{1,4}:|^[ \t]*wr[a-z]{0,4}e?:[ \t]*$|-{2,}\s*original\s*message|-{3,}\s*forwarded"
    r"|^[ \t]*from:[ \t]|^[ \t]*sent:[ \t]|^[ \t]*_{10,})",
    re.I | re.M,
)
SIGNOFF_RE = re.compile(r"sorry for all the typos\s*\.?", re.I)  # his iPhone signature; counted separately


def clean_body(text: str) -> str:
    text = decode_qp(text)
    m = _CUT_RE.search(text)
    if m:
        text = text[: m.start()]
    w = re.search(r"\bwr[a-z]{1,4}:|conversation-id|date-last-viewed|gmail-label-ids", text, re.I)
    if w:  # quoted header ("X <addr> wrote:") -> drop that whole line and everything after
        text = text[: text.rfind("\n", 0, w.start()) + 1] if "\n" in text[: w.start()] else ""
    text = SIGNOFF_RE.sub(" ", text)
    text = BOILER_RE.sub(" ", text)
    text = BATES_RE.sub(" ", text)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(">"):
            continue
        low = s.lower()
        if _DISC_SOLO.match(s) or sum(w in low for w in _DISC_WORDS) >= 2:
            continue
        if re.fullmatch(r"[\W\d_]{0,6}", s):
            continue
        lines.append(s)
    out = "\n".join(lines)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _normalize_headers(text: str) -> str:
    """Turn 'From:\\nValue' layouts into 'From: Value'."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^\s*(from|to|cc|bcc|sent|date|subject|attachments)\s*:\s*$", lines[i], re.I)
        if m and i + 1 < len(lines) and not _HDR_LINE.match(lines[i + 1]):
            out.append(f"{m.group(1)}: {lines[i + 1].strip()}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _parse_segment(seg: str) -> tuple[dict, str]:
    hdr: dict[str, str] = {}
    body_lines: list[str] = []
    in_hdr = True
    seen_any = False
    for ln in seg.splitlines():
        m = _HDR_LINE.match(ln) if in_hdr else None
        if m:
            key = m.group(1).lower()
            val = m.group(2).strip()
            # OCR glitch: "To: Jes Staley <Subject: 5/16/2010 4:08:34 PM"
            inner = re.search(r"(subject|sent|date)\s*:\s*(.*)$", val, re.I)
            if inner and key in ("to", "from", "cc"):
                hdr.setdefault(inner.group(1).lower(), inner.group(2))
                val = val[: inner.start()].strip()
            hdr.setdefault(key, val)
            seen_any = True
            continue
        if in_hdr and seen_any and ln.strip() == "":
            in_hdr = False
            continue
        if in_hdr and seen_any:
            in_hdr = False
        body_lines.append(ln)
    return hdr, "\n".join(body_lines)


def split_document(doc_id: str, text: str, source: str, doc_date: str | None = None) -> list[Message]:
    if not text:
        return []
    text = _normalize_headers(decode_qp(text))
    # cut at quote/forward boundaries; each piece may contain its own headers
    pieces, last = [], 0
    for m in BOUNDARY_RE.finditer(text):
        pieces.append((text[last:m.start()], m.group(0)))
        last = m.end()
    pieces.append((text[last:], None))

    msgs: list[Message] = []
    prev_boundary = None
    for depth, (seg, boundary) in enumerate(pieces):
        hdr, body = _parse_segment(seg)
        sender = hdr.get("from", "")
        ts_raw = hdr.get("sent") or hdr.get("date") or ""
        if not sender and prev_boundary:
            # "On <date>, <name> wrote:" style
            mm = re.match(r"(?is)on (.+?),? (?:at )?(.*?)\s*<?[^<]*wr[a-z]{1,4}:", prev_boundary.replace("\n", " "))
            if mm:
                ts_raw = ts_raw or mm.group(1)
                sender = mm.group(2)
        if depth == 0 and not ts_raw:
            ts_raw = doc_date or ""
        body = clean_body(body)
        prev_boundary = boundary
        if not sender and not body:
            continue
        msgs.append(
            Message(
                doc_id=doc_id,
                source=source,
                order=len(msgs),
                sender=sender,
                recipients=hdr.get("to", ""),
                cc=hdr.get("cc", ""),
                subject=hdr.get("subject", ""),
                ts=parse_date(ts_raw),
                ts_raw=ts_raw,
                body=body,
                is_epstein=is_epstein(sender),
                quoted_depth=depth,
            )
        )
    return msgs
