"""
script_classifier.py -- writing-system detection and cross-script normalisation
views for the ML Challenge 2026 Business Entity Resolution dataset.

Stdlib only. No network, no external data source: the transliteration tables are
derived from Python's own unicodedata and from the parallel layout Unicode gave
the Brahmic scripts.

Public API
----------
    script_of(text)             -> script tag ('latin', 'deva', 'telu', ...)
    transliterate(text)         -> Latin phonetic skeleton
    phonetic(text)              -> spelling-insensitive key
    consonant_skeleton(text)    -> vowel-free key (the cross-script matcher)
    fold_unicode(text)          -> NFKD + Latin-diacritic removal
    script_distribution(paths)  -> counts per file/country, for the report

WHY THE SCRIPT TAG MATTERS
--------------------------
~15-19% of Source 2/3 business names are non-ASCII, and a single ground-truth
cluster routinely holds the same business written in Latin *and* an Indic
script. Any record whose script is not recognised never gets transliterated, so
it can only ever match on address -- a silent recall hole. The classifier is
therefore the gate for the whole cross-script path, and its 'other' bucket is
the thing to watch: see SCRIPT_DISTRIBUTION_PITFALLS below.
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

# Token-level memoisation size, per cache, per process. The token vocabulary
# with real repetition is a few tens of thousands of entries, so a larger cache
# buys almost nothing and costs a lot: three caches at 2**19 entries across 12
# workers is several GB, which on a 16GB machine is enough to push the run into
# paging. That is what made one file in the first full pass run 35x slower than
# its siblings. Override with NORM_CACHE_BITS if you have memory to spare.
_CACHE = 1 << int(os.environ.get("NORM_CACHE_BITS", "16"))

# ---------------------------------------------------------------------------
# 1. Unicode folding
# ---------------------------------------------------------------------------
# Only Latin combining marks (U+0300-U+036F) are stripped. Indic vowel signs,
# viramas and nuktas are combining marks too, and live outside that block: they
# must survive folding or the text is destroyed before it can be transliterated.
_LATIN_COMBINING = re.compile(r"[̀-ͯ]")


def fold_unicode(s):
    """NFKD + Latin-diacritic removal, then NFC. ASCII passes through.

    The trailing NFC matters: NFKD decomposes the two-part Indic vowel signs
    (Odia/Bengali/Tamil/Telugu/Kannada/Malayalam write "o" as an e-sign plus an
    aa-sign) and the syllable scanner consumes only one vowel sign per
    consonant, so without recomposition every such vowel reads as "e".
    """
    if s.isascii():
        return s
    folded = _LATIN_COMBINING.sub("", unicodedata.normalize("NFKD", s))
    return unicodedata.normalize("NFC", folded)


# ---------------------------------------------------------------------------
# 2. Script detection
# ---------------------------------------------------------------------------
# The nine Brahmic blocks present in this dataset, plus Latin. Unicode laid these
# out in parallel (inherited from ISCII), which section 3 exploits.
SCRIPT_BLOCKS = (
    ("deva", 0x0900, 0x097F),   # Devanagari  (Hindi, Marathi)
    ("beng", 0x0980, 0x09FF),   # Bengali
    ("guru", 0x0A00, 0x0A7F),   # Gurmukhi    (Punjabi)
    ("gujr", 0x0A80, 0x0AFF),   # Gujarati
    ("orya", 0x0B00, 0x0B7F),   # Odia
    ("taml", 0x0B80, 0x0BFF),   # Tamil
    ("telu", 0x0C00, 0x0C7F),   # Telugu
    ("knda", 0x0C80, 0x0CFF),   # Kannada
    ("mlym", 0x0D00, 0x0D7F),   # Malayalam
)
INDIC_SCRIPTS = tuple(tag for tag, _, _ in SCRIPT_BLOCKS)
ALL_SCRIPTS = ("latin",) + INDIC_SCRIPTS + ("mixed", "other")


def _block_of(o):
    for tag, lo, hi in SCRIPT_BLOCKS:
        if lo <= o <= hi:
            return tag
    return None


def script_of(s):
    """Classify a string's writing system.

    Returns one of ALL_SCRIPTS. 'mixed' means more than one system is present
    (common here: a Latin brand name beside an Indic legal suffix). 'other'
    means no alphabetic character from a known system was found at all -- a
    digits-only or punctuation-only string, or an empty one.
    """
    if not s:
        return "other"
    seen = set()
    for ch in s:
        o = ord(ch)
        if o < 0x0250:
            if ch.isalpha():
                seen.add("latin")
        else:
            tag = _block_of(o)
            if tag is not None and ch.isalpha():
                seen.add(tag)
    if not seen:
        return "other"
    if len(seen) > 1:
        return "mixed"
    return seen.pop()


def scripts_present(s):
    """Every writing system in the string, for diagnosing 'mixed'."""
    out = set()
    for ch in s:
        o = ord(ch)
        if o < 0x0250:
            if ch.isalpha():
                out.add("latin")
        elif ch.isalpha():
            tag = _block_of(o)
            if tag:
                out.add(tag)
    return out


# ---------------------------------------------------------------------------
# 3. Transliteration
# ---------------------------------------------------------------------------
# Consonants map to the bare consonant sound; the inherent "a" is appended by
# the scanner unless a virama or vowel sign cancels it. This is a phonetic
# skeleton for matching, not a faithful ISO-15919 romanisation.
_DEVA_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "ऩ": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ऱ": "r", "ल": "l",
    "ळ": "l", "ऴ": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h",
}
_DEVA_NUKTA = {
    "क": "q", "ख": "kh", "ग": "g", "ज": "z",
    "ड": "r", "ढ": "rh", "फ": "f", "य": "y",
}
_DEVA_VOWEL = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u",
    "ऊ": "uu", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au", "ऑ": "o", "ऒ": "o", "ऍ": "e", "ऎ": "e",
}
_DEVA_MATRA = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
}
_DEVA_SIGN = {"ं": "n", "ँ": "n", "ः": "h"}
_DEVA_VIRAMA = "्"
_DEVA_NUKTA_CH = "़"
_DEVA_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

# Tamil is the one script that does NOT follow the Devanagari layout: it has a
# reduced inventory with no voiced/aspirated distinctions, so an offset mapping
# would produce nonsense. It gets its own table.
_TAML_CONS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "ny", "ட": "t",
    "ண": "n", "த": "th", "ந": "n", "ப": "p", "ம": "m",
    "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "l",
    "ள": "l", "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh",
    "ஸ": "s", "ஹ": "h",
}
_TAML_VOWEL = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u",
    "ஊ": "uu", "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o",
    "ஓ": "oo", "ஔ": "au",
}
_TAML_MATRA = {
    "ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu",
    "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo",
    "ௌ": "au",
}
_TAML_VIRAMA = "்"
_TAML_DIGITS = {chr(0x0BE6 + i): str(i) for i in range(10)}


def _shift(table, delta):
    """Offset every key into a sibling block. Keys may be multi-character --
    the Indic letter-names used for acronyms are two or three codepoints."""
    return {"".join(chr(ord(c) + delta) for c in k): v
            for k, v in table.items()}


# Points where the parallel-layout assumption breaks down: a sibling script
# either relocated a sign to a non-offset position, or added a letter with no
# Devanagari counterpart to derive from at all. Found by running the
# classifier on real records rather than by inspecting the tables up front --
# both are single real names in the challenge data. Applied as overrides on
# top of the derived table, the same pattern _mlym_fixup already uses below.
_SCRIPT_OVERRIDES = {
    # Gurmukhi's nasalisation mark is TIPPI (U+0A70), not the codepoint at
    # Devanagari anusvara's offset position (which lands on U+0A02, BINDI --
    # a real but rarer Gurmukhi sign). Real record: "ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ".
    "guru": {"sign": {"ੰ": "n"}},
    # Odia YYA (U+0B5F) is an additional consonant for loanwords with no
    # Devanagari source letter to offset from -- nothing at the parallel
    # position derives it. Real record: "ବିଜୟ କନଷ୍ଟ୍ରକସନ୍ସ୍".
    "orya": {"cons": {"ୟ": "y"}},
}


def _build_script_tables():
    """Derive every Brahmic table from Devanagari by codepoint offset.

    Unicode encoded these scripts in parallel positions, so Bengali's KA sits
    exactly 0x80 above Devanagari's KA, Telugu's 0x300 above, and so on. That
    makes seven of the eight sibling tables derivable instead of hand-written --
    less code and far less opportunity for a typo. Tamil is supplied directly.
    A handful of signs and extra letters break the parallel layout; those are
    patched in via _SCRIPT_OVERRIDES rather than trusted to the offset.
    """
    tables = {}
    for tag, lo, _hi in SCRIPT_BLOCKS:
        if tag == "taml":
            tables[tag] = dict(cons=_TAML_CONS, vowel=_TAML_VOWEL,
                               matra=_TAML_MATRA, sign={},
                               virama=_TAML_VIRAMA, digits=_TAML_DIGITS,
                               nukta=None, nukta_ch=None)
            continue
        d = lo - 0x0900
        tables[tag] = dict(
            cons=_shift(_DEVA_CONS, d), vowel=_shift(_DEVA_VOWEL, d),
            matra=_shift(_DEVA_MATRA, d), sign=_shift(_DEVA_SIGN, d),
            virama=chr(ord(_DEVA_VIRAMA) + d),
            digits={chr(0x0966 + d + i): str(i) for i in range(10)},
            nukta=_shift(_DEVA_NUKTA, d), nukta_ch=chr(ord(_DEVA_NUKTA_CH) + d),
        )
        for field, extra in _SCRIPT_OVERRIDES.get(tag, {}).items():
            tables[tag][field].update(extra)
    return tables


_TABLES = _build_script_tables()


def _translit_scan(s, cons, vowel, matra, sign, virama, digits, nukta=None,
                   nukta_ch=None):
    """Syllable-aware scan, shared by every Brahmic script."""
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in cons:
            base = cons[ch]
            i += 1
            if nukta_ch and i < n and s[i] == nukta_ch:
                base = (nukta or {}).get(ch, base)
                i += 1
            if i < n and s[i] == virama:
                out.append(base)
                i += 1
            elif i < n and s[i] in matra:
                out.append(base + matra[s[i]])
                i += 1
            else:
                out.append(base + "a")            # inherent vowel
            while i < n and s[i] in sign:
                out.append(sign[s[i]])
                i += 1
        elif ch in vowel:
            out.append(vowel[ch])
            i += 1
            while i < n and s[i] in sign:
                out.append(sign[s[i]])
                i += 1
        elif ch in digits:
            out.append(digits[ch])
            i += 1
        elif ch in matra or ch == virama or ch in sign or (nukta_ch and ch == nukta_ch):
            i += 1                                # stray sign, ignore
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Latin letter names written in Indic script: the Devanagari spelling of "SS" is
# the letter-name for S twice -- an acronym, not a word. A token is read as an
# acronym only when it decomposes *entirely* into letter names, which stops the
# rule firing inside ordinary words.
_DEVA_LETTER = {
    "ए": "a", "बी": "b", "सी": "c", "डी": "d",
    "ई": "e", "एफ": "f", "जी": "g", "एच": "h",
    "आई": "i", "जे": "j", "के": "k",
    "एल": "l", "एम": "m", "एन": "n", "ओ": "o",
    "पी": "p", "क्यू": "q", "आर": "r",
    "एस": "s", "टी": "t", "यू": "u",
    "वी": "v", "डब्ल्यू": "w",
    "एक्स": "x", "वाई": "y",
    "जेड": "z",
}
# Tamil has no separate P/B letter; LLP / PVT / P LTD dominate B-initialisms in
# this data, so the ambiguity resolves to p.
_TAML_LETTER = {
    "ஏ": "a", "பி": "p", "சி": "c", "டி": "d",
    "ஈ": "e", "எஃப்": "f", "ஜி": "g",
    "எச்": "h", "ஐ": "i", "ஜே": "j",
    "கே": "k", "எல்": "l", "எம்": "m",
    "என்": "n", "ஓ": "o", "ஆர்": "r",
    "எஸ்": "s", "யூ": "u", "வி": "v",
    "எக்ஸ்": "x", "ஒய்": "y",
}
_LETTER_TABLES = {"taml": _TAML_LETTER}
for _tag, _lo, _hi in SCRIPT_BLOCKS:
    if _tag not in ("taml",):
        _LETTER_TABLES[_tag] = _shift(_DEVA_LETTER, _lo - 0x0900)
_LETTER_MAX = {k: max(len(x) for x in v) for k, v in _LETTER_TABLES.items()}

_FINAL_SCHWA = re.compile(r"a\b")

# Malayalam pre-pass. The chillu letters are bare consonants carrying no
# inherent vowel and sit outside the Devanagari-parallel range, and the RRA
# geminate (U+0D31 virama U+0D31) is pronounced "tt", not "rr".
_MLYM_CHILLU = {
    "ൺ": "ണ്", "ൻ": "ന്",
    "ർ": "ര്", "ൽ": "ല്",
    "ൾ": "ള്", "ൿ": "ക്",
}


def _mlym_fixup(tok):
    tok = tok.replace("റ്റ", "ട്ട")
    for a, b in _MLYM_CHILLU.items():
        if a in tok:
            tok = tok.replace(a, b)
    return tok


def _as_acronym(tok, table, maxlen):
    """Greedy longest-match decomposition into Latin letter names."""
    out = []
    i, n = 0, len(tok)
    while i < n:
        for ln in range(min(maxlen, n - i), 0, -1):
            piece = tok[i:i + ln]
            if piece in table:
                out.append(table[piece])
                i += ln
                break
        else:
            return None
    return "".join(out) if len(out) >= 2 else None


@lru_cache(maxsize=_CACHE)
def _translit_token(tok):
    """One whitespace-free token, Indic -> Latin."""
    tag = None
    for ch in tok:
        tag = _block_of(ord(ch))
        if tag:
            break
    if tag is None:
        return tok
    if tag == "mlym":
        tok = _mlym_fixup(tok)
    acr = _as_acronym(tok, _LETTER_TABLES[tag], _LETTER_MAX[tag])
    if acr:
        return acr
    t = _translit_scan(tok, **_TABLES[tag])
    # Final-schwa deletion. Cosmetic for the consonant skeleton (which drops all
    # vowels anyway) but it keeps the readable forms honest: Hindi and its
    # northern siblings do not pronounce the trailing inherent vowel.
    if tag in ("deva", "beng", "guru", "gujr", "orya"):
        t = _FINAL_SCHWA.sub("", t)
    return t


def transliterate(s):
    """Indic -> Latin phonetic skeleton. Latin input passes through unchanged."""
    if s.isascii():
        return s
    return " ".join(_translit_token(t) for t in s.split())


# ---------------------------------------------------------------------------
# 4. Phonetic reduction
# ---------------------------------------------------------------------------
# Collapses the spelling differences that survive transliteration and typos:
# aspirated digraphs, vowel length, English c/k/s ambiguity, and the p/b merge
# the Tamil script forces.
_DIGRAPHS = (
    ("chh", "\x01"), ("ch", "\x01"), ("sh", "s"), ("ph", "f"), ("th", "t"),
    ("dh", "d"), ("bh", "b"), ("gh", "g"), ("jh", "j"), ("kh", "k"),
    ("zh", "j"), ("ck", "k"), ("qu", "k"), ("wr", "r"), ("kn", "n"),
    ("wh", "v"), ("mb", "m"),
)
_SINGLES = str.maketrans({"q": "k", "w": "v", "z": "s", "y": "i", "b": "p"})
_SOFT_C = re.compile(r"c(?=[eiy])")
_RUNS = re.compile(r"(.)\1+")
_VOWELS = re.compile(r"[aeiou]")


@lru_cache(maxsize=_CACHE)
def _phon_token(t):
    for a, b in _DIGRAPHS:
        if a in t:
            t = t.replace(a, b)
    t = _SOFT_C.sub("s", t)
    t = t.replace("c", "k").translate(_SINGLES).replace("\x01", "c")
    return _RUNS.sub(r"\1", t)


def phonetic(s):
    """Spelling-insensitive phonetic key. Vowels kept, but normalised."""
    return " ".join(_phon_token(t) for t in s.lower().split())


@lru_cache(maxsize=_CACHE)
def _cons_token(tok):
    t = _phon_token(tok)
    if not t:
        return ""
    # A vowel-initial token keeps a placeholder rather than its own first
    # vowel: transliteration is unreliable about vowel quality, so Gurmukhi
    # "iisat" and Latin "east" must not diverge on that character alone.
    head = "a" if t[0] in "aeiou" else t[0]
    return _RUNS.sub(r"\1", head + _VOWELS.sub("", t[1:]))


def consonant_skeleton(s):
    """Vowel-free skeleton -- the representation that survives transliteration.

    The Devanagari for "private" romanises to "praaivet" and the Latin form is
    "private"; both reduce to "prvt".
    """
    return " ".join(_cons_token(t) for t in s.lower().split())


# ---------------------------------------------------------------------------
# 5. Script distribution report
# ---------------------------------------------------------------------------
SCRIPT_DISTRIBUTION_PITFALLS = """
Two ways this report degenerates to 100% 'other', both observed:

1. Classifying a POST-NORMALISATION column. script_of() must see the raw
   business_name. Run it on a normalised column and every record comes back
   wrong, because normalisation has already transliterated the text to ASCII
   ('latin' everywhere) or blanked it. Measured on 20k rows of
   train_source2: name_nums -> 100% 'other' (digits only), name_translit ->
   94% 'other' (empty by design for Latin records), name_script -> 100%
   'latin' (it is a tag, not text). Only the raw name gives a real answer.

2. An incomplete script table. Any block the classifier does not know falls
   into 'other'. This dataset carries NINE Brahmic scripts, not two: before
   Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia and Gurmukhi were
   added, ~221k alphabetic characters per 450k rows were being bucketed as
   'other' and silently skipped by transliteration.

The report below guards against both: it resolves the column BY HEADER NAME and
refuses a column that is not raw text, and it flags any run where 'other'
dominates.
"""

_RAW_NAME_COLUMN = "business_name"
_DERIVED_HINT = ("name_", "addr_")


def script_distribution(paths, column=_RAW_NAME_COLUMN, limit_rows=None,
                        by_country=True):
    """Tally writing systems over one or more raw source TSVs.

    Reads with an explicit UTF-8 encoding -- the Windows default (cp1252) raises
    UnicodeDecodeError on this data rather than degrading quietly, so it fails
    loudly, but only if the encoding is stated.
    """
    if column.startswith(_DERIVED_HINT):
        raise ValueError(
            f"{column!r} is a derived column; script_of() must run on the raw "
            f"{_RAW_NAME_COLUMN!r}. See SCRIPT_DISTRIBUTION_PITFALLS.")

    report = {}
    for path in paths:
        key = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        overall = Counter()
        per_country = defaultdict(Counter)
        with open(path, encoding="utf-8", newline="") as fh:
            header = fh.readline().rstrip("\r\n").split("\t")
            if column not in header:
                raise ValueError(f"{path}: no {column!r} column in {header}")
            ci = header.index(column)
            cc = header.index("country") if "country" in header else None
            for n, line in enumerate(fh):
                if limit_rows and n >= limit_rows:
                    break
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) <= ci:
                    continue
                tag = script_of(fields[ci])
                overall[tag] += 1
                if by_country and cc is not None and len(fields) > cc:
                    per_country[fields[cc]][tag] += 1
        total = sum(overall.values())
        entry = {
            "rows": total,
            "counts": dict(overall.most_common()),
            "pct": {k: round(100 * v / total, 4) for k, v in overall.most_common()}
            if total else {},
        }
        if by_country:
            entry["by_country"] = {
                c: dict(t.most_common()) for c, t in sorted(per_country.items())
            }
        # the degenerate-report guard
        if total and overall.get("other", 0) / total > 0.95:
            entry["warning"] = (
                "over 95% 'other' -- almost certainly a wrong column or an "
                "incomplete script table; see SCRIPT_DISTRIBUTION_PITFALLS")
        report[key] = entry
    return report


__all__ = [
    "fold_unicode", "script_of", "scripts_present", "transliterate",
    "phonetic", "consonant_skeleton", "script_distribution",
    "ALL_SCRIPTS", "INDIC_SCRIPTS", "SCRIPT_BLOCKS",
    "SCRIPT_DISTRIBUTION_PITFALLS",
]
