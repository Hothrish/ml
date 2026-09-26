# %% [markdown]
# # 01 - Text normalisation
#
# Turns the six raw source files into normalised, matching-ready records.
#
# The dataset is ~24M rows over ~2.4 GB and this machine has well under 8 GB of
# usable RAM, so nothing here loads a whole file into memory. Every stage
# streams, and the heavy work runs in worker processes over byte ranges of the
# input.
#
# **What this notebook produces** - for each of the six source files, a set of
# gzipped TSV parts in `data/normalized/` with 25 columns: the original
# `entity_id` and `country`, plus 12 name representations and 11 address
# representations.
#
# **Why several representations instead of one canonical string.** No single
# normalisation survives every noise pattern in this data at once. Dropping
# street types fixes `Rd`/`Road`/`Saint` but throws away signal; stripping
# vowels defeats transliteration but collides more; sorting tokens fixes word
# order but loses it as evidence. So each field is emitted at several levels of
# aggression, and the matcher downstream picks which view to trust for which
# comparison.
#
# The logic lives in `src/` rather than in cells because `multiprocessing` on
# Windows uses *spawn*: worker processes re-import the module, and functions
# defined in a notebook cell cannot be pickled to them.

# %%
import os
import sys
import glob
import gzip
import time
import random
from pathlib import Path

PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC = PROJECT / "src"
sys.path.insert(0, str(SRC))

DATA = PROJECT / "dataset"
OUT = PROJECT / "data" / "normalized"
OUT.mkdir(parents=True, exist_ok=True)

# Windows consoles default to cp1252 and this data is full of Devanagari and
# Tamil; force UTF-8 so printing a record never raises.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("project:", PROJECT)
print("cpus   :", os.cpu_count())

# %% [markdown]
# ## 1. The normalisation pipeline
#
# Two modules:
#
# | module | responsibility |
# | --- | --- |
# | `script_classifier.py` | Unicode folding, script detection, Indic transliteration, phonetics |
# | `normalization.py` | lookup tables, name/address/country normalisation, parallel driver |
#
# ### Order of operations
#
# The sequence matters more than any individual step:
#
# 1. **Fold Unicode (NFKD)** and strip *Latin* combining marks only. Indic vowel
#    signs and the virama are combining marks too - stripping those destroys the
#    text before it can be transliterated.
# 2. **Transliterate** all nine Indic scripts to a Latin phonetic skeleton, so
#    the rest of the pipeline only ever sees Latin.
# 3. **Lower-case, strip punctuation.** Apostrophes close up (`Orelee's` ->
#    `orelees`); everything else becomes a space, so `PAYNE-ENRTPRMISES` and
#    `Payne Enterprises` tokenise alike.
# 4. **Preserve numbers**: ordinal suffixes are dropped (`45th`, `45ND` -> `45`)
#    and leading zeros stripped (`AF-0684` -> `af 684`), so the same house number
#    written three ways compares equal.
# 5. **Canonicalise legal forms**, then split them off into their own field.
# 6. **Derive the representations.**

# %%
import script_classifier as text_norm
import normalization as norm_tables
import normalization as record_norm
from normalization import normalize_record, normalize_name, normalize_address

print("name fields   :", len(record_norm.NAME_FIELDS))
print("address fields:", len(record_norm.ADDR_FIELDS))
print()
for f in record_norm.ALL_FIELDS:
    print("   ", f)

# %% [markdown]
# ### Transliteration
#
# The hardest noise pattern in this dataset: ~15-19% of Source 2 names are
# non-ASCII, and a cluster routinely holds the same business written in Latin
# and in an Indic script.
#
# **This data carries nine Brahmic scripts, not two.** Devanagari and Tamil are
# the obvious ones; Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia and
# Gurmukhi are also present in volume. A classifier that knows only two buckets
# the other seven as `other`, and anything tagged `other` is never
# transliterated - a silent recall hole. See section 2.5.
#
# Unicode laid the Brahmic blocks out in parallel (inherited from ISCII), so
# seven of the nine transliteration tables are *derived* from the Devanagari one
# by codepoint offset rather than hand-written. Tamil is the exception: it has a
# reduced inventory with no voiced/aspirated distinctions, so it needs its own
# table.
#
# Two mechanisms do the work. A **syllable-aware scan** adds the inherent `a`
# after a consonant unless a virama or vowel sign cancels it, then deletes the
# word-final schwa the way Hindi actually pronounces it. And an **acronym
# decomposition**: the Devanagari for `SS` is the letter-name for S written
# twice, not a word, so a token that decomposes *entirely* into letter names is
# read as an acronym.

# %%
samples = [
    "एसएस फूड प्राइवेट लिमिटेड",            # Devanagari
    "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",       # Tamil
    "గోల్డ్ ప్రొడ్యూసర్ స్టోర్స్",              # Telugu
    "ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್",                  # Kannada
    "গোল্ড প্রডিউসার স্টোর্স লিমিটেড",        # Bengali
    "શક્તિ અર્બન પ્રોડક્ટ્સ",                  # Gujarati
    "ശക്തി ഇംപെക്സ്",                        # Malayalam
    "ଭିଜନ୍ ଟେକ୍ନୋଲୋଜିସ୍",                     # Odia
    "ਸਕਾਈ ਈਸਟ ਇਸਟੇਟ",                      # Gurmukhi
    "Payne Énterprises",                     # Latin with stray diacritic
]
for s in samples:
    print(f"{text_norm.script_of(s):6} {s}")
    print(f"       -> {text_norm.transliterate(text_norm.fold_unicode(s))}")

# %% [markdown]
# ### The phonetic and consonant keys
#
# Transliteration gets the scripts into one alphabet but not into one spelling:
# the Devanagari for *private* comes out as `praaivet`. Two further reductions
# close that gap - a phonetic key (aspirated digraphs collapsed, `c` resolved to
# `s`/`k`, vowel length normalised) and a **consonant skeleton** that drops
# vowels entirely.
#
# The consonant skeleton is the representation that actually survives
# transliteration, and it is the one to reach for when blocking across scripts.

# %%
pairs = [
    ("एसएस फूड प्राइवेट लिमिटेड", "Ss Food Private Limited"),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "Ram Marketing Private Limited"),
    ("मॉडर्न फाइनेंस", "Modern Finance"),
    ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "Raj Investments LLP"),
    ("Payne Etrepndiels", "Payne Enterprises"),
]
print(f"{'indic / noisy':24} {'latin':24} {'skeletons equal?':>18}")
for a, b in pairs:
    ka = record_norm.normalize_name(a)["name_cons"]
    kb = record_norm.normalize_name(b)["name_cons"]
    verdict = "yes" if ka == kb else ("nospace" if ka.replace(" ", "") ==
                                      kb.replace(" ", "") else "no - typo")
    print(f"{ka:24} {kb:24} {verdict:>18}")

# %% [markdown]
# The two failures are genuine typos (`Etrepndiels` for `Enterprises`). No
# deterministic key fixes those - they need edit-distance scoring at the
# matching stage. That is the honest division of labour: normalisation collapses
# *systematic* variation, fuzzy matching handles *random* corruption.

# %% [markdown]
# ### Addresses
#
# Addresses carry a positional signal that names do not, so components are split
# on commas **before** punctuation is destroyed. Then:
#
# - **Region** is found by scanning every component from the tail for an exact
#   match against the country's administrative divisions - not just the last
#   component, because components in this data are freely reordered
#   (`KANSAS CITY, MO, 630 45ND TERRACE`).
# - **Locality** is the last *all-alphabetic* component after the region is
#   removed. "Last component" alone lands on a PIN code or a street line about
#   as often as on the city.
# - **Landmarks** (`Near Mother India Public School`) are pulled into their own
#   field: they are weak evidence, and leaving them inline drowns the
#   discriminative tokens.
# - **Street types are removed entirely** in `addr_core`. That one move
#   neutralises `Rd`/`Road`, `St`/`Street` and the over-expanded `Saint` in the
#   data, without needing to disambiguate them.
# - The region is held out of the body while locality is detected, then its
#   **canonical code is folded back into the core**. Without that, `Uttar
#   Pradesh` and `UP` leave different cores despite being the same place.
# - A division written in its **own script** resolves by transliterating the
#   component and comparing consonant skeletons, so the Tamil and Devanagari
#   spellings of `Tamil Nadu` both reach `tn`. Enumerating nine spellings of
#   thirty-six divisions by hand is both error-prone and incomplete - an
#   earlier version did exactly that, covered two scripts, and quietly lost
#   5pp of `addr_region` agreement on the other seven. The skeleton lookup is
#   restricted to non-ASCII components so an ordinary Latin street name cannot
#   collide with a state on a lossy key.

# %%
addr_cases = [
    ("Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh", "India"),
    ("AF-0684, Uttar Pradesh, GHAZIABAD, 9487203", "India"),
    ("3315 FREMONT SAINT, PEORIA, IL", "US"),
    ("Fremont St, Peoria, Illinois", "US"),
    ("KANSAS CITY, MO, 630 45ND TERRACE, null", "US"),
    ("630 45th Terrace, Kansas City, MO", "US"),
    ("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
]
for raw, country in addr_cases:
    d = record_norm.normalize_address(raw, country)
    print(f"RAW  {raw[:74]}")
    print(f"     core={d['addr_core']!r} house={d['addr_house']!r} "
          f"region={d['addr_region']!r} locality={d['addr_locality']!r}")
    if d["addr_landmark"]:
        print(f"     landmark={d['addr_landmark']!r}")
    print()

# %% [markdown]
# ## 2. Does it actually help?
#
# A representation earns its place only if true matches agree on it far more
# often than random pairs do. `validate_norm.py` samples ground-truth clusters,
# pulls the member records out of the three source files in one streaming pass,
# and reports hit rate, false-positive rate and lift for each key - with the
# untouched `raw` fields as the baseline.

# %%
import normalization as validate_norm

t0 = time.time()
rows, n_pairs, n_rand, n_found, n_wanted = validate_norm.run(
    data_dir=str(DATA / "train"), n_clusters=8000)
print(validate_norm.report(rows, n_pairs, n_rand))
print(f"\nrecords resolved: {n_found:,}/{n_wanted:,}   {time.time() - t0:.0f}s")

# %% [markdown]
# Read this table as the justification for every field that follows. The `raw`
# rows are what you would get without normalising; the gap between `name_raw`
# and `name_core` is the entire point of the exercise.
#
# High `fp%` is not automatically bad - `addr_region` matches ~3.5% of random
# pairs because there are only ~50 regions, and it is meant as a *partition*,
# not as evidence. The compound keys at the bottom are what blocking should
# actually use: high hit rate with a false-positive rate near zero.
#
# One caveat on reading this table: it scores *exact string equality*, which is
# the harshest possible test. Folding the region code back into `addr_core`
# costs about 0.1pp here - it adds a token that one record of a pair may lack -
# but it stops a record from losing its only geographic token, which matters
# more for the token-overlap scoring blocking actually uses.
#
# Note also that no single key is close to sufficient on its own: the best name
# key agrees on under half of true pairs. Blocking should take the **union** of
# several keys, not pick a winner.

# %% [markdown]
# ## 2.5 Why the script_distribution report returned "Other" for everything
#
# Two independent causes, both reproduced against real data.
#
# **Cause 1 - classifying a post-normalisation column.** `script_of()` must see
# the *raw* `business_name`. Run it downstream of normalisation - the natural
# place to put a reporting step - and the answer is meaningless, because
# normalisation has already transliterated the text to ASCII or blanked it.
# Measured on 20k rows of `train_source2`:
#
# | column classified | result |
# | --- | --- |
# | `business_name` (raw) | latin 91%, deva 4.9%, other 3.1%, taml 0.6% |
# | `name_translit` | **other 94%** (empty by design for Latin records) |
# | `name_nums` | **other 100%** (digits only) |
# | `name_script` | latin 100% (it is a tag, not text) |
#
# An empty or numeric column yields exactly the reported symptom. Note the
# default Windows encoding is *not* the cause: reading this data as cp1252
# raises `UnicodeDecodeError` rather than degrading quietly.
#
# **Cause 2 - an incomplete script table.** Anything the classifier does not
# know falls into `other`. The original table knew Devanagari and Tamil only,
# so seven more scripts were being silently bucketed. Counting the alphabetic
# characters inside names tagged `other`, over 450k sampled rows:

# %%
from collections import Counter
import unicodedata

blocks, examples = Counter(), {}
for path in [DATA / "train" / "train_source2.tsv",
             DATA / "train" / "train_source3.tsv",
             DATA / "test" / "test_source2.tsv"]:
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= 150_000:
                break
            nm = line.rstrip("
").split("	")[1]
            if text_norm.script_of(nm) != "other":
                continue
            for ch in nm:
                if ch.isalpha():
                    try:
                        blk = unicodedata.name(ch).split()[0]
                    except ValueError:
                        blk = "UNNAMED"
                    blocks[blk] += 1
                    examples.setdefault(blk, nm)

if blocks:
    print("scripts still hiding in the 'other' bucket:")
    for b, c in blocks.most_common(10):
        print(f"   {b:12} {c:8,}   e.g. {examples[b][:44]}")
else:
    print("nothing alphabetic left in 'other' - every script is recognised")

# %% [markdown]
# With all nine scripts supported, `other` should now contain only genuinely
# non-alphabetic names (digits, punctuation, empty). The report generator
# guards both causes: it resolves the column **by header name** and refuses a
# derived one, and it flags any run where `other` exceeds 95%.

# %% [markdown]
# ## 3. Mining the geographic vocabulary from the data
#
# The fair-play rule forbids external lookups, and the test set adds **France**,
# which never appears in training. Rather than hard-code 101 French departments
# from outside knowledge, mine the trailing address components from the provided
# files and feed them back into the region table with `load_derived()`.
#
# This keeps the geographic vocabulary inside the supplied data, and it adapts
# automatically to any further country the organisers add.

# %%
from collections import Counter


def mine_trailing_components(path, country, limit_rows=400_000):
    """Frequency of the last comma component, for records of one country."""
    counts = Counter()
    with open(path, encoding="utf-8", newline="") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= limit_rows:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4 or parts[3] != country or not parts[2]:
                continue
            tail = parts[2].rsplit(",", 1)[-1].strip().lower()
            if tail and not any(ch.isdigit() for ch in tail):
                counts[record_norm.basic_clean(tail)] += 1
    return counts


fr_counts = mine_trailing_components(DATA / "test" / "test_source2.tsv", "France")
known = norm_tables.REGIONS_BY_COUNTRY["France"]
unknown = [(name, n) for name, n in fr_counts.most_common(60) if name not in known]

print(f"distinct trailing components (France): {len(fr_counts):,}")
print(f"\ntop components not in the seed region table:")
for name, n in unknown[:25]:
    print(f"   {n:>7,}  {name}")

# %%
# Persist the mined vocabulary and load it into the region table. Anything
# appearing often enough to be an administrative division rather than a street
# name is kept; the threshold is deliberately conservative.
MIN_COUNT = 200
derived_dir = PROJECT / "data" / "derived"
derived_dir.mkdir(parents=True, exist_ok=True)
fr_path = derived_dir / "regions_France.txt"

with open(fr_path, "w", encoding="utf-8") as fh:
    for name, n in fr_counts.items():
        if n >= MIN_COUNT and name not in known:
            fh.write(f"{name}\t{name.replace(' ', '')}\n")

added = norm_tables.load_derived("France", fr_path)
print(f"added {added} mined French divisions to the region table")

d = record_norm.normalize_address("18 RUE JEN ZAY, Dunkerque, Nord", "France")
print("\nafter mining:", d["addr_region"], "|", d["addr_locality"], "|", d["addr_core"])

# %% [markdown]
# Mining cannot tell a department from a city - `gironde` and `bordeaux` both
# show up near the top. That is tolerable precisely because the canonical region
# code is folded back into `addr_core`: a record ending `Bordeaux, Gironde` and
# one ending `Bordeaux` still share a geographic token, even though the region
# *slot* resolves differently for each. Treat `addr_region` as a partition hint
# for France, not as a clean administrative label the way it is for US states.

# %% [markdown]
# ## 4. Run over the whole dataset
#
# Each worker takes a byte range, streams it, and writes its own gzip part, so
# nothing large is ever pickled between processes and peak memory stays flat no
# matter how big the input is.
#
# Expect roughly **50k rows/s** on 12 cores - about 8 minutes for all 24M rows.

# %%
import normalization as normalize_run

FILES = [
    DATA / "train" / "train_source1.tsv",
    DATA / "train" / "train_source2.tsv",
    DATA / "train" / "train_source3.tsv",
    DATA / "test" / "test_source1.tsv",
    DATA / "test" / "test_source2.tsv",
    DATA / "test" / "test_source3.tsv",
]

# Set to False for a quick smoke test on a slice instead of the full 24M rows.
RUN_FULL = True

# %%
summary = []
t_start = time.time()
for path in FILES:
    if not path.exists():
        print(f"missing: {path}")
        continue
    info = normalize_run.normalize_file(str(path), str(OUT), workers=os.cpu_count())
    summary.append(info)
    print(f"{path.name:26} {info['rows']:>9,} rows  {info['seconds']:>6.1f}s  "
          f"{info['rows_per_sec']:>9,.0f} rows/s  malformed={info['malformed']}")

print(f"\ntotal {sum(s['rows'] for s in summary):,} rows in "
      f"{time.time() - t_start:.0f}s")

# %% [markdown]
# ## 5. Check the output
#
# Row counts must match the inputs exactly, every row must have all 25 columns,
# and no record may come out with an empty name core - a record normalised into
# nothing is unmatchable and would silently cost recall.

# %%
EXPECTED = {
    "train_source1": 2_206_821, "train_source2": 5_034_616,
    "train_source3": 5_285_603, "test_source1": 1_732_544,
    "test_source2": 4_887_273, "test_source3": 5_082_316,
}

ncols = len(normalize_run.OUT_COLUMNS)
for stem, expected in EXPECTED.items():
    parts = sorted(glob.glob(str(OUT / f"{stem}.part-*.tsv.gz")))
    if not parts:
        print(f"{stem:16} no parts found")
        continue
    rows = empty_name = empty_addr = ragged = 0
    for i, p in enumerate(parts):
        with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
            if i == 0:
                header = fh.readline().rstrip("\n").split("\t")
                idx_name = header.index("name_core")
                idx_addr = header.index("addr_core")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) != ncols:
                    ragged += 1
                    continue
                rows += 1
                if not f[idx_name]:
                    empty_name += 1
                if not f[idx_addr]:
                    empty_addr += 1
    ok = "OK" if rows == expected and ragged == 0 else "MISMATCH"
    print(f"{stem:16} {rows:>9,}/{expected:,} {ok:>9}  ragged={ragged}  "
          f"empty_name={empty_name:,}  empty_addr={empty_addr:,}")

# %% [markdown]
# `empty_addr` is expected to be non-zero: 2.6-3.4% of Source 2 and Source 3
# records ship with a blank address. `empty_name` should be zero - no record
# should normalise away to nothing.

# %% [markdown]
# ## 6. Spot-check a real cluster end to end
#
# The final sanity check: take a ground-truth cluster and look at every member's
# normalised form side by side. This is where a subtly wrong rule shows itself.

# %%
def show_cluster(s1_id, ids):
    wanted = {s1_id, *ids}
    paths = [str(DATA / "train" / f"train_source{i}.tsv") for i in (1, 2, 3)]
    recs = validate_norm.collect_records(paths, wanted)
    order = [s1_id] + [i for i in ids if i in recs]
    for eid in order:
        if eid not in recs:
            continue
        name, address, country = recs[eid]
        d = normalize_record(name, address, country)
        print(f"[{eid.split('-')[0]}] {name[:46]}")
        print(f"      raw addr : {address[:78]}")
        print(f"      name_cons: {d['name_cons']:<28} legal: {d['name_legal']}")
        print(f"      addr_core: {d['addr_core'][:60]:<60}")
        print(f"      house={d['addr_house']!r} region={d['addr_region']!r} "
              f"locality={d['addr_locality']!r}")
        print()


clusters = validate_norm.load_clusters(str(DATA / "train" / "train_ground_truth.tsv"), 40)
s1_id, ids = max(clusters.items(), key=lambda kv: len(kv[1]))
print(f"cluster {s1_id} with {len(ids)} matches\n")
show_cluster(s1_id, ids)

# %% [markdown]
# ## What comes next
#
# The normalised parts in `data/normalized/` are the input to candidate
# generation. Three findings from this notebook shape that stage:
#
# 1. **`country` is a safe hard partition.** All 7,638,365 training links are
#    within one country, so blocking can be run per country and the France
#    records - which have no training signal - can be given their own, more
#    conservative threshold without disturbing US/India tuning.
# 2. **`name_cons` is the cross-script key.** It is the only name representation
#    that survives Devanagari and Tamil transliteration intact; block on it
#    rather than on `name_norm` or the raw name.
# 3. **Compound keys are what blocking wants.** `name_cons + addr_locality` and
#    `name_cons + addr_house` carry a high hit rate at a false-positive rate near
#    zero, which is exactly the shape needed for an F0.5 metric that punishes
#    false merges twice as hard as misses.
#
# Two limits worth carrying forward. The postcode extractor takes the *last* run
# of digits of the country's length, which is a heuristic - 5-digit US house
# numbers can be picked up as ZIPs, so treat `addr_postcode` as evidence rather
# than truth. And typos are deliberately left alone: they are random, not
# systematic, and belong to edit-distance scoring at the matching stage.
