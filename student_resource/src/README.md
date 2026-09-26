# Normalisation stage — ML Challenge 2026 Business Entity Resolution

Turns the six raw source TSVs into normalised, matching-ready records.
**Stdlib only** — Python 3.8+, no pip install, no network, no external data.

## Files

| file | role |
| --- | --- |
| `script_classifier.py` | Unicode folding, writing-system detection, Indic transliteration, phonetic keys |
| `normalization.py` | Lookup tables, name/address/country normalisation, parallel driver, report builder |

`normalization.py` imports `script_classifier.py`. Both must travel together.

## Verify before you trust it

```bash
python -m unittest discover -s tests
```

64 tests, runs in under a second. Do this first — it confirms the transfer is
intact before you spend 13 minutes on a full run.

## Regenerate the normalised data

```bash
python src/normalization.py --workers 10 \
  --out data/normalized \
  --in dataset/train/train_source1.tsv dataset/train/train_source2.tsv \
       dataset/train/train_source3.tsv dataset/test/test_source1.tsv \
       dataset/test/test_source2.tsv dataset/test/test_source3.tsv
```

~13 minutes for all 24,229,173 rows on 12 cores. Output is
`data/normalized/<stem>.part-NN.tsv.gz` — gzipped TSV, 26 columns, ~2.2 GB.

Deterministic: same inputs plus same code give byte-identical output, which is
why the code ships rather than the 2.2 GB of derived parts.

### Memory

Each worker streams a byte range and never holds a file in memory, so peak RSS
is a few hundred MB per worker. The token caches are the only thing that grows;
they are capped at 2^16 entries each. Raise with `NORM_CACHE_BITS=18` if you
have RAM to spare, but note that 2^19 across 12 workers was enough to push a
16 GB machine into paging and made one file run 24x slower than its siblings.

Use `--workers 6` on a machine with less than 16 GB.

## Regenerate the report

```bash
python src/normalization.py --report normalization_report.json
```

Scans all six files for the script distribution and scores 8,000 ground-truth
clusters for the lift table. Takes a few minutes.

## Consuming the output

```python
import sys, glob
sys.path.insert(0, "src")
from normalization import iter_normalized
from script_classifier import consonant_skeleton

parts = sorted(glob.glob("data/normalized/test_source2.part-*.tsv.gz"))
for rec in iter_normalized(parts):       # streams dicts, never loads the file
    key = (rec["addr_region"], rec["name_cons"])
```

Do **not** re-normalise in the next stage. The parts already carry all 24
derived fields per record.

## Output columns

`entity_id`, `country`, `country_norm`, then:

**Name (12)** — `name_norm`, `name_core`, `name_sorted`, `name_nospace`,
`name_acronym`, `name_translit`, `name_phon`, `name_cons`, `name_legal`,
`name_nums`, `name_script`, `name_is_domain`

**Address (11)** — `addr_norm`, `addr_core`, `addr_sorted`, `addr_cons`,
`addr_nums`, `addr_house`, `addr_postcode`, `addr_region`, `addr_locality`,
`addr_landmark`, `addr_street`

Several representations rather than one canonical string, because no single
normalisation survives every noise pattern at once: dropping street types fixes
`Rd`/`Road`/`Saint` but throws away signal, stripping vowels defeats
transliteration but collides more, sorting tokens fixes word order but loses it
as evidence. The matcher picks which view to trust per comparison.

## Which fields to block on

Measured on 8,000 ground-truth clusters (29,206 true pairs) against a
random-pair control — see `normalization_report.json`:

| field | true-pair agreement | random-pair |
| --- | ---: | ---: |
| `addr_region` | 93.50% | 3.55% |
| `addr_house` | 69.74% | 0.22% |
| `addr_locality` | 65.85% | 0.22% |
| `name_acronym` | 60.07% | 0.12% |
| `name_cons` | 49.59% | 0.00% |
| `name_sorted` | 49.27% | 0.00% |
| *(raw name, for comparison)* | *10.65%* | *0.00%* |
| *(raw address, for comparison)* | *7.12%* | *0.00%* |

No single key covers even half the true pairs, so block on the **union** of
several rather than picking a winner.

`addr_region` is a **partition**, not evidence — its 3.55% random-pair rate is
just the ~50 regions. All 7,638,365 training links are same-country, so
partitioning on country loses nothing.

## Two facts that shape the next stage

1. **Source 1 is 100% Latin**, in both train and test. Cross-script matching is
   always S1(Latin) ↔ S2/S3(Indic), never Indic ↔ Indic.
2. **Non-Latin text appears only in India records.** France and US are 100%
   Latin; India is 76% Latin, 24% Indic across eight scripts.

## Known limits

- `addr_postcode` takes the last digit-run of the country's length. US ZIPs and
  5-digit house numbers are indistinguishable this way — treat it as evidence,
  not truth.
- Odia and Malayalam cross-script token overlap is 14% and 60%, against 100%
  for the other seven scripts. Odia writes "v" with the bh letter and both use
  different loanword spellings. A Soundex-style b/p/v/w merge would lift Odia
  to 33% but was rejected: not worth collapsing four consonants across the
  whole key under a precision-heavy F0.5.
- **Typos are deliberately not handled.** `Payne Etrepndiels` does not reduce to
  `Payne Enterprises`. That is random corruption, not systematic variation, and
  belongs to edit-distance scoring at the matching stage. Normalisation's job
  is the systematic part.
- French region vocabulary is mined from the data (`load_derived`), so it mixes
  departments and cities. The canonical region code is folded back into
  `addr_core` so a record never loses its only geographic token when the two
  resolve differently.
