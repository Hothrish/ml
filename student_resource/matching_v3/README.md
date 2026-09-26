# Matching version 3

## Run

Keep the previous full prediction stopped. The installer copies this package and the prebuilt compact caches into the existing project, runs tests, predicts every test S1, then validates the output. It does not repeat normalization, SQLite indexing or training.

Run the accompanying `start_matching_v3.ps1` in PowerShell. Its default project is `C:\student_resource\student_resource`, with 8 workers. No manual copying is required. The installer and `matching_v3_assets` folder must remain alongside `matching_v3_update.zip` until installation finishes.

After installation, the full command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\student_resource\student_resource\matching_v3\run_matching_v3.ps1 -Stage predict -Workers 8
```

The installer already runs that command. Use it again only to resume an interrupted version 3 run. Completed 1,000-query shards are reused; an unfinished shard is recomputed. A model or input change is rejected when a checkpoint exists. Previous version 2 shards cannot be resumed with this model.

To run only a 5,000-query timing check, pass `-BenchmarkOnly` to the installer, or use the installed runner with `-Stage benchmark`.

## Files to submit

- Portal upload: `C:\student_resource\student_resource\output_v3\matching_results.tsv`
- Candidate file: `C:\student_resource\student_resource\output_v3\candidate_pairs.tsv`
- Both files contain every test S1 when the full run finishes. Benchmark files are partial and must not be submitted.
- For the final source archive, put these two files under the challenge's required `output/` directory and include this code, model, dependencies list and methodology. The local `output_v3` name prevents mixing old and new predictions.

## Accuracy

On the same 2,000 held-out training S1 records:

| Metric | Previous model | Delivered version 3 |
|---|---:|---:|
| Macro F0.5 | 0.910636 | 0.945437 |
| Pair precision | 0.964238 | 0.987606 |
| Pair recall | 0.834617 | 0.880105 |
| False positives | 213 | 76 |

Version 3 correctly predicts no match for 103 of 109 true singletons. The delivered model uses a reranking pool of 300, chosen on calibration for speed; pool 1000 scored 0.946739 on the audit but was slower. This is an internal validation result, not a leaderboard prediction. France has no labeled audit examples.

See `reports/final_report.json` for the completed runtime benchmark and extrapolation. A sub-30-minute run on this laptop has not been achieved. Keep the laptop awake and powered while processing. Eight workers are the measured setting; adding more workers can increase memory and disk contention.

## Changes

- Exact compact arrays replace repeated SQLite LSH lookups; memory-mapped normalized records replace repeated database fetches.
- Name/address reranking uses native batched RapidFuzz comparisons, with a union of high-vote and rare-bucket candidates.
- A cheap 22-feature classifier screens every candidate. Only plausible candidates receive the full 153-feature classifier. Candidate output includes all screened pairs, including rejected pairs.
- The full model adds IDF-weighted token comparisons, number agreement and additional edit-distance features.
- Threshold calibration optimizes the challenge macro F0.5, including empty predictions for true singletons.
- TSVs are written as bounded-memory shards, then combined. The validator streams candidates instead of loading hundreds of millions of pairs into a Python dictionary.

## Normalization audit

The existing normalization is already parallel and streams compressed chunks. It preserves unknown country labels and provides France-specific rules. In a diagnostic scan of the first 50,000 training S1 records, no nonempty address became an empty normalized address, and no name core was empty.

Two limitations were found: leading zeros are removed before fixed-width postcode extraction, and landmark fragments are stored separately but not used by the current matcher. The address body still retains its numeric tokens, and the landmark column remains in the normalized files. Changing these model inputs requires a fresh retrieval/training validation. This package keeps the normalization used by the validated model; do not rerun normalization into the existing indexed directory during prediction.

The measured pairwise sparse TF-IDF experiment produced equivalent cosine scores but was slower after including CSR construction. A full prebuilt sparse retrieval index was not benchmarked. There are no Pandas joins in inference, so Polars would not remove the measured retrieval and classification costs. A strict first-three-letter or postcode filter was not adopted because it can exclude spelling/transliteration variants.

## Rebuilding and training

`src/compact_lsh.py` builds compact postings from the completed LSH SQLite index. `src/compact_records.py` builds memory-mapped records from the indexed normalization in exactly the same row order. `matching_v2.py build-country-cache` creates the small country lookup. Input manifests are checked on startup.

`src/train_v3.py prepare` creates 153-feature candidate data from a 12,000-S1 sample, split deterministically into 8,000 fitting, 2,000 calibration and 2,000 audit records. `train` fits the full model. `src/train_cascade_v3.py` fits the cheap first stage. Training reports and final inference settings accompany the saved model. Python 3.12 and the pinned packages in `matching_requirements_v3.txt` were used.
