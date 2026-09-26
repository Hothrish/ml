# Blocking stage after normalization

This folder reads the normalized files you already generated and produces candidate pairs for the later classification model. It includes retrieval, labeled sampling, recall evaluation, tests, and a Windows runner.

The classifier and its name/address comparison features come after this stage. The saved retrieval routes, ranks, BM25 scores and rank-fusion score can become classifier features alongside Jaccard, edit similarity, numeric agreement, missingness and other features discussed.

## Run on your Windows machine

Extract `blocking_stage.zip`. Open PowerShell **inside the extracted `blocking_stage` folder**. The default data root is your existing `C:\student_resource\student_resource`; the runner finds Python or the bundled Codex Python automatically.

```powershell
# 1. Check the environment and run the tests.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage doctor
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage tests

# 2. Build the FULL training S2/S3 index once, select 1,000 random S1
#    queries including singletons, generate candidates and evaluate recall.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage validate

# 3. Compare a more generous retrieval budget on the SAME tuning sample.
#    This reuses the index and writes a fresh output directory.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage validate -Config high_recall

# 4. After choosing a configuration, audit it on disjoint S1 records.
#    Use the chosen -Config value here; default is shown.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage audit -Config default

# 5. Generate test candidates using the chosen configuration.
#    This first builds the FULL test S2/S3 index, including France.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage test -Config default

# Optional: generate all training candidates for the later classifier.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage train -Config default
```

`validate` and `test` can be substantial jobs on this dataset. Complete validation and inspect both recall and runtime before starting all test queries. Choose one configuration after tuning, then use the disjoint audit sample once to assess it.

Options apply to all runner stages:

```powershell
# Put large indexes on an SSD with space; change roots when needed.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage validate -DataRoot "C:\student_resource\student_resource" -RunRoot "D:\blocking_runs" -Workers 4 -SampleSize 1000

# Specify an interpreter if automatic discovery is unsuitable.
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage doctor -PythonPath "C:\path\to\python.exe"
```

`SampleSize` is used when a sample is first created. Existing samples are reused for fair comparisons. Use a fresh `RunRoot` or a new sample directory with the direct CLI to change sample size. The audit sample excludes the tuning sample IDs. Reserve both samples from future classifier training when reporting their final pipeline performance.

## Files read and written

Expected existing inputs:

```text
C:\student_resource\student_resource\
  data\normalized\
    train_source1.part-00.tsv.gz ...
    train_source2.part-00.tsv.gz ...
    train_source3.part-00.tsv.gz ...
    test_source1.part-00.tsv.gz ...
    test_source2.part-00.tsv.gz ...
    test_source3.part-00.tsv.gz ...
  dataset\train\train_ground_truth.tsv
```

The reader handles the supplied normalizer's header in part 00 and headerless later parts. It also accepts one normalized `<split>_sourceN.tsv` per source. Missing part numbers, malformed rows and wrong source prefixes cause an error. It reads existing normalization; it does not require rerunning normalization.

Generated under `blocking_stage/runs/` by default:

```text
train.sqlite                       reusable full training index
test.sqlite                        reusable full test index
tune_sample/                        random S1 queries + corresponding labels
audit_sample/                       disjoint S1 queries + labels
validate-default-<timestamp>/
  candidate_pairs.tsv
  candidate_evidence.tsv.gz
  blocking_stats.json
  blocking_report.json
test-default-<timestamp>/
  candidate_pairs.tsv
  candidate_evidence.tsv.gz
  blocking_stats.json
```

`candidate_pairs.tsv` uses the challenge's two columns:

```text
source1_entity_id<TAB>candidate_entity_ids
S1-101<TAB>S2-204,S3-310,S2-811
S1-102<TAB>
```

Every processed S1 gets one row, even with zero candidates. The full test command processes all test S1 records. Candidate IDs are deduplicated and come from the indexed S2/S3 records. This is a candidate file, **not final matching_results.tsv**.

`candidate_evidence.tsv.gz` has one row per candidate pair with routes, per-route ranks, BM25 scores and a rank-fusion score. These are retrieval evidence, not probabilities. Candidate lists follow rank-fusion order. Preserve the exact set the later classifier actually scores: if you add further filtering, the final submitted candidate file must reflect that filtering.

## Retrieval method

Each S1 query searches S2 and S3 separately so one source cannot consume the other source's budget. Results are unioned by entity ID.

| Route | Representation | Default budget per source |
|---|---|---:|
| Exact | `name_sorted`, `name_nospace`, `name_cons`, within country | Every member of buckets of at most 100 records |
| Fuzzy name | Character trigrams of `name_core` | 30 |
| Fuzzy address | Character trigrams of `addr_norm` | 30 |
| Fuzzy transliteration | Character trigrams of `name_cons` | 20 |
| Global name rescue | Name trigrams in other/unknown countries | 5 |

Exact keys exclude empty values. Oversized exact buckets are refined by house number or locality; the fuzzy routes still run. A missing query country causes the main fuzzy routes to search all countries. Known countries get same-country main searches plus the cross-country name route. Country labels are open strings; France requires no trained country encoder.

Character grams are indexed inside words with boundary markers, and are insensitive to token order. They do not require matching a fixed prefix. They are scored with **SQLite FTS5 BM25**, not TF-IDF cosine. This is an explicit implementation choice: persistent indexes, bounded Python memory and no pip dependencies. Full-vocabulary character TF-IDF cosine can still be added later as a pairwise classifier feature.

The default query uses up to 16 rare indexed grams per representation, chosen by document frequency. This is an approximation for speed and may affect recall. `query_terms: 0` searches all available grams. The `high_recall` configuration tries all grams and larger budgets; its name describes intent, not a measured guarantee.

The default has **no extra cap after union**. It can produce substantially more than 50–80 candidates per S1: fuzzy budgets alone allow up to 170 before overlap, and exact routes can add more. Do not assume the union is a fixed size. A positive `max_candidates` applies a rank-fusion cap; evaluate its recall after capping. Even without a union cap, per-route top-K and oversized-bucket handling already limit retrieval.

## Evaluation and choosing budgets

Use the full training S2/S3 pool for every tuning and audit query. `sample` chooses random S1 rows including singletons. Sampling does not restrict the candidate pool to their known positives. Labels are read only by sampling/evaluation; `generate` has no ground-truth argument.

Read `blocking_report.json`:

- `pair_recall`: all retrieved true links / all true links in the query sample.
- `macro_recall_non_singletons`: mean per-S1 recall over queries with true matches.
- `all_links_retrieved_fraction_non_singletons`: fraction of matched S1 entities whose complete true match set survived.
- `oracle_macro_f0_5_ceiling`: the score a perfect classifier could achieve from these candidates, including singleton credit. This is a ceiling, not a trained model result.
- `mean_candidates`, `p95_candidates`, `p99_candidates`, `max_candidates`: classifier workload.
- `slices`: recall by query country, S2/S3, matched record script and country agreement.
- `retrieved_true_links_by_route`, `true_links_only_retrieved_by_this_route`: route contribution after any final cap. The second is a leave-one-route-out marginal count for that output; route overlaps mean these counts do not sum to all hits.
- `missing_index_true_links`: should be zero for a complete, correct training index. Missing records remain in recall denominators.

Singletons do not have a pair-recall denominator. Generating candidates for a singleton is expected; the later classifier must reject them. France has no supplied training labels, so this evaluation cannot establish France recall or guarantee generalization. Sharing a name or parent company across countries is not by itself proof of the challenge's entity identity.

Compare `config/default.json` and `config/high_recall.json` on the same tuning queries. Tune one budget at a time if necessary. Aim for recall as close to 100% as practical, examine missed links, then choose a budget whose runtime and pair count are affordable. Evaluate the chosen configuration on the disjoint audit sample before running all S1 records. Larger samples are needed for reliable script-specific conclusions when a script is rare.

## Scale, limits and recovery

The pipeline streams normalized data into disk indexes and streams output pairs. It never materializes an all-pairs similarity matrix. Retrieval workers share the read-only database, each with an approximately 128 MiB SQLite cache plus Python overhead and a bounded term cache. Start with four workers on an SSD; lower this if disk contention or RAM use rises. Index building currently uses one process.

Indexes, candidate TSVs and evidence can occupy many GB. The real-data smoke test described in `VERIFICATION.md` is not a full-dataset speed benchmark. Measure full-index sample throughput first: `blocking_stats.seconds / s1_records` gives measured seconds per query at the selected worker count. Multiplying by 1,732,544 gives a rough test-generation estimate; country distributions and caches can change it. SQLite BM25 queries over common grams can still be expensive. Do not infer full-dataset latency from the small smoke index.

An existing index is never overwritten. Ready status and input file sizes/mtimes are checked when reused through the runner. If normalization changes, use a new `RunRoot` to rebuild. Interrupted builds leave an unusable index marked `building`; use a new index path. Generation publishes the final TSV only on completion; interrupted runs leave `.partial` files. Automatic resume is not implemented. Fresh runner invocations use timestamped output directories.

The optional `--limit-per-source` build flag is for smoke tests only. Such indexes are marked partial, and full generation through `--normalized-dir` rejects them. Evaluation identifies partial indexes and keeps missing positives in its denominators. Never use partial-index recall to choose production budgets.

## Direct Python CLI (Windows or Linux)

Python 3.10+ with SQLite FTS5 is required. No package installation is needed. Replace `python` with your interpreter path or `py -3` if needed.

```powershell
python src/blocking.py doctor
python -m unittest discover -s tests -v

python src/blocking.py build --normalized-dir "C:\student_resource\student_resource\data\normalized" --split train --index runs/train.sqlite
python src/blocking.py sample --normalized-dir "C:\student_resource\student_resource\data\normalized" --ground-truth "C:\student_resource\student_resource\dataset\train\train_ground_truth.tsv" --out runs/tune_sample --size 1000 --seed 17
python src/blocking.py generate --index runs/train.sqlite --queries runs/tune_sample/queries.tsv --split train --out runs/tune_default --config config/default.json --workers 4
python src/blocking.py evaluate --index runs/train.sqlite --queries runs/tune_sample/queries.tsv --ground-truth runs/tune_sample/ground_truth.tsv --candidates runs/tune_default/candidate_pairs.tsv --evidence runs/tune_default/candidate_evidence.tsv.gz --report runs/tune_default/blocking_report.json

python src/blocking.py build --normalized-dir "C:\student_resource\student_resource\data\normalized" --split test --index runs/test.sqlite
python src/blocking.py generate --index runs/test.sqlite --normalized-dir "C:\student_resource\student_resource\data\normalized" --split test --out runs/test_default --config config/default.json --workers 4
```

Use `python src/blocking.py <command> --help` for all options. `--no-evidence` saves output space if route/rank features are not needed. Keep evidence while evaluating route contribution or preparing classifier features.

Retrieval implementation reference: [SQLite FTS5: ranking and indexes](https://www.sqlite.org/fts5.html).
