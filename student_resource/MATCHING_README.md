# Matching stage and first submission

This package adds a trained classifier to the completed blocking stage. Extract its contents into `C:\student_resource\student_resource`. The package contains the trained model, so you can start with test indexing rather than repeat training.

## Measured validation result

A fresh uniform sample of 3,000 training S1 entities excludes the 1,000 entities used for blocker tuning. The model fits on 2,100 S1 entities, selects its threshold on 450 different S1 entities, and is evaluated on the remaining 450. All pairs belonging to one S1 stay together. The fitting data contains 6,569 retrieved positive pairs and up to 30 randomly sampled candidate negatives per fitting entity. Threshold selection and evaluation use complete candidate lists, including singleton queries and true links missed by blocking.

The held-out macro F0.5 is **0.865035**. Micro pair precision is 0.937594 and micro pair recall is 0.812907. India macro F0.5 is 0.816200 over 173 entities; US is 0.895535 over 277. These are training-data validation results, not a leaderboard estimate. France has no training labels. The report, model, and split IDs are in `models/`. Model fitting, feature extraction and threshold evaluation took about 118 seconds after sampled candidates and their comparison fields were prepared.

The MIT-licensed model is logistic regression trained with NumPy, using nonlinear transforms and interactions of name character/token overlap, consonant/phonetic similarities, full address character/token overlap, number overlap, house/postcode/region/locality agreement, missing fields, and saved retrieval evidence. No country-specific encoder is learned. Model scores are not calibrated probabilities because fitting negatives are sampled. The threshold is selected by macro F0.5 including singletons. Levenshtein, Jaro-Winkler, and vocabulary TF-IDF are not part of this initial model.

## Run these stages in order

Run from the existing project root. The runner uses the installed Codex Python runtime when available, otherwise Python from your system. Outside that runtime, install `matching_requirements.txt` first. No GPU is used.

```powershell
cd C:\student_resource\student_resource
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage tests
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage setup-test
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage benchmark -Workers 8
```

`setup-test` builds `runs/test_match.sqlite`, a compact test record index with the fields needed by the classifier, then `runs/test_match_lsh_b8.sqlite`. The training index cannot retrieve test IDs. The compact builder avoids the old FTS index because inference uses LSH. It logs its loading and indexing progress. The LSH build is still a substantial one-time job over all test S2/S3 records. Both completed indexes are reused.

`benchmark` processes only the first 1,000 test S1 entities using the full test indexes and includes blocking, features, model scoring, and output writing. Its files under `runs/matching_benchmark_*` are partial and must not be submitted. Use the measured rate to judge whether the full run fits your remaining time. Early test rows need not represent the full country distribution. Eight workers use more memory than four; the full-data benchmark is the deciding measurement. Use `-Workers 4` if eight workers cause memory or disk contention.

For full predictions after the benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage predict -Workers 8
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage validate-output
```

The full run writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file records every pair scored by the classifier. Final matches are a subset, every processed S1 gets one row, and IDs come from the test index. Only `matching_results.tsv` goes to the leaderboard portal; the final challenge ZIP also needs `candidate_pairs.tsv`, source code, pinned environment information and the filled methodology template. The provided official validator checks output format, not predictive accuracy. Its default invocation does not load all S2/S3 IDs; the inference code obtains candidate IDs directly from the test index.

No submission is uploaded automatically.

## Interruption and restart

Prediction saves completed shards of 1,000 S1 entities under `output/shards/`. After an interruption, rerun the identical `-Stage predict` command. Completed shards are reused; the interrupted shard is recomputed. A changed model or inputs are rejected against the checkpoint. Do not delete `output/shards` until the full output is finished and checked. Prediction needs disk space for both the shards and the assembled outputs. The runner refuses to overwrite predictions from a different run; pass `-OutputRoot` to choose another output directory.

Compact record loading can resume. LSH loading in this release cannot resume: if its build was interrupted before ready status, only that incomplete LSH sidecar must be removed and rebuilt. A completed LSH index is never rebuilt by setup-test. Keep the compact record index.

## Reproduce model training

To reproduce the included model using the training indexes already completed:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_matching.ps1 -Stage train -Workers 4
```

This creates `runs/matcher_training_v1`, samples 3,000 new S1 entities with seed 41 while excluding `runs/tune_sample/queries.tsv`, generates their candidates, extracts comparison fields from the supplied normalized sources, then fits and evaluates the model. It copies the trained model and report into `models/`. Replacing the model invalidates any existing prediction checkpoint; use a new output directory for another model. The optional `-PythonPath`, `-DataRoot`, `-RunRoot`, and `-OutputRoot` parameters support other machines and locations.

To reproduce the entire challenge solution, retain your original normalization source and its dependencies along with these files. The final submission code folder must include the normalization entry point and its exact input/output command as well as this matching stage. Training and inference use only the challenge data.

## Verification

The tests cover the challenge macro metric and singleton handling, missing-value comparisons, model serialization and learning, compact index creation, open-country behavior, full candidate/match inference, prediction checkpoint resume and rejection of changed models. Standalone inference was also checked against all 450 held-out predictions. See `models/inference_verification.json` for the measured benchmark and parity checks.
