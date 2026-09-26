# LightGBM matching update

Extract this ZIP into `C:\student_resource\student_resource`. It supplies a trained model. The existing normalized data, compact test record index and test LSH index are required.

## Run

From the project PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_matching_v2.ps1 -Stage tests
powershell -ExecutionPolicy Bypass -File .\run_matching_v2.ps1 -Stage predict -Workers 8
powershell -ExecutionPolicy Bypass -File .\run_matching_v2.ps1 -Stage validate-output
```

The first prediction run creates a 20 MB country lookup from the completed test record index. This took about 25 seconds on this laptop. It then scores all test S1 records. Progress is printed every 100 records, with completed output saved every 1,000 records. If interrupted, run the identical predict command to resume completed shards. Keep the model, normalized files and shard settings unchanged during a run. A changed model or input fingerprint is rejected on resume.

Results are `output/matching_results.tsv` and `output/candidate_pairs.tsv`. After validation, upload `matching_results.tsv` to the portal. The final code ZIP must also contain both output files and the completed methodology document. Candidate output includes every pair scored by the classifier, and all predicted matches are among those candidates. Validation reads candidates one row at a time, checks formatting and match membership, then calls the supplied official validator for matches and complete S1 coverage. This avoids loading hundreds of millions of candidate IDs into memory. S2/S3 ID existence is not separately scanned: inference gets those IDs from the test record index. Validation does not measure predictive accuracy.

An optional `-Stage benchmark -Workers 8` processes only the first 1,000 test S1 records and writes a separate run directory. Benchmark outputs are partial and must not be submitted. `-OutputRoot` selects a different prediction directory when needed.

## Measured accuracy

| Model | Held-out macro F0.5 | Pair precision | Pair recall |
|---|---:|---:|---:|
| Previous logistic model | 0.865035 | 0.937594 | 0.812907 |
| Supplied LightGBM model | 0.903525 | 0.960699 | 0.860495 |

Both evaluations use the same 450 audit S1 entities. Training uses 2,100 other S1 entities and all their 349,054 retrieved candidate pairs, including 6,569 positives. Another 450 S1 entities select model complexity and the score threshold (0.56) by the challenge's macro F0.5 metric. The original 1,000 blocker-tuning entities are excluded from this 3,000-entity sample. Splits are by S1 entity; they are not guaranteed to be disjoint connected components of the full entity graph. Audit results include true links missed by blocking and singleton entities.

The upgraded model uses the original name, address, number, phonetic and retrieval features plus native Levenshtein, Jaro-Winkler and fuzzy token comparisons. LightGBM learns nonlinear combinations of these features. Fitting and feature extraction took about 147 seconds using the already prepared sample. Training does not require generating candidates for every training S1.

India audit macro F0.5 is 0.861590 over 173 S1 entities; US is 0.929715 over 277. These are small training-data estimates, not test leaderboard scores. France has no labeled training examples. The current blocker still misses some true links, especially in Indian scripts, and limits attainable recall.

## Speed change and verification

Profiling found that cross-country retrieval repeatedly read countries from the large record database. An exact array mapping record IDs to country codes replaces those reads and is shared through a memory-mapped file. Candidate comparison fields are also fetched with their record IDs, avoiding a second lookup by entity ID. Retrieval still allows cross-country candidates.

The first upgraded 1,000-record test benchmark took 90.375 seconds. The optimized version took 26.437 seconds. SHA-256 comparisons confirmed both candidate and match files were byte-for-byte identical. All 450 saved audit predictions also matched the full pipeline's predictions using eight Windows worker processes. A longer 5,000-record test benchmark took 182.859 seconds (27.34 S1/s), implying about 17.6 hours for 1,732,544 records if that rate holds. This is an extrapolation, and later records can run differently. Detailed results are in `models/benchmark_v2.json`; runtime remains sensitive to record distribution, memory, disk activity and laptop power settings.

## Dependencies and retraining

The runner uses the Codex bundled Python 3.12 runtime by default and loads installed packages from the project's `.matching_deps`. The exact versions are in `matching_requirements_v2.txt`. For another machine, install those packages into a Python environment and pass its executable using `-PythonPath`. No GPU or external entity data is needed.

`-Stage train` recreates or reuses the 3,000-query training sample, candidate lists and comparison fields, then writes a new model directory under `runs`. Use `-ModelPath` to choose that model for later prediction. Existing prediction checkpoints cannot be reused with a different model.

The MIT license covers the supplied pipeline code. NumPy, SciPy, LightGBM and RapidFuzz retain their own package licenses.
