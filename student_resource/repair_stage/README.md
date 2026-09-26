# Repair for the interrupted training index

The second normalization used six workers, while the first used ten. The six new parts `00`–`05` were written, and old parts `06`–`09` remained. The blocker read the mixed set and encountered a repeated S2 entity ID. The six new parts have the expected row counts for five sources. The new `test_source3` is missing one row because the old normalizer skipped a row when a byte range started exactly at a line boundary.

Extract this repair package anywhere. In PowerShell, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\repair_stage\repair.ps1
```

If your terminal is inside the extracted `repair_stage` folder, run `powershell -ExecutionPolicy Bypass -File .\repair.ps1` instead. The default project location is `C:\student_resource\student_resource`; pass `-ProjectRoot` if needed.

The script:

1. Backs up your current `src\normalization.py`, `src\blocking.py`, and `run.ps1` with timestamped suffixes.
2. Installs corrected versions. The normalizer now preserves exact byte-boundary rows and removes old tail parts after a successful run with fewer workers. The blocker can resume an interrupted SQLite build.
3. Regenerates only `test_source3` with six workers.
4. Fully reads all six new normalized sources and checks row counts against the known raw counts. Only after every check passes does it remove the verified older `06`–`09` parts.
5. Resumes the interrupted `runs\train.sqlite` build from its committed 5,030,000 S2 records, then runs the original training validation. The S2 pass will still scan its normalized rows, but it will not re-index the committed records.

The script can take a while because it must regenerate one 5-million-row source, verify all 24-million normalized rows, finish the index, and run validation. It reports each stage as it progresses. `-SkipValidate` performs the repair and stops before resuming blocking; run `powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage validate` from the project root later.

If a verification fails, stale files remain and the index is not resumed. The checker is safe to run without deletion:

```powershell
py -3 .\repair_stage\verify_parts.py C:\student_resource\student_resource\data\normalized
```
