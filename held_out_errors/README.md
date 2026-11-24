# Counting errors on the held-out development videos

Table 4 and Figure 3 compare the first uniform report with the four-report archived MAMC prediction for the 103 held-out MLVU development questions. Both use the same reference counts. A correction changes a wrong first answer to a correct one. An introduced error changes a correct first answer to a wrong one. The final error direction compares the aggregated count with the reference count.

The script applies the original grid and split from the package's `reproduce.py`. It sorts the 206 video filenames lexicographically and permutes them with NumPy generator seed zero. It selects the parameters on the first 103 indices, preserving the original grid order on ties, and verifies the recomputed left-to-right binary64 predictions against all 206 archived counts. All reported errors and strata in this folder use the remaining 103 indices. The original all-development statistics remain in the top-level replay.

Run from the package root with Python and NumPy.

```sh
python -B held_out_errors/analyze.py --csv data/action_count.csv --out-dir held_out_replay
cmp held_out_replay/results.json held_out_errors/results.json
cmp held_out_replay/items.csv held_out_errors/items.csv
```

`results.json` gives the overall and stratified rates, paired transitions, exact two-sided McNemar test, bootstrap difference interval and figure bar heights. The default percentile interval resamples 103 paired outcomes with replacement in each of 200,000 draws using seed zero. `items.csv` records all 103 questions, their four reports, the aggregate, error direction and absolute error. The `difference` fields subtract the first-call metric from the archived four-report metric. Accuracy and undercount and overcount rate differences are in percentage points.
