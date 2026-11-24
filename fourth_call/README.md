# Paired fourth-call frame selection

Both arms start with the same three archived uniform-frame reports. A new fourth call receives either uniform frames or frames selected for visual change. The study covers 206 development questions, with the original 103 held-out questions as the primary cohort.

Run from the repository root after installing `requirements.txt`.

```sh
python -B fourth_call/analyze.py fourth_call/terminal_counts.jsonl --archived-csv data/action_count.csv --annotations fourth_call/options.json --scope all --out replay_fourth.json --items-out replay_fourth.csv
```

Compare the generated files with [results.json](results.json) and [paired_items.csv](paired_items.csv). Both arms returned usable fourth answers for 80 held-out questions. On those questions, the change-selected and uniform arms gave 69 and 67 correct aggregate counts, respectively. The results also record response availability and bounds across the full planned cohort.

The [archived collection plan](protocol.md) retains paths from the original working directory. The command above uses the paths in this repository. [Collection notes](acquisition_notes.md) describe retries and missing responses. The [reproduction guide](../docs/reproduction.md#paired-fourth-call-frame-selection-experiment) provides input reconstruction and fresh-inference commands.
