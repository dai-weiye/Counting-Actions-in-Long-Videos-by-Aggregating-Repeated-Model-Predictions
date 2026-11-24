# Open-weight counting on the official MLVU test split

Each of the 60 official counting videos was assigned four independent generations using the pinned Qwen2-VL-2B-Instruct model. The 16-frame settings are shared with the development panel in `open_weight_check/`. The six numeric options and official source file SHA-256 values are included in `cohort.json`. All 240 generated responses, frame indices, hashes and final text are stored in `predictions.jsonl`. `prediction_lock.json` identifies the frozen ledger used for scoring. The model and exact stopping code are shared across experiments so that the same functions run in each setting.

## Offline scoring

Use Python and NumPy for the frozen offline replay. The package includes the 60 public numeric answers in `official_counts.json`. To verify the labels independently, install PyArrow and download [the official published test annotations at the pinned dataset revision](https://huggingface.co/datasets/sy1998/MLVU_Test/resolve/4a9826be2bc6c86f9cab9222a06a55c224098958/mlvu_test/test-00000-of-00001.parquet). The scoring command verifies the annotation SHA-256 and the complete frozen prediction ledger before mapping the public answer letters to the six numerical options. Only aggregate scores are written.

```sh
python -B open_weight_test/score.py --out replay_qwen_test.json
cmp replay_qwen_test.json open_weight_test/results.json
python -B open_weight_test/score.py --annotations "$MLVU_TEST_PARQUET" --export-labels rebuilt_counts.json --out replay_from_public_source.json
cmp rebuilt_counts.json open_weight_test/official_counts.json
cmp replay_from_public_source.json open_weight_test/results.json
```

## Model execution

On Apple Silicon, install the versions in `runtime.json` and use the model verification and download program from `open_weight_check/`. Supply the 60 source videos from the revision and archive referenced in the test study. `collect.py` verifies each video against the hash in `cohort.json`.

```sh
python open_weight_check/fetch_weights.py --model-dir "$QWEN_MODEL_DIR"
python -B open_weight_test/collect.py --model-dir "$QWEN_MODEL_DIR" --video-dir "$MLVU_TEST_VIDEOS" --run-dir "$QWEN_TEST_RUN"
python -B open_weight_test/freeze.py --run-dir "$QWEN_TEST_RUN" --out "$QWEN_TEST_OUTPUT"
```

The collection program reads only the label-free roster, input videos and checkpoint files. The fixed development duration predictor and its test predictions are described in the study plan. The API model, input budget and generated outputs belong to a separate experiment.
