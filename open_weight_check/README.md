# Open-weight counting experiment

The 32 videos, four slots, fixed model revision, frame budgets, parser and aggregation rule are specified in [the study plan](protocol.md). The input and numerical output for each generation are recorded in [predictions.jsonl](predictions.jsonl). Each digest in that file is a client-side checksum of ordered, resized RGB images (width and height as two big-endian unsigned 32-bit integers followed by pixel bytes, repeated for each frame) or of the UTF-8 formatted chat prompt. The [runtime](runtime.json) records the software versions used on the Apple GPU. The release includes the prediction text and archived pixels’ indices and hashes.

## Offline scoring

Run the following command from the top-level anonymous reproduction package.

```sh
python -B open_weight_check/score.py --labels data/action_count.csv --out replay_open_weight.json --items-out replay_open_weight.csv
```

Compare these outputs with [the stored report](results.json) and [the item-level results](items.csv). Scoring needs only Python and NumPy.

## Running the checkpoint

The video-generation path uses the public Qwen2-VL-2B-Instruct 4-bit checkpoint, publicly released MLVU development videos, the original MLVU `4_count.json` annotation file, and Apple Silicon with MLX. Install the versions listed in `runtime.json`. Specify local directories for the checkpoint and generated records.

```sh
python open_weight_check/fetch_weights.py --model-dir "$QWEN_MODEL_DIR"
python -B open_weight_check/collect.py --model-dir "$QWEN_MODEL_DIR" --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --run-dir "$QWEN_RESULTS_DIR"
```

The downloader verifies every file against the pinned revision, including the publisher's SHA-256 for the weights. The optional `--modelscope` flag fetches identical files from the pinned mirror revision. The runner checks each video's SHA-256 and performs four separate generations with greedy decoding. To replay the four slots of a single example, add `--item count_121` and use a fresh results directory.
