# Open-weight frame-sampling groups

For each of the 32 prespecified videos, 24 independently seeded input sequences form six groups of four. Each generation uses the 16 evenly spaced positions within an ordered 128-frame sequence. The pinned Qwen2-VL-2B-Instruct 4-bit model receives 16 local RGB images resized to a maximum side of 224 pixels. Each slot executes a fresh greedy generation, with the same prompt, checkpoint, runtime and reset MLX seed. The local experiment runs separately from the 128-frame API study.

From the extracted artifact's top-level directory, reproduce the complete statistical report with Python and NumPy.

```sh
python open_weight_seed/score_groups.py --design open_weight_seed/design.json --counts open_weight_seed/predictions.jsonl --csv data/action_count.csv --out qwen_seed_replay.json
```

The report measures the dispersion of six aggregation gains on common valid videos and the two whole-cohort answer-yield rules. It also counts how often different input selections change a model report or a four-report estimate. The prediction table carries every final response, its source indices, ordered RGB input and prompt hashes, and output status.

To independently derive the same slots, set MLVU_ANNOTATIONS to the development Action Count JSON and MLVU_VIDEOS to the directory of original MP4s. To replay inference, obtain the pinned checkpoint with the shared open-weight study's fetch script and dependencies. The model-directory manifest and input videos are verified before generation. The weights and videos are external inputs.

```sh
python open_weight_seed/derive_design.py --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --out reconstructed_qwen_seed_inputs.json
python open_weight_seed/collect_groups.py --reference-dir open_weight_check --model-dir "$LOCAL_CHECKPOINT" --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --design open_weight_seed/design.json --run-dir qwen_seed_run
```

The optional replay_checks.py performs two frozen verification generations with a cold-loaded model, using the first video in group 1/call 1 and the last video in group 6/call 4. It compares RGB and prompt hashes, exact final strings, termination reasons and generated-token counts against the completed ledger. The target slots and verification procedure are recorded in replay_plan.md.
