# Independent MLVU Test Action Count evaluation

The MLVU authors released test answers in their 18 March 2025 update. We used all 60 test counting videos from the dataset distributed through the accompanying lmms-eval task. The original task presents six numeric options in the test split. The study evaluates four model reports and a duration-only predictor trained on the 206 development questions. Parameters and model requests were fixed before any test answers were read.

- [MLVU project and release announcement](https://github.com/JUNJIE99/MLVU)
- [MLVU Test task in lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mlvu/mlvu_test.yaml)
- [Pinned test files](https://huggingface.co/datasets/sy1998/MLVU_Test/tree/4a9826be2bc6c86f9cab9222a06a55c224098958)
- [Primary dataset card and research-use terms](https://huggingface.co/datasets/MLVU/MLVU_Test)

From the root of the extracted supplement, reproduce the paired statistics and independently refit the development-only duration predictor:

```sh
python external_eval/verify_release.py
python external_eval/score_test.py --out test_replay.json
cmp test_replay.json external_eval/results.json
```

The analysis uses NumPy and scikit-learn. The calculator reads the 60 item IDs, offered numbers, public counts, durations and fixed duration predictions in `items.json`, the 240 terminal slot observations in `calls.jsonl`, and the development counts and durations in `data/action_count.csv`. Its paired test and score use the same implementation as the fourth-call comparison. Slots U1, U2 and U3 receive the same 128 uniformly sampled frames; slot C4 receives 128 change-selected frames. A complete four-report prediction requires four valid final answers. The prespecified fallback scores the reports that arrived with a valid final number. See `protocol.md` and `acquisition_notes.md` for the original endpoints and additional candidate-range comparisons.

The two unique visual inputs per question are in `selected_frames.jsonl`, with temporal indices and exact request SHA-256 digests. Raw model traffic and the videos are excluded. `video_sources.json` contains the official test archive member names, archive checksums, byte offsets, compressed sizes and extracted CRC32. `test_video_digests.json` and `dev_video_digests.json` record file SHA-256 values for an overlap check. These identify exact file matches. The release does not annotate whether source footage within different files is shared.

The service paused after 160 model responses when five submissions returned HTTP 402. The 160 responses and 80 outstanding slots at that point are archived in `checkpoint_at_pause/`. At resumption, the collector submitted only outstanding slots with the frozen request digests. It used three concurrent videos and retained all earlier responses, including blank finals. `verify_release.py` compares the checkpoint with the completed run, checks all 240 request digests and all 120 recorded frame selections, and verifies the source file overlap checks.

All 60 source videos were hashed before 38 completed videos were removed from the working cache. The export matched both planned frame inputs for each video to its previously recorded file digest and rehashed each of the 22 files still available. `run_settings.json` records these counts. `prepare_scores_from_frozen.py` performs this cache-aware export. The original preparation and scoring scripts are preserved under their recorded digests in `analysis_lock.json`.

To reconstruct the inputs from the pinned release, install requests, PyArrow, scikit-learn, NumPy, OpenCV and Pillow. From an extracted writable copy of the supplement:

```sh
python external_eval/prepare_inputs.py --cache "$VIDEO_CACHE" --inventory
```

This fetches the annotation and only the ZIP members for the 60 videos into the chosen cache. Each MP4 is published to the collector only after its expected length and CRC32 both match. Inputs and the duration predictor are prepared without reading test answer columns. `questions.json` is generated in `external_eval/`. Eliminate remaining temporary files by removing only the dedicated cache created by this command once verification is complete.

An optional independent API run uses these downloaded videos and sends the same prompts and frames to the official DeepSeek service. Set `MAMC_API_KEY` in your local environment and run:

```sh
python external_eval/collect_test.py --videos "$VIDEO_CACHE" --run-dir "$RUN_DIR" --jobs 4
python external_eval/prepare_scores.py --annotations "$VIDEO_CACHE/annotations.parquet" --videos "$VIDEO_CACHE" --run-dir "$RUN_DIR" --out "$RUN_RESULTS"
python external_eval/score_test.py --items "$RUN_RESULTS/items.json" --calls "$RUN_RESULTS/calls.jsonl" --out "$RUN_RESULTS/results.json"
```

Do not reuse the same run directory for two separate studies. The collector can resume interrupted slots in an existing study. A completed model response closes a slot even when the final number is blank. The baseline and the two uniform repeats are three separate inference calls with byte-identical inputs. New provider runs may return different predictions. The published `calls.jsonl` supports exact offline score replay.

Test question IDs, options, durations and reference answers are derived from MLVU by Junjie Zhou and colleagues, CVPR 2025, under the benchmark's CC BY-NC-SA 4.0 research-use terms. Videos remain with their original rightsholders.
