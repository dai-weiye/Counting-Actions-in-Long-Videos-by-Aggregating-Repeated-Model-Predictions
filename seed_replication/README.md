# Seed replication

The design ranks the original 103 held-out MLVU Action Count filenames with a fixed hash and takes the first 32. Six independent groups of four random frame subsets each use the same 32 questions. For each video, all six groups share one pool of 256 uniformly spaced frame positions. Each call samples 128 positions without replacement and presents the frames in temporal order. The coverage panel re-evaluates the six videos without historical random-sampling predictions, using the first four new streams. The two panels have disjoint videos.

From the top of the extracted artifact, run the two offline analyses with NumPy and Python 3.11 or newer.

```sh
python seed_replication/analyze.py --design seed_replication/main_design.json --counts seed_replication/main_terminal_counts.jsonl --csv data/action_count.csv --out replay_main.json
python seed_replication/analyze.py --design seed_replication/recovery_design.json --counts seed_replication/recovery_terminal_counts.jsonl --csv data/action_count.csv --out replay_recovery.json
```

The resulting JSON files match the two bundled reports byte for byte. The main report gives all six gains and their sample standard deviation on the same complete cases. The all-question analyses include all 32 in every group. They distinguish the four-valid-reports rule and the prespecified fallback that uses the available numerical reports. The seed-group spread reflects both input sampling and variation between model executions on this fixed set. Only the historical evaluation used process-salted seeds.

For input replay, provide the public MLVU Action Count development annotations and source MP4 files, using the NumPy, OpenCV and Pillow versions reported for request replay in the artifact. The compact design stores the source-video SHA-256, frame counts, per-stream pool positions and per-call frame indices. Each terminal record with an encoded request contains its one-way payload digest.

```sh
python seed_replication/design.py --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --out replay_design.json
python seed_replication/verify_payload.py --design seed_replication/main_design.json --counts seed_replication/main_terminal_counts.jsonl --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --item count_121 --group 1 --call 1 --inference-protocol inference_protocol.py
```

Use --recovery with design.py to reconstruct the six-video input panel. The optional collect.py sends model requests to the original service with a caller-supplied MAMC_API_KEY. Its --run directory holds private diagnostic records and should be kept separate from the released anonymous files.

The main panel resumed from 648 terminal slots after an interruption in service billing. Before resuming we verified the hashes for all 32 original videos and rebuilt each request independently from the 256-frame pool. All earlier terminal records were preserved. The main_coverage.json file separates model replies and exhausted transport attempts across the complete set of assigned slots.
