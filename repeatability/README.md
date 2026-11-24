# Repeatability of identical requests

For each of the 103 development questions held out from tuning, this study paired one new uniform-frame call with the uniform call from the earlier fourth-input experiment. Each row of [records.csv](records.csv) contains both final counts or completion statuses, both request SHA-256 digests, and a comparison of the provider-reported fingerprints. The same fingerprint was returned for all 103 pairs. Among the 83 questions with two numerical answers, 19 answers changed. The two rounds had 9 and 15 blank finals. The [report](results.json) gives both the response-status table and the conditional disagreement interval.

Run the following from the anonymous package root.

```sh
python repeatability/analyze_repeats.py repeatability/records.csv --out replay_repeats.json
```

The [protocol](protocol.md) defines the cohort, endpoint and treatment of unusable responses. The [run notes](acquisition_notes.md) describe the change in upload timeouts. The collector reads the held-out IDs in the published split and checks every request digest before making an inference call. An optional new acquisition requires the original dev videos, annotations and an API credential in the environment.

```sh
python repeatability/collect_repeats.py --fourth-call fourth_call --annotations "$MLVU_ANNOTATIONS" --video-dir "$MLVU_VIDEOS" --run-dir new_repeats --jobs 3
```

The model was requested through a reasoning-enabled API. The published inference settings distinguish the submitted temperature field from effective decoding. [DeepSeek thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/#input-and-output-parameters).
