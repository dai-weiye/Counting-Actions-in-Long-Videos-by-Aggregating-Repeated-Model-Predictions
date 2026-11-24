# Acquisition notes

The local analysis plan is preserved in `protocol.md` with SHA-256 `43a96974d2d695b365b7c25c7a17f2b28eae25a76957c67dd6ce849777067409`.

At 18:46 UTC on 23 November 2026, before inspecting accuracy in either new fourth-call arm or the difference between arms, the investigators confirmed collection for the optional 206-question secondary cohort based on available credentials, budget and recoverable videos. The 103-question held-out cohort remains primary, with its original scoring and comparison rules.

During acquisition, a network failure prompted an operational change to replay the identical payload after an unavailable response. Each item and arm permits at most five client attempts in total for transport failures and HTTP 429, 500, 502, 503 or 504. HTTP 401, 402 or 403 pauses collection for the job. A nonretryable HTTP error receives the terminal status `http_error`. Exhausting the retry allowance without a usable HTTP response receives `transport_exhausted`. A lost response leaves the server's execution status unknown.

The first complete model response closes the item and arm, including a blank or unsupported final answer. New fourth-call outputs never use reasoning text or an earlier response as a replacement. All terminal failure statuses and unrecorded arm slots are counted separately from the matched valid pairs.

At 18:59 UTC on 23 November 2026, after seeing the frequency of missing final answers and before examining the primary accuracy contrast, the investigators added a secondary bounds analysis. For each missing report, this analysis enumerates the four offered answers and applies the fixed prefix and smaller-maximizer rule to each. The minimum and maximum possible correctness for that arm bound change minus uniform on all planned questions. These bounds describe hypothetical valid completions from the listed options. They do not assign a response to a blank call. The original unconstrained bounds and complete-pair comparison remain the prespecified analyses.
