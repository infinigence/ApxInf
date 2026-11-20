# Historical ApxInf Review Patterns

Simplified Chinese version:
[historical-patterns.zh-CN.md](historical-patterns.zh-CN.md).

This reference records durable failure patterns, not a permanent verdict on the
linked PRs. It was derived on 2026-11-02 from 83 merged pull requests, including
55 independent inline review roots, 41 thread replies, 13 non-empty review
bodies, and 52 merged-PR issue comments across 27 PRs. Re-check current code;
many linked findings were fixed before merge.

## Recurrent patterns and evidence

1. **Asynchronous ordering and lifetime dominate device bugs.** A process-wide
   allocation cache published buffers while work was in flight ([#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4044354105)); a migration from blocking to non-blocking CUDA streams invalidated default-stream copy/clear assumptions ([#117](https://github.com/infinigence/ApxInf/pull/117#issuecomment-6468340954)); and a prepared graph could outlive the runtime its executions referenced ([#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)). Always trace stream order and ownership across the full handle graph.

2. **Prepared execution identity is frequently incomplete.** Exact token shapes
   were hidden behind a padded capacity ([#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225493)); shared plan caches did not match `thread_local`
   native state, and fused/plain tactics could alias one key in the retired
   legacy JSON tuning path ([#35](https://github.com/infinigence/ApxInf/pull/35#pullrequestreview-5075611780)). The old database mechanism is no longer a
   current cuda-new contract, but its durable lesson remains: derive recipe and
   execution keys from every fact that changes preparation or provider selection.

3. **Global switches conflict with instance semantics.** Model configuration was
   written to a global atomic and process environment ([#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225504)). If a backend supports only process-wide policy, conflicting loads must fail rather than silently change existing models.

4. **Checkpoint identity must span languages and all behavior-changing assets.**
   Rust and Python selected/sorted checkpoint shards differently ([#29](https://github.com/infinigence/ApxInf/pull/29#pullrequestreview-5074847186)); calibration identity omitted processor/statistics files ([#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4121159495)). Identity rules need one canonical, cross-language test vector.

5. **Fallback and capability checks must be structural.** A fused route selected
   hardware shapes without checking whether its backend was compiled ([#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4120106199)); an operator without fallback pushed a duplicated architecture predicate into model code ([#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)); vision weights were made mandatory for text-only checkpoints ([#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194595)). Capabilities should follow loaded assets and available providers.

6. **Safe public paths must reject malformed checkpoints and requests before
   expensive work.** Weight loading used indexing, `unwrap` and assertions
   ([#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225512)); image prefill omitted a KV-capacity guard until after the vision tower ([#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194586)); and a rejected request could permanently poison later DSpark requests ([#112](https://github.com/infinigence/ApxInf/pull/112#issuecomment-6074838568)). Validation errors should remain recoverable and side-effect-free.

7. **Build and feature coverage often misses the actual changed backend.** A
   full CUDA migration passed hosted checks while `--features cuda` did not
   compile ([#117](https://github.com/infinigence/ApxInf/pull/117#issuecomment-6468340954)); feature unification changed enum availability in unrelated crates ([#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4119947568)); a model-only AOT bundle could break a second CUDA crate ([#107](https://github.com/infinigence/ApxInf/pull/107#issuecomment-6095939102)). Review the real feature matrix, partial bundles and workspace unification.

8. **Memory and hidden hot-path work are commonly larger than kernel math.**
   Disabled diagnostics still performed D2H copies and synchronization ([#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4055999271)); prepared sessions and duplicate fused weights retained many GiB ([#107](https://github.com/infinigence/ApxInf/pull/107#issuecomment-6095939102)); a fused winner still allocated/tuned decomposed resources ([#41](https://github.com/infinigence/ApxInf/pull/41#pullrequestreview-5089082728)). Count retained and transient resources per profile and layer.

9. **End-to-end evidence catches failures that operator tests miss.** Model
   outputs changed with output-buffer clearing despite passing FP64 oracles
   ([#78](https://github.com/infinigence/ApxInf/pull/78#issuecomment-5748251587)); non-square vision differences required controlled per-layer analysis ([#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194606)); a backend migration repaired multiple self-introduced semantic regressions without an old/new model oracle ([#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)). Require real-checkpoint, exact-route comparisons after migrations.

10. **Cross-layer option plumbing and boundary semantics regress silently.** A
    configured planning-step override was stored and reported but never forwarded
    ([#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4056011288)); EOS in the final token-budget slot was misreported as truncation ([#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4320978650)); a new HIP alias bypassed variant admission ([#164](https://github.com/infinigence/ApxInf/pull/164#issuecomment-6410487397)). Test defaults, aliases and non-default overrides through public entry points.

11. **Scope and layer hygiene prevent unreviewed production surface.** Reviewers
    removed probes, experimental reports, unused four-layer operator stacks and
    unreachable perception scaffolding from one PR ([#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4044289465)); unrelated Llama edits were reverted from a CUDA performance PR ([#107](https://github.com/infinigence/ApxInf/pull/107#discussion_r4236999862)); vendor kernel provenance and adapter/kernel placement needed correction ([#130](https://github.com/infinigence/ApxInf/pull/130#issuecomment-6263373023)). Trace every new path to production and enforce repository ownership.

12. **Architecture checks and tests can accidentally bless exceptions.** A
    migration retargeted guards away from a still-supported legacy backend and
    added source-text tests that froze a registry bypass ([#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)). Compare policy/check scripts against the base and prefer behavioral tests over source grep.

## Applying the history

Use these patterns to choose where to spend review depth. Do not report a defect
merely because a historical PR had one. Reproduce the relevant invariant on the
current pinned comparison, provide a concrete trigger and impact, and record when
the current design already closes the old failure mode.
