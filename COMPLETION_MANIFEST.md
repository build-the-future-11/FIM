# FIM — COMPLETION_MANIFEST

## Research Question
Are the maintained memory, retrieval, and salience-gating components generally necessary for improved performance in the frozen component study?

## Final Hypothesis
Removing each maintained component should systematically degrade performance if that component is necessary under the tested frozen protocol.

## Contribution
A bounded, reproducible component-level test of the maintained FIM implementation, including negative/null mechanism evidence rather than a state-of-the-art claim.

## Evidence
- 24/24 frozen component cells completed.
- Six paired reduced-minus-full comparisons retained.
- Exact sign tests retained.
- 42 passing tests, 7 explicit hardware/optional-path skips, 0 failures in the recorded verification.
- Evidence checksum sidecar and compiled seven-page preprint recorded by the repository status.

## Primary Result
The favorable mean direction splits 3/6 for the full model and 3/6 for removals; the smallest exact sign-test p-value is 0.5.

## Disposition
**NEGATIVE**

## Scope
The result applies to the frozen maintained component study. It does not establish natural-domain superiority, state-of-the-art performance, or universal uselessness of memory mechanisms.

## Limitations
The study is bounded and does not answer whether a newly preregistered mechanism-recovery design on independent public datasets would produce a different result.

## Reproduction
Use the repository's frozen ablation runner, analysis script, manifest validator, tests, and checksum verification exactly as recorded in `PROJECT_STATUS.md` / `REPRODUCE.md`.

## Integrity
Do not restore the historical `no_spectral_mixing` label to current-code interpretation; it has no valid maintained-code mapping. Preserve the negative result and do not rerun the frozen study to seek significance.

## Release
Treat the current repository as a completed negative-result package; any stronger claim belongs to a new successor study.
