# FIM manuscript revision: inference and provenance

20 October 2026. Internal review only. No new experiment, raw-artifact reproduction, external submission, or change to a protected study is made by this revision.

## Source boundary

This revision refers to repository snapshot `142739465f0c3aa8efc674f64c793b818c9e48e5`, specifically `results/current_component_ablations/analysis.json` (Git blob `f11aed63bfa51b81693615f07fa02157ec544fd9`) and `paper/fim-current-evidence.md`. The September analysis records executed source identity `e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`. A repository commit and an executed source-bundle digest are different identifiers; neither substitutes for the other.

The August 29 adjudication describes a separate run at checkout `d4fdca0189f071ba4ce38f874301ae8be8e586ba` with different results. Do not combine it with the September matrix, choose favorable values from each, or count their cells as independent domain replications. The historical mini-suite and paper-reference values are also separate. No ITFM v9 M4 Kaggle output is included here.

## Replacement abstract

A model containing an external memory bank does not establish the usefulness of that bank merely by producing forecasts. We examine a retained component study of the Fabric-Induced Memory implementation that crosses two generated benchmarks, three optimization seeds, and four explicitly defined interventions. The September study's committed analysis records 24 cells under one executed source identity. On delayed recall, mean rollout mean-squared error is 0.804509 with the full system and 0.804421 without memory. On Lorenz-96, the corresponding means are 0.568927 and 0.569006, with paired differences changing sign across seeds. Removing salience-gated writing increases mean Lorenz-96 error by 0.040930, but two of three seeds favor removal; one adverse seed drives the mean. The no-memory and no-retrieval variants have identical stored error statistics. These results do not demonstrate a stable benefit of the tested memory pathway. They also do not establish equivalence or the absence of effects: the study has only three seeds per benchmark and no verified equivalence margin. This revision separates distinct historical studies, reports direction and magnitude without significance inflation, and identifies the evidence needed to support a stronger mechanism claim. No training or evaluation was performed for this revision.

## Corrected result table

Delta is reduced variant minus full; negative favors removal. All values below come from the September analysis, not a new computation on checkpoints.

| Benchmark / removal | Seed 11 | Seed 23 | Seed 37 | Mean delta |
|---|---:|---:|---:|---:|
| Delayed / memory | -0.000011444 | -0.000252426 | +0.000000477 | -0.000087798 |
| Delayed / retrieval | -0.000011444 | -0.000252426 | +0.000000477 | -0.000087798 |
| Delayed / salience gate | 0.000000000 | -0.000280976 | -0.000037074 | -0.000106017 |
| Lorenz / memory | +0.003006995 | -0.031505704 | +0.028738081 | +0.000079791 |
| Lorenz / retrieval | +0.003006995 | -0.031505704 | +0.028738081 | +0.000079791 |
| Lorenz / salience gate | -0.002717316 | +0.155525923 | -0.030019760 | +0.040929615 |

The positive delayed-recall seed-37 contrast must not be rounded into a tie when counting directions. The zero salience contrast must remain a tie.

## Required interpretation changes

1. Replace unqualified statements that this study "falsifies necessity" with: **the retained study does not establish a consistent benefit from the tested component under its settings**. No equivalence margin, equivalence test, or adequate-power claim has been verified.
2. Identical stored no-memory/no-retrieval error statistics do not alone prove bitwise prediction identity, correct storage behavior, or a bug. Do not present these duplicate contrasts as independent evidence.
3. With three nonzero seed pairs, the minimum two-sided exact sign-test p-value is 0.25. Large p-values are not proof of no effect. The tied salience comparison has only two nonzero observations.
4. Describe standard deviations as variation across optimization seeds, not population confidence intervals or variation across independent domains.
5. Do not treat equal epochs/updates as equal compute. No new GPU latency, throughput, memory-use, or efficiency evidence was obtained.
6. Describe the endpoint as the endpoint used by the committed analysis unless its prospective selection is authenticated. A retrospective revision cannot create preregistration.
7. Preserve the separate mini-suite baseline win as context, not a matched comparison with this matrix. Do not pool its parameters, losses, or ranking into the component results.

## Evidence needed for a stronger claim

A prospective successor should distinguish whether a relevant trace was available, retrieved, and used. Oracle retrieval, random retrieval, erased values, and a capacity-matched simple baseline are discriminating controls. Delay, distractor density, capacity, and query noise may be varied only within a separately bounded development protocol. External transfer requires admitted public data and a protected evaluation. These proposals are NOT RUN and do not amend the closed September study.

## Release boundary

The repository remains an internal archival/manuscript line, not an independent replication or a new campaign host. Full executed-source and raw-artifact verification, authorship, independent technical review, and an explicit release decision remain open. Existing source manifests and result files are unchanged. No GPU or CPU research run, paid compute, or external submission was initiated by this revision.
