# Historical FIM: complete retained-analysis presentation

**Separate candidate for SIAM DS27; not submitted.** This package covers the
closed 24-cell historical shared-minibatch study in this repository: Lorenz96
and delayed recall, three seeds and four component configurations. It is
distinct from FIM Burgers V2, delayed-recall V2 and the held trajectory-isolated
successor in the sibling runtime repository. It must not replace those lineages
in a project register or be represented as their source recovery.

## Result and terminal scope

The retained result remains negative. Mean directions favor the full model in
three of six paired comparisons and removals in three. The smallest exact
two-sided sign-test p-value is 0.5. No-memory and no-retrieval have identical
retained rollout-MSE sequences across the six benchmark/seed combinations;
matching errors do not establish component necessity or prediction identity.

The retained-report analysis and presentation assets are complete. This work
does not reopen the experiment, access the held successor outcomes, establish
equivalence, or make a compute-efficiency or universal no-benefit claim. It
preserves the original compiled preprint and every original result artifact.

## What is included

| File | Purpose |
| --- | --- |
| `abstract.txt`, `abstract_title.txt`, `abstract_metadata.json` | Complete separate historical-study candidate abstract, title and submission inputs |
| `poster_text.md` | Complete poster text, methods, result table and limitations |
| `historical_paired_comparisons.svg`, `.pdf`, `.png` | All paired seed contrasts and their means; independent benchmark scales |
| `cells.csv`, `paired_comparisons.csv` | All 24 seed-level cells and six paired descriptive comparisons |
| `analysis.json` | Protocol-scoped machine-readable analysis |
| `SOURCE_ARTIFACTS.json` | Verification record for all 96 retained text artifacts |
| `MANIFEST.json` | Source script and generated asset SHA-256 identities |

## Reproduce without scientific execution

Install Matplotlib for figure rendering and pytest for the focused tests, then
run from the repository root:

```bash
python scripts/build_historical_ds27_presentation.py
python -m pytest tests/test_component_ablation_analysis.py tests/test_historical_text_evidence.py -q
```

The new builder writes only this presentation directory. It does not execute
the existing runner, rebuild the preprint, overwrite the original analysis,
deserialize checkpoints or evaluate a model. The existing analyzer now rejects
incomplete or duplicate matrices, coerced seed identities and invalid errors.
Its retained-text loader checks 96 hashes and matches manifest errors to both
cell summary and run-results JSON before admitting the analysis. Previously,
calling the reducer on a 23-cell matrix silently produced a mixture of two- and
three-seed comparisons. An empty sign test also returned 1.0; it now fails.

Local focused validation passed **24 tests**. The tests cover complete seeded
pairing, malformed data, source/report disagreement even after a digest update,
path escapes, protocol drift, and zero checkpoint reads.

## Provenance and limits

All source artifacts were retrieved at immutable repository base
[`142739465f0c3aa8efc674f64c793b818c9e48e5`](https://github.com/build-the-future-11/FIM/tree/142739465f0c3aa8efc674f64c793b818c9e48e5).
The retained manifest's actual byte SHA-256 is
`6350bd2252e75a4d1886f9f6f6f817ea1ff3256f30549a81b27396e37c555e9a`.
All **96/96** text files (24 summaries, 24 run-results, 24 configurations and
24 logs) match their manifest hashes. Both error reports per cell agree with
the manifest. These files support the reported arithmetic and provenance.

The historical manifest declares source-tree SHA-256
`e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46`
and execution Git commit `UNKNOWN`. This work preserves that distinction; it
does not claim to reconstruct the original executed source tree. The 24
checkpoint identities remain in the original manifest, but no checkpoint was
read, hash-audited or replayed by this presentation workflow. This is a text
artifact audit, not a full execution replay.

The protocol retains batch size 64 and a four-step training rollout. This
historical shared-minibatch lineage must not be described as a trajectory-
isolated mechanism test. Three training seeds are the paired observational
units. They do not supply independent-domain replication; small exact sign-test
p-values are unattainable with three nonzero pairs (minimum two-sided p=0.25),
and the observed minimum is 0.5. Mean directions and tiny error differences
alone do not establish mechanism recovery, equivalence or general usefulness.

## Conference compatibility and remaining author inputs

The candidate is framed as an honest historical component-analysis and
reproducibility poster for a dynamical-systems audience. Author review must
confirm SIAM DS27 topic fit, the presenter, affiliation and availability under
the meeting's presentation rules. The complete abstract is below the stated
1,500-character limit; the title is supplied separately. Approval to submit,
registration and attendance are separate from this repository deliverable.
See the official [SIAM DS27 submission page](https://www.siam.org/conferences-events/siam-conferences/ds27/submissions/).

The originally planned **Burgers V2** candidate remains a separate closed
negative development result. Its exact private archive could not be recovered
through authorized materialization in this session; no replacement evidence
or new Burgers result is asserted here.
