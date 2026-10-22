# Retained Component Ablations in FIM: A Historical Three-Seed Evaluation

**Separate historical-study poster draft · author and affiliation fields require author input**

## Question and scope

Does removing a maintained FIM component consistently worsen performance in
the retained historical study? We examine the closed 24-cell shared-minibatch
lineage using its committed artifacts. The scientific disposition remains
negative, and the analysis does not generate new outcomes.

This lineage contains Lorenz96 and delayed recall. It is distinct from FIM
Burgers V2, delayed-recall V2 and the later held trajectory-isolated protocol.

## Retained design and verification

There are two benchmarks, three training seeds (11, 23 and 37) and four
configurations: full, no memory, no retrieval and no salience gating. The
protocol records 12 epochs, batch size 64, dataset size 2,048, four-step training
rollouts and 30-step evaluation rollouts.

We verify all 96 retained text-artifact hashes: each cell's summary, run-results
JSON, resolved configuration and log. We also cross-check rollout MSE and MAE
between both cell reports and the manifest, together with seed, experiment and
rollout-horizon identities. These checks admit all 24 cells. Checkpoints are
not loaded, evaluated or claimed to have been audited.

## Paired analysis

Within each benchmark and seed, we compute the removal's rollout MSE minus the
full model's MSE. Positive differences favor the full model; negative
differences favor removal. Means and sample deviations describe the three
paired seed differences. The exact two-sided sign test counts positive and
negative differences, omits exact zero ties, and doubles the smaller binomial
tail with a cap of one. With three nonzero pairs its smallest possible p-value
is 0.25; it cannot establish strong confirmatory evidence here.

| Benchmark | Removal | Mean removal − full MSE | Seeds favoring full | Exact sign-test p |
| --- | --- | ---: | ---: | ---: |
| Lorenz96 | Memory | +0.00007979075113932292 | 2/3 | 1.0 |
| Lorenz96 | Retrieval | +0.00007979075113932292 | 2/3 | 1.0 |
| Lorenz96 | Salience gating | +0.04092961549758911 | 1/3 | 1.0 |
| Delayed recall | Memory | −0.00008779764175415039 | 1/3 | 1.0 |
| Delayed recall | Retrieval | −0.00008779764175415039 | 1/3 | 1.0 |
| Delayed recall | Salience gating | −0.0001060167948404948 | 0/3, with one tie | 0.5 |

Mean directions split three comparisons in favor of the full model and three
in favor of removals. The smallest observed exact sign-test p-value is 0.5.
The largest mean advantage of the full model, on Lorenz96 without salience
gating, is driven by seed 23; the other two seeds favor removal. The paired
figure shows this heterogeneity rather than presenting only the mean.

## Identifiability and limitations

The no-memory and no-retrieval rollout-MSE sequences coincide for every
retained seed in both benchmarks. Identical error sequences do not prove that
the predictions or implementations are identical, and they do not provide
distinct positive evidence for each component's necessity.

The shared-minibatch protocol is not a trajectory-isolated memory mechanism
test. Three training seeds per benchmark do not imply independent-domain
replication. Nonsignificant sign tests do not prove equivalence or universal
absence of benefit; no equivalence margin or compute-efficiency analysis is
established. The retained source identity is a declared source-tree hash with
Git commit recorded as `UNKNOWN`, so this work does not claim full executed-
source reconstruction or checkpoint replay.

## Contribution and reproducibility

The contribution is a complete, checksum-bound analysis of a closed negative
component study with every paired seed contrast retained. The code rejects
missing or duplicate cells and cross-checks the manifest's numbers against the
original cell reports before producing CSVs, figures and the candidate
abstract. Complete commands and byte identities are included in the package.

**Figure caption.** Dots show all retained removal-minus-full seed differences
and horizontal bars show their arithmetic means. The dashed reference is zero.
The two benchmark panels use independent vertical scales. The figure reports
historical outcomes, without new experimental execution or checkpoint replay.
