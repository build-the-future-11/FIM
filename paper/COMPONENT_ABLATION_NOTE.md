# FIM Component Ablation Note

Negative/mixed component study. Source: results/current_component_ablations/analysis.json

```json
{
  "schema_version": 1,
  "source_identity": "e8071b2243247361e3ea0da1eaa412e69f204593f80102fe5f837dead5300f46",
  "manifest": "results/current_component_ablations/manifest.json",
  "inference_boundary": "Three paired training seeds per benchmark. Seed variation is reported descriptively; it is not independent-domain evidence, and the exact sign test has minimum two-sided p=0.25.",
  "summaries": [
    {
      "benchmark": "delayed_recall",
      "variant": "full",
      "n_seeds": 3,
      "rollout_mse_mean": 0.8045086661974589,
      "rollout_mse_std": 0.01748645379428087,
      "rollout_mse_min": 0.7846519947052002,
      "rollout_mse_max": 0.8176088333129883
    },
    {
      "benchmark": "delayed_recall",
      "variant": "no_memory",
      "n_seeds": 3,
      "rollout_mse_mean": 0.8044208685557047,
      "rollout_mse_std": 0.017389733169115434,
      "rollout_mse_min": 0.7846524715423584,
      "rollout_mse_max": 0.8173564076423645
    },
    {
      "benchmark": "delayed_recall",
      "variant": "no_retrieval",
      "n_seeds": 3,
      "rollout_mse_mean": 0.8044208685557047,
      "rollout_mse_std": 0.017389733169115434,
      "rollout_mse_min": 0.7846524715423584,
      "rollout_mse_max": 0.8173564076423645
    },
    {
      "benchmark": "delayed_recall",
      "variant": "no_salience_gating",
      "n_seeds": 3,
      "rollout_mse_mean": 0.8044026494026184,
      "rollout_mse_std": 0.01740272104076419,
      "rollout_mse_min": 0.7846149206161499,
      "rollout_mse_max": 0.8173278570175171
    },
    {
      "benchmark": "lorenz96",
      "variant": "full",
      "n_seeds": 3,
      "rollout_mse_mean": 0.5689265429973602,
      "rollout_mse_std": 0.16800350481858484,
      "rollout_mse_min": 0.3750159442424774,
      "rollout_mse_max": 0.6707999110221863
    },
    {
      "benchmark": "lorenz96",
      "variant": "no_memory",
      "n_seeds": 3,
      "rollout_mse_mean": 0.5690063337484995,
      "rollout_mse_std": 0.16730578459268886,
      "rollout_mse_min": 0.3780229389667511,
      "rollout_mse_max": 0.6897018551826477
    },
    {
      "benchmark": "lorenz96",
      "variant": "no_retrieval",
      "n_seeds": 3,
      "rollout_mse_mean": 0.5690063337484995,
      "rollout_mse_std": 0.16730578459268886,
      "rollout_mse_min": 0.3780229389667511,
      "rollout_mse_max": 0.6897018551826477
    },
    {
      "benchmark": "lorenz96",
      "variant": "no_salience_gating",
      "n_seeds": 3,
      "rollout_mse_mean": 0.6098561584949493,
      "rollout_mse_std": 0.22774700679652723,
      "rollout_mse_min": 0.37229862809181213,
      "rollout_mse_max": 0.8263258337974548
    }
  ],
  "paired": [
    {
      "benchmark": "delayed_recall",
      "comparison": "no_memory_minus_full",
      "variant": "no_memory",
      "n_pairs": 3,
      "mean_delta_rollout_mse": -8.779764175415039e-05,
      "std_delta_rollout_mse": 0.00014269659440194264,
      "min_delta_rollout_mse": -0.0002524256706237793,
      "max_delta_rollout_mse": 4.76837158203125e-07,
      "seeds_variant_better": 2,
      "seeds_full_better": 1,
      "exact_sign_pvalue": 1.0,
      "seed_deltas": [
        {
          "seed": 11,
          "delta": -1.1444091796875e-05
        },
        {
          "seed": 23,
          "delta": -0.0002524256706237793
        },
        {
          "seed": 37,
          "delta": 4.76837158203125e-07
        }
      ],
      "interpretation": "negative favors the reduced variant; positive favors full"
    },
    {
      "benchmark": "delayed_recall",
      "comparison": "no_retrieval_minus_full",
      "variant": "no_retrieval",
      "n_pairs": 3,
      "mean_delta_rollout_mse": -8.779764175415039e-05,
      "std_delta_rollout_mse": 0.00014269659440194264,
      "min_delta_rollout_mse": -0.0002524256706237793,
      "max_delta_rollout_mse": 4.76837158203125e-07,
      "seeds_variant_better": 2,
      "seeds_full_better": 1,
      "exact_sign_pvalue": 1.0,
      "seed_deltas": [
        {
          "seed": 11,
  
```
