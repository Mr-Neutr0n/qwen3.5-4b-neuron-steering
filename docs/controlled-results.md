# Controlled neuron steering study

The recipe increased harmful endorsement on these held-out families. The paired model-rating difference is 96.9 percentage points [95% family bootstrap interval 90.6, 100.0].

96 prompts, 24 families, 2016 condition outputs. Two same-model condition-blinded reviewers, with separate multi-item packet histories, each scored 800 outputs. These are model ratings, not human validation.

Original preregistered selection: 384 outputs per rater. Supplementary selection: 416 additional outputs declared after generation began, before aggregate review unblinding. The added ratings are secondary descriptive outcomes; original primary endpoints remain unchanged. [Secondary review plan](../study/secondary-review.json), [deviations](../study/deviations.json).

| Paired harmful effect | Mean [95% family bootstrap interval] |
| --- | --- |
| Harmful endorsement | 96.9% [90.6%, 100.0%] |
| Actionable harmful assistance | 64.1% [39.1%, 85.9%] |
| Coherence | -0.844 [-0.969, -0.688] |
| Relevance | -0.578 [-0.828, -0.297] |

| Control type | Recipe minus mean controls [95% CI] | Rank |
| --- | --- | --- |
| neuron | 66.7% [58.3%, 75.0%] | 1 |
| residual | 88.0% [82.8%, 93.2%] | 1 |

Secondary descriptive dose comparison. Harmful and capability outcomes use separate 32-prompt sets.

| Condition | Harmful n | Endorsement [95% CI] | Coherence [95% CI] | Capability n | Strict | First answer | Format |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 32 | 3.1% [0.0%, 9.4%] | 2.000 [2.000, 2.000] | 32 | 56.3% | 56.3% | 93.8% |
| recipe_0.125 | 32 | 3.1% [0.0%, 9.4%] | 1.953 [1.891, 2.000] | 32 | 53.1% | 53.1% | 90.6% |
| recipe_0.25 | 32 | 46.9% [21.9%, 68.8%] | 1.781 [1.672, 1.906] | 32 | 40.6% | 40.6% | 84.4% |
| recipe_0.5 | 32 | 100.0% [100.0%, 100.0%] | 1.531 [1.406, 1.656] | 32 | 18.8% | 18.8% | 71.9% |
| recipe_1 | 32 | 100.0% [100.0%, 100.0%] | 1.156 [1.031, 1.312] | 32 | 0.0% | 0.0% | 18.8% |

| Capability condition | n | Strict | First answer | Format | Capped |
| --- | --- | --- | --- | --- | --- |
| baseline | 32 | 56.3% | 56.3% | 93.8% | 3.1% |
| noop | 32 | 56.3% | 56.3% | 93.8% | 0.0% |
| recipe_0.125 | 32 | 53.1% | 53.1% | 90.6% | 3.1% |
| recipe_0.25 | 32 | 40.6% | 40.6% | 84.4% | 0.0% |
| recipe_0.5 | 32 | 18.8% | 18.8% | 71.9% | 0.0% |
| recipe_1 | 32 | 0.0% | 0.0% | 18.8% | 21.9% |
| L14_only | 32 | 0.0% | 9.4% | 0.0% | 93.8% |
| L13_only | 32 | 15.6% | 15.6% | 37.5% | 0.0% |
| reversed | 32 | 0.0% | 0.0% | 6.3% | 96.9% |
| prefill_only | 32 | 9.4% | 9.4% | 25.0% | 3.1% |
| decode_only | 32 | 37.5% | 40.6% | 71.9% | 21.9% |
| ablation | 32 | 56.3% | 56.3% | 93.8% | 0.0% |
| ablation_prefill | 32 | 56.3% | 56.3% | 93.8% | 3.1% |
| baseline_rescue | 32 | 56.3% | 56.3% | 93.8% | 0.0% |
| engineering_rescue | 32 | 37.5% | 40.6% | 71.9% | 21.9% |
| random_neuron_00 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_neuron_01 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_neuron_02 | 8 | 0.0% | 0.0% | 0.0% | 62.5% |
| random_neuron_03 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_neuron_04 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_neuron_05 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_neuron_06 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_neuron_07 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_neuron_08 | 8 | 0.0% | 12.5% | 0.0% | 100.0% |
| random_neuron_09 | 8 | 0.0% | 0.0% | 12.5% | 12.5% |
| random_neuron_10 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_neuron_11 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_residual_00 | 8 | 0.0% | 0.0% | 0.0% | 87.5% |
| random_residual_01 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_residual_02 | 8 | 0.0% | 0.0% | 0.0% | 100.0% |
| random_residual_03 | 8 | 0.0% | 0.0% | 12.5% | 87.5% |
| random_residual_04 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_residual_05 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_residual_06 | 8 | 0.0% | 0.0% | 12.5% | 87.5% |
| random_residual_07 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_residual_08 | 8 | 0.0% | 0.0% | 12.5% | 37.5% |
| random_residual_09 | 8 | 0.0% | 12.5% | 0.0% | 100.0% |
| random_residual_10 | 8 | 0.0% | 0.0% | 12.5% | 100.0% |
| random_residual_11 | 8 | 0.0% | 0.0% | 0.0% | 87.5% |

Resampling uses 8 harmful-family clusters; these authored families are not a random population sample. The 95% family bootstrap intervals do not cover rater uncertainty, correlated same-model bias or hardware variability. 124 outputs have raw reviewer disagreements, all retained in the HTML report.

First-token KL uses the same-batch baseline at a fixed prompt. Prefill replacement and baseline rescue are engineering checks, not proof of natural mediation. Strict correctness, first-answer content and format are separate conservative parser checks. All capped outputs remain included.

[Full report](index.html), [earlier exploratory report](exploratory.html), [preregistration receipt](../study/registration.json), [registered commit](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering/commit/0b1729df5d121bacc15035c8a0be522e21ff65d9), [machine summary](../results/controlled-v1/machine-summary.json), [review summary](../results/controlled-v1/review-summary.json).
