# Credit Data Fairness

This repository contains our fairness and interpretability study on the UCI
German Credit dataset. The main work is in:

- `notebooks/fairness_notebook_Lina.ipynb`

## Goal

The objective is to predict credit risk while checking whether the model behaves
differently across sensitive groups. In the dataset:

- `0` means good credit
- `1` means bad credit

Since credit approval is the favorable outcome, the notebook also checks
fairness from the point of view of being predicted as good credit.

## Content

The notebook is organized as a small audit:

- data loading, encoding, train/test split and exploratory plots;
- Logistic Regression and Random Forest baselines;
- cost-sensitive evaluation;
- fairness metrics for age and gender/personal-status groups;
- mitigation with reweighing and Fairlearn;
- comparison of performance and fairness tradeoffs;
- interpretability with feature importance, permutation importance, partial
  dependence and a local LIME-style explanation;
- a check of sensitive attributes and possible proxies;
- a comparison with a model trained without direct age/gender features;
- final limitations on privacy and responsible AI.

## Running The Notebook

Install the environment:

```bash
uv sync
```

Then run:

```text
notebooks/fairness_notebook_Lina.ipynb
```

The dataset is loaded with `ucimlrepo`, so the first run may require internet
access.

## Main Takeaway

The Random Forest with combined reweighing gives the best compromise in our
experiments. It keeps good predictive performance and reduces several fairness
gaps compared with the baseline.

The result is still not perfect. Some sensitive attributes or proxies can remain
important, and the interpretability plots should be read as model diagnostics,
not as causal explanations.

## Limits

This is an academic project, not a deployable credit-scoring system. Differential
privacy is discussed but not implemented. A real system would need stronger
governance, privacy guarantees, legal checks and monitoring over time.
