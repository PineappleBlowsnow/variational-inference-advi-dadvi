# ADVI and DADVI: Bayesian inference experiments

Joint M1 Applied Mathematics / Statistics research project at Université Paris Dauphine–PSL (2025), by **Ying JIN and Bethlehem MESSELU**, supervised by **Joshua J. Bon**.

The accompanying report, *Testing New Variational Approximations for Bayesian Inference* (28 April 2025), studies how stochastic and fixed-sample variational objectives behave on small models with interpretable or analytically available posteriors. This repository contains the project notebooks and aggregate trial data; the report and defence slides are not included here. The joint authorship applies to the project as a whole; individual notebook ownership is not separately documented.

## Research questions

- How does redrawing Monte Carlo samples during optimization compare with optimizing a fixed sample-average objective?
- How accurately do the fitted Gaussian approximations recover posterior means and variances in conjugate examples?
- What happens when a unimodal variational family approximates a multimodal posterior?
- How do parameter transformations and likelihood aggregation affect a two-probability binomial example?

The experiments explore ideas from [ADVI](https://jmlr.org/papers/v18/16-107.html) and [DADVI](https://jmlr.org/papers/v25/23-1015.html). They are educational implementations, not complete reproductions of those papers. Some notebooks estimate gradients with NumPy/SciPy finite differences; the meta-analysis notebook also uses PyTorch automatic differentiation. The experiments do not establish a general accuracy or runtime advantage for either method.

## Repository guide

| File | Contents |
|---|---|
| [Mémoire 3.1..ipynb](M%C3%A9moire%203.1..ipynb) | Conjugate Gaussian examples, analytic posterior comparisons, and shared or independent scale parameterizations. |
| [Multimodel_seperate bumps.ipynb](Multimodel_seperate%20bumps.ipynb) | Gaussian-mixture example with separated modes. The original filename spelling is retained. |
| [Multimodel_close bumps.ipynb](Multimodel_close%20bumps.ipynb) | Mixture example with closer modes. |
| [Multimodel_ overlapping bumps.ipynb](Multimodel_%20overlapping%20bumps.ipynb) | Mixture example with overlapping modes. |
| [Meta analysis.ipynb](Meta%20analysis.ipynb) | Binomial observations from two trial arms, logit transformations, and Gaussian variational approximations. |
| [data.csv](data.csv) | 22 rows of aggregate trial counts: study index, control deaths/total, and treated deaths/total. |

In the report, “three-/four-dimensional variational family” refers to the number of optimization parameters: two means with one shared or two independent log scales. It should not be interpreted as evidence of large-scale or high-dimensional inference.

## Data and model

The accompanying report attributes the 22-trial beta-blocker example to [Yusuf et al. (1985)](https://pubmed.ncbi.nlm.nih.gov/2858114/). The CSV contains study-level counts, not patient-level records. Its original download source and any intervening preprocessing are not recorded in this repository; the exact rows have not been independently reconciled against the publication.

The implemented statistical setup uses one shared probability per arm across trials, with independent Beta(1,1) priors. It is a simple pooled binomial example, not a hierarchical random-effects meta-analysis. The exact Beta posteriors provide a reference for numerical checks. A Gaussian approximation is fitted in logit space, so its probability-space transform is logistic-normal.

## Opening and running the notebooks

Use Python 3 and a separate environment. The following is a starting environment based on the notebook imports, not a tested or pinned reconstruction of the original environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install numpy scipy pandas matplotlib torch jupyterlab
python -m jupyter lab
```

Start Jupyter in the repository root so that `pd.read_csv('data.csv')` resolves correctly. Open a notebook, restart its kernel, and inspect/run cells in order. Function names are reused between sections, making execution order relevant. Read the implementation notes below before using any printed objective values for a comparison.

There is no environment lockfile, automated test suite, or command-line experiment runner in this snapshot. The September 2026 documentation update checked notebook structure, imports, file references and selected objective definitions; it did **not** rerun the full experiments or verify stored outputs.

## Implementation notes and comparison limits

These findings were checked against repository snapshot [`e7ac211`](https://github.com/PineappleBlowsnow/Dauphine-M1-Memoire_ADVI-vs-DADVI/tree/e7ac2118af22566254fd643fb52c53edff0c7e3a) on 23 September 2026. Cell indices below are **zero-based**. This README leaves the original notebooks, outputs and data unchanged.

### Trial aggregation in the meta-analysis notebook

In [cell 5 of `Meta analysis.ipynb`](https://github.com/PineappleBlowsnow/Dauphine-M1-Memoire_ADVI-vs-DADVI/blob/e7ac2118af22566254fd643fb52c53edff0c7e3a/Meta%20analysis.ipynb), the NumPy `log_joint` sums the trial log-likelihoods and adds the transformed prior once. The Torch `log_joint_advi` instead returns a vector over trials. Its caller stacks these vectors and takes `torch.mean` across both Monte Carlo and trial dimensions.

Writing each trial's log-likelihood as `ell_j`, the transformed prior as `r`, and the number of trials as `J = 22`, the two objectives therefore use:

```text
NumPy: E_q[sum_j ell_j + r]       + entropy(q)
Torch: E_q[(sum_j ell_j) / J + r] + entropy(q)
```

This changes the relative likelihood, prior and entropy weights; it is not a constant rescaling of the complete objective. The objectives need to be aligned before attributing differences to ADVI versus DADVI. Omitting binomial coefficient constants in one implementation does not resolve this aggregation difference.

### Gaussian objective checks

In [`Mémoire 3.1..ipynb`](https://github.com/PineappleBlowsnow/Dauphine-M1-Memoire_ADVI-vs-DADVI/blob/e7ac2118af22566254fd643fb52c53edff0c7e3a/M%C3%A9moire%203.1..ipynb), cell 34 uses `sum(log_std)` for the entropy contribution. When a single log scale is shared across two latent dimensions, the corresponding contribution should be twice that log scale. The displayed ELBO calls in cells 38 and 40 also pass `sigma2, tau2` where the function signature expects `tau2, sigma2`; the optimizer call in cell 40 uses the expected order. These issues require controlled checks before treating printed ELBOs as benchmark results.

### Historical results and next validation steps

Stored notebook outputs are historical artifacts. The report describes 4,838 ADVI iterations in the trial example, while the current notebook specifies 200 iterations and 200 fixed DADVI samples. The exact report-producing execution state has not been reconstructed. The findings above should therefore not be used to retroactively explain every reported numerical difference.

A reliable follow-up comparison would first align objectives and transformations, check them against analytic posteriors and finite-difference gradients, use independent evaluation samples, and record seeds, objective/gradient calls and wall-clock time. Matching ADVI iterations to the number of fixed DADVI samples alone does not establish equal computational budgets. These checks and reruns are future work, not completed validation claimed by this README.
