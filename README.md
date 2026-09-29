# Certified Bounded Query Release

Reproducible experiments for finite, unbiased differential-privacy mechanisms on small categorical histograms. The implementation jointly optimizes a public linear workload, repairs rounded probabilities with a rational anchor, and checks privacy, moments, and upper/lower risk bounds using exact arithmetic.

## Quick start

Use Python 3.12 and a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --workers 8
```

The full run recomputes all 59 mechanisms, analytical comparisons, public-data cohort results, tuning checks, exact certificates, and seeded sampling checks. Eight workers completed mechanism construction in approximately 155 seconds on a 12-core Apple M3 Pro with 18 GB RAM. Runtime depends on the machine and solver build. Individual linear solves use one CPU thread; independent designs run concurrently. The workload uses sparse optimization and arbitrary-precision integers.

To validate the included results and perform the sampling check:

```sh
python reproduce.py --verify-only
```

For separate stages:

```sh
python run_experiments.py --workers 8 --force
python evaluate.py
python robustness.py
python validate.py
python -m unittest discover -s tests -v
```

Without `--force`, `run_experiments.py` reuses completed designs with certified bounds. The full reproduction command explicitly recomputes them. It updates files in `results/`.

## Experiment design

`config/main.json` defines every public input and design identifier:

| Family | Cases | Parameters |
|---|---:|---|
| Main | 30 | Three categories; n = 4, 6, 8; epsilon = 0.25, 0.5, 1, 2, 4; balanced and weighted workloads |
| Graph ablation | 10 | n = 8; both workloads; coordinate-box adjacency |
| Dimensional extension | 9 | Four categories; n = 2, 3, 4; epsilon = 0.5, 1, 2 |
| Tail expansion | 9 | n = 8; balanced workload; expansion factors 1.25, 1.5, 2 |
| Larger cohort | 1 | Three categories; n = 10; epsilon = 1 |

The public state is the first r counts of an (r+1)-category histogram with known total n. One protected change replaces one category in a fixed public record slot. The true replacement graph has coordinate changes `±e_j` and `e_j-e_k`. The box ablation adds feasible simultaneous changes in both retained coordinates.

The balanced workload reports the three category counts and the sum of the first two. The weighted workload reports `3*x1`, `x2`, `x1+x2`, and `x1-x2`. The factor three makes first-coordinate error more expensive. The four-category extension reports the four counts and three successive prefixes. All risk comparisons use mean squared error per workload row.

The finite output grid contains `-b, 0, 1, ..., n, n+b` in each coordinate. Outputs preserve affine count identities and unbiasedness, with an expanded range that can contain negative counts. The optimization uses a public grid and worst-state loss before observing a confidential histogram.

## Included results

For each design, `results/<id>.json` records dimensions, timings, risk, repair size, and exact-bound summaries. `results/<id>.npz` contains states, outputs, probabilities, workloads, and solver multipliers. `results/<id>.certificate.json.gz` contains integer probability numerators, their common denominator, rational outputs, the complete graph, and exact risk fractions.

The main summary files are:

- `comparisons.csv`: worst-state and uniform-state risks for each baseline.
- `public_cohorts.csv`: exact expected risks averaged over public-data cohorts.
- `workload_transfer.csv`: evaluation of each channel on both workloads.
- `cpm_refinement.json`: a 65-point scalar-budget search within the joint CPM family.
- `exact_verification.json`: verification of all privacy certificates and risk intervals.
- `sampler_validation.json`: 900,000 seeded draws compared with exact conditional risks.
- `environment.json` and `run_summary.json`: software, platform, configuration hash, and timing information.

The recorded maximum relative optimality gap is below 0.01%. This interval bounds the optimum on the specified grid, under the stated workload and privacy graph. Solver tolerances do not determine certificate acceptance: integer checks do. Floating-point optimization can choose different near-optimal channels on another solver build; the included exact certificates and rational risk bounds identify the archived results precisely. Execution timings vary between runs.

## Source organization

- `src/bounded_queries.py`: domains, adjacency, workloads, sparse optimization, and analytical risk.
- `src/anchor.py`: a strictly positive rational unbiased anchor.
- `src/certify.py`: exact moment completion, privacy repair, exhaustive checking, and integer sampling.
- `src/dual_certificate.py`: rational upper and lower risk bounds.
- `src/baselines.py`: bounded geometric, CPM, projected noise, matrix-Laplace, and continuous geometric comparators.
- `evaluate.py`: lattice comparator and public-cohort evaluation.
- `robustness.py`: workload transfer and CPM tuning refinement.
- `validate.py`: exact certificate recomputation and seeded sampling.
- `tests/`: checks of adjacency, scalar moments, lattice-shell identities, anchors, and certificate rejection.

## Data and randomness

`data/adult.zip` contains the UCI Adult dataset, credited to Barry Becker and Ronny Kohavi (1996), DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20). Its three education-number categories are at most 9, 10–12, and at least 13. The 48,842 records yield 6,105 complete cohorts of eight.

`data/bank_marketing.zip` contains UCI Bank Marketing, credited to Sérgio Moro, Paulo Rita, and Paulo Cortez (2012), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306). The experiment uses `bank-full.csv` and the categories married, single, and divorced. Its 45,211 records yield 5,651 complete cohorts of eight.

A data-independent permutation uses NumPy seed 20260928. Incomplete final cohorts are omitted. Dataset download URLs and SHA-256 hashes appear in `data/sources.json`. Both datasets are distributed by UCI under CC BY 4.0; attribution and license links appear in `THIRD_PARTY_NOTICES.md`.

The sampler validation uses fixed seeds for repeatable experiments. `ExactSampler` defaults to the operating-system random source when no seed is supplied. Its certified output is a finite index, decoded using rational output coordinates. The privacy model concerns the output distribution; timing behavior and surrounding application software are separate implementation properties.

## Baseline provenance

The CPM implementation follows the equations and monotone region-density constraints in A. Yamamoto and T. Shibuya, “Central Piecewise Mechanism for Numeric Queries,” 2026, DOI [10.1109/TP.2026.3669533](https://doi.org/10.1109/TP.2026.3669533). Joint-family risk is optimized directly, including cross moments. The 17- and 65-budget grids retain the published default scalar range rule.

The continuous staircase comparator follows A. Kulesza, A. T. Suresh, and Y. Wang, “General Staircase Mechanisms for Optimal Differential Privacy,” AISTATS 2025, [PMLR 258:4564–4572](https://proceedings.mlr.press/v258/kulesza25a.html). The lattice comparator uses the exact hexagonal sensitivity norm and analytical shell sums, checked independently by enumeration.

## License and citation

The experiment code is licensed under MIT; see `LICENSE`. Dataset licensing is described in `THIRD_PARTY_NOTICES.md`. `CITATION.cff` provides the project title, version, and author identifiers.
