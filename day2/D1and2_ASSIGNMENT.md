# Days 1–2 — Take-Home Assignment

## From Messy Raw Data to a Defensible Tabular Model

**Course:** Machine Learning in Industry · Cardo AI / MSCA Digital Doctoral Network

**Modules:** Day 1 — Data Preprocessing & Feature Engineering · Day 2 — Tabular Data Modeling, Evaluation & Diagnostics

**Effort:** ~1 week (budget 15–20 hours)

**Submission:** see [Deliverables & Submission](#5-deliverables--submission)

---

## 1. Scenario

A raw extract has landed on your desk: `day1/generated/adult_income_issues.csv` — an
income-classification dataset (a proxy for an affordability / income-verification
signal in a lending pipeline). It is **deliberately messy**: duplicate records,
duplicate entities, mixed types, row-wise column misalignment, impossible values,
missingness of several kinds, leakage columns, KYC/process artifacts, and constant
columns. Your task: turn this raw file into a **trustworthy, reproducible model**, and
**defend every decision with evidence**.

The raw target column is `class` (`>50K` vs `<=50K`); your pipeline should derive a
binary `target` (1 = `>50K`) from it. About three in ten rows are positive
(2,796 / 9,271 ≈ 30.2%) — i.e. the classes are **imbalanced**, which has consequences
for how you split, evaluate, and threshold.

This is **not** a "re-run the lesson notebooks" exercise. The Day 1 and Day 2 notebooks
*demonstrate techniques* on this and related datasets — they are your reference. The
assignment grades something the notebooks do **not** contain: **independent judgment,
quantified trade-offs, and diagnostics.** You may consult the lesson notebooks freely,
but for every preprocessing and modeling decision you must (a) justify *why*, and
(b) where the task asks, **measure its effect on held-out performance**. Copying a
cleaning step without justification or measurement earns no points.

> The course's spine question: *"We trained a model" is not an acceptable answer.*
> By the end you should be able to defend your pipeline to a skeptical reviewer with
> numbers, not adjectives.

---

## 2. Ground rules

- **Reproducibility is graded.** Fix `SEED = 42` everywhere, set every estimator's
  `random_state`, pin your environment, and make every number in your report
  re-creatable from a command. Two runs of your final pipeline must give identical metrics.
- **The held-out test set is sacred.** The file has a `split` column (`train` / `test`).
  Treat `split == "test"` as a locked vault: **no fitting, no imputation statistics, no
  encoding tables, no threshold selection, no feature decisions** may use it. You touch
  it **once**, at the very end, to report your final number. Carve any validation you
  need out of the `train` split.
- **No data leakage — two kinds, both graded:**
  - *Entity leakage:* `person_id` repeats across rows. Splits and cross-validation must
    be **entity-aware** (no person in two folds). Use grouped splitting.
  - *Target leakage:* some columns encode the outcome or post-decision information
    (e.g. `post_adjudication_risk_code`). Identify and handle them — and in Task B6 you
    will **quantify** how badly leaving one in inflates your metrics.
- **Fit on train, apply to test.** Every transformer (imputer, encoder, scaler, target
  encoder) is *fit on training data only* and *applied* to validation/test. Prefer a
  single `sklearn` `Pipeline` / `ColumnTransformer` so this is structurally guaranteed.
- **LLM / AI tools** are allowed for scaffolding and debugging, but every number, plot,
  and claim must reproduce from your committed code, and you must be able to defend each
  decision orally. State in your report where you used AI assistance.
- Work on a branch in your fork. Do not edit the original lesson notebooks; put your work
  under `day2/assignment/` (and/or a new notebook).

---

## 3. What you are handed (your toolbox)

| Reference | What it gives you |
|---|---|
| [day1/first_eda.ipynb](../day1/first_eda.ipynb) | A catalog of 19 data-quality issues and how to *spot* them |
| [day1/datapreprocessing_featureengineering.ipynb](../day1/datapreprocessing_featureengineering.ipynb) | Worked encoding, imputation, outliers, scaling, FE, selection, end-to-end `ColumnTransformer` |
| [day1/DIY-...ipynb](../day1/) | Self-study exercises on encoding pitfalls, imputation, missingness-as-signal, scaling, PCA traps |
| [day2/modeling.ipynb](modeling.ipynb) | Splitting strategy, metrics (ROC-AUC / PR-AUC / threshold / the accuracy trap), the 4-family model zoo, overfitting, tuning, diagnostics |
| [day2/advanced_modeling.ipynb](advanced_modeling.ipynb) | CV mathematics & stratification, AUPRC under changing prevalence, covariate shift, **adversarial validation**, Optuna, **SHAP**, **contrast model / error slicing**, stacking |
| `day1/generated/adult_income_issues.csv` | Your raw dataset (9,271 rows × 33 cols; `split`, `person_id`, raw `class`; you create `target`) |

You may reuse helper code from the notebooks, but the pipeline you submit must be yours
and must run end-to-end.

---

## 4. Tasks

Two halves mirroring the two days, plus a set of cross-cutting experiments that carry
most of the marks. Point weights guide effort and grading.

### Part A — Data: audit, clean, engineer (Day 1) — 35 pts

**A1. Data-quality audit (8 pts).** Profile the raw file and produce a **decision log**:
a table listing every data-quality issue you find (type, affected columns, evidence —
a count or example), the action you will take, and a one-line justification. Cover at
least: exact duplicates, duplicate `person_id`s, mixed/incorrect dtypes, impossible
values, row-wise column misalignment, inconsistent category labels, constant columns,
and ID/process/leakage columns. Aim for breadth and correctness, not prose.

**A2. Splitting strategy (5 pts).** Define your splits. Respect the dataset's
`split` column for the final test set; carve a validation scheme out of `train` that is
**entity-aware** (grouped by `person_id`). State explicitly why a naive row-level random
split would leak here, and how big the leak is (A-Cross experiment below quantifies it).

**A3. Encoding (6 pts).** Choose and justify an encoding strategy per categorical
feature (one-hot, ordinal, frequency, or out-of-fold target encoding). Address
high-cardinality columns and **unseen categories at test time**. Justify *why* per column
— e.g. when frequency encoding causes collisions, when target encoding needs OOF to
avoid leakage.

**A4. Missing values (6 pts).** Diagnose the *mechanism* (MCAR/MAR/MNAR is a useful
lens) before choosing a remedy. Decide per column among drop / constant fill / median-mode
/ group-wise / model-based imputation, **plus missing-indicator flags where missingness
may itself be predictive.** Justify, and fit all imputation statistics on train only.

**A5. Outliers, scaling & feature engineering (10 pts).** Apply rule-based validation
and an outlier policy (clip / cap / flag / leave — your choice, justified). Apply numeric
transforms/scaling where the *model family* warrants it (note: tree models don't need
scaling — say so). Engineer **at least 4 new features** with a stated hypothesis for each
(interactions, ratios, group aggregates fit on train, binning, or datetime-derived from
`record_written_at`). Assemble everything into a single fit-on-train `Pipeline` /
`ColumnTransformer`.

**Deliverable:** the decision log (A1) + a reproducible preprocessing pipeline that
turns the raw CSV into a model-ready matrix, with no test-set leakage.

---

### Part B — Modeling, evaluation & diagnostics (Day 2) — 40 pts

**B1. Baselines first (4 pts).** Report a majority-class baseline and a simple logistic
regression. Every later model is judged against these.

**B2. Model zoo (8 pts).** Train **≥4 model families**: logistic regression, a single
decision tree, a random forest, and a gradient-boosting model (sklearn GBM or LightGBM).
Use your Part A pipeline. Compare them with **entity-aware cross-validation** (grouped
by `person_id`), reporting **mean ± std** per metric — not a single split.

**B3. Metric selection under imbalance (6 pts).** At this prevalence, justify which
metric you optimize. Demonstrate, with your own numbers, **the accuracy trap** and the
difference between **ROC-AUC and PR-AUC** (Day 2 Advanced §2). Pick a primary metric and
defend the choice as a *business* decision (what does a false positive vs false negative
cost a lender?).

**B4. Threshold from a cost matrix (5 pts).** Your model outputs probabilities; a
decision needs a threshold. Define a cost matrix (state your assumed cost ratio), then
select the operating threshold that minimizes expected cost on validation — **not** the
default 0.5. Report the confusion matrix at your chosen threshold.

**B5. Tuning & overfitting (6 pts).** Tune your best family's hyperparameters with
`GridSearchCV` or **Optuna** (Day 2 Advanced §6), inside the entity-aware CV (no test
leakage). Show an overfitting curve (e.g. `n_estimators` or tree depth vs train/val gap)
and state where you stop and why.

**B6. Diagnostics & explanation (6 pts).** For your final model: permutation importance
**and** SHAP (Day 2 Advanced §7); comment on agreement/disagreement. Then **error
slicing / contrast model** (Day 2 Advanced §8): find at least one **subgroup where the
model underperforms** (slice by `workclass`, `occupation`, `native_country`, or
`marital_status`) and discuss the operational implication.

**B7. Final honest evaluation (5 pts).** Refit your chosen pipeline on all `train`,
evaluate **once** on `split == "test"`, and report your primary metric **with a 95%
confidence interval** (bootstrap the test rows). Compare to your CV estimate: did the
held-out number land inside the CV mean ± std? Discuss any gap.

**Deliverable:** the modeling notebook/script, the CV comparison table, tuning evidence,
diagnostics plots, and the single final test number with its CI.

---

### Cross-cutting experiments (carry the rigor) — 20 pts

These are the questions the lesson notebooks do **not** answer for you. Each wants a
**number with uncertainty**, not a sentence.

**X1. Quantify the two leakage traps (8 pts).**
- *Target leakage:* train your model **with** `post_adjudication_risk_code` included,
  then **without** it. Report the held-out (CV) AUC/PR-AUC difference. Explain why the
  inflated number is a lie.
- *Entity leakage:* evaluate your model under a **naive row-level random split** vs your
  **entity-aware grouped split**. Report the optimism gap (how many AUC points the naive
  split buys you for free) and explain the mechanism.

**X2. Preprocessing ablation (7 pts).** Pick **≥3** of your Part A decisions (e.g.
target vs frequency encoding; median vs group-wise imputation; with vs without missing
indicators; with vs without your engineered features). For each, hold everything else
fixed and measure the change in held-out PR-AUC under entity-aware CV, reporting
**mean ± std**. Which decisions actually mattered? Which were cosmetic? Be honest when a
choice you liked made no measurable difference.

**X3. Is the test set drawn from the same distribution? (5 pts).** Run **adversarial
validation** (Day 2 Advanced §4): train a classifier to distinguish `train` from `test`
rows. If its AUC ≫ 0.5, the splits differ — report which features drive the distinction
and discuss what that implies for trusting your B7 number.

---

### Part C — Bonus (up to +10 pts)

Pick **one**:
- **Stacking/blending** (Day 2 Advanced §9): build an out-of-fold stacked ensemble of
  your base models and show whether it beats the best single model — with CV uncertainty,
  not a lucky single split.
- **Transfer challenge:** apply your *exact* preprocessing+modeling pipeline to a second
  messy dataset (`day1/generated/ames_housing_issues.csv`, regression, or
  `day1/generated/retail_panel_issues.csv`, time-aware). Report what broke and what you
  had to change — a test of how general your pipeline really is.
- **Calibration:** assess calibration (reliability diagram + Brier) of your final model,
  apply a calibrator (Platt / isotonic) fit on validation, and show the effect on both
  calibration and your cost-based threshold from B4.

---

## 5. Deliverables & Submission

Submit a link to your fork (branch) containing:

1. **`day2/assignment/REPORT.md`** (or PDF) — the narrative tying it together: the
   decision log (A1), the CV model-comparison table, the metric/threshold justification,
   diagnostics, the three cross-cutting experiment results (X1–X3) with numbers + CIs,
   and your single final test result. End with a short **"What I would not trust about
   this model"** paragraph — the limitations you can defend.
2. **Code** — a runnable notebook *or* scripts that reproduce every number and figure,
   seeded, fit-on-train-only.
3. **`HOW_TO_REPRODUCE.md`** — the exact commands, in order, to regenerate your results
   from a clean checkout, plus your environment spec (e.g. `requirements.txt` / `uv.lock`
   / `environment.yml`).

Keep prose tight. A reviewer should be able to follow your reasoning and re-run your
results without talking to you.

---

## 6. Grading rubric

| Part | Pts | What earns the marks |
|---|---|---|
| A — Data | 35 | Thorough, correct audit; justified per-column decisions; leakage-safe, reproducible pipeline; ≥4 hypothesis-driven features |
| B — Modeling | 40 | Entity-aware CV with mean±std; correct imbalance-aware metric choice; cost-based threshold; honest tuning/overfitting analysis; SHAP+permutation; a real underperforming slice; one clean final test number with CI |
| X — Cross-cutting | 20 | Both leakage traps quantified; ≥3 ablations with uncertainty and honest verdicts; adversarial validation interpreted |
| C — Bonus | +10 | Correct method, evaluated with CV uncertainty, clearly written |
| Report quality | (folded in) | Clarity, intellectual honesty, defensible limitations |

**Penalties:** any test-set leakage (fitting/selecting on `test`); non-entity-aware
splits or CV; unfixed seeds / non-reproducible numbers; claims without supporting
numbers; reporting a single split where the task asked for CV mean ± std.

---

## 7. Hints & pitfalls

- **Build the pipeline first, then experiment.** If preprocessing lives in one
  `ColumnTransformer`, swapping an encoder for an ablation (X2) is a one-line change and
  leakage is structurally prevented. Ad-hoc cleaning in loose cells makes X1–X2 painful.
- **Grouped CV is not optional here.** `sklearn`'s `GroupKFold` / `StratifiedGroupKFold`
  with `groups=person_id` is your friend. A `StratifiedKFold` that ignores `person_id`
  is the entity-leakage bug you're being asked to *measure* in X1.
- **ROC-AUC can look healthy while PR-AUC is poor** at ~30% prevalence. Report both;
  optimize the one that matches the cost of the decision.
- **Target encoding leaks if naive.** Use out-of-fold target encoding (see the lesson's
  `oof_target_encode`) or fit it inside CV folds — never on the whole dataset.
- **A non-result is a result.** If an ablation (X2) shows your favorite feature did
  nothing, say so with the number. Honesty about null effects is part of the grade.
- **Touch `test` once.** If you find yourself re-evaluating on `test` to "improve," you
  have already contaminated it — go back to validation.
