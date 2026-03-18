---
marp: true
theme: default
paginate: true
style: |
  /* ── Cardo AI corporate template ────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600&display=swap');

  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 1.15rem;
    background: #ffffff;
    color: #1a1a1a;
    padding-bottom: 3em;
  }

  /* Footer: "CARDO AI  |  page" on every slide */
  section::after {
    content: 'CARDO AI  ·  ' attr(data-marpit-pagination);
    position: absolute;
    bottom: 0.7em;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #C5532A;
    border-top: 1.5px solid #C5532A;
    padding-top: 0.4em;
    margin: 0 2em;
  }

  /* Title slide: white background, orange headings, dark body text */
  section.title {
    background: #ffffff;
    color: #1a1a1a;
    text-align: left;
  }
  section.title h1 {
    font-family: 'Oswald', Arial, sans-serif;
    font-size: 2.6rem;
    color: #C5532A;
    border-bottom: 3px solid #C5532A;
    padding-bottom: 0.3em;
    margin-bottom: 0.2em;
  }
  section.title h2 {
    color: #444;
    font-size: 1.1rem;
    font-weight: normal;
    border: none;
  }
  section.title h3 {
    color: #666;
    font-size: 1rem;
    font-weight: normal;
  }
  section.title p { color: #1a1a1a; font-size: 1rem; }
  section.title strong { color: #1a1a1a; font-weight: 700; }

  /* Section header slides: orange background, white text */
  section.lead {
    background: #C5532A;
    color: #ffffff;
    text-align: left;
  }
  section.lead h1 {
    font-family: 'Oswald', Arial, sans-serif;
    font-size: 2.4rem;
    color: #ffffff;
    border-bottom: 2px solid rgba(255,255,255,0.5);
    padding-bottom: 0.3em;
  }
  section.lead h2 {
    color: rgba(255,255,255,0.85);
    font-size: 1.2rem;
    font-weight: normal;
  }
  section.lead h3 {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    font-weight: normal;
  }
  section.lead p { color: rgba(255,255,255,0.9); }
  section.lead strong { color: #ffffff; }
  section.lead blockquote {
    background: rgba(255,255,255,0.15);
    border-left-color: rgba(255,255,255,0.6);
    color: #ffffff;
    font-style: italic;
  }
  section.lead::after { color: rgba(255,255,255,0.7); border-top-color: rgba(255,255,255,0.4); }

  /* Content slides */
  h1 { color: #C5532A; font-family: 'Oswald', Arial, sans-serif; font-size: 1.7rem; }
  h2 { color: #1a1a1a; font-size: 1.2rem; border-bottom: 1.5px solid #C5532A; padding-bottom: 0.2em; }
  strong { color: #C5532A; }
  a { color: #C5532A; }
  code {
    background: #f4f4f4;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.9em;
    color: #1a1a1a;
  }
  pre code { background: #f8f8f8; padding: 0.8em; display: block; color: #1a1a1a; }
  table { font-size: 0.9rem; width: 100%; }
  th { background: #C5532A; color: #ffffff; }
  td { border-bottom: 1px solid #e0e0e0; }
  blockquote {
    border-left: 4px solid #C5532A;
    background: #fdf6f4;
    padding: 0.5em 1em;
    color: #555;
    font-style: italic;
  }

  /* Pill labels for tool names */
  .tool {
    background: #C5532A;
    color: #ffffff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.8em;
    font-weight: bold;
  }
---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 1 — TITLE
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: title -->

# Intro to MLOps
# Reproducibility & Model Monitoring

## Machine Learning in Industry
### Cardo AI · MSCA Digital Doctoral Network · Day 4

**Gennaro Di Brino** · Head of Data Science, Cardo AI
March 19, 2026

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 2 — AGENDA
     ══════════════════════════════════════════════════════════════════════════ -->

# Today's Roadmap

| Block | Time | Topic | Deliverable |
|---|---|---|---|
| **1** | 9:30–10:30 | Why MLOps? Reproducibility | Concepts + motivation |
| **2** | 10:30–11:00 | Experiment tracking with MLflow | `01_mlflow_tracking.ipynb` |
| *break* | 10:45 | | |
| **3** | 11:00–11:45 | Data drift + monitoring (Evidently) | `02_drift_monitoring.ipynb` |
| **4** | 11:45–12:30 | Docker + CI/CD overview | Pre-built walkthrough |
| | 13:30–15:00 | **Workshop** | Steps 1–3 (+ stretch: CI/CD) |
| | 15:00–16:30 | **Project work** | Day 5 prep |

> All code lives in `day4/`. Start from `make -f day4/Makefile help`.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BLOCK 1 — SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Block 1

# Why MLOps?

## Reproducibility as the Foundation

*9:30 – 10:30 · 60 minutes*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 4 — FAILURE STORY 1
     ══════════════════════════════════════════════════════════════════════════ -->

# Failure Story 1 — "It works on my machine"

**The situation**

A data scientist trains a credit scoring model on their laptop. AUC looks great. The model is deployed to the production server. No errors are thrown.

Two weeks later, a colleague notices the forecasted default rates for a certain loan pool are slightly off. Investigation reveals:

- Dev machine: scikit-learn **1.1** — `OneHotEncoder(sparse=False)` works as expected
- Prod server: scikit-learn **1.2** — the `sparse` parameter was **renamed to `sparse_output`**
- Passing the old `sparse=False` keyword is silently ignored; the encoder returns a sparse matrix instead of a dense one
- Downstream arithmetic on a sparse matrix produces no exception — just subtly wrong values

**Result:** Three weeks of wrong predictions. No error. No alert. No paper trail.

> The model worked. The *system* failed.

*"Only a small fraction of real-world ML systems is composed of the ML code. The required surrounding infrastructure is vast and complex."* — [Sculley et al. (2015), "Hidden Technical Debt in ML Systems" (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

<!-- notes: This story is generic/composite — don't attribute it to a specific company. The key teaching point: scikit-learn 1.2 deprecated `sparse` in OneHotEncoder and introduced `sparse_output`; passing the old keyword name raises no error, it is simply ignored. Silent keyword-mismatch failures are the hardest to catch without pinned dependencies. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 5 — FAILURE STORY 2
     ══════════════════════════════════════════════════════════════════════════ -->

# Failure Story 2 — Silent Model Decay

**The situation**

A consumer lending platform deploys a loan default classifier in early 2019. It performs well for 18 months.

In Q2 2020, employment patterns shift dramatically: furlough schemes, gig economy collapse, remote work. The distribution of `hours_per_week`, `occupation`, and `income` — three key model features — shifts substantially (**covariate shift**: P(X) changes). Simultaneously, the relationship between income and default risk changes (**concept drift**: P(Y|X) changes).

The model keeps returning predictions. They just become progressively less accurate.

**Nobody notices** until Non-Performing Loan (NPL) rates spike and the credit committee asks for a performance report.

**Result:** The model had been underperforming for ~6 months before anyone looked.

> Without monitoring, a deployed model is a black box. You are flying blind.

See: [Lipton, Wang & Smola (2018), "Detecting and Correcting for Label Shift with Black Box Predictors"](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf)

<!-- notes: This is the central motivation for Block 3. Students should feel the operational cost of late detection before we show them how to prevent it. The formal drift taxonomy (covariate shift, concept drift, prior probability shift) will be expanded in slide 26. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 6 — FAILURE STORY 3
     ══════════════════════════════════════════════════════════════════════════ -->

# Failure Story 3 — The Regulatory Audit

**The situation**

A bank uses an internal rating-based (IRB) model for credit risk capital calculation. The ECB supervisory team requests a review.

Questions asked:
- Which version of the model is currently in production?
- What training data was used? Which version?
- What were the validation metrics at deployment?
- Has the model been monitored since deployment? Evidence?
- What changed between model version 3 and version 4?

**The answer given:** *"We can reconstruct most of that from emails and shared drives."*

**Result:** Remediation plan required. Model placed under enhanced monitoring. Regulatory capital add-on applied.

> "We trained a model" is usually not an acceptable answer in the financial sector, or anywhere else.

**Key regulations:** US Fed SR 11-7 (Supervisory Guidance on Model Risk Management), ECB Guide to Internal Models, EBA GL/2017/16. All require documented model development, validation, and ongoing monitoring. MLflow's run logs and model registry directly address these requirements.

<!-- notes: EBA/ECB guidelines (especially the ECB guide to internal models) explicitly require documentation of model development, validation, and ongoing monitoring. SR 11-7 in the US is the equivalent. This slide connects MLOps to regulatory compliance — a language students going into finance need to speak. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 7 — THE COMMON THREAD
     ══════════════════════════════════════════════════════════════════════════ -->

# The Common Thread

All three failures share a root cause:

```
  ML Development          Production Reality
  ───────────────    ≠    ────────────────────
  Laptop / notebook        Versioned server
  "It ran correctly"       Undocumented environment
  Ad-hoc experiment log    Regulatory paper trail
  No monitoring plan       Silent performance decay
```

**MLOps closes this gap.**

It is not a set of tools. It is a set of *practices* — supported by tools — that make ML systems:
- **Reproducible** — any result can be recreated from a run ID
- **Auditable** — every decision is logged with its evidence
- **Observable** — model performance is measured continuously in production

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 8 — THE MLOPS LIFECYCLE
     ══════════════════════════════════════════════════════════════════════════ -->

# The MLOps Lifecycle

```
  Data ──────► Feature ──────► Model ──────► Model ──────► Model
  Collection   Engineering     Development   Deployment    Monitoring
       ▲                                                      │
       └───────────────── feedback loop ◄─────────────────────┘
```

*Source: [ml-ops.org/content/mlops-principles](https://ml-ops.org/content/mlops-principles)*

**Today we cover from Model Deployment onwards:**

| Stage | What we build today |
|---|---|
| Model Development | LightGBM pipeline (same as Day 2) |
| Experiment Tracking | MLflow — Block 2 |
| Model Registry | MLflow Registry — Block 2 |
| Monitoring | Evidently drift + PSI — Block 3 |
| Deployment | Docker + FastAPI — Block 4 |
| CI/CD | GitHub Actions — Block 4 |

---

<!-- ══════════════════════════════════════════════════════════════════════════
     REPRODUCIBILITY SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Reproducibility

# The Foundation

> *"If you can't reproduce it,*
> *you don't understand it."*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 10 — WHAT REPRODUCIBILITY MEANS
     ══════════════════════════════════════════════════════════════════════════ -->

# What Reproducibility Means

A result is reproducible if, given the same inputs, you always get the same output — on **any machine**, at **any time**.

In ML, that requires five things:

| # | Requirement | Common failure mode |
|---|---|---|
| 1 | **Fixed random seeds** | `train_test_split` returns different rows each run |
| 2 | **Pinned dependencies** | Library update silently changes behaviour |
| 3 | **Environment isolation** | "Works on my machine" |
| 4 | **Deterministic data splits** | "Same person" in train and test (leakage) |
| 5 | **Log everything** | Can't reconstruct which features a model used |
| 6 | **Cross-validation** | Single-split variance inflates or deflates metrics |

> None of this is glamorous. All of it is load-bearing. But reproducibility is at the heart of the scientific method — it's how we build trust in our findings. In MLOps, it's how we build trust in our models.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 11 — RANDOM SEEDS
     ══════════════════════════════════════════════════════════════════════════ -->

# Random Seeds

**The problem:**

```python
# Without random_state — different result every run
X_train, X_test = train_test_split(X, test_size=0.2)
model.fit(X_train, y_train)
# AUC: 0.8731 ... 0.8619 ... 0.8802  ← which one do you report?
```

**The fix:**

```python
SEED = 42  # defined once, used everywhere

X_train, X_test = train_test_split(X, test_size=0.2, random_state=SEED)

model = LGBMClassifier(n_estimators=300, random_state=SEED)
```

**Rules:** define `SEED` once at module level; pass `random_state=SEED` to every sklearn object; call `np.random.seed(SEED)` at script entry.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 12 — PINNED DEPENDENCIES
     ══════════════════════════════════════════════════════════════════════════ -->

# Pinned Dependencies

**Underspecified requirement (risky):**

```
pandas>=1.0
scikit-learn>=0.24
```

A CI job running 18 months later installs `pandas 3.x`. The `DataFrame.groupby()` sort order might have changed. Silent wrong results.

**Pinned requirement (safe):**

```
pandas==2.2.1
scikit-learn==1.5.2
```

**In this course:** `pyproject.toml` declares bounds (`>=`); `uv.lock` pins exact versions automatically. Install with `cd day4 && uv sync`.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 13 — ENVIRONMENT ISOLATION
     ══════════════════════════════════════════════════════════════════════════ -->

# Environment Isolation

**The problem:** your global Python environment has 200 packages. Upgrading one for project A breaks project B.

**Solutions (in order of isolation strength):**

| Tool | Scope | Command |
|---|---|---|
| `uv` | per-project venv + pinned deps | `uv venv && uv sync` |
| `conda` | env + non-Python deps | `conda create -n myenv python=3.12` |
| **Docker** | full OS + runtime | `docker build -t myapp .` |

**Docker = the nuclear option for isolation.** Same OS, same Python, same glibc, same everything. The environment is *defined in code* (`Dockerfile`) and version-controlled.

> Today: we use `uv` (with `pyproject.toml`) for the notebooks, Docker for the serving layer. In production, everything would be containerized.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 14 — DETERMINISTIC DATA SPLITS
     ══════════════════════════════════════════════════════════════════════════ -->

# Deterministic Data Splits

**The problem — data leakage from non-entity-aware splits:**

```
Person 42 appears in 3 rows (one application per year).
Random split puts 2 rows in train, 1 in test.
The model "memorises" Person 42.
Test AUC looks great. Production AUC is much worse.
```

**The fix — entity-aware splitting:**

```python
person_ids = df["person_id"].unique()
train_ids, val_ids = train_test_split(person_ids,
                                      test_size=0.2,
                                      random_state=SEED)
train_df = df[df["person_id"].isin(train_ids)]
val_df   = df[df["person_id"].isin(val_ids)]
# guaranteed: no person appears in both sets
```

> Revisited from Days 1–2. Now you see why it matters operationally, not just statistically.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 15 — LOG EVERYTHING
     ══════════════════════════════════════════════════════════════════════════ -->

# Log Everything

**Minimum viable log entry for a model run:**

| What | Why | How |
|---|---|---|
| Hyperparameters | Reproduce the exact model | `mlflow.log_params(params)` |
| Metrics (train + val) | Verify performance claims | `mlflow.log_metrics(metrics)` |
| Feature list | Know what the model sees | log as a JSON artifact |
| Data version / hash | Know what it was trained on | log as a param or artifact |
| Git commit hash | Know the exact code used | `git rev-parse HEAD` |
| Serialised model | Load and predict from it | `mlflow.sklearn.log_model(pipe, ...)` |
| Calibration plot | Verify score trustworthiness | `calibration_curve()` → log as artifact |
| Per-subgroup metrics | Detect fairness gaps | AUC by `sex`, `race` → `mlflow.log_metrics(...)` |

> A run without these fields is not reproducible. It's just a memory.

**MLflow does this for you** — with three function calls. That's Block 2.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 15b — BEYOND POINT ESTIMATES
     ══════════════════════════════════════════════════════════════════════════ -->

# Beyond Point Estimates

A single `val_auc = 0.923` tells you nothing about:

| | Variance | Calibration | Subgroups | Ranking |
|---|---|---|---|---|
| **Report** | AUC mean ± std (k-fold CV) | Brier score + reliability diagram | Per-demographic AUC | Bootstrap CI on AUC diff |

**How noisy is a single AUC?** Hanley–McNeil (1982):

```
SE(AUC) = √[ (A(1−A) + (n₊−1)(Q₁−A²) + (n₋−1)(Q₂−A²)) / (n₊·n₋) ]
where A = AUC,  n₊/n₋ = positive/negative counts,  Q₁ = A/(2−A),  Q₂ = 2A²/(1+A)
```

Our test set (AUC = 0.923, n₊ = 360, n₋ = 1,140): **SE ≈ 0.010, 95% CI [0.903, 0.943]**.

Two configs differing by 0.006 → gap is within the noise. A single train/test partition cannot tell whether it is real — you need cross-validation or bootstrap CIs.

> [*Bouthillier et al. (2021), MLSys*](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) — without variance estimates, model selection is noise mining.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 15c — MULTIPLE COMPARISONS IN MODEL SELECTION
     ══════════════════════════════════════════════════════════════════════════ -->

# Multiple Comparisons in Model Selection

**The problem:** you sweep *k* hyperparameter configs and pick `argmax(val_auc)`.

From the previous slide, two configs whose true AUC is identical will still differ by ~0.01 due to noise. With *k* configs, the **maximum** of *k* random fluctuations grows — even if no config is truly better, one will *look* better by pure chance.

How likely is a spurious winner? Each config has some probability *p* of producing a misleadingly high AUC. The probability that **at least one** of *k* configs is a false winner:

```
P(at least one false best) = 1 − (1 − p)^k

Example with p = 0.05:
k =  4 → 1 − 0.95^4  ≈ 18.5%
k = 20 → 1 − 0.95^20 ≈ 64.2%
```

The exact *p* depends on your data and metric — the point is the **exponential growth with *k***. More configs = more lottery tickets.

**Mitigations:**

- **Cross-validation** — report mean ± std across folds to estimate true variance
- **One-standard-error rule** — pick the simplest model within 1 SE of the best
- **Bonferroni correction** — divide significance threshold by the number of comparisons
- **Bayesian optimisation** — model the objective surface instead of grid search

---

<!-- ══════════════════════════════════════════════════════════════════════════
     TAXONOMY SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# The MLOps Taxonomy

## Five capabilities. One pipeline.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 17 — 5 CAPABILITIES
     ══════════════════════════════════════════════════════════════════════════ -->

# Five Capabilities, One Pipeline

```
  Experiment     Model        Model        CI / CD      Deployment
  Tracking   →  Versioning →  Monitoring →  Pipeline  →  & Serving
  ──────────    ──────────    ──────────    ────────     ──────────
  MLflow        MLflow        Evidently     GitHub       Docker +
  Tracking      Registry                    Actions      FastAPI
```

**How they connect:**
1. You train a model → **Tracking** records the run
2. The best run → **Registry** promotes it to Production
3. Production traffic → **Monitoring** checks for drift
4. Drift detected → **CI/CD** triggers retraining
5. New model → **Serving** replaces the old container

> Today you will see all five. The workshop gives you hands-on experience with steps 1–3.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 18 — WHAT WE WON'T COVER
     ══════════════════════════════════════════════════════════════════════════ -->

# What We Won't Cover Today — And Where To Learn It

| Topic | Why cut | Where to learn |
|---|---|---|
| Full cloud deployment | Requires cloud accounts; tool-specific | Chip Huyen — *Designing ML Systems*, Ch. 6 on |
| Online / continual learning | Needs streaming data infrastructure | Chip Huyen — *Designing ML Systems*, Ch. 9 |
| A/B testing and shadow mode | Requires production traffic splitting | Chip Huyen — Ch. 9 |
| Model explainability at scale | SHAP covered in Day 2 | SHAP docs, *Interpretable ML* (Molnar) |
| Multiple monitoring tools | Depth over breadth | Evidently docs, NannyML, DeepChecks |
| Data versioning (DVC, Git LFS, LakeFS) | Orthogonal to today's focus | [DVC docs](https://dvc.org/doc), [Git LFS](https://git-lfs.com) |

> The syllabus for this week is designed for depth, not breadth — use the resources above to go further.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 19 — BLOCK 1 WRAP-UP
     ══════════════════════════════════════════════════════════════════════════ -->

# Block 1 — Summary

**What we established:**

- Three failure modes that MLOps prevents (environment mismatch, silent decay, regulatory gap)
- The MLOps lifecycle and where today's tools sit in it
- Five reproducibility practices that are non-negotiable: seeds, pinned deps, isolation, deterministic splits, logging
- The five-capability taxonomy: Tracking → Versioning → Monitoring → CI/CD → Deployment

**What comes next:**

> Block 2 builds the **Tracking** and **Versioning** layers live.
> Open `day4/notebooks/01_mlflow_tracking.ipynb`.

*Start the MLflow server now (separate terminal):*

```bash
make -f day4/Makefile mlflow-server
```

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BLOCK 2 — SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Block 2

# Experiment Tracking
# with MLflow

## `01_mlflow_tracking.ipynb`

*10:30 – 11:00 · 30 minutes*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 21 — THE PROBLEM MLFLOW SOLVES
     ══════════════════════════════════════════════════════════════════════════ -->

# The Problem MLflow Solves

**Before MLflow** — a team's shared drive:

```
models/
  model_v1.pkl
  model_v2.pkl
  model_v2_scaled.pkl
  model_v2_scaled_FINAL.pkl
  model_v2_scaled_FINAL_really_final.pkl   ← which one is in production?
  model_v3_experiment.pkl                  ← what were the params?
```

No record of: which data it was trained on, what the validation AUC was,
which feature set was used, or who approved it.

**After MLflow** — every run has an ID:

```
Run a4f3e2b1 | 2024-03-01 | n_estimators=300 lr=0.1 | val_auc=0.923
Run 9c7d1a23 | 2024-03-05 | n_estimators=500 lr=0.05 | val_auc=0.931 ← PRODUCTION
Run f2e8c9d4 | 2024-03-10 | n_estimators=100 lr=0.2 | val_auc=0.908
```

Any run is reproducible from its ID. Any production model can be traced to its training run.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 22 — MLFLOW'S 4 COMPONENTS
     ══════════════════════════════════════════════════════════════════════════ -->

# MLflow's Four Components

| Component | What it does | Used today |
|---|---|---|
| **Tracking** | Logs runs: params, metrics, tags, artifacts | ✅ Core demo |
| **Projects** | Reproducible run packaging (`MLproject` file) | ✅ `day4/MLproject` |
| **Models** | Serialisation with "flavors" (sklearn, lgbm, …) | ✅ `predict.py` |
| **Registry** | Model versioning + stage lifecycle | ✅ Demo + workshop |

**The three API calls you need:**

```python
with mlflow.start_run(run_name="baseline"):
    mlflow.log_params({"n_estimators": 300, "lr": 0.1})  # what you chose
    mlflow.log_metrics({"val_auc": 0.923, "val_f1": 0.71})  # what you measured
    mlflow.sklearn.log_model(pipeline, artifact_path="model")  # the model itself
```

**Backend store URI:** `sqlite:///day4/mlflow.db`

- `sqlite://` = use SQLite as the database engine
- `///day4/mlflow.db` = file path relative to where the server runs (repo root)
- Production equivalent: `postgresql://user:pass@host/mlflow`
- Your training code never sees this — it only calls `mlflow.set_tracking_uri("http://127.0.0.1:5000")`

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 23 — MODEL REGISTRY
     ══════════════════════════════════════════════════════════════════════════ -->

# The Model Registry — Stage Lifecycle

```
  None            Staging         Production      Archived
  (just logged) ──────────────► (candidate) ──► (live) ──► (retired)
```

**What each stage means organisationally:**

| Stage | Meaning | Who acts |
|---|---|---|
| **None** | Run logged, model not yet reviewed | Data scientist |
| **Staging** | Candidate for production — under validation | Model risk / peer review |
| **Production** | Live model serving predictions | Approved by model risk |
| **Archived** | Retired; still accessible for audit | Automatic or manual |

> Staging → Production transition generally requires a documented approval. MLflow captures who promoted it, when, and from which run. This is your audit trail.

<!-- notes: The key pedagogical point: the Registry is not just version control, it's a governance tool. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 24 — START THE SERVER
     ══════════════════════════════════════════════════════════════════════════ -->

# Start the Server — Follow Along

**Terminal 1 (leave running):**

```bash
make -f day4/Makefile mlflow-server
# equivalent: mlflow server --host 127.0.0.1 --port 5000 \
#   --backend-store-uri sqlite:///day4/mlflow.db \
#   --default-artifact-root ./day4/mlartifacts
```

Open **http://127.0.0.1:5000** in your browser.

**Notebook:**

```
day4/notebooks/01_mlflow_tracking.ipynb
```

We will:
1. Log a first training run with three `mlflow.*` calls
2. Run a 4-config hyperparameter sweep and compare in the UI
3. Register the best model and transition it to Production
4. Load the Production model and verify predictions

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BLOCK 3 — SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Block 3

# Data Drift &
# Model Monitoring

## `02_drift_monitoring.ipynb`

*11:00 – 11:45 · 45 minutes*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 26 — THREE TYPES OF DRIFT
     ══════════════════════════════════════════════════════════════════════════ -->

# Three Types of Drift

| Type | Formal definition | Credit risk example | Detectable without labels? |
|---|---|---|---|
| **Covariate shift** | P(X) changes, P(Y\|X) stable | Applicant age distribution shifts after a marketing campaign | ✅ Yes — PSI, KS test, univariate/multivariate drift |
| **Concept drift** | P(Y\|X) changes — relationship between features and outcome shifts | Remote work makes `commute_distance` irrelevant for income | ❌ No — requires ground truth labels |
| **Prior probability shift** | P(Y) changes — base rate of outcome shifts | Recession increases the default rate | ⚠️ Partially — score distribution may shift, but confirmation requires labels |

> **Practical implication:** Covariate shift is detectable early with PSI and statistical tests. Concept drift is invisible to all input-based monitoring — label collection is non-negotiable.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 27 — WHY DRIFT DETECTION IS HARD
     ══════════════════════════════════════════════════════════════════════════ -->

# Why Drift Detection Is Hard in Credit Risk

**Ground truth latency:**

```
  Loan originated ──► 6–24 months ──► Default / Full repayment (label arrives)
```

You cannot wait for labels to know if your model is degrading.
By the time you have enough labels to confirm performance, the portfolio has already been mispriced.

**What you can observe immediately:**
- The distribution of input features (P(X))
- The model's predicted probability distribution
- The score distribution over time

**What Evidently gives you:**
- Feature-level drift alerts (statistical tests per feature) without waiting for labels
- Rich HTML reports with per-feature distribution overlays
- An actionable signal weeks or months before ground truth arrives

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 28 — SCORE DISTRIBUTION MONITORING
     ══════════════════════════════════════════════════════════════════════════ -->

# Score Distribution Monitoring

Track how the model's **output distribution** changes over time — the simplest, most robust label-free signal.

**What to track:**

| Metric | What it tells you | Threshold |
|---|---|---|
| Score PSI | Overall shift in predicted probability distribution | < 0.10 stable, > 0.25 action |
| Score mean | Average confidence trending up or down | Domain-specific |
| Score std | Spread of predictions narrowing or widening | Domain-specific |
| Score percentiles | Tail behaviour (P5, P50, P95) over time | Domain-specific |

**Why this works:** if the input distribution shifts (covariate shift), the score distribution will shift too. Score PSI captures this in a single number.

**Why this isn't enough:** if P(Y|X) changes but P(X) stays the same (concept drift), the score distribution is unchanged. You need labels to detect this.

> Production monitoring = **feature PSI** + **score PSI** + **univariate drift tests** + **label collection when available**.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 29 — UNIVARIATE VS MULTIVARIATE
     ══════════════════════════════════════════════════════════════════════════ -->

# Drift Detection — Two Levels

**Univariate (per-feature):**

| Feature type | Common tests | Evidently default (n > 1000) |
|---|---|---|
| Continuous | KS, Wasserstein, Anderson-Darling | Wasserstein distance |
| Categorical | Chi-squared, Jensen-Shannon, Z-test | Jensen-Shannon distance |

Output: distance or p-value per feature (method-dependent).

**Multivariate (joint distribution):**

A **domain classifier** trains a simple model to distinguish reference from current data. If it succeeds (AUC >> 0.5), the joint distribution has shifted — even if no individual feature crosses a threshold. This is a single test: no multiple testing correction needed.

> Evidently does not provide this for tabular data out of the box — we implement it in ~15 lines of scikit-learn. See notebook 02, Section 7C.
> **Use both.** Univariate catches obvious per-feature shifts. The domain classifier catches subtle correlated shifts that per-feature tests miss.

**Multiple testing warning:** Testing 19 features independently gives a high chance of false alerts under no drift. Evidently does **NOT** auto-correct. Apply Bonferroni or Benjamini-Hochberg when using p-value-based tests (KS, chi-squared) in production.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 29b — PSI — THE INDUSTRY STANDARD
     ══════════════════════════════════════════════════════════════════════════ -->

# PSI — Population Stability Index


```
PSI = Σ (p_analysis_i - p_reference_i) × ln(p_analysis_i / p_reference_i)
```

Bin both distributions into deciles, compare proportions per bin.

| PSI value | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 – 0.25 | Moderate shift | Investigate |
| > 0.25 | Significant shift | Retrain / escalate |

**Why PSI over p-values?**
- Interpretable — thresholds are universal across the industry
- Applicable to features (input PSI) and predicted scores (score PSI)
- ~15 lines of code — no library dependency
- Well-understood by regulators (SR 11-7, ECB Guide to Internal Models)

> In the notebook, we implement PSI from scratch and compare it to Evidently's statistical tests.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 29c — CONCEPT DRIFT
     ══════════════════════════════════════════════════════════════════════════ -->

# When Input Monitoring Fails — Concept Drift

**All input-distribution-based monitoring** (PSI, KS, chi-squared) monitors P(X) or the score distribution. **Concept drift** changes P(Y|X) — the mapping between features and outcome.

**Example:** credit model trained pre-pandemic. Post-pandemic: same income distributions, same age distributions — but the default relationship has changed. Every monitoring metric stays green. Actual performance degrades.

```
  Score PSI:  0.02  (stable)     ← scores haven't changed
  Feature KS: no alerts          ← features haven't changed
  Actual AUC: 0.74  (degraded)   ← labels reveal the truth
```

**Takeaway:** input monitoring is necessary but not sufficient. In production:
1. Pursue **label collection** — even partial labels are informative
2. Use **early proxies** (30/60/90 day delinquency) when final labels are delayed
3. **Periodic backtesting** against realised outcomes is non-negotiable

See: [*Gama et al. (2014), "A Survey on Concept Drift Adaptation"*](https://dl.acm.org/doi/10.1145/2523813)

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 30 — FAILING LOUDLY PAPER
     ══════════════════════════════════════════════════════════════════════════ -->

# "Failing Loudly" — The Statistical Foundation

[**Rabanser, Günnemann & Lipton · NeurIPS 2019**](https://arxiv.org/abs/1810.11953)

**What they did:** Systematic empirical comparison of 11 drift detection methods across 7 datasets and 9 perturbation types.

**Key findings:**
- Univariate tests (KS, chi²) are easy to implement but miss multivariate drift
- **Maximum Mean Discrepancy (MMD)** with learned representations performs best overall
- Dimensionality reduction before testing helps
- No single test dominates across all drift types — use multiple

**Why it matters for us:** Evidently's univariate drift tests + PSI approach is grounded in this empirical evidence. You're not just using a tool — you're applying peer-reviewed methodology.

> Read after the course. Title is the thesis: production ML should *fail loudly*, not silently.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 31 — REFERENCE VS ANALYSIS SETS
     ══════════════════════════════════════════════════════════════════════════ -->

# Evidently — Reference and Analysis Sets

```
  Reference set                    Analysis set
  ─────────────────────────────    ──────────────────────────────────
  Training / validation data       Production data (new observations)
  Ground truth available ✅        Ground truth NOT yet available ❌
  Used to FIT the estimators       Used to ESTIMATE + DETECT DRIFT

  Columns needed:                  Columns needed:
    • features (X)                   • features (X)
    • y_pred_proba                   • y_pred_proba
    • y_pred                         • y_pred
    • y_true                         • timestamp
    • timestamp
```

**In today's demo:** reference = train split (7,400 rows), analysis = test split (1,850 rows).

We simulate production conditions: Evidently never sees the test-set labels. We use PSI and univariate drift tests to monitor for distributional changes.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 32 — MONITORING AS A CI STEP
     ══════════════════════════════════════════════════════════════════════════ -->

# Monitoring as a CI Step

**What `ml-pipeline.yml` implements:**

```
  Push trigger (day4/** changed) or manual dispatch
         │
         ▼
  train.py  → log model to file-based MLflow
         │
         ▼
  drift_check.py
  ├── Load model from the training run (runs:/<run_id>/model)
  ├── Score data → y_pred_proba
  ├── Compute PSI → score + feature drift
  ├── Run univariate drift → alert flags
  ├── Generate HTML reports + log to MLflow
  └── Exit code 1 if threshold exceeded (--fail-on-drift)
         │
         ▼  (always)
  MLflow artifacts + drift reports uploaded as GitHub Actions artifacts
  If drift check failed → workflow fails
```

**In production** you would add: nightly cron schedule, remote MLflow server, Slack/PagerDuty notifications on failure.

> Block 4 shows the CI/CD side. `02_drift_monitoring.ipynb` shows the monitoring side.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 33 — OPEN THE NOTEBOOK
     ══════════════════════════════════════════════════════════════════════════ -->

# Block 3 — Open the Notebook

```
day4/notebooks/02_drift_monitoring.ipynb
```

**What we will do:**
1. Build reference and analysis sets from the Adult Income dataset
2. Compute **PSI** from scratch — the industry-standard drift metric
3. Run **univariate drift detection** — identify drifted features (with Bonferroni correction)
4. Run **multivariate drift detection** — dataset-level drift
5. **Simulate drift** — perturb features, watch alerts fire; simulate **concept drift** and show monitoring blindness
6. **Log** the HTML reports to MLflow as artifacts

> The MLflow server from Block 2 must still be running.
> If it's not: `make -f day4/Makefile mlflow-server`

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BLOCK 4 — SECTION HEADER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Block 4

# From Local to Production

## Docker & CI/CD

*11:45 – 12:30 · 45 minutes*
*(Pre-built examples — no live coding)*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 35 — WHY CONTAINERS
     ══════════════════════════════════════════════════════════════════════════ -->

# Why Containers?

| Benefit | Without Docker | With Docker |
|---|---|---|
| **Reproducibility** | "Works on my machine" | Same OS + runtime everywhere |
| **Isolation** | Global package conflicts | Each app has its own dependencies |
| **Deployment** | "Ship the code, hope it works" | Ship one artifact — it always works |
| **Auditability** | Env is implicit, undocumented | Env is defined in `Dockerfile`, version-controlled |

**Docker = the nuclear option for reproducibility.**

It extends `pyproject.toml` from "Python packages" to "the entire operating system."

> If MLflow is the reproducibility tool for *experiments*, Docker is the reproducibility tool for *environments*.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 36 — OUR DOCKERFILE
     ══════════════════════════════════════════════════════════════════════════ -->

# Our Dockerfile — Annotated

```dockerfile
FROM python:3.12-slim                # ← base image: pinned Python version

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*   # ← LightGBM needs OpenMP

RUN pip install --no-cache-dir \
    mlflow fastapi "uvicorn[standard]" pydantic \
    pandas scikit-learn lightgbm      # ← runtime deps only (no dev tools)

COPY day4/src/predict.py src/predict.py
COPY day4/src/__init__.py src/__init__.py

COPY day4/outputs/model/ /app/model/ # ← model extracted before build
ENV MODEL_URI="/app/model"           # ← loaded from local path, no server

EXPOSE 8080
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Build flow:** `make train` logs the model to MLflow → `make docker-build` extracts the artifact to `day4/outputs/model/` → Docker `COPY` bakes it into the image.

**But the image also supports runtime override:** set `MODEL_URI` and `MLFLOW_TRACKING_URI` as env vars to pull from MLflow instead — that's what `docker-compose.yml` does.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 37 — DOCKER COMPOSE
     ══════════════════════════════════════════════════════════════════════════ -->

# Docker Compose — Two Services

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.21.3   # pre-built, pinned version
    volumes: ["./day4:/app/day4"]            # SQLite DB persists on host
    ports: ["5000:5000"]
    healthcheck: ...                         # ready gate for model-svc
    restart: unless-stopped                  # self-heal on crash

  model-svc:
    image: adult-income-predictor:latest
    environment:                             # override baked-in model path
      - MODEL_URI=models:/adult-income-classifier@champion
      - MLFLOW_TRACKING_URI=http://mlflow:5000   # Docker internal DNS
    ports: ["8080:8080"]
    healthcheck: ...                         # orchestrator can detect failures
    depends_on:
      mlflow: { condition: service_healthy } # startup ordering
    restart: unless-stopped
```

**Patterns that transfer to K8s / ECS:** health checks, restart policies, env-based config, service dependencies.

**Start:** `make -f day4/Makefile docker-compose-up`
**Endpoints:** `localhost:5000` (MLflow UI) · `localhost:8080` (API) · `localhost:8080/docs`

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 38 — CI/CD WORKFLOW
     ══════════════════════════════════════════════════════════════════════════ -->

# CI/CD — What the Workflow Does

`.github/workflows/ml-pipeline.yml` — triggers: push to `day4/`, manual dispatch
*(In production you'd add a cron schedule for nightly retraining on fresh data.)*

```
  1. checkout code              (actions/checkout@v4)
        │
  2. setup Python 3.12          (actions/setup-python@v5)
        │
  3. cache uv deps              (actions/cache@v4, keyed on uv.lock)
        │
  4. uv sync                    (install exact dependency versions)
        │
  5. TRAIN MODEL                (train.py → file-based mlruns/)
        │
  6. CHAMPION / CHALLENGER GATE (compare AUC vs baseline_metrics.json)
        │
  7. DRIFT CHECK                (drift_check.py, ref=val, analysis=test)
        │                       (continue-on-error — uploads first, fails later)
  8. upload MLflow artifacts     (mlruns/ → Actions artifact, 7 days)
        │
  9. upload drift reports        (if: always() — even on failure)
        │
 10. fail if drift exceeded      (propagate exit code from step 7)
```

Runner: `ubuntu-latest` (4 vCPU, 16 GB RAM — free on public repos).

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 38b — THE PRODUCTION CI/CD PIPELINE
     ══════════════════════════════════════════════════════════════════════════ -->

# What a Production Pipeline Adds

Our demo stops at "upload artifacts." A production pipeline **closes the loop**:

| | Step | Tool |
|---|---|---|
| **Our demo** | `train.py` | MLflow |
| | `drift_check.py` | Evidently |
| | Upload artifacts | GitHub Actions |
| **Production adds** | `docker build` | Dockerfile |
| | `docker push` | Container registry (ECR, GCR, GHCR) |
| | Deploy | Kubernetes / ECS / ArgoCD |

Docker is not a standalone tool — it is the **packaging step** that turns a trained model into a deployable artifact, automated by CI/CD.

> In our course: `make docker-build` is manual. In production, the CI runner does it on every successful training run.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 39 — THE KEY CI/CD INSIGHT
     ══════════════════════════════════════════════════════════════════════════ -->

# The Key CI/CD Insight

**In the workflow:**

```yaml
env:
  MLFLOW_TRACKING_URI: "file://./mlruns"   # artifacts stay on the runner
```

MLflow writes to a local directory on the GitHub Actions runner. After the job, we upload `mlruns/` as a GitHub Actions artifact. You download it and run `mlflow ui` locally to explore it.

**In production:**

```yaml
env:
  MLFLOW_TRACKING_URI: "https://mlflow.company.com"   # shared remote server
```

Same code. Same train.py. Only the URI changes. CI jobs and local developer runs see the same experiment history.

> **This is the most common misconception:** CI/CD does not require a fundamentally different codebase. The abstraction is in the tracking URI.

<!-- notes: Spend 2 minutes on this slide. It's the architectural insight that connects Blocks 2, 3, and 4 into a coherent system. Make sure everyone understands why file:// works in CI and what would change in production. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 40 — SCHEDULED RETRAINING
     ══════════════════════════════════════════════════════════════════════════ -->

# Scheduled Retraining

**Adding a cron trigger to the workflow:**

```yaml
on:
  push:
    paths: ["day4/**"]
  schedule:
    - cron: "0 2 * * *"   # run every night at 2 AM UTC
  workflow_dispatch:        # manual trigger from GitHub UI
```

**The full retraining loop:**

```text
 ┌─────────────┐   ┌──────────────────┐   ┌───────────┐   ┌──────────────┐
 │ Cron 2 AM   ├──►│ drift_check.py   ├──►│ train.py  ├──►│ Compare AUCs ├──► promote
 └──────┬──────┘   └────────┬─────────┘   └───────────┘   └──────┬───────┘
        ▲                   │                                    │
        │                no drift                          AUC ≤ current
        │                   ▼                                    ▼
        │                  stop                               archive
        └──────────────────────── next night ────────────────────┘
```

Implementing this loop is a **stretch goal for the group project.**

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 40b — GITOPS
     ══════════════════════════════════════════════════════════════════════════ -->

# GitOps — Git as the Single Source of Truth

**Core idea:** every operational artifact is a file in your repo.

| Artifact | File | What changes trigger |
|---|---|---|
| Python environment | `pyproject.toml` + `uv.lock` | CI re-installs deps |
| Model training | `src/train.py` | CI retrains the model |
| Container definition | `Dockerfile` | CI rebuilds the image |
| Service topology | `docker-compose.yml` | Cluster re-deploys services |
| Pipeline definition | `.github/workflows/ml-pipeline.yml` | CI itself changes |
| Monitoring config | `src/drift_check.py` | Drift thresholds update |

**You already have all of these in `day4/`.** That's GitOps.

The pattern: **push to `main` (or another specific branch, like `day4`) → CI detects changed files → runs the right pipeline steps → deploys the result.** No manual SSH, no "run this script on the server," **no folklore**.

> In mature setups, tools like **ArgoCD** or **Flux** watch the repo and automatically sync the production cluster to match the latest commit.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 41 — MLOPS LIFECYCLE REVISITED
     ══════════════════════════════════════════════════════════════════════════ -->

# The MLOps Lifecycle — Revisited

We started this morning with an abstract diagram. Here it is again, with concrete tools:

```text
 ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  Data    ├─►│  Feature ├─►│  Model   ├─►│  Deploy  ├─►│  Monitor │
 │Collection│  │   Eng.   │  │   Dev    │  │  Docker  │  │ Evidently│
 │          │  │ Days 1-2 │  │  MLflow  │  │  FastAPI │  │ PSI+KS   │
 └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
      ▲                                                         │ drift
      └──────────────── retrain (cron · Block 4) ───────────────┘
```

Every component you built today connects to this diagram. The workshop gives you the opportunity to run the full loop yourself.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 42 — WHAT'S NEXT
     ══════════════════════════════════════════════════════════════════════════ -->

# What's Next

## This afternoon — Workshop (1:30–4:30)

| Time | Activity |
|---|---|
| 1:30–3:00 | Steps 1–3: MLflow → Evidently → Docker (+ stretch: CI/CD) |
| 3:00–4:30 | Group project time — MLOps integration discussion |

**Before you leave today, your group should have decided:**
- Which MLflow experiment name you'll use for your project
- What your reference and analysis sets will be
- What a drift alert will trigger (retrain? alert? manual review?)

## Recommended reading

- Chip Huyen — *Designing Machine Learning Systems* (O'Reilly 2022)
- Sculley et al. — *Hidden Technical Debt in ML Systems* (NeurIPS 2015)
- Rabanser et al. — *Failing Loudly* (NeurIPS 2019)
- Bouthillier et al. — *Accounting for Variance in ML Benchmarks* (MLSys 2021)
- Gama et al. — *A Survey on Concept Drift Adaptation* (ACM Computing Surveys 2014)

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BACK COVER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Machine Learning in Industry
## Day 4 — Appendix & References

**Course repo:** github&#46;com/CardoAI/ml&#95;industry&#95;course

- MLflow — mlflow.org/docs/latest
- Evidently — docs.evidentlyai.com
- Sculley et al. (2015) — Hidden Technical Debt in ML Systems
- Rabanser et al. (2019) — Failing Loudly (arxiv.org/abs/1810.11953)
- Bouthillier et al. (2021) — Accounting for Variance in ML Benchmarks
- Gama et al. (2014) — A Survey on Concept Drift Adaptation
- Docker — docs.docker.com/get-started
- GitHub Actions — docs.github.com/en/actions

**Questions?** gennaro&#46;dibrino&#64;cardoai&#46;com
