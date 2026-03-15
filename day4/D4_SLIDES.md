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
| **3** | 11:00–11:45 | Data drift + monitoring (NannyML) | `02_nannyml_monitoring.ipynb` |
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

<!-- notes: This story is generic/composite — don't attribute it to a specific company. The key teaching point: scikit-learn 1.2 deprecated `sparse` in OneHotEncoder and introduced `sparse_output`; passing the old keyword name raises no error, it is simply ignored. Silent keyword-mismatch failures are the hardest to catch without pinned dependencies. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 5 — FAILURE STORY 2
     ══════════════════════════════════════════════════════════════════════════ -->

# Failure Story 2 — Silent Model Decay

**The situation**

A consumer lending platform deploys a loan default classifier in early 2019. It performs well for 18 months.

In Q2 2020, employment patterns shift dramatically: furlough schemes, gig economy collapse, remote work. The distribution of `hours_per_week`, `occupation`, and `income` — three key model features — shifts substantially.

The model keeps returning predictions. They just become progressively less accurate.

**Nobody notices** until Non-Performing Loan (NPL) rates spike and the credit committee asks for a performance report.

**Result:** The model had been underperforming for ~6 months before anyone looked.

> Without monitoring, a deployed model is a black box. You are flying blind.

<!-- notes: This is the central motivation for Block 3. Students should feel the operational cost of late detection before we show them how to prevent it. -->

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

> "We trained a model" is usually not an acceptable answer in the financial sector.

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
       ▲                                                       │
       └───────────────── feedback loop ◄─────────────────────┘
```

*Source: [ml-ops.org/content/mlops-principles](https://ml-ops.org/content/mlops-principles)*

**Today we cover from Model Deployment onwards:**

| Stage | What we build today |
|---|---|
| Model Development | LightGBM pipeline (same as Day 2) |
| Experiment Tracking | MLflow — Block 2 |
| Model Registry | MLflow Registry — Block 2 |
| Monitoring | NannyML drift + CBPE — Block 3 |
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

A result is reproducible if, given the same inputs, you always get the same output — on any machine, at any time.

In ML, that requires five things:

| # | Requirement | Common failure mode |
|---|---|---|
| 1 | **Fixed random seeds** | `train_test_split` returns different rows each run |
| 2 | **Pinned dependencies** | Library update silently changes behaviour |
| 3 | **Environment isolation** | "Works on my machine" |
| 4 | **Deterministic data splits** | "Same person" in train and test (leakage) |
| 5 | **Log everything** | Can't reconstruct which features a model used |

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

**Rules:** define `SEED` once at module level; pass `random_state=SEED` to every sklearn object; call `np.random.seed(SEED)` at script entry. In all Day 4 code: `SEED = 42`.

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

> Today: we use `uv` (with `pyproject.toml`) for the notebooks, Docker for the serving layer. In production, everything would be containerised.

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

> A run without these fields is not reproducible. It's just a memory.

**MLflow does this for you** — with three function calls. That's Block 2.

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
  MLflow        MLflow        NannyML       GitHub       Docker +
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
| Multiple monitoring tools | Depth over breadth | NannyML docs, Evidently, DeepChecks |

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

**Backend store URI:** `sqlite:///mlflow.db`

- `sqlite://` = use SQLite as the database engine
- `///mlflow.db` = file path relative to where the server runs (`./mlflow.db`)
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
#                           --backend-store-uri sqlite:///mlflow.db
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

## `02_nannyml_monitoring.ipynb`

*11:00 – 11:45 · 45 minutes*

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 26 — THREE TYPES OF DRIFT
     ══════════════════════════════════════════════════════════════════════════ -->

# Three Types of Drift

| Type | Formal definition | Credit risk example | Detectable without labels? |
|---|---|---|---|
| **Covariate shift** | P(X) changes, P(Y\|X) stable | Applicant age distribution shifts after a marketing campaign | ✅ Yes — NannyML univariate/multivariate |
| **Concept drift** | P(Y\|X) changes — relationship between features and outcome shifts | Remote work makes `commute_distance` irrelevant for income | ⚠️ Partially — CBPE detects the effect; cause needs labels |
| **Prior probability shift** | P(Y) changes — base rate of outcome shifts | Recession increases the default rate | ✅ Yes — CBPE tracks estimated vs actual performance |

> **Practical implication:** Covariate shift is easiest to detect early. Concept drift may only become visible when labels arrive. Build monitoring for both.

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

**What NannyML gives you:**
- Estimated performance from predicted probabilities alone (CBPE - Confidence-based Performance Estimation)
- Feature-level drift alerts without waiting for labels
- An actionable signal weeks or months before ground truth arrives

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 28 — CBPE
     ══════════════════════════════════════════════════════════════════════════ -->

# CBPE — Performance Estimation Without Ground Truth

**Confidence-Based Performance Estimation** works in two phases:

```
  Phase 1: FIT on reference data (ground truth available)
  ───────────────────────────────────────────────────────
  Reference set:  X_ref, y_pred_proba_ref, y_true_ref
  Learn: calibrated mapping  y_pred_proba → realised performance (AUC, F1)

  Phase 2: ESTIMATE on analysis data (no ground truth needed)
  ──────────────────────────────────────────────────────────
  Analysis set:  X_new, y_pred_proba_new  (no y_true)
  Apply the learned mapping → estimated AUC, estimated F1
```

**The assumption:** the model's probability calibration has not changed (i.e., no concept drift). If calibration has shifted, CBPE estimates degrade. In practice, CBPE is your early warning system — confirmation still requires labels.

<!-- notes: IMPORTANT — make this point explicitly. Students often think CBPE "solves" the label latency problem entirely. It doesn't; it gives you an early signal, not a certainty. The key use case is: flag the problem 6 months before labels arrive, so you can investigate. -->

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 29 — UNIVARIATE VS MULTIVARIATE
     ══════════════════════════════════════════════════════════════════════════ -->

# Drift Detection — Two Levels

**Univariate (per-feature):**

| Feature type | Test used | What it detects |
|---|---|---|
| Continuous | Kolmogorov-Smirnov (KS) | Distribution shape changes |
| Categorical | Chi-squared | Category proportion changes |

Output: p-value + alert flag per feature per chunk.

**Multivariate (joint distribution):**

NannyML uses **PCA reconstruction error** — project all numeric features into a lower-dimensional space, measure how well the reference reconstruction fits the analysis data. A spike in reconstruction error = the correlation structure has changed, even if individual features look stable.

> **Use both.** Univariate catches obvious shifts. Multivariate catches subtle correlated shifts that per-feature tests miss.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 30 — FAILING LOUDLY PAPER
     ══════════════════════════════════════════════════════════════════════════ -->

# "Failing Loudly" — The Statistical Foundation

**Rabanser, Günnemann & Lipton · NeurIPS 2019**
[arxiv.org/abs/1810.11953](https://arxiv.org/abs/1810.11953)

**What they did:** Systematic empirical comparison of 11 drift detection methods across 7 datasets and 9 perturbation types.

**Key findings:**
- Univariate tests (KS, chi²) are easy to implement but miss multivariate drift
- **Maximum Mean Discrepancy (MMD)** with learned representations performs best overall
- Dimensionality reduction before testing helps (hence NannyML's PCA approach)
- No single test dominates across all drift types — use multiple

**Why it matters for us:** NannyML's univariate + PCA reconstruction approach is grounded in this empirical evidence. You're not just using a tool — you're applying peer-reviewed methodology.

> Read after the course. Title is the thesis: production ML should *fail loudly*, not silently.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 31 — REFERENCE VS ANALYSIS SETS
     ══════════════════════════════════════════════════════════════════════════ -->

# NannyML — Reference and Analysis Sets

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

We simulate production conditions: NannyML never sees the test-set labels when estimating performance. At the end, we compare the CBPE estimate to the actual AUC as a sanity check.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 32 — MONITORING AS A CI STEP
     ══════════════════════════════════════════════════════════════════════════ -->

# Monitoring as a CI Step

**The production architecture:**

```
  Nightly cron / push trigger
         │
         ▼
  nannyml_check.py
  ├── Load production model (MLflow Registry)
  ├── Score new data → y_pred_proba
  ├── Run CBPE → estimated AUC
  ├── Run univariate drift → alert flags
  ├── Generate HTML reports
  ├── Log metrics + reports to MLflow
  └── Exit code 1 if threshold exceeded
         │
         ▼  (on failure)
  GitHub Actions step fails
  NannyML report uploaded as artifact
  Team notified via GitHub notification
```

This is exactly what `.github/workflows/ml-pipeline.yml` implements.

> Block 4 shows the CI/CD side. `02_nannyml_monitoring.ipynb` shows the monitoring side.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 33 — OPEN THE NOTEBOOK
     ══════════════════════════════════════════════════════════════════════════ -->

# Block 3 — Open the Notebook

```
day4/notebooks/02_nannyml_monitoring.ipynb
```

**What we will do:**
1. Build reference and analysis sets from the Adult Income dataset
2. Run **CBPE** — compare estimated AUC vs actual AUC
3. Run **univariate drift detection** — identify drifted features
4. Run **multivariate drift detection** — PCA reconstruction error
5. **Simulate drift** — perturb `hours_per_week`, watch the alerts fire
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

**Production pattern:** `MODEL_URI` points to a remote registry; model pulled at startup.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 37 — DOCKER COMPOSE
     ══════════════════════════════════════════════════════════════════════════ -->

# Docker Compose — Two Services

```yaml
services:
  mlflow:
    image: python:3.12-slim
    command: >
      bash -c "pip install mlflow -q &&
      mlflow server --host 0.0.0.0 --port 5000
        --backend-store-uri sqlite:///day4/mlflow.db"
    volumes: ["./day4:/app/day4"]
    ports: ["5000:5000"]
    healthcheck: ...

  model-svc:
    image: adult-income-predictor:latest   # pre-built image
    ports: ["8080:8080"]
    depends_on:
      mlflow: { condition: service_healthy }
```

**Start everything:** `make -f day4/Makefile docker-compose-up`

| URL | Service |
| --- | --- |
| `localhost:5000` | MLflow UI |
| `localhost:8080` | Prediction API |
| `localhost:8080/docs` | Swagger / OpenAPI |

> The model is already baked into the `model-svc` image. The MLflow container is for the UI and experiment history only.

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 38 — CI/CD WORKFLOW
     ══════════════════════════════════════════════════════════════════════════ -->

# CI/CD — What the Workflow Does

`.github/workflows/ml-pipeline.yml` runs on every push to `day4/`:

```
  1. checkout code          (actions/checkout@v4)
        │
  2. setup Python 3.12      (actions/setup-python@v5)
        │
  3. cache uv deps          (actions/cache@v4, keyed on day4/uv.lock)
        │
  4. cd day4 && uv sync     (isolated day4 environment)
        │
  5. TRAIN MODEL            (train.py → file-based mlruns/ on runner)
        │
  6. DRIFT CHECK            (nannyml_check.py, continue-on-error: true)
        │
  7. upload MLflow artifacts   (mlruns/ → GitHub Actions artifact, 7 days)
        │
  8. upload NannyML reports    (if: always() — even on failure)
        │
  9. fail if drift exceeded    (propagate exit code from step 6)
```

Runner: `ubuntu-latest` (4 vCPU, 16 GB RAM — free on public repos).

---

<!-- ══════════════════════════════════════════════════════════════════════════
     SLIDE 38b — THE PRODUCTION CI/CD PIPELINE
     ══════════════════════════════════════════════════════════════════════════ -->

# What a Production Pipeline Adds

Our demo workflow stops at "upload artifacts." A production pipeline **closes the loop**:

```text
  ┌─────────────────────── our demo (slides 38) ──────────────────────────┐
  │                                                                       │
  train.py ──► nannyml_check.py ──► upload artifacts                      │
                                         │                                │
  ┌──────────────────────────────────────────── production adds ──────────┐│
  │                                      ▼                               ││
  │                               docker build                           ││
  │                                      │                               ││
  │                               docker push  → container registry      ││
  │                                      │        (ECR, GCR, GHCR)       ││
  │                               deploy trigger → Kubernetes / ECS      ││
  └──────────────────────────────────────────────────────────────────────┘│
  └───────────────────────────────────────────────────────────────────────┘
```

**The key point:** Docker is not a standalone tool. It is the **packaging step** that turns a trained model into a deployable artifact, automated by CI/CD.

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

> **This is the most common misconception:** students think CI/CD requires a fundamentally different codebase. It doesn't. The abstraction is in the tracking URI.

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
 │ Cron 2 AM   ├──►│ nannyml_check.py ├──►│ train.py  ├──►│ Compare AUCs ├──► promote
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
| Monitoring config | `src/nannyml_check.py` | Drift thresholds update |

**You already have all of these in `day4/`.** That's GitOps.

The pattern: **push to `main` → CI detects changed files → runs the right pipeline steps → deploys the result.** No manual SSH, no "run this script on the server," no tribal knowledge.

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
 │Collection│  │   Eng.   │  │   Dev    │  │  Docker  │  │  NannyML │
 │          │  │ Days 1-2 │  │  MLflow  │  │  FastAPI │  │   CBPE   │
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
| 1:30–3:00 | Steps 1–3: MLflow → NannyML → Docker (+ stretch: CI/CD) |
| 3:00–4:30 | Group project time — MLOps integration discussion |

**Before you leave today, your group should have decided:**
- Which MLflow experiment name you'll use for your project
- What your reference and analysis sets will be
- What a drift alert will trigger (retrain? alert? manual review?)

## Recommended reading

- Chip Huyen — [*Designing Machine Learning Systems*](https://a.co/d/8zYS4eg)
- Rabanser et al. — [*Failing Loudly*](https://arxiv.org/abs/1810.11953) (NeurIPS 2019)
- [MLOps Principles](https://ml-ops.org/content/mlops-principles) — ml-ops.org

---

<!-- ══════════════════════════════════════════════════════════════════════════
     BACK COVER
     ══════════════════════════════════════════════════════════════════════════ -->
<!-- _class: lead -->

# Machine Learning in Industry
## Day 4 — Appendix & References

**Course repo:** github&#46;com/CardoAI/ml&#95;industry&#95;course

- MLflow — mlflow.org/docs/latest
- NannyML — nannyml.readthedocs.io
- Failing Loudly — arxiv.org/abs/1810.11953
- MLOps Principles — ml-ops.org/content/mlops-principles
- Docker — docs.docker.com/get-started
- GitHub Actions — docs.github.com/en/actions

**Questions?** gennaro&#46;dibrino&#64;cardoai&#46;com
