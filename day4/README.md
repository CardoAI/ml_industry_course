# Day 4 — Intro to MLOps, Reproducibility, and Model Monitoring

**Course:** Machine Learning in Industry · Cardo AI / MSCA Digital Doctoral Network

**Schedule:** 9:30–12:30 Lectures · 1:30–4:30 Workshop

---

## Recommended flow

Whether you are teaching this material or working through it independently, the intended sequence is:

```text
Slides (Blocks 1–4)
    └─ Block 1  Reproducibility (slides 3–19)
    └─ Block 2  Experiment tracking — follow along: notebooks/01_mlflow_tracking.ipynb
    └─ Block 3  Drift & monitoring — follow along: notebooks/02_nannyml_monitoring.ipynb
    └─ Block 4  Containers & CI/CD (pre-built examples, no live coding)

Workshop (afternoon)
    └─ notebooks/03_workshop_skeleton.ipynb  ← students fill in the TODOs
    └─ Steps 1–3 guided (90 min), Step 4 stretch goal
    └─ Group project discussion (final 90 min)
```

The two lecture notebooks (`01`, `02`) are fully worked demonstrations — run them top-to-bottom during the lecture. The workshop skeleton (`03`) has the same structure but with `TODO` blanks for students to complete.

---

## Repository layout

```text
day4/
├── D4_SLIDES.md                  # Marp slide deck (43 slides, all 4 blocks)
├── Makefile                      # All workshop commands (run from repo root)
├── Dockerfile                    # Prediction service image
├── docker-compose.yml            # MLflow server + prediction service
├── MLproject                     # MLflow Projects entry point
├── src/
│   ├── train.py                  # Training script — MLflow logging, model registration
│   ├── nannyml_check.py          # Drift detection script — CI gate (exit 1 on drift)
│   └── predict.py                # FastAPI prediction service
└── notebooks/
    ├── 01_mlflow_tracking.ipynb  # Block 2 demo — experiment tracking (30 min)
    ├── 02_nannyml_monitoring.ipynb # Block 3 demo — drift detection (45 min)
    └── 03_workshop_skeleton.ipynb  # Afternoon workshop — TODOs for students
```

The CI/CD workflow lives at `.github/workflows/ml-pipeline.yml`.

---

## Rendering the slides

The slides are written in [Marp](https://marp.app) Markdown. To render them:

```bash
# VS Code: install the "Marp for VS Code" extension, open D4_SLIDES.md, click Preview
# CLI:
npx @marp-team/marp-cli day4/D4_SLIDES.md --html -o day4/slides.html
```

---

## Prerequisites

- **Python ≥ 3.13**
- **uv** — install with:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- **Docker Desktop** (Step 3 only) — [docs.docker.com/get-started](https://docs.docker.com/get-started/)
- **GitHub account** (Step 4 only)

---

## Setup

> **Windows users:** The `make` commands require WSL or Git Bash. If neither is available, replace any `make -f day4/Makefile <target>` command with the equivalent `python` call shown in the Makefile for that target — the Python scripts themselves are fully cross-platform. For the NannyML compat environment, run `cd day4\nannyml-compat && uv sync` the same way — `uv` works natively on Windows.

### 1. Install dependencies

From the **repo root**:

```bash
uv sync --extra day4
# or
make -f day4/Makefile install
```

> **NannyML note:** NannyML 0.14.x requires `pandas < 2`, which conflicts with the main project's `pandas >= 3`. If `uv sync --extra day4` raises a conflict error, use the isolated compat environment instead:
>
> ```bash
> cd day4/nannyml-compat && uv sync
> # or via Makefile:
> make -f day4/Makefile nannyml-install
> ```
>
> Then prefix `nannyml_check.py` calls with `uv run --project day4/nannyml-compat`.

### 2. Start the MLflow tracking server

Run this in a **separate terminal** and leave it running throughout the workshop:

```bash
make -f day4/Makefile mlflow-server
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## Workshop steps

### Step 1 — Experiment tracking with MLflow

Open [notebooks/01_mlflow_tracking.ipynb](notebooks/01_mlflow_tracking.ipynb) for the lecture demo, or run from the command line:

```bash
make -f day4/Makefile train
```

This trains a LightGBM classifier on the Adult Income dataset and logs:

- Hyperparameters (`n_estimators`, `learning_rate`, `max_depth`)
- Metrics (`val_auc`, `val_f1`, `val_accuracy`)
- The serialised model as an MLflow artifact
- Registers the best model in the MLflow Model Registry (stage: Production)

The run ID is saved to `day4/outputs/last_run_id.txt` for use in later steps.

**Explore in the MLflow UI:**

- Compare multiple runs side by side
- Inspect the logged model artifact
- Check Model Registry → `adult-income-classifier` → Production

### Step 2 — Drift detection with NannyML

Open [notebooks/02_nannyml_monitoring.ipynb](notebooks/02_nannyml_monitoring.ipynb) for the lecture demo, or run from the command line:

```bash
make -f day4/Makefile drift-check
```

This will:

- Load the registered Production model
- Build reference (train) and analysis (test) sets
- Run CBPE to estimate performance without ground truth
- Run univariate drift detection across all features
- Save HTML reports to `day4/outputs/`
- Log reports to MLflow as artifacts

**Check the MLflow UI:** navigate to the `adult-income-monitoring` experiment to see the logged reports.

### Step 3 — Docker

First build the image (requires Step 1 to be complete):

```bash
make -f day4/Makefile docker-build
```

Then start the container:

```bash
make -f day4/Makefile docker-run
```

Test the service:

```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{
           "age": 35,
           "workclass": "Private",
           "education": "Bachelors",
           "education_num": 13,
           "marital_status": "Married-civ-spouse",
           "occupation": "Prof-specialty",
           "relationship": "Husband",
           "race": "White",
           "sex": "Male",
           "capital_gain": 0,
           "capital_loss": 0,
           "hours_per_week": 45,
           "native_country": "United-States"
         }'
```

The interactive API docs are at **[http://localhost:8080/docs](http://localhost:8080/docs)**.

> **Design note:** The Dockerfile bakes the trained model into the image using a local `file://` tracking URI. This avoids needing a shared MLflow server at runtime and keeps the workshop self-contained. In production you would pass `MODEL_URI` as an environment variable at runtime and pull the artifact from a remote registry.

### Step 4 (Stretch) — CI/CD with GitHub Actions

1. **Fork** this repository on GitHub.
2. **Push** any change to the `day4/` directory to trigger the workflow.
3. Navigate to the **Actions** tab on your fork.
4. Wait for the `ML Pipeline` workflow to complete.
5. Download the uploaded artifacts:
   - `mlflow-run-<sha>` — the full MLflow run directory (explore locally with `mlflow ui`)
   - `nannyml-report-<sha>` — HTML drift and CBPE reports

The workflow runs on a free `ubuntu-latest` GitHub-hosted runner — no cloud account needed.

> **Architecture note:** In the workflow, `MLFLOW_TRACKING_URI` is set to `file://./mlruns` so artifacts stay on the runner and are uploaded as GitHub Actions artifacts. In production, both the training job and the CI workflow would point at a shared remote tracking server (e.g., backed by S3 or Azure Blob).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` on port 5000 | macOS/Linux: `lsof -ti:5000 \| xargs kill` · Windows: `netstat -ano \| findstr :5000` then `taskkill /PID <pid>` |
| `Address already in use` on port 8080 | macOS/Linux: `lsof -ti:8080 \| xargs kill` · Windows: `netstat -ano \| findstr :8080` then `taskkill /PID <pid>` |
| NannyML `ImportError` | Run `cd day4/nannyml-compat && uv sync` (see Setup note above) |
| `docker: command not found` | Install Docker Desktop from [docs.docker.com/get-started](https://docs.docker.com/get-started/) |
| Docker build fails: `mlruns/ not found` | Run `make -f day4/Makefile train` first |
| MLflow model not in Production stage | Run `make -f day4/Makefile train` (registers automatically), then promote via UI if needed |

---

## Key references

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [NannyML Documentation](https://nannyml.readthedocs.io/)
- [Failing Loudly — Rabanser et al., NeurIPS 2019](https://arxiv.org/abs/1810.11953)
- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [Docker Getting Started](https://docs.docker.com/get-started/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Chip Huyen — Designing Machine Learning Systems](https://a.co/d/8zYS4eg)
