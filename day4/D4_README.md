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
    └─ Block 3  Drift & monitoring — follow along: notebooks/02_drift_monitoring.ipynb
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
│   ├── drift_check.py            # Drift detection — Evidently reports
│   └── predict.py                # FastAPI prediction service
└── notebooks/
    ├── 01_mlflow_tracking.ipynb  # Block 2 demo — experiment tracking (30 min)
    ├── 02_drift_monitoring.ipynb # Block 3 demo — drift detection (45 min)
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

> **Before you start:** open a clean terminal with **no active Python venv or conda environment**. Docker must be installed and running (needed for Step 3).

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11, < 3.14 | 3.12 recommended |
| [uv](https://docs.astral.sh/uv/) | latest | replaces pip/venv |
| [Docker Desktop](https://docs.docker.com/get-started/) | latest | Step 3 only |
| GitHub account | — | Step 4 stretch goal only |

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell (or run `source $HOME/.local/bin/env`) so that the `uv` command is on your `PATH`.

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell after installation. All subsequent commands in this guide that use `uv` work natively on Windows.

</details>

---

## Setup

> **Windows users:** The `make` commands require WSL or Git Bash. If neither is available, replace any `make -f day4/Makefile <target>` command with the equivalent shown in the [command reference](#command-reference) table below. The Python scripts themselves are fully cross-platform. `uv` works natively on Windows.
>
> **`make` on Windows:** available via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), [Git for Windows](https://gitforwindows.org/) (Git Bash), or [Chocolatey](https://community.chocolatey.org/packages/make) (`choco install make`).

### 1. Install dependencies

From the **repo root**:

```bash
make -f day4/Makefile install
# or: cd day4 && uv sync
```

`uv` will create an isolated virtual environment under `day4/.venv/` and install all locked dependencies. You do **not** need to activate the environment manually — subsequent `make` calls use it automatically via `uv run --project day4 --frozen`.

> **Why a separate project?** `day4` has its own `pyproject.toml` with `pandas>=2.2.0,<3` (required by MLflow), which conflicts with the root's `pandas>=3`. The `day4/.venv` environment is completely isolated from the root environment.

### 1b. Register the Jupyter kernel

```bash
make -f day4/Makefile kernel-install
```

<details>
<summary><strong>Windows without make</strong></summary>

```powershell
cd day4
uv run python -m ipykernel install --user --name mlops-industry-course --display-name "MLOps Industry Course"
cd ..
```

</details>

After running it, open any `.ipynb` in VS Code and select **"MLOps Industry Course"** from the kernel picker.

### 2. Start the MLflow tracking server

Run this in a **separate terminal** and leave it running throughout the workshop:

```bash
make -f day4/Makefile mlflow-server
```

<details>
<summary><strong>Windows without make</strong></summary>

```powershell
cd day4
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
cd ..
```

</details>

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
- Registers the model in the MLflow Model Registry with the `@champion` alias

The run ID is saved to `day4/outputs/last_run_id.txt` for use in later steps.

**Explore in the MLflow UI:**

- Compare multiple runs side by side
- Inspect the logged model artifact
- Check Model Registry → `adult-income-classifier` → `@champion` alias

### Step 2 — Drift detection with Evidently

Open [notebooks/02_drift_monitoring.ipynb](notebooks/02_drift_monitoring.ipynb) for the lecture demo, or run from the command line:

```bash
make -f day4/Makefile drift-check
```

This will:

- Load the `@champion` model from the registry
- Build reference (train) and analysis (test) sets
- Compute PSI for features and score distribution
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

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
# Health check
Invoke-RestMethod http://localhost:8080/health

# Prediction
$body = @{
    age=35; workclass="Private"; education="Bachelors"; education_num=13
    marital_status="Married-civ-spouse"; occupation="Prof-specialty"
    relationship="Husband"; race="White"; sex="Male"
    capital_gain=0; capital_loss=0; hours_per_week=45
    native_country="United-States"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8080/predict -Method Post `
    -ContentType "application/json" -Body $body
```

</details>

The interactive API docs are at **[http://localhost:8080/docs](http://localhost:8080/docs)**.

> **Design note:** The Dockerfile bakes the trained model into the image using a local `file://` tracking URI. This avoids needing a shared MLflow server at runtime and keeps the workshop self-contained. In production you would pass `MODEL_URI` as an environment variable at runtime and pull the artifact from a remote registry.

### Step 4 (Stretch) — CI/CD with GitHub Actions

1. **Fork** this repository on GitHub.
2. **Push** any change to the `day4/` directory to trigger the workflow.
3. Navigate to the **Actions** tab on your fork.
4. Wait for the `ML Pipeline` workflow to complete.
5. Download the uploaded artifacts:
   - `mlflow-run-<sha>` — the full MLflow run directory (explore locally with `mlflow ui`)
   - `drift-report-<sha>` — HTML drift report

The workflow runs on a free `ubuntu-latest` GitHub-hosted runner — no cloud account needed.

> **Architecture note:** In the workflow, `MLFLOW_TRACKING_URI` is set to `file:./mlruns` so artifacts stay on the runner and are uploaded as GitHub Actions artifacts. In production, both the training job and the CI workflow would point at a shared remote tracking server (e.g., backed by S3 or Azure Blob).

---

## Command reference

All `make` targets must be run from the **repo root**.

| Action | macOS / Linux | Windows (no make) |
|---|---|---|
| Install deps | `make -f day4/Makefile install` | `cd day4 && uv sync` |
| Install kernel | `make -f day4/Makefile kernel-install` | `cd day4 && uv run python -m ipykernel install --user --name mlops-industry-course --display-name "MLOps Industry Course"` |
| Train model | `make -f day4/Makefile train` | `cd day4 && uv run python src/train.py --n-estimators 300 --learning-rate 0.1 --max-depth 5 --mlflow-uri http://127.0.0.1:5000 --experiment-name adult-income-lgbm --run-name makefile-run --data-path ../day1/generated/adult_income_issues.csv --register true --model-name adult-income-classifier` |
| Run drift check | `make -f day4/Makefile drift-check` | `cd day4 && uv run python src/drift_check.py --data-path ../day1/generated/adult_income_issues.csv --model-uri models:/adult-income-classifier@champion --mlflow-uri http://127.0.0.1:5000 --output-dir outputs` |
| Build Docker image | `make -f day4/Makefile docker-build` | See note below |
| Run Docker container | `make -f day4/Makefile docker-run` | `docker run --rm -p 8080:8080 adult-income-predictor:latest` |
| Docker Compose (MLflow + API) | `make -f day4/Makefile docker-compose-up` | `docker compose -f day4/docker-compose.yml up` |
| Clean outputs | `make -f day4/Makefile clean` | `rmdir /s /q day4\outputs` |

> **Docker build note:** `make docker-build` extracts the model artifact from the running MLflow server and bakes it into the image. The MLflow server must be running when you build. On Windows without `make`, the equivalent requires extracting the model manually — use WSL or Git Bash with `make` instead.

---

## Troubleshooting

| Problem | macOS / Linux fix | Windows fix |
|---|---|---|
| Port 5000 already in use | `lsof -ti:5000 \| xargs kill` | `netstat -ano \| findstr :5000` then `taskkill /PID <pid> /F` |
| Port 8080 already in use | `lsof -ti:8080 \| xargs kill` | `netstat -ano \| findstr :8080` then `taskkill /PID <pid> /F` |
| `uv: command not found` | Re-run install script and restart shell | Restart PowerShell after install |
| `make: command not found` | Install via brew: `brew install make` | Use Git Bash, WSL, or raw Python commands (see table above) |
| `docker: command not found` | Install [Docker Desktop](https://docs.docker.com/get-started/) | Install [Docker Desktop](https://docs.docker.com/get-started/) |
| Docker build fails: `RESOURCE_DOES_NOT_EXIST` | Run `make mlflow-server` (separate terminal) then `make train` before `docker-build`. The run ID in `day4/outputs/last_run_id.txt` must match a run in the MLflow server. | Same |
| MLflow model not in Production stage | `make -f day4/Makefile train` (registers automatically) | Run train step first, then promote via UI if needed |
| `uv run` resolution error | Run `cd day4 && uv sync` first, then use `make` targets (they use `--frozen`) | Same |

---

## Key references

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Failing Loudly — Rabanser et al., NeurIPS 2019](https://arxiv.org/abs/1810.11953)
- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [Docker Getting Started](https://docs.docker.com/get-started/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Chip Huyen — Designing Machine Learning Systems](https://a.co/d/8zYS4eg)
