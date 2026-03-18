# Day 4 — Setup Guide

> **Before you start:** open a clean terminal with **no active Python venv or conda environment**. Docker must be installed and running (needed for Step 3).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11, < 3.14 | 3.12 recommended |
| [uv](https://docs.astral.sh/uv/) | latest | replaces pip/venv |
| [Docker Desktop](https://docs.docker.com/get-started/) | latest | Step 3 only |
| GitHub account | — | Step 4 stretch goal only |

### Install uv

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell (or run `source $HOME/.local/bin/env`) so that the `uv` command is on your `PATH`.

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell after installation. All subsequent commands in this guide that use `uv` work natively on Windows.

---

## 1. Install Python dependencies

From the **repo root**, run:

**macOS / Linux**

```bash
make -f day4/Makefile install
```

**Windows (PowerShell or Command Prompt)**

```powershell
cd day4
uv sync
cd ..
```

`uv` will create an isolated virtual environment under `day4/.venv/` and install all locked dependencies. You do **not** need to activate the environment manually — subsequent `make` calls use it automatically via `uv run --project day4 --frozen`.

> **Why a separate project?** `day4` has its own `pyproject.toml` with `pandas>=2.2.0,<3` (required by MLflow), which conflicts with the root's `pandas>=3`. The `day4/.venv` environment is completely isolated from the root environment.

### 1b. Register the Jupyter kernel

From the **repo root**:

**macOS / Linux**

```bash
make -f day4/Makefile kernel-install
```

**Windows — option A: WSL or Git Bash**

```bash
make -f day4/Makefile kernel-install
```

**Windows — option B: PowerShell (no make)**

```powershell
cd day4
uv run python -m ipykernel install --user --name mlops-industry-course --display-name "MLOps Industry Course"
cd ..
```

After running it, open any `.ipynb` in VS Code and select **"MLOps Industry Course"** from the kernel picker.

---

## 2. Start the MLflow tracking server

Open a **second terminal** (keep it running throughout the workshop) and run:

**macOS / Linux**

```bash
make -f day4/Makefile mlflow-server
```

**Windows — option A: WSL or Git Bash**

```bash
make -f day4/Makefile mlflow-server
```

**Windows — option B: PowerShell (no make)**

```powershell
cd day4
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
cd ..
```

Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser once the server is ready.

---

## 3. Workshop commands

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

> **Windows make:** The `make` utility is available via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), [Git for Windows](https://gitforwindows.org/) (Git Bash), or [Chocolatey](https://community.chocolatey.org/packages/make) (`choco install make`). If none of these are available, use the equivalent Python commands shown in the table above — the scripts themselves are fully cross-platform.

---

## 4. Testing the prediction service (Step 3)

**macOS / Linux**

```bash
# Health check
curl http://localhost:8080/health

# Prediction
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"age":35,"workclass":"Private","education":"Bachelors","education_num":13,"marital_status":"Married-civ-spouse","occupation":"Prof-specialty","relationship":"Husband","race":"White","sex":"Male","capital_gain":0,"capital_loss":0,"hours_per_week":45,"native_country":"United-States"}'
```

**Windows (PowerShell)**

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

Interactive API docs: **[http://localhost:8080/docs](http://localhost:8080/docs)**

---

## Troubleshooting

| Problem | macOS / Linux fix | Windows fix |
|---|---|---|
| Port 5000 already in use | `lsof -ti:5000 \| xargs kill` | `netstat -ano \| findstr :5000` then `taskkill /PID <pid> /F` |
| Port 8080 already in use | `lsof -ti:8080 \| xargs kill` | `netstat -ano \| findstr :8080` then `taskkill /PID <pid> /F` |
| `uv: command not found` | Re-run install script and restart shell | Restart PowerShell after install |
| `make: command not found` | Install via brew: `brew install make` | Use Git Bash, WSL, or raw Python commands (see table above) |
| `docker: command not found` | Install [Docker Desktop](https://docs.docker.com/get-started/) | Install [Docker Desktop](https://docs.docker.com/get-started/) |
| Docker build fails: model not found | Ensure MLflow server is running, then `make -f day4/Makefile train` before `docker-build` | Same |
| MLflow model not in Production stage | `make -f day4/Makefile train` (registers automatically) | Run train step first, then promote via UI if needed |
| `uv run` resolution error | Run `cd day4 && uv sync` first, then use `make` targets (they use `--frozen`) | Same |
