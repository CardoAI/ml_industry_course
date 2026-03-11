# Environment Setup

## Prerequisites

- **Python 3.13+** — download from [python.org](https://www.python.org/downloads/)
- **uv** (recommended) or **pip**

Install uv if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install and Run

```bash
# Clone the repository
git clone <repo-url>
cd ml_industry_course

# Install dependencies
uv sync

# Register the environment as a Jupyter kernel
uv run python -m ipykernel install --user --name ml-industry-course --display-name "ML Industry Course"

# Launch JupyterLab
uv run jupyter lab
```

If you prefer plain pip:

```bash
pip install -e .
python -m ipykernel install --user --name ml-industry-course --display-name "ML Industry Course"
jupyter lab
```

## Verify

1. In JupyterLab, open any notebook.
2. Select the **"ML Industry Course"** kernel (top-right corner or *Kernel > Change Kernel*).
3. Run the first cell. If no import errors appear, you're good to go.
