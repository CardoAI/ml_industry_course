# Day 4 — Take-Home Assignment

## From Notebook to a Reproducible, Monitored, Audited ML Service

**Course:** Machine Learning in Industry · Cardo AI / MSCA Digital Doctoral Network

**Module:** MLOps · Reproducibility · Model Monitoring

**Effort:** ~1 week (budget 15–20 hours)

 **Submission:** see [Deliverables & Submission](#5-deliverables--submission)

---

## 1. Scenario

You have just joined the ML team at a consumer-lending fintech. A colleague left
behind a notebook that trains an income-classification model on the *Adult Income*
dataset (here a proxy for an income-verification / affordability signal feeding a
credit decision). The model "works on their machine." Your job is to turn it into
something the company — and a regulator — can actually trust.

Concretely, you will take the Day 4 pipeline and (a) select a model *with statistical
rigor*, (b) document it for governance, (c) design and stress-test a monitoring
system, and (d) ship it behind a CI gate that blocks bad models from deploying.

This is **not** a "re-run the workshop notebooks" exercise. The worked notebooks
(`01`, `02`) and the workshop skeleton (`03`) are your *toolbox*. The assignment
asks you to **compose, extend, and reason about** those tools, and to write up the
decisions an MLOps engineer is accountable for. Re-running cells earns few points;
the analysis and justification earn most of them.

> Re-read the three failure stories in [D4_SLIDES.md](D4_SLIDES.md) (slides 4–6). Each
> part of this assignment closes one of those gaps. Keep them in mind — your final
> report explicitly answers the regulator from Failure Story 3.

---

## 2. Ground rules

- **Start from a fork** of the course repo. All work lives in your fork, on a branch
  (e.g. `assignment-<yourname>`). Do **not** modify `day1/` or `day2/`.
- Work inside `day4/`. You may add new files under `day4/` (e.g. `day4/assignment/`)
  and you may edit `day4/src/`, the `Dockerfile`, `baseline_metrics.json`, and
  `.github/workflows/ml-pipeline.yml`. Keep the existing scripts importable.
- **Reproducibility is graded.** Fix `SEED = 42` everywhere, pin dependencies
  (the repo already uses `uv.lock` — do not loosen it), and make every result
  re-creatable from a command + a git SHA.
- Dataset: `day1/generated/adult_income_issues.csv` (raw CSV: 9,271 × 33).
  `load_data` in [src/train.py](src/train.py) adds `target`, so the modeling frame
  is 9,271 × 34. Target `class` → `target` (0/1); entity column `person_id`; split
  column `split`. **Splits must be entity-aware** (see `split_data` in
  [src/train.py](src/train.py)).
- **LLM / AI tools:** allowed for code scaffolding and debugging, but every number,
  plot, and claim in your report must be reproducible from your committed code, and
  you must be able to defend every design choice orally. State in your report where
  you used AI assistance.

---

## 3. What you are handed (your toolbox)

| File | What it gives you |
|---|---|
| [src/train.py](src/train.py) | `train_and_log`, `train_and_log_cv` (GroupKFold), MLflow logging, registry promotion |
| [src/drift_check.py](src/drift_check.py) | `compute_psi`, `run_psi_check`, Evidently univariate drift, Bonferroni helper, CI-gate `check_thresholds` |
| [src/predict.py](src/predict.py) | FastAPI service (`/health`, `/predict`) |
| [notebooks/01–03](notebooks/) | Worked MLflow + drift demos (`01`, `02`) and the workshop skeleton (`03`, including TODOs for domain classifier, PCA-Wasserstein, concept-drift, adversarial injection) |
| [Dockerfile](Dockerfile) · [docker-compose.yml](docker-compose.yml) · [Makefile](Makefile) | Containerization + commands |
| [.github/workflows/ml-pipeline.yml](../.github/workflows/ml-pipeline.yml) | CI: train → champion/challenger gate → drift check → upload |
| [baseline_metrics.json](baseline_metrics.json) | Champion/challenger baseline (currently a placeholder `0.5` — you will set it properly) |

Setup instructions are in [D4_README.md](D4_README.md). Confirm `make -f day4/Makefile mlflow-server` and `make -f day4/Makefile train` work before you start Part 1. If working on a Windows machine, remember that you can still use WSL to run the `make` commands, including MLflow server and the training commands.

---

## 4. Tasks

Five parts. Parts 1–4 are required; Part 5 is a bonus. Point weights are a guide to
effort and to grading.

### Part 0 — Reproducibility baseline (5 pts)

1. Fork, set up the environment, start the MLflow server, and run one training run.
2. Run the **identical** training command twice. Confirm the validation metrics are
   bit-for-bit identical. If they are not, find and fix the source of nondeterminism.
3. Produce a **reproducibility manifest** (a small JSON or table) capturing: `SEED`,
   git commit SHA, `uv.lock` hash, a content hash of the data file, Python version,
   and the resulting `val_auc`/`val_f1`. Log it to MLflow as an artifact.

**Deliverable:** the manifest + a 3–5 sentence note explaining *why* each field is
needed to reproduce the run (tie back to the five reproducibility practices, slides 10–15).

---

### Part 1 — Model selection with variance awareness (25 pts)

The notebook reports a single `val_auc` from one split. That is "noise mining"
(slides 15b–15c). Do it properly.

1. **Sweep** at least **8** hyperparameter configurations (vary `n_estimators`,
   `learning_rate`, `max_depth`, `num_leaves`). Use **GroupKFold CV**
   (`train_and_log_cv`, grouped by `person_id`). Report **mean ± std AUC** per config
   in a table, all logged to one MLflow experiment.
2. For your top-2 configs, estimate the **uncertainty of the test-set AUC**:
   - Compute the Hanley–McNeil analytic SE (formula on slide 15b), **and**
   - Compute a **bootstrap 95% CI** on the AUC *difference* between the two configs
     (resample test rows with replacement, ≥1,000 iterations).
   - State plainly: **is the difference statistically distinguishable from zero, or is
     it within the noise?** Justify your conclusion.
3. **Multiple comparisons:** you evaluated *k* configs. Estimate the probability that
   your apparent winner is a false winner (slide 15c). Then apply a principled
   selection rule — the **one-standard-error rule** *or* a Bonferroni-adjusted
   comparison — and pick your final model. Defend the choice in 2–4 sentences.
4. Register the chosen model in the MLflow Model Registry under the `@champion` alias.

**Deliverable:** the sweep table, a CI plot (AUC ± CI per top config), and a
~1-page justification of your final pick. The grade is mostly on the *reasoning*,
not on getting the highest AUC.

---

### Part 2 — Governance: subgroup performance, calibration & a Model Card (20 pts)

A regulator will ask not just "how good is it" but "good for whom, and how trustworthy
are the scores."

1. **Subgroup performance:** compute test AUC (and one more metric of your choice:
   FPR, FNR, or selection rate) **per group** for at least `marital_status` and
   `relationship` -the current generated CSV does not contain `sex` or `race`, which you may still find in some of the Day 4 example notebooks-. Report the gaps. Log per-subgroup metrics to MLflow.
2. **Calibration:** report Brier score and a reliability diagram for the champion.
   Comment on whether the predicted probabilities can be read as real probabilities,
   and what that means for a downstream lending threshold.
3. **Model Card** (1–2 pages) covering: intended use & out-of-scope use, training
   data and its known issues, headline + subgroup metrics, calibration, limitations,
   and ethical/fairness considerations. Treat it as a document you would hand to a
   model-risk reviewer.
4. **Audit trail:** describe the registry promotion (which run ID became champion,
   when, and what evidence justified promotion). Connect this to the
   Staging→Production lifecycle (slide 23).

**Deliverable:** the Model Card + the subgroup/calibration results (tables + plots),
committed and logged to MLflow.

---

### Part 3 — Monitoring design and drift stress-test (30 pts) — *the core*

The workshop showed *individual* drift tools. Here you design a **monitoring system**
and **characterize when it works and when it fails**.

1. **Reference vs analysis design.** State precisely what your reference and analysis
   sets are and *why* (slide 31). Justify the choice operationally, not just statistically.

2. **Build a drift simulator.** Write a function that injects, into a copy of the
   analysis set, each of the three drift types from slide 26 at **tunable severity**:
   - **Covariate shift** — shift the distribution of one or more input features (P(X)).
   - **Prior probability shift** — change the base rate of the positive class (P(Y)).
   - **Concept drift** — change the feature→label relationship (P(Y|X)) while leaving
     the marginal feature distributions ~unchanged.

3. **Run the full detector suite** on each (drift type × at least 3 severity levels):
   feature PSI, **score PSI**, univariate drift (with a multiple-testing correction),
   and a **domain-classifier** multivariate test (notebook `03`, TODO section 2i,
   or notebook `02`, section 7C). Where you
   have labels, also report the *true* AUC degradation.

4. **Produce the detection matrix.** A table with rows = (drift type × severity) and
   columns = each detector's verdict (fired / silent) plus realized AUC. **Explicitly
   demonstrate the concept-drift blind spot**: a setting where every input-based monitor
   stays green while real AUC drops (slide 29c). Discuss the operational consequence.

5. **Calibrate your thresholds — don't just inherit the defaults.** With **no drift
   injected**, resample the analysis set many times and estimate the **false-alarm rate**
   of your chosen thresholds (score-PSI cutoff, max-drifted-features, corrected α).
   Pick thresholds that trade off sensitivity vs false alarms, and justify them with
   your measured numbers. Note explicitly how multiple testing across ~19 features
   inflates false alerts if uncorrected (slide 29).

**Deliverable:** a **Monitoring Plan** (1–2 pages: reference/analysis design, the
metrics you monitor, thresholds + their measured false-alarm rates, and the documented
limits — i.e. what your monitoring *cannot* catch) plus the detection matrix and the
code that produced it.

---

### Part 4 — Serving and a working CI gate (20 pts)

1. **Serve.** Build the Docker image and run the prediction service. Show a successful
   `/health` and at least two `/predict` calls (one likely-positive, one likely-negative
   profile). Briefly explain the baked-in-model vs `MODEL_URI`-override design (slide 36).
2. **Make the CI gate real.** Currently [baseline_metrics.json](baseline_metrics.json)
   is a `0.5` placeholder and the drift step does not block. You will:
   - Set `baseline_metrics.json` to your champion's actual metrics with a sensible
     `tolerance`, so the **champion/challenger gate** is meaningful.
   - Turn the drift check into a **hard gate** (`--fail-on-drift`) with the thresholds
     you justified in Part 3.
3. **Demonstrate the gate firing both ways.** Produce **two** CI runs on your fork:
   - one that **passes** (healthy model + no drift), and
   - one that **fails** — caused either by a deliberately worse challenger (regressed
     AUC) **or** by feeding a drifted analysis set that breaches your threshold.
   Capture the Actions run links/screenshots and explain in 3–5 sentences exactly which
   gate fired and why.

**Deliverable:** the two CI run links/screenshots, your edited workflow +
`baseline_metrics.json`, and the serving evidence.

---

### Part 5 — Close the loop (bonus, up to +10 pts)

Implement the retrain-on-drift loop sketched on slide 40: a script (or workflow job)
that, on a drift signal, **retrains**, runs the **champion/challenger** comparison, and
**promotes the challenger only if it beats the champion** beyond tolerance — otherwise
keeps the champion and logs the decision. A cron trigger and a notification stub
(printed/logged "alert") are welcome. Document the control flow with a small diagram.

---

## 5. Deliverables & Submission

Submit a link to your fork (branch) containing:

1. **`day4/assignment/REPORT.md`** (or PDF) — the written report tying everything
   together. Must include, at the end, a **one-page "Regulator Q&A"** answering the five
   questions from Failure Story 3 (slide 6): which model version is in production, what
   data/version trained it, deployment-time validation metrics, monitoring evidence, and
   what changed between versions. Answer them *using your own run IDs and artifacts*.
2. **Code** for every part (selection sweep, drift simulator + detector suite,
   threshold-calibration script, any retrain-loop code), runnable and seeded.
3. **The Model Card** and the **Monitoring Plan** (may be sections of the report).
4. **MLflow evidence** — either a committed `mlflow.db` / `mlruns/` export or screenshots
   of the relevant experiments, registry, and logged artifacts.
5. **Two CI run links/screenshots** (pass + fail).
6. A short **`HOW_TO_REPRODUCE.md`**: the exact commands (in order) to regenerate your
   results from a clean checkout.

Keep prose tight. A reviewer should be able to follow your reasoning and re-run your
results without talking to you.

---

## 6. Grading rubric

| Part | Pts | What earns the marks |
|---|---|---|
| 0 — Reproducibility baseline | 5 | Identical reruns; complete, justified manifest |
| 1 — Model selection | 25 | Correct CV; valid CIs/SE; honest "within noise?" call; principled multiple-comparison handling; defensible final pick |
| 2 — Governance | 20 | Correct subgroup + calibration analysis; a genuine, reviewer-ready Model Card; clear audit trail |
| 3 — Monitoring | 30 | Working 3-type drift simulator; complete detection matrix; **demonstrated concept-drift blind spot**; **threshold false-alarm calibration with measured numbers** |
| 4 — Serving + CI gate | 20 | Service runs; meaningful baseline; gate demonstrated **both** passing and failing, correctly explained |
| 5 — Retrain loop | +10 | Correct promote-only-if-better logic, documented |
| Report quality | (folded in) | Clarity, honesty about limitations, the Regulator Q&A |

**Penalties:** results that don't reproduce from your commands; loosened/unpinned
dependencies; non-entity-aware splits (leakage); unfixed seeds; claims unsupported by
committed artifacts.

---

## 7. Hints & pitfalls

- **Entity leakage** is the classic trap: split by `person_id`, never by row (slide 14).
  Your CV already does this via `GroupKFold` — keep it that way in any new split.
- **PSI direction:** distance-based Evidently tests flag drift when value ≥ threshold;
  p-value tests when value ≤ threshold. The handling is in `run_feature_drift`
  ([src/drift_check.py](src/drift_check.py)) — read it before extending it.
- **Bonferroni helper scope:** `apply_bonferroni` in
  [src/drift_check.py](src/drift_check.py) logs the corrected alpha and family-wise
  error-rate calculation. It does not filter Evidently results by p-value; implement
  that explicitly if your detector suite needs corrected p-value decisions.
- **Concept drift is invisible to PSI/KS.** If your concept-drift injection moves the
  feature distributions, you have built covariate shift instead. Verify P(X) is ~unchanged
  (PSI ≈ 0) while AUC drops — that's the whole point of Part 3.4.
- **Bootstrap the *difference*, not two separate CIs.** Two overlapping marginal CIs do
  not imply the difference is insignificant. Resample once, compute both AUCs on the same
  resample, take the difference.
- **CI `MLFLOW_TRACKING_URI`:** `file:./mlruns` (single slash, relative) on the runner —
  see the comments in the workflow. Don't change it to a server URI.
- **Make the failing CI run intentional and explained** — a red ✗ you can't account for
  is worth less than a red ✗ you predicted.

---
