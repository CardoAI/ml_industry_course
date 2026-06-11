# Days 1–2 — Take-Home Assignment

## Loan Default Prediction — Competition + Report

**Course:** Machine Learning in Industry · Cardo AI / MSCA Digital Doctoral Network

**Modules:** Day 1 — Data Preprocessing & Feature Engineering · Day 2 — Tabular Data Modeling, Evaluation & Diagnostics

**Effort:** ~1 week (budget 15–20 hours)

---

## 1. The task

You work for a consumer-lending platform. You are given an **anonymized extract of
historical personal loans** with a binary outcome: did the loan end in default
(`target = 1`) or was it repaid in full (`target = 0`)?

Your job has two deliverables of equal seriousness:

1. **Predictions.** Produce a probability of default for every loan in a **scoring set
   you have no labels for**. We score your submission with **ROC-AUC against the true
   outcomes — exactly the same way the in-course competition
   ([homework_credit_competition.ipynb](homework_credit_competition.ipynb)) was scored.**
   Top score is the goal.
2. **A report.** A written account of *how you got there*: what you found in the data,
   what you decided and why, how you validated, what worked and what failed. A high
   score with an empty report fails the assignment — and so does a beautiful report
   wrapped around a careless model.

> The course's spine question: *"we trained a model" is not an acceptable answer.*
> The leaderboard rewards the model; the report is where you prove you understand
> *why* it scores what it scores.

---

## 2. Data

Both files live in [day2/assignment/data/](assignment/data/) and load directly with
`pd.read_csv(...)` (pandas handles `.gz` natively).

| File | Shape | Contents |
|---|---|---|
| `d2assignment_dataset.csv.gz` | 96,112 × 54 | `row_id`, 52 features, `target` — your training data |
| `d2assignment_test.csv.gz` | 23,888 × 53 | `row_id`, 52 features — **no target**; this is the scoring set |

Overall default rate in the training file: ≈ 14.4%. The classes are **imbalanced** —
that has consequences for how you validate and what you can read from raw accuracy.

The data has been anonymized: identifiers removed, units and codes recoded, categories
relabeled. The column names are descriptive of what each field *means*; the values are
not in their original units. As with any operational extract, not every field is
populated for every record.

### Column dictionary

**Loan application**

| Column | Description |
|---|---|
| `principal_req` | Requested principal amount |
| `pricing_rate` | Interest rate assigned to the loan |
| `monthly_payment` | Contractual monthly installment |
| `risk_band` | Internal risk grade (1 = best … 7 = worst) |
| `loan_reason_code` | Purpose of the loan (coded) |
| `app_kind` | Application type (coded) |

**Applicant**

| Column | Description |
|---|---|
| `emp_years` | Years in current employment |
| `housing_status` | Housing situation (coded) |
| `income_annual` | Self-reported annual income |
| `income_verif` | Income verification status (coded) |
| `region_code` | Geographic region (coded) |
| `area_code` | Sub-region area code (coded, nested in region) |
| `debt_ratio` | Debt-to-income ratio |

**Credit bureau profile**

| Column | Description |
|---|---|
| `bureau_score` | Credit bureau score at application |
| `credit_age_m` | Age of the credit file, months |
| `open_tradelines` / `total_tradelines` | Open / total credit accounts |
| `active_cards` | Active credit card accounts |
| `mortgage_ct` | Mortgage accounts |
| `new_accts_12m` / `new_accts_24m` | Accounts opened in the last 12 / 24 months |
| `inquiries_6m` | Credit inquiries in the last 6 months |
| `m_since_inquiry` | Months since the most recent inquiry |
| `m_since_card_open` | Months since the most recent card was opened |
| `m_since_delinq` | Months since the most recent delinquency |
| `delinq_24m` | Delinquencies in the last 24 months |
| `public_records` | Derogatory public records |
| `bankruptcies` | Public-record bankruptcies |
| `collections_12m` | Collections in the last 12 months (excl. medical) |
| `pct_clean_history` | % of tradelines never delinquent |
| `oldest_revolving_m` | Age of the oldest revolving account, months |
| `newest_account_m` | Months since the most recent account opening |

**Balances and limits**

| Column | Description |
|---|---|
| `revolving_bal` | Total revolving balance |
| `revolving_util` | Revolving utilization, % |
| `card_util` | Bankcard utilization, % |
| `card_headroom` | Unused bankcard credit |
| `pct_cards_hi_util` | % of bankcards above 75% utilization |
| `total_balance` / `avg_balance` | Total / average current balance across accounts |
| `total_credit_limit` | Total high credit / limit across accounts |
| `card_limit_total` | Total bankcard limit |
| `balance_ex_mortgage` | Total balance excluding mortgage |
| `install_limit_total` | Total installment high credit / limit |

**Account / servicing fields**

| Column | Description |
|---|---|
| `principal_recv` | Principal received on the account to date |
| `payments_total` | Total payments recorded on the account to date |
| `last_txn_amt` | Amount of the most recent transaction on the account |
| `late_fees` | Late fees accrued on the account to date |
| `residual_amt` | Residual amounts recovered on the account |
| `fee_adj` | Fee adjustments recorded on the account |
| `score_recent` | Most recent credit score on file |
| `review_gap_m` | Months between origination and the most recent account review |
| `account_flag` | Internal account status flag |

---

## 3. Submission

You submit **three things** (fork link or archive, instructor will specify the channel):

### 3.1 Predictions — `submission.csv`

A CSV with **exactly one row per `row_id` in the test file**:

```
row_id,predicted_probability
96,0.1234
178,0.0567
...
```

- `predicted_probability` = your P(default), a float in **[0, 1]**
- all 23,888 test `row_id`s present, no duplicates, no extras

Validate the format yourself before submitting — the same checks the grader runs are in
[score_submission.py](assignment/score_submission.py):

```bash
python day2/assignment/score_submission.py \
    --submission your_submission.csv --labels your_own_validation_labels.csv
```

(You don't have the test labels, but you can — and should — use the script to score
your model on a validation split you carve out yourself. The grader runs the identical
script against the held-back test labels.)

**One submission per student. Make it count.** Choose your final model with your own
validation evidence, not by guessing at the leaderboard.

### 3.2 Report — `REPORT.md` (or PDF, ~4–8 pages)

The report must let a skeptical reviewer retrace your path. Required sections:

1. **Data audit** — what you found when you profiled the data: distributions, missing
   values and their patterns, suspicious columns, anything that made you stop and
   think. Evidence (counts, plots), not adjectives.
2. **Preprocessing & feature decisions** — what you cleaned / encoded / imputed /
   engineered / **dropped**, and *why*. For decisions that mattered, show the measured
   effect on validation performance; for choices that turned out not to matter, say so.
3. **Validation design** — how you split, why that design gives you an honest estimate
   of performance on the unseen scoring set, and what could still make your estimate
   optimistic. State your expected test AUC **before** seeing the result.
4. **Modeling** — what you tried (start simple: a baseline and a linear model before
   anything fancy), how you tuned, how you picked the final model. Comparison table
   with uncertainty (mean ± std across folds), not single lucky numbers.
5. **What failed** — at least two things you tried that did not help, and your best
   explanation of why. A null result honestly reported earns marks; a report that only
   contains successes is not credible.
6. **Final model summary** — features used, hyperparameters, expected vs (if known)
   actual performance, and a short *"what I would not trust about this model"*
   paragraph.

### 3.3 Code + `HOW_TO_REPRODUCE.md`

Runnable code (notebook or scripts) that regenerates your `submission.csv` from the two
provided data files, plus the exact commands to do it from a clean checkout. Fix your
seeds (`SEED = 42`, `random_state` on every estimator) and pin your environment
(`requirements.txt` / `uv.lock` / `environment.yml`). Two runs must produce the same
submission file.

---

## 4. Rules

- **No external data.** Only the two provided files. No attempts to identify or
  reconstruct the original source of the data — that is out of scope and out of bounds.
- **No label leakage into your pipeline.** Every transformer (imputer, encoder, scaler)
  is fit on training data only. The test file may be used for *inspection* (you may
  look at its feature distributions — a production system sees its inputs too) but
  never for fitting anything that touches the target.
- **Any model, any library** (sklearn, LightGBM, XGBoost, CatBoost, …).
- **LLM / AI tools** allowed for scaffolding and debugging; every number and claim must
  reproduce from your committed code, and you must be able to defend every decision at
  the oral. Disclose where you used AI assistance.
- **One final submission.** No leaderboard probing.

---

## 5. Grading

| Component | Weight | What earns the marks |
|---|---|---|
| **Leaderboard score** | 50 | ROC-AUC on the held-back test labels, scored in bands — beating the field by chasing decimals matters less than being solidly in the top group. A submission that scores *below* the trivial baseline signals a process failure and caps this component. |
| **Report** | 40 | Audit depth and evidence; justified decisions with measured effects; honest validation design with a stated expectation; failures reported; limitations owned. |
| **Reproducibility** | 10 | Clean checkout → same `submission.csv`. Seeds, pinned env, working instructions. |

**Penalties:** invalid submission format (the scorer must accept it); claims in the
report not supported by committed code; non-reproducible results; external data.

**A warning worth its own line:** the single biggest determinant of your final score is
whether your validation estimate *transfers* to the unseen scoring set. The course
taught you why a model can look brilliant in-sample and collapse in production — and
how to catch it before it happens. Apply that. If your validation AUC and your final
test AUC disagree wildly, the explanation belongs in your report (the oral exam will
start there).

---

## 6. Hints

- **Profile before you model.** The Day 1 catalog
  ([first_eda.ipynb](../day1/first_eda.ipynb)) exists for a reason. This extract is
  messy in ways the course taught you to find — and not every useful check is about
  missing values.
- **For every feature, ask the Day 1–2 question: *would I know this value at the moment
  I have to make the prediction?*** Read the column dictionary carefully.
- **Compare the training file and the scoring file.** You have both. The course showed
  you more than one way to check whether two datasets look alike — and what it means
  for your validation design when they don't ([advanced_modeling.ipynb](advanced_modeling.ipynb), §3–4).
- **Baselines first.** A majority-class baseline and a logistic regression before any
  gradient boosting. If your fancy model can't beat the linear one, your time is better
  spent on features and data quality than on hyperparameters.
- **Trust mean ± std across folds, never one split.** Day 2 showed how much a single
  holdout AUC moves on its own.
- **Budget your week:** ~2 days data audit + preprocessing, ~2 days modeling +
  validation, ~1 day final selection + submission, ~1–2 days report. The report written
  in the last hour reads like it.

