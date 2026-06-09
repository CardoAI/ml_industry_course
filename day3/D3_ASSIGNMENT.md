# Day 3 — Take-Home Assignment — Dependence Modeling & Portfolio Risk in Consumer Credit

## From a Raw Loan Tape to a Validated, Simulated Credit-Portfolio Risk View

**Course:** Machine Learning in Industry · Cardo AI / MSCA Digital Doctoral Network

**Module:** Copula-Based Portfolio Risk · Marginal & Dependence Modeling · Model Validation

**Effort:** ~2 weeks (budget 30–40 hours)

 **Submission:** see [Deliverables & Submission](#5-deliverables--submission)

---

## 1. Scenario

You have just joined the **Structured Credit & Risk Analytics desk** of a mid-sized European bank, *Meridian Capital Partners*. The desk is about to act as **arranger** on a new asset-backed security (ABS) collateralized by a pool of **unsecured personal consumer loans** originated across several regions. Before the deal can be marketed, the desk must deliver an **independent risk view** of the collateral pool to three audiences with very different concerns:

- **The originator**, who claims the book is "well diversified and low risk" and wants the lowest possible credit enhancement.
- **The investors** (senior and mezzanine note buyers), who want to know how much they can lose and under what stress.
- **The internal Model Validation unit**, who will sign off only if your dependence assumptions are *justified, tested, and documented* — not pulled from a textbook.

Your head of desk hands you a single CSV — the **loan-level tape** of the pool — and says:

> *"Here's the book. I don't care what model you saw in a paper — I care which one this data actually supports. Tell me how correlated these defaults are, what a bad year looks like, and where the pool is dangerously concentrated. And tell me where you might be fooling yourself. We present to the rating agency in two weeks."*

This is **not** a "fit a copula from the textbook" exercise. The tape contains realized `default_date` and `prepayment_date` fields because the originator seasoned the book before securitizing. Your job is to **reverse-engineer the risk**: recover the marginal timing of default, recover the **dependence structure** linking obligors, and **project the pool forward** to quantify portfolio loss. You do **not** know which marginal distributions, which copula, or which clustering of obligors generated this data. Discovering and *defending* those choices is where most of the marks live.

> Keep the three audiences above in mind throughout. Your final report explicitly answers the Model Validation unit and the rating agency: every modeling choice must be one you can defend.

---

## 2. Ground rules

- **Start from a fork** of the course repo. All work lives in your fork, ona branch (e.g. assignment-day3-<yourname>). Do not modify day1/ or day2/.
- **Work inside day3/**. You may add new files under day3/ (e.g. day3/assignment/) and you may edit day3/src/.
- **Reproducibility is graded**. Fix SEED = 42 everywhere, pin dependencies (the repo already uses uv.lock — do not loosen it), and make every result re-creatable from a command + a git SHA.
- **Dataset**: day3/generated/loan_tape.csv (raw CSV: yy × yy).
- **LLM / AI tools:** allowed for code scaffolding and debugging, but every number, plot, and claim in your report must be reproducible from your committed code, and you must be able to defend every design choice. State in your report where you used AI assistance.

---

## 3. What you are handed (the data)

You are given a single file, `loan_tape.csv`, with **one row per loan**. The *intended* schema is below — but be warned: this is a **raw production export**, and the actual file deviates from this specification in ways you must discover and fix (see Part 1). Field types, formats, domains, and logical relationships are **not** guaranteed to hold on the raw tape.

| Field | Type | Description |
|---|---|---|
| `loan_id` | int | Unique obligor/loan identifier |
| `issue_date` | date | Origination date |
| `term_months` | int | Contractual maturity in months |
| `issue_amount` | float | Original principal (EUR) |
| `interest_rate` | float | Fixed annual nominal rate |
| `amortization_type` | str | `french` (constant annuity), `bullet`, or `linear` |
| `region` | str | Origination region (e.g., `North`, `South`, `Center`, `Islands`, `Overseas`) |
| `fico` | int | Credit score at origination (300–850 scale) |
| `income_bracket` | str | `low` / `mid` / `high` |
| `default_date` | date or NaN | Date of default; `NaN` if never defaulted within observation window |
| `prepayment_date` | date or NaN | Date of full prepayment; `NaN` if not prepaid |
| `obs_end_date` | date | End of observation window (right-censoring date) |

> **Censoring.** Many loans neither defaulted nor prepaid by `obs_end_date`. These are **right-censored** observations and must be handled as such. Treating censored loans as "non-defaulters" is a modeling error you are expected to avoid and to discuss.

> **Competing risks.** Default and prepayment are **competing terminations**: a loan that prepays can no longer default, and vice versa.

The data were generated from a marginal-plus-copula data-generating process, with obligors grouped into latent **clusters**. The clustering scheme, the marginal families, and the copula/latent structure are **hidden from you**. Recovering them — and quantifying your uncertainty about them — is the assignment.

---

## 4. Tasks

Six parts. Parts 1–6 are all required. Point weights are a guide to effort and to grading. Each part ends with an explicit **Deliverable**.

### Part 1 — Data cleaning, EDA & data integrity (10 pts)

The file you receive is a **raw production tape**, not a curated teaching dataset. It was exported from the originator's loan-management system and concatenated across vintages and regional sub-ledgers. Like any real tape, it is **dirty**: it contains the kinds of defects that arise from manual entry, system migrations, inconsistent regional conventions, and reporting lags. **No modeling result is trustworthy until the tape is cleaned** — a model fitted on uncleaned data produces confident, wrong risk numbers, and Model Validation will catch it.

1. **Clean & validate (do this before anything else).** Profile the raw tape and build a defensible, *documented* cleaning pipeline: detect anomalies, decide a principled treatment for each (correct / impute / flag / drop), and **quantify the impact** of your decisions. At minimum, investigate:
   For each defect class report **how many records are affected**, your remediation rule, and a **before/after comparison** of the headline statistics (default rate, prepay rate, mean exposure) so the impact is transparent and auditable.
2. **Descriptive profile** (on the *cleaned* tape): distributions of `issue_amount`, `interest_rate`, `term_months`, `fico`, `region`, `income_bracket`. Report outstanding balance, not just issued amount, as of `obs_end_date` (you will need the schedule from step 4).
3. **Rates.** Quantify the **censoring rate**, **realized default rate**, and **realized prepayment rate**. Cross-tabulate each against `region`, `fico` bands, and `income_bracket`. Form an early hypothesis about which covariates drive risk.
4. **Amortization & EAD.** Reconstruct each loan's **amortization schedule** from `amortization_type`, `term_months`, `interest_rate`, `issue_amount` (constant annuity for `french`; equal principal for `linear`; interest-only + balloon for `bullet`). Compute **outstanding principal at any date** — this is the **exposure at default (EAD)** you need later. State assumptions (payment frequency, day-count).
5. **Critical question.** Beyond the mechanical defects above, what *structural* data-quality and survivorship issues could contaminate inference. List at least three and state the **direction** of each bias.

**Deliverable:** a clean, reproducible dataset; a short **data-quality log** (defect → count → rule → before/after impact); the EAD tables/plots; and the amortization/EAD function.

---

### Part 2 — Marginal estimation (15 pts)

1. **Time scale.** Define your time origin and scale — *calendar time* vs *loan age (months since issue)*. Justify the choice; it propagates everywhere downstream.
2. **Competing risks.** Treating **default** as the event of interest with **prepayment + administrative end** as competing/censoring, estimate the marginal **time-to-default** distribution. Handle right-censoring explicitly, and decide and justify whether you model default and prepayment as **competing risks** (cause-specific hazards / cumulative incidence) or treat one as **independent censoring** — and state the consequence of getting this wrong.
3. **Family selection.** Fit and compare parametric families per event (e.g., Exponential, Weibull, Log-Normal, Log-Logistic, Gamma, Gompertz) plus a **non-parametric** benchmark (Kaplan–Meier / Aalen–Johansen). Select using information criteria (AIC/BIC) **and** fit diagnostics (QQ vs fitted, cumulative-hazard plots).
4. **Covariates vs segments.** Decide whether marginals should be **covariate-dependent** (AFT or Cox on `fico`, `region`, `income_bracket`, `interest_rate`) or **segment-specific** (one marginal per cluster). You revisit this in Part 4 — the two choices interact.

**Deliverable:** the per-event family comparison (tables + diagnostic plots) and the selected marginals with parameters.

---

### Part 3 — Copula selection & dependence (40 pts) — *the core*

1. **Dependence axis.** State explicitly *what* is being coupled and pick/justify one (or contrast both):
   - **(i)** within-obligor dependence between time-to-default;
   - **(ii)** **cross-obligor** dependence of default timing driven by a shared/systematic factor — the one that matters for portfolio tail risk.

   Your portfolio risk in Parts 5–6 hinges on (ii); keep the modeling choice consistent.
2. **Specify & justify the latent dependence structure.** Do **not** pick a copula off the shelf — first decide *what generates the co-movement*, then let that drive the family. Articulate and defend the **latent architecture**, e.g.:
   - a **single-factor** structure (one common systematic driver loading on all obligors — the credit-portfolio workhorse) vs a **multi-factor** structure (region- or FICO-specific factors, possibly plus a global factor);
   - **flat / exchangeable** dependence vs **hierarchical / nested** dependence (segments correlated internally and across segments through a higher-level factor);
   - the **tail behavior** the structure implies (symmetric vs asymmetric tail dependence) and *why it suits default co-movement*.

   Compare competing latent specifications (e.g., one-factor vs two-factor vs nested/hierarchical), estimate each by a justified method (IFM / two-step or full MLE), and **report parameters with uncertainty**. Explain the reasoning: how many factors, what each represents economically, how they map onto your Part 4 segmentation, and what you lose by going simpler or more complex. The copula *family* follows from this structural argument, not the reverse.
3. **Goodness of fit.** Select among your latent structures (and the family each implies) using a *combination* of: log-likelihood, AIC, BIC; a formal GoF test (Cramér–von Mises / Rosenblatt-transform-based, or parametric-bootstrap GoF); **empirical vs fitted tail-dependence** coefficients ($\lambda_L, \lambda_U$) — argue that tail dependence, not central fit, is the decision-relevant criterion for credit; and visual diagnostics ($\chi$-plots, K-plots, empirical-vs-model exceedance).

**Deliverable:** the competing-structure comparison (parameters + uncertainty), the GoF battery (tests + tail-dependence + plots), and the selected structure with its economic rationale.

---

### Part 4 — Clustering / segmentation (10 pts)

1**Critical question.** The originator claims the pool is "diversified by region." Using your fitted dependence, evaluate whether regional diversification actually reduces tail risk or whether a shared systematic factor undermines it. Quantify the diversification benefit (or lack thereof).

**Deliverable:** Diversification analysis.

---

### Part 5 — Monte Carlo portfolio loss & risk measures (20 pts)

You now have fitted marginals, a selected latent structure, a segmentation, and per-loan EAD. Build the simulation engine.

1. **Simulation design.** For a chosen risk horizon $H$ (state it — e.g. 12 / 24 / 36 months): (a) draw correlated uniforms $\mathbf{U}$ from the fitted copula across obligors (respecting segmentation / common factor); (b) map each $u_i$ through the inverse marginal to a simulated time-to-default; (c) determine whether default occurs **before** $H$; (d) compute $\text{Loss}_i = \text{EAD}_i(\tau_i)\times\text{LGD}$ using the **outstanding balance at the simulated default time** (state and justify your LGD — fixed is acceptable if defended, stochastic earns more); (e) aggregate $L=\sum_i \text{Loss}_i$. Repeat for $N$ paths, and **justify $N$** via the Monte Carlo standard error on the tail quantile.
2. **Risk measures.** From the simulated loss distribution, at $\alpha \in \{95\%, 99\%, 99.9\%\}$, compute **Expected Loss (EL)**, **VaR$_\alpha$**, **Expected Shortfall ES$_\alpha$** (CVaR), and **Economic Capital** $= \text{VaR}_\alpha - \text{EL}$. Report Monte Carlo confidence intervals, especially on the 99.9% quantile.
3. **Marginal & Component VaR.** Define and compute **Marginal VaR** ($\partial\text{VaR}/\partial w_i$) and **Component VaR** ($\text{CVaR}_i = w_i\,\partial\text{VaR}/\partial w_i$, verifying $\sum_i \text{CVaR}_i = \text{VaR}$, Euler allocation) per loan/segment. Explain how to estimate these from the *same* simulation (tail-conditional estimators) without re-running, and discuss estimator noise.
4. **Critical question.** Compare your copula-based VaR to a **naïve independence** benchmark (same marginals, independent obligors). Quantify how much capital independence would *understate* — this number is the economic value of the entire dependence-modeling exercise. Interpret it for the investor audience.

**Deliverable:** the simulation code (seeded), the loss-distribution plot, the risk-measure table with MC confidence intervals, the MVaR/CVaR attribution, and the independence-benchmark comparison.

---

### Part 6 — Concentration, risk attribution & critique (5 pts)

1. **Concentration indices.** Using the Component-VaR / MVaR output, construct indices beyond a simple exposure-weighted HHI: a **risk-based HHI** on Component-VaR shares (vs the EAD-based HHI); a **diversification ratio** ($\sum_i$ standalone risk / portfolio risk); and **name/segment concentration** (share of 99.9% tail loss from the top-$k$ obligors or each `region × fico` segment).
2. **Locate the risk.** Identify *where* the portfolio is most dangerous — which segment(s)/region(s)/FICO band(s) contribute disproportionately to tail loss relative to their exposure share. Produce a clear risk-attribution table/figure and reconcile it with your Part 1 hypothesis: did the data confirm it?
3. **Final critique (mandatory, heavily weighted).** Address candidly:
   - **Model risk** — sensitivity of EL, VaR$_{99.9}$, ES to (a) marginal family, (b) latent structure / copula, (c) segmentation, (d) LGD. Provide a sensitivity table.
   - **Estimation risk** — with this much censoring, how reliable is the *tail* of your fitted copula? Could you even identify tail dependence at this sample size? Propose how you'd quantify parameter uncertainty (e.g., bootstrap the whole pipeline).
   - **Gap to a real use case** — what does production credit risk modeling have that this omits? Cover at least: point-in-time vs through-the-cycle / macro conditioning, stochastic & downturn LGD, dynamic prepayment, regulatory frameworks (Basel IRB, IFRS 9 ECL), macro covariate data, model governance & backtesting, and the fact that *you assumed a copula generated this data* whereas reality offers no such guarantee.
   - **Recommendation to the desk** — in one paragraph, what credit enhancement / capital would you advise, and with what caveats?

**Deliverable:** the concentration indices, the risk-attribution table/figure, and the critique (model risk + estimation risk + gap-to-reality + recommendation).

---

## 5. Deliverables & Submission

Submit a link to your fork (branch) containing:

1. **`day3/assignment/REPORT.md`** (or PDF) — the written report tying everything together, addressed to the **desk head and Model Validation**: justify every choice. Must end with a **one-page Executive Summary** for the investor audience (EL, economic capital at 99.9%, top concentration, headline caveat).
2. **Code** for every part (cleaning pipeline + data-quality log, marginal fitting, copula/latent-structure fitting + GoF, segmentation comparison, the Monte Carlo engine, concentration/attribution), runnable end-to-end from the **raw** `loan_tape.csv` and **seeded**.
3. **The data-quality log** and the **clean dataset** (or the command that regenerates it).
4. **Figures & tables** referenced in the report, regenerated by the committed code.
5. A short **`HOW_TO_REPRODUCE.md`**: the exact commands (in order) to regenerate every result from a clean checkout, including the seed.

Recommended tooling: Python (`lifelines` / `scikit-survival`, `copulas` / `statsmodels` / `scipy`, `numpy`). Use any library, but understand and defend what it does.

Keep prose tight. A reviewer should be able to follow your reasoning and re-run your results without talking to you.

---

## 6. Grading rubric

| Part | Pts         | What earns the marks |
|---|-------------|---|
| 1 — Data cleaning & EDA | 10          | Defects detected and correctly classified; censoring **not** confused with error; auditable data-quality log with before/after impact; correct amortization/EAD |
| 2 — Marginals | 15          | Explicit censoring & competing-risks handling; ≥3 families + non-parametric benchmark; selection on IC **and** diagnostics; valid PIT check |
| 3 — Copula & latent structure | 40          | Pseudo-obs under censoring; **justified latent architecture (factor count / hierarchy) with economic reasoning**; ≥3 specifications compared; **tail-aware** GoF; AIC-vs-tail demonstration |
| 4 — Segmentation | 10          | Hypothesis ladder; penalized / out-of-sample comparison; overfitting controlled; inter-segment / diversification analysis |
| 5 — Simulation & risk measures | 20          | Correct, seeded MC respecting the copula; EL/VaR/ES with MC CIs; correct MVaR & Component VaR (Euler additivity); independence-benchmark comparison |
| 6 — Concentration & critique | 5           | Risk-based concentration indices; clear attribution; honest model-risk, estimation-risk & gap-to-reality critique; defensible recommendation |
| Report quality | (folded in) | Clarity, honesty about limitations, the investor Executive Summary |

**Penalties:** results that don't reproduce from your commands; hand-edited data instead of a reproducible cleaning pipeline; censoring treated as missing data (or vice versa); pseudo-observations built ignoring censoring; a latent structure asserted without justification; claims unsupported by committed code/artifacts.

> A technically flawless report that uncritically trusts its own 99.9% number will **not** earn top marks. The senior desk values the analyst who knows where the model breaks.

---