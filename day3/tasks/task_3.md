# Task

Using the **same dataset** as in the maximum-likelihood task (same observations $k = 1,\ldots,K$ and the same covariates/features, if any), fit survival models in `lifelines` and select the best-performing model.

1. **Model fitting.** Prepare the data with a duration (time-to-event) variable and an event indicator ($1$ = event occurred, $0$ = right-censored). Fit one or more candidate survival models (e.g., Cox PH, Weibull AFT, Log-logistic AFT).

2. **Choose and justify an evaluation metric.** Evaluate and compare models using an appropriate metric, such as:

   - **Concordance index (C-index)** for predictive discrimination (higher is better), and/or
   - **AIC/BIC** for model comparison (lower is better) for parametric/AFT models.

3. **Confidence intervals.** Report parameter estimates together with confidence intervals at a chosen level (default: $95\%$, i.e. $\alpha = 0.05$). Where relevant, include confidence bands for estimated survival curves.