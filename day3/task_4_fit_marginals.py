import pandas as pd
import numpy as np
from operator import attrgetter
from lifelines import GeneralizedGammaFitter
import plotly.graph_objects as go
import json
from dotenv import dotenv_values


def from_loan_book_to_surv_df(loan_book_df, today="2026-01-01"):
    """
    Returns a dataframe with columns:
      - durations (in months, integer)
      - event_observed (bool): True if charged_off/default_date is present
      - cluster_id (sector as string)

    Notes:
    - Uses vectorized datetime parsing (no .apply / lambdas).
    - Keeps dates as Period[M] for clean month arithmetic.
    """
    df = loan_book_df.copy()

    # Parse to monthly Periods (NaT stays NaT; no need for negative-int cleanup)
    orig = pd.to_datetime(df["origination_date"], errors="coerce").dt.to_period("M")
    mat = pd.to_datetime(df["maturity_date"], errors="coerce").dt.to_period("M")
    dflt = pd.to_datetime(df["charged_off"], errors="coerce").dt.to_period("M")

    # Drop loans missing origination or maturity
    mask_valid = orig.notna() & mat.notna()
    orig = orig[mask_valid]
    mat = mat[mask_valid]
    dflt = dflt[mask_valid]
    sector = df.loc[mask_valid, "sector"]

    # "today" as monthly Period scalar
    today_p = pd.Period(pd.to_datetime(today), freq="M")
    today_s = pd.Series(today_p, index=orig.index)

    # lifetime = min(maturity, default, today) per row
    lifetime = pd.concat([mat.rename("maturity"),
                          dflt.rename("default"),
                          today_s.rename("today")],
                         axis=1).min(axis=1, skipna=True)

    # durations in months (Period difference is integer count of months)
    durations = (lifetime - orig).map(attrgetter("n")).astype("int64")

    out = pd.DataFrame({
        "durations": durations,
        "event_observed": dflt.notna().to_numpy(),
        "cluster_id": sector.astype(str).to_numpy(),
    }, index=orig.index)

    return out


def _extract_ci_bounds(ci_df: pd.DataFrame):
    """
    Return (lower_series, upper_series) from a lifelines CI dataframe, trying to be robust
    to column naming differences across lifelines versions.
    """
    cols = [c.lower() for c in ci_df.columns]

    # common patterns: "*_lower_0.95", "*_upper_0.95" or "lower-bound"/"upper-bound"
    lower_candidates = [ci_df.columns[i] for i, c in enumerate(cols) if "lower" in c]
    upper_candidates = [ci_df.columns[i] for i, c in enumerate(cols) if "upper" in c]

    if not lower_candidates or not upper_candidates:
        raise ValueError(f"Could not infer CI bound columns from: {list(ci_df.columns)}")

    lower_col = lower_candidates[0]
    upper_col = upper_candidates[0]
    return ci_df[lower_col], ci_df[upper_col]


def fit_generalized_gamma_by_cluster(
        df: pd.DataFrame,
        duration_col="durations",
        event_col="event_observed",
        cluster_col="cluster_id",
        alpha=0.05,
        n_timeline=200,
):
    # Basic cleaning / typing
    d = df[[duration_col, event_col, cluster_col]].dropna().copy()
    d[duration_col] = pd.to_numeric(d[duration_col], errors="coerce")
    d = d.dropna(subset=[duration_col])
    # add small value to 0 durations
    d.loc[d[duration_col] <= 0, duration_col] = 0.01
    d[event_col] = d[event_col].astype(bool)

    t_max = float(d[duration_col].max())
    timeline = np.linspace(0, t_max, n_timeline)

    models = {}
    curves = {}  # cluster -> (survival, lower, upper)

    for cluster, g in d.groupby(cluster_col):
        gg = GeneralizedGammaFitter(alpha=alpha)
        try:
            gg.fit(
                durations=g[duration_col].values,
                event_observed=g[event_col].values,
                timeline=timeline,
                label=str(cluster),
            )
        except Exception as e:
            print(f"[WARN] Cluster '{cluster}': could not fit GeneralizedGammaFitter: {e}")
            continue

        # Survival curve
        sf = gg.survival_function_.iloc[:, 0]  # series indexed by timeline

        # Confidence interval for survival function (available in most recent lifelines)
        if hasattr(gg,
                   "confidence_interval_survival_function_") and gg.confidence_interval_survival_function_ is not None:
            ci_df = gg.confidence_interval_survival_function_
            lower, upper = _extract_ci_bounds(ci_df)
        else:
            # If your lifelines version doesn't expose CI for survival function, you can:
            #  - upgrade lifelines, OR
            #  - implement bootstrap CIs here.
            raise RuntimeError(
                "Your lifelines version does not provide confidence_interval_survival_function_. "
                "Please upgrade lifelines (recommended) or add a bootstrap CI implementation."
            )

        models[cluster] = gg
        curves[cluster] = (sf, lower, upper)

    return models, curves, timeline


def plot_survival_curves_plotly(curves, alpha=0.05, title="Generalized Gamma Survival by cluster"):
    fig = go.Figure()

    for cluster, (sf, lower, upper) in curves.items():
        x = sf.index.values.astype(float)
        y = sf.values.astype(float)
        lo = lower.values.astype(float)
        up = upper.values.astype(float)

        # CI band (fill between upper and lower)
        fig.add_trace(
            go.Scatter(
                x=x, y=up,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=f"{cluster} CI upper",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.12)",  # neutral band; change if you want per-cluster colors
                showlegend=False,
                hoverinfo="skip",
                name=f"{cluster} CI lower",
            )
        )

        # Survival function
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(width=2),
                name=f"{cluster} (CI {1 - alpha:.0%})",
                hovertemplate="t=%{x:.2f}<br>S(t)=%{y:.4f}<extra>" + str(cluster) + "</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Survival probability S(t)",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        legend_title_text="Cluster",
    )
    return fig


# --------------------
# Usage (df is your dataframe with columns: durations, event_observed, cluster_id)
# --------------------
config = dotenv_values(".env")
_path = config['PATH_PHD_COURSE']

df = from_loan_book_to_surv_df(
    pd.read_parquet(f"{_path}/clean_loanbook.parquet")).sample(frac=0.9)
models, curves, timeline = fit_generalized_gamma_by_cluster(df, alpha=0.05)
#save parameters
res = {}
for k, v in models.items():
    res[k] = {
        "lambda": v.lambda_,
        "ln_sigma": v.ln_sigma_,
        "mu_": v.mu_,
    }

#save
with open(f"{_path}/survival_model_params.json", "w") as f:
    json.dump(res, f)

#plot
fig = plot_survival_curves_plotly(curves, alpha=0.05)
fig.show()
