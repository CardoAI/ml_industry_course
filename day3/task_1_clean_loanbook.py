import pandas as pd
from dotenv import dotenv_values

def clean_loanbook_sanity(df, *, drop_if_both_terminal_events=True, return_report=True):
    """
    Clean a noisy loan book without assuming which noise was introduced.

    Drops rows that violate basic timeline / completeness constraints, e.g.:
      - origination_date or maturity_date is null
      - maturity_date <= origination_date
      - charged_off < origination_date
      - charged_off > maturity_date
      - prepayment_date outside [origination_date, maturity_date]
      - (optional) both charged_off and prepayment_date populated (mutually exclusive terminal events)

    Parameters
    ----------
    df : pd.DataFrame
    drop_if_both_terminal_events : bool
        If True, drop rows where both charged_off and prepayment_date are present.
        (Set False if your data allows both for operational reasons.)
    return_report : bool
        If True, returns (clean_df, report_dict). Otherwise returns clean_df.

    """

    out = df.copy()

    # --- Parse date columns if present
    date_cols = ["origination_date", "maturity_date", "charged_off", "prepayment_date"]
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    # Helper masks (only evaluate comparisons when both sides non-null)
    has_orig = out["origination_date"].notna()
    has_mat  = out["maturity_date"].notna()

    invalid_orig_null = ~has_orig
    invalid_mat_null  = ~has_mat
    invalid_mat_le_orig = has_orig & has_mat & (out["maturity_date"] <= out["origination_date"])

    # charged_off checks
    if "charged_off" in out.columns:
        has_co = out["charged_off"].notna()
        invalid_co_before_orig = has_co & has_orig & (out["charged_off"] < out["origination_date"])
        invalid_co_after_mat   = has_co & has_mat  & (out["charged_off"] > out["maturity_date"])
    else:
        has_co = pd.Series(False, index=out.index)
        invalid_co_before_orig = pd.Series(False, index=out.index)
        invalid_co_after_mat   = pd.Series(False, index=out.index)

    # prepayment_date checks
    if "prepayment_date" in out.columns:
        has_pp = out["prepayment_date"].notna()
        invalid_pp_before_orig = has_pp & has_orig & (out["prepayment_date"] < out["origination_date"])
        invalid_pp_after_mat   = has_pp & has_mat  & (out["prepayment_date"] > out["maturity_date"])
    else:
        has_pp = pd.Series(False, index=out.index)
        invalid_pp_before_orig = pd.Series(False, index=out.index)
        invalid_pp_after_mat   = pd.Series(False, index=out.index)

    # Optional: terminal-event consistency
    invalid_both_terminal = (has_co & has_pp) if drop_if_both_terminal_events else pd.Series(False, index=out.index)

    # Optional: if both exist and one occurs after the other in an impossible way
    # (only applied when not dropping both; otherwise redundant)
    invalid_pp_after_co = pd.Series(False, index=out.index)
    if (not drop_if_both_terminal_events) and ("charged_off" in out.columns) and ("prepayment_date" in out.columns):
        invalid_pp_after_co = has_co & has_pp & (out["prepayment_date"] >= out["charged_off"])

    # Combine invalids
    invalid_any = (
        invalid_orig_null
        | invalid_mat_null
        | invalid_mat_le_orig
        | invalid_co_before_orig
        | invalid_co_after_mat
        | invalid_pp_before_orig
        | invalid_pp_after_mat
        | invalid_both_terminal
        | invalid_pp_after_co
    )

    clean = out.loc[~invalid_any].copy()

    #remove the data for Originator_A with a default rate = 1
    vintage_year = clean["origination_date"].dt.year
    target_block = (
            (clean["originator"] == "Originator_A")
            & vintage_year.between(1980, 1985, inclusive="both")
    )
    clean = clean.loc[~target_block].copy()


    if not return_report:
        return clean

    report = {
        "input_rows": int(len(out)),
        "output_rows": int(len(clean)),
        "dropped_rows": int(invalid_any.sum()),
        "dropped_pct": float(invalid_any.mean()),
        "reasons": {
            "origination_date_null": int(invalid_orig_null.sum()),
            "maturity_date_null": int(invalid_mat_null.sum()),
            "maturity_le_origination": int(invalid_mat_le_orig.sum()),
            "charged_off_before_origination": int(invalid_co_before_orig.sum()),
            "charged_off_after_maturity": int(invalid_co_after_mat.sum()),
            "prepayment_before_origination": int(invalid_pp_before_orig.sum()),
            "prepayment_after_maturity": int(invalid_pp_after_mat.sum()),
            "both_charged_off_and_prepayment": int(invalid_both_terminal.sum()),
            "prepayment_on_or_after_chargeoff": int(invalid_pp_after_co.sum()),
        },
    }
    return clean, report

if __name__ == '__main__':
    config = dotenv_values(".env")
    _path = config['PATH_PHD_COURSE']

    raw_df = pd.read_parquet(f"{_path}/raw_loanbook.parquet")
    clean_df, report = clean_loanbook_sanity(raw_df)
    clean_df.to_parquet(f"{_path}/clean_loanbook.parquet")