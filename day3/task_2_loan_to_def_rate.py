import pandas as pd
import plotly.express as px
from dotenv import dotenv_values

config = dotenv_values(".env")
_path = config['PATH_PHD_COURSE']

df = pd.read_parquet(f"{_path}/clean_loanbook.parquet")

# --- dates + vintage (monthly) ---
df = df.copy()
df["origination_date"] = pd.to_datetime(df["origination_date"])
df["charged_off"] = pd.to_datetime(df["charged_off"])

# monthly vintage as month-start timestamp (e.g., 1980-02-01)
df["vintage"] = df["origination_date"].dt.to_period("M")

# --- 1) number of loans issued per (vintage, sector) ---
issued = (
    df.groupby(["vintage", "sector"])["loan_id"]
      .nunique()
      .unstack("sector", fill_value=0)
      .sort_index()
)

# --- 2) number of those loans charged off within 1 year of origination ---
# window: [origination_date, origination_date + 1 year)
charged_off_1y_mask = (
    df["charged_off"].notna()
    & (df["charged_off"] >= df["origination_date"])
    & (df["charged_off"] < (df["origination_date"] + pd.DateOffset(years=5)))
)

charged_off_1y = (
    df.loc[charged_off_1y_mask]
      .groupby(["vintage", "sector"])["loan_id"]
      .nunique()
      .unstack("sector", fill_value=0)
      .sort_index()
)

# (optional) align columns/index so both frames have same shape
charged_off_1y = charged_off_1y.reindex(index=issued.index, columns=issued.columns, fill_value=0)

# output dataframes:
alives_df = issued            # counts issued at vintage
charged_off_df = charged_off_1y  # counts charged off within 1 year

alives_df.to_parquet(f"{_path}/alive_train.parquet")
charged_off_df.to_parquet(f"{_path}/target_train.parquet")

#plot
df = charged_off_df.div(alives_df)
df = df.dropna()
df.reset_index(inplace=True)

# Convert wide -> long for Plotly Express
df["vintage"] = pd.to_datetime(df["vintage"].astype(str))  # "1980-02" -> 1980-02-01

# --- wide -> long ---
df_long = df.melt(
    id_vars="vintage",
    var_name="sector",
    value_name="rate"
)

# --- plot ---
fig = px.line(
    df_long,
    x="vintage",
    y="rate",
    color="sector",
    title="Charged-off within 1Y / Issued (12M rolling mean) by sector",
    labels={"vintage": "Vintage", "rate": "Rate", "sector": "Sector"},
)

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    yaxis=dict(tickformat=".2%")  # remove if your rates are already 0-100
)

fig.show()
print()
