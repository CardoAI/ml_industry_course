# Task

Transform the provided loan book into a dataset of **12-month default rates** segmented by **monthly vintage (origination month)** and **cluster (economic sector)**.

Before calculating default rates, perform a **critical data assessment** to verify the dataset is fit for purpose and to surface issues that could bias the results. Your work should include:

- **Data audit and validation:** confirm required fields exist and are internally consistent (e.g., loan ID uniqueness, origination date validity, sector/cluster assignment, default definition, default/charge-off date timing). Identify missing values, duplicates, implausible timelines (e.g., default before origination), and inconsistent formats.

- **Output dataset:** produce a clean table at the **vintage × sector** level with counts/exposures originated, number of **12-month defaults**, and the resulting **default rate**, plus any flags/meta-data needed to interpret limited coverage.

- **Limitations and remediation:** summarize the key data limitations you discovered, explain why each limitation matters for bias/interpretability (e.g., censoring, definition drift, selection effects), and propose concrete fixes (filters, exclusions, standardised definitions, additional fields/data sources, or sensitivity checks).