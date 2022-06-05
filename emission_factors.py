"""
Prepare emission factors.
"""

import pandas as pd

df = pd.read_csv(r"data\emission_factor_db.csv")

# Fill nan values
df["hp_max"] = df["hp_max"].fillna(9999)
df["model_year_min"] = df["model_year_min"].fillna(0)
df["model_year_max"] = df["model_year_max"].fillna(9999)

# CI engines without a tier are uncontrolled and are tier 0
df.loc[(df["engine_type"] == "ci") & (df["standard"].isna()), "standard"] = "t0"

# Check if any col contain na
if any(df.isna().any()):
    print("Check if col contain na:")
    print(df.isna().any())
    raise Exception(
        f"Missing values in column(s): {list(df.isna().any()[df.isna().any() == True].index)}."
    )

df = df.set_index(
    [
        "ref",
        "ref_note",
        "hp_min",
        "hp_max",
        "model_year_min",
        "model_year_max",
        "standard",
        "engine_type",
    ]
)

# Unpivot the df for emission factor and det rate
df = df.melt(ignore_index=False)

# Identify pollutant and type of factor (emission or det)
df["pollutant"] = df["variable"].str.split("_").str[0]
df["type"] = df["variable"].str.split("_").str[1]

df = df.reset_index().set_index(
    [
        "ref",
        "ref_note",
        "hp_min",
        "hp_max",
        "model_year_min",
        "model_year_max",
        "standard",
        "pollutant",
        "engine_type",
    ]
)

# Filter for each factor type and join together to create a col for each factor type
df = df.loc[df["type"] == "ef"].join(
    df.loc[df["type"] == "dr"], lsuffix="_ef", rsuffix="_dr"
)
df = df.reset_index()
df = df.rename(columns={"value_ef": "ef", "value_dr": "dr"})
df = df[
    [
        "ref",
        "ref_note",
        "engine_type",
        "hp_min",
        "hp_max",
        "model_year_min",
        "model_year_max",
        "standard",
        "pollutant",
        "ef",
        "dr",
    ]
]

df.to_csv(r"output\emission_factors.csv", index=False)
