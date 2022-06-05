from datetime import datetime

import ce
import pandas as pd
import plotly.express as px
import streamlit as st


def horizontal_line():
    st.markdown("---")


# ADDING A TITLE AND FAVICON
st.set_page_config(page_title="Grant Calculator - Multiple Replacement", page_icon="🚛")

st.write(
    """
# Calculator for 🚛🚛💨 projects
Calculate potential grant amounts for one or multiple baselines.
"""
)

emi_df = pd.read_csv(r"output/emission_factors.csv")
load_df = pd.read_csv(r"output/load_factors.csv")

num_base = st.number_input(label="Number of 🚛", min_value=1)

horizontal_line()

preselect = {
    "base": {"ci": 0, "lsi-g": 3, "lsi-a": 3},
    "red": {"ci": 4, "lsi-g": 3, "lsi-a": 3, "ze": 0},
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reduced")
    red_unit = st.text_input(f"Reduced unit")
    red_engine = st.text_input(f"Reduced engine")
    red_engine_type = st.selectbox(
        label="Reduced engine type",
        options=emi_df["engine_type"].unique(),
        key="red_engine_type",
    )
    red_standard = st.selectbox(
        label="Reduced emission standard",
        options=sorted(
            list(
                emi_df.loc[
                    emi_df["engine_type"] == red_engine_type, "standard"
                ].unique()
            )
        ),
        index=preselect["red"][red_engine_type],
        key="red_standard",
    )
    red_engine_my = st.number_input(
        label="Reduced engine model year",
        min_value=1900,
        value=datetime.now().year,
        step=1,
        key="red_emy",
    )
    red_hp = st.number_input(
        label="Reduced horsepower", min_value=25, value=95, key="red_hp"
    )
    red_cost = st.number_input(
        label="Reduced cost", min_value=1, value=100000, step=1000, key="red_cost"
    )

with col2:
    st.subheader("Parameters")
    load_factor = st.number_input(
        label="Load factor",
        min_value=0.01,
        max_value=1.0,
        value=0.31,
        help="See load factor tables",
        key="load_factor",
    )
    year_1 = st.number_input(
        label="Year in operation",
        min_value=1998,
        value=datetime.now().year,
        step=1,
        key="year_1",
    )
    cel = st.number_input(
        label="Cost-effectiveness limit", min_value=0, value=33000, step=1500, key="cel"
    )
    max_percent = st.number_input(
        label="Max percent",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05,
        key="max_percent",
    )
    rate = st.number_input(
        label="Discount rate", min_value=0.01, max_value=1.0, value=0.01, key="rate"
    )
    project_life = st.select_slider(
        label="Project life",
        options=range(1, 11),
        value=(1, 10),
    )
    match red_engine_type:
        case "ze":
            st.info("Multi-step advanced only!")
            adv_cel = st.number_input(
                label="Advanced cost-effectiveness limit",
                min_value=0,
                value=100900,
                step=1500,
                key="adv_cel",
            )
            adv_project_life = st.select_slider(
                label="Advanced project life",
                options=range(1, 11),
                value=(1, 10),
            )

with st.expander("Emission factors"):
    container = st.container()

with st.expander("Load factors"):
    lf_type_key = {"ci": "d-7", "lsi": "d-10"}
    lf_type = st.radio(label="Load factor type", options=["ci", "lsi"])
    lf_cat = st.multiselect(
        label="Category",
        options=load_df.loc[
            load_df["ref_note"] == lf_type_key[lf_type], "category"
        ].unique(),
        default=load_df.loc[
            load_df["ref_note"] == lf_type_key[lf_type], "category"
        ].unique(),
    )
    lf_equip = st.multiselect(
        label="Equipment type",
        options=load_df.loc[
            (load_df["ref_note"] == lf_type_key[lf_type])
            & (load_df["category"].isin(lf_cat)),
            "equipment_type",
        ].unique(),
        default=load_df.loc[
            (load_df["ref_note"] == lf_type_key[lf_type])
            & (load_df["category"].isin(lf_cat)),
            "equipment_type",
        ].unique(),
    )
    st.write(
        load_df.loc[
            (load_df["ref_note"] == lf_type_key[lf_type])
            & (load_df["equipment_type"].isin(lf_equip))
        ][load_df.columns[2:]].set_index("category")
    )

horizontal_line()

# BASELINE SECTION
st.subheader(f"Baseline {'🚛'*int(num_base)}")

# Hold base data
base_data = {
    "unit": [],
    "engine_type": [],
    "emy": [],
    "annual_activity": [],
    "engine": [],
    "standard": [],
    "hp": [],
    "percent_op": [],
}

bcol1, bcol2 = st.columns(2)
for count, base in enumerate(range(1, int(num_base) + 1), start=1):
    with bcol1:
        st.markdown("---")
        st.caption(f"{'🚛'*count}")
        base_unit = st.text_input(f"Baseline unit {count}")
        base_engine_type = st.selectbox(
            label=f"Baseline engine type {count}",
            options=emi_df.loc[emi_df["engine_type"] != "ze"]["engine_type"].unique(),
            key="base_engine_type",
        )
        base_engine_my = st.number_input(
            label=f"Baseline engine model year {count}",
            min_value=1900,
            value=2004,
            step=1,
            key="base_emy",
        )
        base_annual_act = st.number_input(
            label=f"Baseline annual activity {count}", min_value=1, value=500, step=20
        )
    with bcol2:
        st.markdown("---")
        st.caption(f"{'🦕'*count}")
        base_engine = st.text_input(f"Baseline engine {count}")
        base_standard = st.selectbox(
            label=f"Baseline emission standard {count}",
            options=sorted(
                list(
                    emi_df.loc[
                        emi_df["engine_type"] == base_engine_type, "standard"
                    ].unique()
                )
            ),
            index=preselect["base"][base_engine_type],
            key="base_standard",
        )
        base_hp = st.number_input(
            label=f"Baseline horsepower {count}", min_value=25, value=75
        )
        base_percent_op = st.number_input(
            label=f"Baseline percent operation {count}",
            min_value=0.01,
            max_value=1.0,
            value=1.0,
            key="percent_op",
        )

    # Store base data
    base_data["unit"] += [base_unit]
    base_data["engine_type"] += [base_engine_type]
    base_data["emy"] += [base_engine_my]
    base_data["annual_activity"] += [base_annual_act]
    base_data["engine"] += [base_engine]
    base_data["standard"] += [base_standard]
    base_data["hp"] += [int(base_hp)]
    base_data["percent_op"] += [base_percent_op]

# Instantiate reduced unit
red = ce.OffRoadEquipment(
    unit_id=red_unit,
    engine_id=red_engine,
    engine_type=red_engine_type,
    engine_my=red_engine_my,
    hp=red_hp,
    standard=red_standard,
    cost=red_cost,
    emissions_table=emi_df,
)

container.caption("Reduced 🟢")
container.write(
    red.emission_factors[["standard", "pollutant", "ef", "dr"]].set_index("pollutant")
)

# Instantiate baseline units and append objects to a list
base_list = []

for count, i in enumerate(range(0, int(num_base)), start=1):
    base = ce.OffRoadEquipment(
        unit_id=base_data["unit"][i],
        engine_id=base_data["engine"][i],
        engine_type=base_data["engine_type"][i],
        engine_my=base_data["emy"][i],
        hp=base_data["hp"][i],
        standard=base_data["standard"][i],
        emissions_table=emi_df,
    )
    base_list.append(base)
    container.caption(f"Baseline {count} {'🚛'*count}")
    container.write(
        base.emission_factors[["standard", "pollutant", "ef", "dr"]].set_index(
            "pollutant"
        )
    )


# SINGLE STEP CALCULATIONS
df = pd.DataFrame()

for t in range(project_life[0], project_life[1] + 1):
    surplus = ce.calc_surplus_emissions(
        base_list,
        red,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=base_data["annual_activity"],
        percent_op=base_data["percent_op"],
        project_life=t,
    )
    df = pd.concat([df, pd.DataFrame(data=[surplus])])

# Calculate one for one cost-effectiveness
df = df.reset_index(drop=True)
df["equip_cost"] = red.cost
df["max_per"] = max_percent
df["pot_grant_inc"] = ce.round_down_hundred(max_percent * red.cost)
df["cel_base"] = cel
df["pot_grant_cel"] = ce.round_down_hundred(
    df["cel_base"] * df["weighted"] / ce.calc_crf(rate, df["project_life"])
)
df.loc[df["pot_grant_inc"] >= df["pot_grant_cel"], "grant"] = df["pot_grant_cel"]
df.loc[df["pot_grant_inc"] < df["pot_grant_cel"], "grant"] = df["pot_grant_inc"]
df["percent_cost"] = df["grant"] / df["equip_cost"]
df["total_activity"] = df["project_life"] * df["annual_activity"]

df_short = df[
    [
        # "base",
        # "red",
        # "base_count",
        "project_life",
        "per_nox_red",
        "nox",
        "rog",
        "pm",
        "weighted",
        # "percent_op",
        "annual_activity",
        "total_activity",
        # "equip_cost",
        # "max_per",
        # "pot_grant_inc",
        # "cel_base",
        # "pot_grant_cel",
        "grant",
        "percent_cost",
    ]
].set_index("project_life")

horizontal_line()


# CALCULATE AND DISPLAY MULTI-STEP PROJECTS
if red_engine_type == "ze":
    st.subheader("Multi-Step Results")
    st.caption(
        f"Results for {int(num_base)} baseline unit(s) ({', '.join([typ + '-' + std for typ, std in (zip(base_data['engine_type'],base_data['standard']))])}) to conventional for {adv_project_life[0]} years at {cel:,.0f} limit then up to {adv_project_life[1]} years for conventional to {red_engine_type}-{red_standard} at {adv_cel:,.0f} limit"
    )
    df_adv = pd.DataFrame()

    for t in range(adv_project_life[0], adv_project_life[1] + 1):
        adv_tmp = ce.calc_surplus_emissions_2s(
            base_list,
            red,
            year_1=year_1,
            load_factor=load_factor,
            annual_activity=base_data["annual_activity"],
            percent_op=base_data["percent_op"],
            project_life_s1=adv_project_life[0],
            project_life=t,
            emissions_table=emi_df,
        )
        adv = pd.DataFrame(data=[adv_tmp])
        adv_s1 = pd.DataFrame(data=[adv_tmp.s1])
        adv_s1.columns = [col + "_s1" for col in adv_s1.columns]
        adv_s2 = pd.DataFrame(data=[adv_tmp.s2])
        adv_s2.columns = [col + "_s2" for col in adv_s2.columns]
        adv = pd.concat([adv_s1, adv_s2, adv], axis=1)
        df_adv = pd.concat([df_adv, adv])
        df_adv = df_adv.reset_index(drop=True)

    df_adv["equip_cost"] = red_cost
    df_adv["max_per"] = max_percent
    df_adv["pot_grant_inc"] = ce.round_down_hundred(max_percent * red_cost)
    df_adv["cel_base"] = cel
    df_adv["cel_at"] = adv_cel
    df_adv["pot_grant_cel_s1"] = ce.round_down_hundred(
        df_adv["cel_base"]
        * df_adv["weighted_s1"]
        / ce.calc_crf(rate, df_adv["project_life_s1"])
    )
    df_adv["pot_grant_cel_s2"] = ce.round_down_hundred(
        df_adv["cel_at"]
        * df_adv["weighted_s2"]
        / ce.calc_crf(rate, df_adv["project_life_s2"])
    )
    df_adv["pot_grant_cel_s1s2"] = (
        df_adv["pot_grant_cel_s1"] + df_adv["pot_grant_cel_s2"]
    )
    df_adv.loc[
        df_adv["pot_grant_inc"] >= df_adv["pot_grant_cel_s1s2"], "grant"
    ] = df_adv["pot_grant_cel_s1s2"]
    df_adv.loc[
        df_adv["pot_grant_inc"] < df_adv["pot_grant_cel_s1s2"], "grant"
    ] = df_adv["pot_grant_inc"]
    df_adv["percent_cost"] = df_adv["grant"] / df_adv["equip_cost"]

    df_adv_short = df_adv[
        [
            "project_life_s1",
            "weighted_s1",
            "project_life_s2",
            "weighted_s2",
            "weighted",
            "pot_grant_cel_s1",
            "pot_grant_cel_s2",
            "pot_grant_cel_s1s2",
            "grant",
            "percent_cost",
        ]
    ]
    df_adv_short = df_adv_short.rename(
        columns={
            "weighted_s1": "wgt_s1",
            "weighted_s2": "wgt_s2",
            "weighted": "wgt",
            "pot_grant_cel_s1": "pot_cel_s1",
            "pot_grant_cel_s2": "pot_cel_s2",
            "pot_grant_cel_s1s2": "pot_cel_s1s2",
        }
    )

    # Display multi-step results
    st.dataframe(
        df_adv_short.set_index(["project_life_s1", "project_life_s2"]).style.format(
            {
                "pot_cel_s1": "{:,.0f}",
                "pot_cel_s2": "{:,.0f}",
                "pot_cel_s1s2": "{:,.0f}",
                "grant": "{:,.0f}",
                "percent_cost": "{:.2f}",
            },
            precision=4,
        )
    )


# DISPLAY SINGLE STEP RESULTS
st.subheader("Results")
st.caption(
    f"**SINGLE STEP ONLY**: Results for {int(num_base)} baseline unit(s) ({', '.join([typ + '-' + std for typ, std in (zip(base_data['engine_type'],base_data['standard']))])}) to {red_engine_type}-{red_standard} from {project_life[0]} to {project_life[1]} years at {cel:,.0f} limit"
)
st.dataframe(
    df_short.style.format(
        {
            "per_nox_red": "{:.2f}",
            "percent_op": "{:.2f}",
            "annual_activity": "{:,.0f}",
            "total_activity": "{:,.0f}",
            "grant": "{:,.0f}",
            "percent_cost": "{:.2f}",
        },
        precision=4,
    )
)


# MINIMIZE SINGLE STEP PROJECTS
st.subheader("Minimize Activity")
st.caption(
    f"**SINGLE STEP ONLY**: Minimize activity subject to grant less than or equal to {ce.round_down_hundred(max_percent * red_cost):,.0f} at {cel:,.0f} limit for {base_engine_type}-{base_standard} to {red_engine_type}-{red_standard}"
)

df_min = pd.DataFrame()

for t in range(project_life[0], project_life[1] + 1):
    min_act = ce.min_annual_act(
        base_list,
        red,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=base_data["annual_activity"],
        percent_op=base_data["percent_op"],
        ce_limit=cel,
        cost_red_equip=red_cost,
        max_percent=max_percent,
        rate=rate,
        project_life=t,
    )
    df_min = pd.concat([df_min, pd.DataFrame(data=[min_act])])

df_min["annual_activity"] = df_min["annual_activity"].apply(
    lambda x: sum([a for a in x])
)
df_min["total_activity"] = df_min["total_activity"].apply(lambda x: sum([a for a in x]))

df_min_short = df_min[
    [
        "project_life",
        "per_nox_red",
        "nox",
        "rog",
        "pm",
        "weighted",
        "annual_activity",
        "total_activity",
        "grant",
        "percent_cost",
    ]
].sort_values(by="project_life")

# Display minimize results
st.dataframe(
    df_min_short.set_index("project_life").style.format(
        {
            "per_nox_red": "{:.2f}",
            "annual_activity": "{:,.0f}",
            "total_activity": "{:,.0f}",
            "grant": "{:,.0f}",
            "percent_cost": "{:.2f}",
        },
        precision=4,
    )
)


# GRANT CHANGE CALCS AND SIDEBAR
st.subheader("Grant Change Results")
st.caption(
    f"**SINGLE STEP ONLY**: Change to grant from previous run for project lengths {project_life[0]} to {project_life[1]} years"
)

# Calculate change to grant
# Store state of run prior to being rerun
if "df_multi_grant" not in st.session_state:
    st.session_state["df_multi_grant"] = df[["project_life", "grant"]]

grant_change = df.set_index("project_life").join(
    st.session_state["df_multi_grant"].set_index("project_life")["grant"],
    rsuffix="_prev",
)
grant_change["per_delta"] = (
    grant_change["grant"] - grant_change["grant_prev"]
) / grant_change["grant_prev"]
grant_change["act_delta"] = grant_change["grant"] - grant_change["grant_prev"]
st.dataframe(
    grant_change[["grant_prev", "grant", "act_delta", "per_delta"]].style.format(
        {
            "grant_prev": "{:,.0f}",
            "grant": "{:,.0f}",
            "act_delta": "{:,.0f}",
            "per_delta": "{:.2f}",
        }
    )
)

# Metric for sidebar
st.sidebar.header(f"Grant Estimates")
st.sidebar.write(f"Potential grant amount and the change from previous run")
st.sidebar.caption(f"**SINGLE STEP ONLY**")

# Display metric in sidebar
for count, t in enumerate(range(project_life[0], project_life[1] + 1), start=1):
    grant_value = grant_change.loc[t, "grant"]
    grant_delta = grant_change.loc[t, "act_delta"]
    match grant_delta:
        case 0:
            delta_color = "off"
        case _:
            delta_color = "normal"
    with st.sidebar:
        st.metric(
            label=f"{t}-year project life",
            value=f"{grant_value:,.0f}",
            delta=f"{grant_delta:,.0f}",
            delta_color=delta_color,
        )

# Update session state after another run occurred (i.e., a param changed)
# Current run will be the previous run for the next run
st.session_state["df_multi_grant"] = df[["project_life", "grant"]]


# Plot minimize results
st.subheader("Grant Chart")
st.caption(
    f"**SINGLE STEP ONLY**: Grant amount for project lengths {project_life[0]} to {project_life[1]} years"
)
fig = px.bar(
    df_min_short,
    x="project_life",
    y="grant",
)

st.plotly_chart(fig)
