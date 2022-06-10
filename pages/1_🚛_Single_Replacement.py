from datetime import datetime
from pathlib import Path

import ce
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import *


def horizontal_line():
    st.markdown("---")


def project_selector():
    # Get list of projects for selector
    with engine.connect() as conn:
        project_list = [
            row[0]
            for row in conn.execute(select(projects.columns["project_name"]).distinct())
        ]
    project_list = [""] + project_list

    st.selectbox(
        label="Projects",
        options=project_list,
        key="project_select",
        on_change=get_db_params,
    )


def example_button():
    if st.button(label="Get Started", on_click=example_params):
        st.success("Example values loaded!")


def example_params():
    st.session_state["project_name"] = ""
    st.session_state["base_unit"] = ""
    st.session_state["base_engine"] = ""
    st.session_state["base_engine_type"] = "ci"
    st.session_state["base_standard"] = "t2"
    st.session_state["base_emy"] = 2004
    st.session_state["base_hp"] = 170
    st.session_state["red_unit"] = ""
    st.session_state["red_engine"] = ""
    st.session_state["red_engine_type"] = "ci"
    st.session_state["red_standard"] = "t4f"
    st.session_state["red_emy"] = datetime.now().year
    st.session_state["red_hp"] = 340
    st.session_state["red_cost"] = 125000
    st.session_state["annual_activity"] = 500
    st.session_state["load_factor"] = 0.51
    st.session_state["percent_op"] = 1.0
    st.session_state["year_1"] = datetime.now().year
    st.session_state["cel"] = 33000
    st.session_state["max_percent"] = 0.8
    st.session_state["rate"] = 0.01


def get_db_params():
    if st.session_state["project_select"] != "":
        with engine.connect() as conn:
            row = conn.execute(
                select(projects).where(
                    projects.c["project_name"] == st.session_state["project_select"]
                )
            ).first()
        st.session_state["project_select"] = ""
        st.session_state["project_name"] = row.project_name
        st.session_state["base_unit"] = row.base_unit
        st.session_state["base_engine"] = row.base_engine
        st.session_state["base_engine_type"] = row.base_engine_type
        st.session_state["base_standard"] = row.base_emi_std
        st.session_state["base_emy"] = row.base_engine_my
        st.session_state["base_hp"] = row.base_hp
        st.session_state["red_unit"] = row.red_unit
        st.session_state["red_engine"] = row.red_engine
        st.session_state["red_engine_type"] = row.red_engine_type
        st.session_state["red_standard"] = row.red_emi_std
        st.session_state["red_emy"] = row.red_engine_my
        st.session_state["red_hp"] = row.red_hp
        st.session_state["red_cost"] = row.red_cost
        st.session_state["annual_activity"] = row.annual_activity
        st.session_state["load_factor"] = row.load_factor
        st.session_state["percent_op"] = row.percent_op
        st.session_state["year_1"] = row.year_1
        st.session_state["cel"] = row.cel
        st.session_state["max_percent"] = row.percent_max
        st.session_state["rate"] = row.rate


@st.cache
def get_connection(path: str):
    return create_engine(f"sqlite+pysqlite:///{path}", echo=True, future=True)


# ADDING A TITLE AND FAVICON
st.set_page_config(page_title="Grant Calculator - Single Baseline", page_icon="🚛")

URI_SQLITE_DB = Path(__file__).parents[1] / "output/grants.db"


# sqlalchemy
engine = create_engine(f"sqlite+pysqlite:///{URI_SQLITE_DB}", echo=True, future=True)

metadata_obj = MetaData()

projects = Table(
    "projects",
    metadata_obj,
    Column("project_id", Integer, primary_key=True),
    Column(
        "project_name",
        String,
        CheckConstraint("project_name <> ''"),
        nullable=False,
        unique=True,
    ),
    Column("base_unit", String),
    Column("base_engine", String),
    Column("base_engine_type", String),
    Column("base_emi_std", String),
    Column("base_engine_my", Integer),
    Column("base_hp", Integer),
    Column("red_unit", String),
    Column("red_engine", String),
    Column("red_engine_type", String),
    Column("red_emi_std", String),
    Column("red_engine_my", Integer),
    Column("red_hp", Integer),
    Column("red_cost", Integer),
    Column("annual_activity", Integer),
    Column("load_factor", Float),
    Column("percent_op", Float),
    Column("year_1", Integer),
    Column("cel", Integer),
    Column("percent_max", Float),
    Column("rate", Float),
    Column("project_life_min", Integer),
    Column("project_life_max", Integer),
    Column("create_date", DateTime, default=func.now()),
    Column("last_modified", DateTime, onupdate=func.now()),
)

metadata_obj.create_all(engine)

st.write(
    """
# Calculator for 🚛💨 projects
Calculate potential grant amounts. Save your progress and come back later!

**👇 Click on the `Get Started` button to load an example!**
"""
)

example_button()

horizontal_line()

emi_df = pd.read_csv(Path(__file__).parents[1] / "output/emission_factors.csv")
load_df = pd.read_csv(Path(__file__).parents[1] / "output/load_factors.csv")


# PROJECT SELECTOR AND NAME
project_selector()

project_name = st.text_input(label="Project name", key="project_name")

# PROJECT INPUTS
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Baseline")
    base_unit = st.text_input(label="Baseline unit", key="base_unit")
    base_engine = st.text_input(label="Baseline engine", key="base_engine")
    base_engine_type = st.selectbox(
        label="Baseline engine type",
        options=emi_df.loc[emi_df["engine_type"] != "ze"]["engine_type"].unique(),
        key="base_engine_type",
    )
    base_standard = st.selectbox(
        label="Baseline emission standard",
        options=sorted(
            list(
                emi_df.loc[
                    emi_df["engine_type"] == base_engine_type, "standard"
                ].unique()
            )
        ),
        key="base_standard",
    )
    base_engine_my = st.number_input(
        label="Baseline engine model year",
        min_value=1900,
        step=1,
        key="base_emy",
    )
    base_hp = st.number_input(
        label="Baseline horsepower",
        min_value=25,
        key="base_hp",
    )

with col2:
    st.subheader("Reduced")
    red_unit = st.text_input(label="Reduced unit", key="red_unit")
    red_engine = st.text_input(label="Reduce engine", key="red_engine")
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
        key="red_standard",
    )
    red_engine_my = st.number_input(
        label="Reduced engine model year", min_value=1900, step=1, key="red_emy"
    )
    red_hp = st.number_input(label="Reduced horsepower", min_value=25, key="red_hp")
    red_cost = st.number_input(
        label="Reduced cost", min_value=1, step=1000, key="red_cost"
    )


with col3:
    st.subheader("Parameters")
    annual_activity = st.number_input(
        label="Annual activity", min_value=1, step=20, key="annual_activity"
    )
    load_factor = st.number_input(
        label="Load factor",
        min_value=0.01,
        max_value=1.0,
        help="See load factor tables",
        key="load_factor",
    )
    percent_op = st.number_input(
        label="Percent operation", min_value=0.01, max_value=1.0, key="percent_op"
    )
    year_1 = st.number_input(
        label="Year in operation", min_value=1998, step=1, key="year_1"
    )
    cel = st.number_input(
        label="Cost-effectiveness limit", min_value=0, step=1500, key="cel"
    )
    max_percent = st.number_input(
        label="Max percent", min_value=0.0, max_value=1.0, step=0.05, key="max_percent"
    )
    rate = st.number_input(
        label="Discount rate", min_value=0.01, max_value=1.0, key="rate"
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

# SAVE BUTTON
if st.button(label="Save"):
    match project_name:
        case "":
            st.error(f"Please provide a name!")
        case _:
            try:
                with engine.connect() as conn:
                    stmt = insert(projects).values(
                        project_name=project_name,
                        base_unit=base_unit,
                        base_engine=base_engine,
                        base_engine_type=base_engine_type,
                        base_emi_std=base_standard,
                        base_engine_my=base_engine_my,
                        base_hp=base_hp,
                        red_unit=red_unit,
                        red_engine=red_engine,
                        red_engine_type=red_engine_type,
                        red_emi_std=red_standard,
                        red_engine_my=red_engine_my,
                        red_hp=red_hp,
                        red_cost=red_cost,
                        annual_activity=annual_activity,
                        load_factor=load_factor,
                        percent_op=percent_op,
                        year_1=year_1,
                        cel=cel,
                        percent_max=max_percent,
                        rate=rate,
                        project_life_min=project_life[0],
                        project_life_max=project_life[1],
                    )
                    result = conn.execute(stmt)
                    conn.commit()
                    st.success(f"Project {project_name} saved successfully!")
            except Exception as e:
                # st.exception(e)
                st.error("Project already exists! Please provide a unique name!")

# UPDATE BUTTON
if st.button(label="Update"):
    with engine.connect() as conn:
        stmt = (
            update(projects)
            .where(projects.c.project_name == project_name)
            .values(
                project_name=project_name,
                base_unit=base_unit,
                base_engine=base_engine,
                base_engine_type=base_engine_type,
                base_emi_std=base_standard,
                base_engine_my=base_engine_my,
                base_hp=base_hp,
                red_unit=red_unit,
                red_engine=red_engine,
                red_engine_type=red_engine_type,
                red_emi_std=red_standard,
                red_engine_my=red_engine_my,
                red_hp=red_hp,
                red_cost=red_cost,
                annual_activity=annual_activity,
                load_factor=load_factor,
                percent_op=percent_op,
                year_1=year_1,
                cel=cel,
                percent_max=max_percent,
                rate=rate,
                project_life_min=project_life[0],
                project_life_max=project_life[1],
            )
        )
        result = conn.execute(stmt)
        conn.commit()
        if result.rowcount == 1:
            st.success(f"Project {project_name} updated!")
        else:
            st.error(f"Project {project_name} does not exist. You should save it!")


# Instantiate baseline and reduced tech objects
base = ce.OffRoadEquipment(
    unit_id=base_unit,
    engine_id=base_engine,
    engine_type=base_engine_type,
    engine_my=base_engine_my,
    hp=base_hp,
    standard=base_standard,
    emissions_table=emi_df,
)
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

# Expander for emission factors for the technology chosen
with st.expander("Emission factors"):
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.caption("Baseline")
        st.write(
            base.emission_factors[["standard", "pollutant", "ef", "dr"]].set_index(
                "pollutant"
            )
        )
    with ecol2:
        st.caption("Reduced")
        st.write(
            red.emission_factors[["standard", "pollutant", "ef", "dr"]].set_index(
                "pollutant"
            )
        )

# Expander for load factors to help with looking up factor
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


# CALCULATION SECTION
# Calculate and display multi-step projects
if red_engine_type == "ze":
    st.subheader("Multi-Step Results")
    st.caption(
        f"Results for {base_engine_type}-{base_standard} to conventional for {adv_project_life[0]} years at {cel:,.0f} limit then up to {adv_project_life[1]} years for conventional to {red_engine_type}-{red_standard} at {adv_cel:,.0f} limit"
    )
    df_adv = pd.DataFrame()

    for t in range(adv_project_life[0], adv_project_life[1] + 1):
        adv_tmp = ce.calc_surplus_emissions_2s(
            base,
            red,
            year_1=year_1,
            load_factor=load_factor,
            annual_activity=annual_activity,
            percent_op=percent_op,
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


# SINGLE STEP PROJECTS
st.subheader("Results")
st.caption(
    f"Results for {base_engine_type}-{base_standard} to {red_engine_type}-{red_standard} from {project_life[0]} to {project_life[1]} years at {cel:,.0f} limit"
)

# Calculate single step results
df = pd.DataFrame()

# Calculate surplus emissions
for t in range(project_life[0], project_life[1] + 1):
    surplus = ce.calc_surplus_emissions(
        base,
        red,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
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

# Shorten the dataframe
df_short = df[
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
]

# Display single step results
st.dataframe(
    df_short.set_index("project_life").style.format(
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

# MINIMIZE SINGLE STEP PROJECTS
st.subheader("Minimize Activity")
st.caption(
    f"Minimize activity subject to grant less than or equal to {ce.round_down_hundred(max_percent * red_cost):,.0f} at {cel:,.0f} limit for {base_engine_type}-{base_standard} to {red_engine_type}-{red_standard}"
)

df_min = pd.DataFrame()

for t in range(project_life[0], project_life[1] + 1):
    min_act = ce.min_annual_act(
        base,
        red,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        ce_limit=cel,
        cost_red_equip=red_cost,
        max_percent=max_percent,
        rate=rate,
        project_life=t,
    )
    df_min = pd.concat([df_min, pd.DataFrame(data=[min_act])])

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
    f"Change to grant from previous run for project lengths {project_life[0]} to {project_life[1]} years"
)

# Calculate change to grant
# Store state of run prior to being rerun
if "df_grant" not in st.session_state:
    st.session_state["df_grant"] = df[["project_life", "grant"]]

grant_change = df.set_index("project_life").join(
    st.session_state["df_grant"].set_index("project_life")["grant"],
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
st.session_state["df_grant"] = df[["project_life", "grant"]]


# Plot minimize results
st.subheader("Grant Chart")
st.caption(
    f"Grant amount for project lengths {project_life[0]} to {project_life[1]} years"
)
fig = px.bar(
    df_min_short,
    x="project_life",
    y="grant",
)

st.plotly_chart(fig)

horizontal_line()

# DELETE SECTION
if st.button(label="Delete"):
    with engine.connect() as conn:
        stmt = delete(projects).where(projects.c.project_name == project_name)
        result = conn.execute(stmt)
        conn.commit()
        if result.rowcount == 1:
            st.success(f"Project {project_name} deleted!")
        else:
            st.error(f"Project {project_name} does not exist. Nothing to delete!")
