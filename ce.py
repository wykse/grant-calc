"""
Classes and functions for calculating grant amount.
"""

import numbers
import os.path
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd


class Equipment:
    """A class to represent equipment."""

    def __init__(self, unit_id, engine_id):
        """Construct all the necessary attributes for the Equipment object.

        Args:
            unit_id (str): Identifier for unit.
            engine_id (str): Identifier for engine.
        """
        self.unit_id = unit_id
        self.engine_id = engine_id


class OffRoadEquipment(Equipment):
    """A subclass of Equipment."""

    def __init__(
        self,
        unit_id: str,
        engine_id: str,
        engine_type: str,
        engine_my: int,
        hp: int,
        standard: str,
        cost: float = None,
        emissions_table=pd.read_csv(Path(__file__).parents[0] / "output/emission_factors.csv"),
    ):
        """Construct all the necessary attributes for the object.

        Args:
            unit_id (str): Identifier for unit.
            engine_id (str): Identifier for engine.
            engine_type (str): Type of engine, e.g., 'ci'.
            engine_my (int): Engine model year.
            hp (int): Horsepower.
            standard (str): Emission standard.
            cost (float, optional): Unit cost. Defaults to None.
            emissions_table (dataframe, optional): Dataframe of emission factors.
                Defaults to pd.read_csv(Path(__file__).parents[0] / "output/emission_factors.csv").

        Raises:
            Exception: Provided parameters do not return emission factors.
        """
        super().__init__(unit_id, engine_id)
        self.engine_type = engine_type
        self.engine_my = engine_my
        self.hp = hp
        self.standard = standard
        self.cost = cost

        self.emission_factors = emissions_table.loc[
            (emissions_table["engine_type"] == engine_type)
            & (emissions_table["hp_min"] <= hp)
            & (emissions_table["hp_max"] >= hp)
            & (emissions_table["standard"] == standard)
            & (emissions_table["model_year_min"] <= engine_my)
            & (emissions_table["model_year_max"] >= engine_my)
        ]

        if len(self.emission_factors) == 0:
            standards = emissions_table.loc[
                emissions_table["engine_type"] == engine_type, "standard"
            ].unique()
            raise Exception(
                f"Empty emission_factors. Check {self.unit_id} {engine_type=}, {engine_my=}, {standard=} (i.e., {standards}), and {len(emissions_table)=}. Available engine types: {emissions_table['engine_type'].unique()}."
            )

    def calc_annual_emissions(
        self,
        project_life: int,
        year_1: int,
        load_factor: float,
        annual_activity: int,
        percent_op: float,
        baseline: bool = False,
        verbose: bool = False,
    ):
        """Calculate annual emissions for engine.

        Args:
            project_life (int): Project life.
            year_1 (int): First year of operation.
            load_factor (float): Load factor.
            annual_activity (int): Annual activity.
            percent_op (float): Percent of operation in area of interest.
            baseline (bool, optional): Baseline (True) or reduced technology (False). Defaults to False.
            verbose (bool, optional): Print information. Defaults to False.

        Returns:
            dataframe: Dataframe of annual emissions by pollutant.
        """
        if baseline is True:
            det_life = year_1 - self.engine_my + (project_life / 2)
        else:
            det_life = project_life / 2

        total_equip_act = annual_activity * det_life
        if self.engine_type == "ci" and total_equip_act > 12000:
            total_equip_act = 12000
        elif (
            self.engine_type[0:3] == "lsi"
            and self.engine_my <= 2006
            and total_equip_act > 3500
        ):
            total_equip_act = 3500
        elif (
            self.engine_type[0:3] == "lsi"
            and self.engine_my >= 2007
            and total_equip_act > 5000
        ):
            total_equip_act = 5000

        df = self.emission_factors.copy()
        df["total_equip_act"] = total_equip_act
        df["det_prod"] = df["dr"] * df["total_equip_act"]
        df["annual_emissions"] = (
            (df["ef"] + df["det_prod"])
            * self.hp
            * load_factor
            * annual_activity
            * percent_op
            / 907200
        )

        df = df.set_index("pollutant")

        if verbose:
            print(f"unit_id={self.unit_id}")
            print(f"baseline={baseline}")
            print(f"hp={self.hp}")
            print(f"engine_my={self.engine_my}")
            print(f"project_life={project_life}")
            print(f"year_1={year_1}")
            print(f"load_factor={load_factor}")
            print(f"annual_activity={annual_activity}")
            print(f"det_life={det_life}")
            print(f"total_equip_act={total_equip_act}")
            print(f"percent_op={percent_op}")
            print(
                f"{df[['engine_type', 'standard','ef', 'dr', 'det_prod', 'annual_emissions']]} \n"
            )

        return df["annual_emissions"]

    def __repr__(self) -> str:
        return f"OffRoadEquipment(unit_id={self.unit_id}, engine_id={self.engine_id}, engine_type={self.engine_type}, engine_my={self.engine_my}, hp={self.hp}, standard={self.standard}, cost={self.cost})"


def calc_crf(i: float, n: int) -> float:
    """Calculate capital recovery factor given interest rate and years.

    Args:
        i (float): Interest rate.
        n (int): Number of years.

    Returns:
        float: Capital recovery factor.
    """
    return (i * (1 + i) ** n) / ((1 + i) ** n - 1)


def round_down_hundred(number: int | float):
    """Round number down to the nearest hundred.

    Args:
        number (int | float): Number to round.

    Returns:
        int: Rounded number down to the nearest hundred.
    """
    return number - (number % 100)


def calc_pot_grant_cel(cel, annual_emissions, crf):
    return cel * annual_emissions / crf


def calc_pot_grant_cel_2s(
    cel_base, cel_at, surplus_base, surplus_at, project_life_s1, project_life, i=0.01
):
    s1 = cel_base * surplus_base / calc_crf(i, project_life_s1)
    s2 = cel_at * surplus_at / calc_crf(i, project_life)
    return s1 + s2


def calc_surplus_emissions(
    base_equip: OffRoadEquipment | list,
    red_equip: OffRoadEquipment,
    year_1: int,
    load_factor: float,
    annual_activity: int | list,
    percent_op: int | list,
    project_life: int,
    verbose: bool = False,
) -> tuple:
    """Calculates emission reductions for conventional replacement and
    repower projects. This includes one for one or multiple for one.

    If multiple baseline: base_equip, annual_activity, and percent_op
    need to be a list each, and length of each need to be the same.

    Args:
        base_equip (OffRoadEquipment | list): An instance of OffRoadEquipment.
        red_equip (OffRoadEquipment): An instance of OffRoadEquipment.
        year_1 (int): First year of operation
        load_factor (float): The engine load factor.
        annual_activity (int | list): Annual activity.
        percent_op (int | list): Percent of operation in area of interest.
        project_life (int): Project life in years.
        verbose (bool, optional): Print information to console. Defaults to False.

    Returns:
        tuple: Named tuple containing emission reductions and related information.
    """

    SurplusEmissions = namedtuple(
        "SurplusEmissions",
        [
            "base",
            "red",
            "base_count",
            "annual_activity",
            "percent_op",
            "project_life",
            "per_nox_red",
            "nox",
            "rog",
            "pm",
            "weighted",
        ],
    )

    # Put arguments into a list if not already
    if (
        isinstance(base_equip, OffRoadEquipment)
        and isinstance(annual_activity, numbers.Number)
        and isinstance(percent_op, numbers.Number)
    ):
        base_equip_list = [base_equip]
        annual_activity_list = [annual_activity]
        percent_op_list = [percent_op]
    else:
        base_equip_list = base_equip
        annual_activity_list = annual_activity
        percent_op_list = percent_op

    sum_activity = sum(annual_activity_list)
    wgt_percent_op = np.matmul(annual_activity_list, percent_op_list) / sum_activity

    # Initialize an empty dataframe to hold baseline equipment emissions
    base_emi = red_equip.calc_annual_emissions(
        project_life=0,
        year_1=0,
        load_factor=0,
        annual_activity=0,
        percent_op=0,
    )
    # Loop through baseline equipment and sum emissions
    for i, equip in enumerate(base_equip_list):
        if verbose:
            print(
                "--------------------------------"
                + "\n"
                + f"BASELINE EQUIPMENT ({i+1} of {len(base_equip_list)})"
                + "\n"
                + "--------------------------------"
            )
        base_emi += equip.calc_annual_emissions(
            project_life=project_life,
            year_1=year_1,
            load_factor=load_factor,
            annual_activity=annual_activity_list[i],
            percent_op=percent_op_list[i],
            baseline=True,
            verbose=verbose,
        )

    if verbose:
        print(
            "--------------------------------"
            + "\n"
            + f"REDUCED EQUIPMENT"
            + "\n"
            + "--------------------------------"
        )
    red_emi = red_equip.calc_annual_emissions(
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=sum_activity,
        percent_op=wgt_percent_op,
        baseline=False,
        verbose=verbose,
    )

    surplus_emi = base_emi - red_emi

    emission_reductions = SurplusEmissions(
        base=base_equip_list,
        red=red_equip,
        base_count=len(base_equip_list),
        annual_activity=sum_activity,
        percent_op=wgt_percent_op,
        project_life=project_life,
        per_nox_red=(base_emi["nox"] - red_emi["nox"]) / base_emi["nox"],
        nox=surplus_emi["nox"],
        rog=surplus_emi["rog"],
        pm=surplus_emi["pm"],
        weighted=surplus_emi["nox"] + surplus_emi["rog"] + (20 * surplus_emi["pm"]),
    )

    if verbose:
        print(
            "--------------------------------"
            + "\n"
            + f"EMISSION REDUCTIONS"
            + "\n"
            + "--------------------------------"
        )
        print(f"project_life={emission_reductions.project_life}")
        print(f"per_nox_red={emission_reductions.per_nox_red:.2f}")
        print(f"nox={emission_reductions.nox:.6f}")
        print(f"rog={emission_reductions.rog:.6f}")
        print(f"pm={emission_reductions.pm:.6f}")
        print(f"weighted={emission_reductions.weighted:.6f}")

    return emission_reductions


def calc_surplus_emissions_2s(
    base_equip: OffRoadEquipment | list,
    red_equip: OffRoadEquipment,
    year_1: int,
    load_factor: float,
    annual_activity: int | list,
    percent_op: int | list,
    project_life_s1: int,
    project_life: int,
    emissions_table=pd.read_csv(Path(__file__).parents[0] / "output/emission_factors.csv"),
    verbose: bool = False,
) -> tuple:
    """Calculates emission reductions for advanced technology replacement
    and repower projects. This includes one for one and multiple for one.

    Args:
        base_equip (OffRoadEquipment | list): An instance of OffRoadEquipment.
        red_equip (OffRoadEquipment): An instance of OffRoadEquipment.
        year_1 (int): First year of operation
        load_factor (float): The engine load factor.
        annual_activity (int | list): Annual activity.
        percent_op (int | list): Percent of operation in area of interest.
        project_life_s1 (int): First step project life in years.
        project_life (int): Project life in years.
        emissions_table (dataframe, optional): DataFrame of emission factors.
            Defaults to pd.read_csv(Path(__file__).parents[0] / "output/emission_factors.csv").
        verbose (bool, optional): Print information to console. Defaults to False.

    Returns:
        tuple: Named tuple containing emission reductions and related information.
    """

    if verbose:
        print(
            "\n"
            + "-----------------------------------------------------------"
            + "\n"
            + "           CALCULATE SURPLUS EMISSIONS TWO-STEP"
            + "\n"
            + "                        STEP ONE"
            + "\n"
            + "-----------------------------------------------------------"
        )

    # Current emission standards for new engines
    current_standard = {"ci": "t4f", "lsi-g": "c", "lsi-a": "c"}

    if isinstance(base_equip, list) and len(base_equip) >= 1:
        # Get lowest hp
        low_hp = min([equip.hp for equip in base_equip])

        # Get engine type for step 1; pick based on cleanest
        # LSI to CI is ineligible; assume lsi-a is cleaner than lsi-g
        if all(equip.engine_type == base_equip[0].engine_type for equip in base_equip):
            engine_type_s1 = base_equip[0].engine_type
        elif "lsi-a" in [equip.engine_type for equip in base_equip]:
            engine_type_s1 = "lsi-a"
        else:
            engine_type_s1 = "lsi-g"

        # Step 1 reduced baseline information
        red_base_s1 = OffRoadEquipment(
            unit_id=f"Reduced Base",
            engine_id=None,
            engine_type=engine_type_s1,
            engine_my=year_1,
            hp=low_hp,
            standard=current_standard[engine_type_s1],
            emissions_table=emissions_table,
        )

        annual_activity_new = sum(annual_activity)
        percent_op_new = np.matmul(annual_activity, percent_op) / sum(annual_activity)

    else:
        annual_activity_new = annual_activity
        percent_op_new = percent_op

        # Step 1 reduced baseline information
        red_base_s1 = OffRoadEquipment(
            unit_id=f"{base_equip.unit_id} - Reduced",
            engine_id=None,
            engine_type=base_equip.engine_type,
            engine_my=year_1,
            hp=base_equip.hp,
            standard=current_standard[base_equip.engine_type],
            emissions_table=emissions_table,
        )

    # Step 1 reduced baseline
    s1 = calc_surplus_emissions(
        red_equip=red_base_s1,
        base_equip=base_equip,
        project_life=project_life_s1,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=verbose,
    )

    if verbose:
        print(
            "\n"
            + "-----------------------------------------------------------"
            + "\n"
            + "                         STEP TWO"
            + "\n"
            + "-----------------------------------------------------------"
        )

    # Step 2 advanced technology
    s2 = calc_surplus_emissions(
        red_equip=red_equip,
        base_equip=red_base_s1,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity_new,
        percent_op=percent_op_new,
        verbose=verbose,
    )

    TwoStep = namedtuple(
        "TwoStep",
        [
            "project_life",
            "s1",
            "s2",
            "nox",
            "rog",
            "pm",
            "weighted",
        ],
    )

    s1s2 = TwoStep(
        project_life=project_life,
        s1=s1,
        s2=s2,
        nox=(s1.nox * s1.project_life / project_life)
        + (s2.nox * s2.project_life / project_life),
        rog=(s1.rog * s1.project_life / project_life)
        + (s2.rog * s2.project_life / project_life),
        pm=(s1.pm * s1.project_life / project_life)
        + (s2.pm * s2.project_life / project_life),
        weighted=(s1.weighted * s1.project_life / project_life)
        + (s2.weighted * s2.project_life / project_life),
    )

    if verbose:
        print(
            "-----------------------------------------------------------"
            + "\n"
            + "                     TWO-STEP SUMMARY"
            + "\n"
            + "-----------------------------------------------------------"
        )
        df = pd.concat([pd.DataFrame(data=[s1s2.s1]), pd.DataFrame(data=[s1s2.s2])])
        df = df.reset_index(drop=True)
        df.index += 1
        df.index = df.index.rename("step")
        print(f"{df}" + "\n")
        print(f"nox={s1s2.nox:.6f}")
        print(f"rog={s1s2.rog:.6f}")
        print(f"pm={s1s2.pm:.6f}")
        print(f"weighted={s1s2.weighted:.6f}")

    return s1s2


def min_annual_act(
    base_equip: OffRoadEquipment | list,
    red_equip: OffRoadEquipment,
    year_1: int,
    load_factor: float,
    annual_activity: int | list,
    percent_op: float | list,
    ce_limit: int | float,
    cost_red_equip: int | float,
    max_percent: float,
    rate: float = 0.01,
    project_life: int = 3,
    tol: int = 500,
    step: int = 1,
) -> tuple:
    """Minimize annual activity subject to potential grant amount
    approximately equal to incremental cost.

    Function works for one step calculations only. This includes one
    for one and two for one projects. Only repower and replacement
    project types.

    The function uses bisection method to approximate a solution first.
    Then the function adjusts activity in steps until activity is
    minimized.

    Args:
        base_equip (OffRoadEquipment | list): An instance of OffRoadEquipment.
        red_equip (OffRoadEquipment): An instance of OffRoadEquipment.
        year_1 (int): First year of operation
        load_factor (float): The engine load factor.
        annual_activity (int | list): Annual activity.
        percent_op (int | list): Percent of operation in area of interest.
        ce_limit (int | float): Cost-effectiveness limit.
        cost_red_equip (int | float): Cost of reduced technology.
        max_percent (float): Max percentage of eligible costs.
        rate (float, optional): Discount rate. Defaults to 0.01.
        project_life (int, optional): Project life in years. Defaults to 3.
        tol (int, optional): Positive number for tolerance. Defaults to 500.
        step (int, optional): Step size to minimize annual activity. Defaults to 1.

    Returns:
        tuple: Named tuple containing emission reductions and related information.
    """

    ProjectAlts = namedtuple(
        "ProjectAlts",
        [
            "base",
            "red",
            "per_nox_red",
            "nox",
            "rog",
            "pm",
            "weighted",
            "rate",
            "project_life",
            "annual_activity",
            "pot_grant_cel",
            "pot_grant_inc",
            "grant",
            "ce_per_ton",
            "grant_cel_minus_inc",
            "grant_dist",
            "total_activity",
            "percent_cost",
            "cel",
        ],
    )
    surplus = calc_surplus_emissions(
        red_equip=red_equip,
        base_equip=base_equip,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=False,
    )

    crf = calc_crf(rate, project_life)
    pot_grant_cel = round_down_hundred(ce_limit * surplus.weighted / crf)
    pot_grant_inc = round_down_hundred(cost_red_equip * max_percent)
    grant = min(pot_grant_cel, pot_grant_inc)

    hold_alt = ProjectAlts(
        base=base_equip,
        red=red_equip,
        per_nox_red=surplus.per_nox_red,
        nox=surplus.nox,
        rog=surplus.rog,
        pm=surplus.pm,
        weighted=surplus.weighted,
        rate=rate,
        annual_activity=annual_activity,
        project_life=project_life,
        pot_grant_cel=pot_grant_cel,
        pot_grant_inc=pot_grant_inc,
        grant=min(pot_grant_cel, pot_grant_inc),
        ce_per_ton=min(pot_grant_cel, pot_grant_inc) * crf / surplus.weighted,
        grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
        grant_dist=abs(pot_grant_cel - pot_grant_inc),
        total_activity=np.array(annual_activity) * project_life,
        percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_red_equip,
        cel=ce_limit,
    )

    # Bisection method to approximate minimum activity
    # print(
    #     "\n"
    #     + "----------------------------------------"
    #     + "\n"
    #     + f"MINIMIZE ACTIVITY ({project_life=})"
    #     + "\n"
    #     + "----------------------------------------"
    # )
    # print(
    #     f"{annual_activity=}   |   total_activity={hold_alt.total_activity}   |   grant_cel_minus_inc={hold_alt.grant_cel_minus_inc}"
    # )

    # Limit iterations to prevent infinite loop
    n = 1
    nmax = 20

    if pot_grant_cel > grant:
        low_act = 0
        high_act = annual_activity
        bi_act = np.floor(np.add(low_act, high_act) / 2)

    while (abs(pot_grant_cel - grant) >= tol) and (n <= nmax):
        surplus_bi_act = calc_surplus_emissions(
            red_equip=red_equip,
            base_equip=base_equip,
            project_life=project_life,
            year_1=year_1,
            load_factor=load_factor,
            annual_activity=bi_act,
            percent_op=percent_op,
            verbose=False,
        )

        pot_grant_cel = round_down_hundred(ce_limit * surplus_bi_act.weighted / crf)

        hold_alt = ProjectAlts(
            base=base_equip,
            red=red_equip,
            per_nox_red=surplus_bi_act.per_nox_red,
            nox=surplus_bi_act.nox,
            rog=surplus_bi_act.rog,
            pm=surplus_bi_act.pm,
            weighted=surplus_bi_act.weighted,
            rate=rate,
            annual_activity=bi_act,
            project_life=project_life,
            pot_grant_cel=pot_grant_cel,
            pot_grant_inc=pot_grant_inc,
            grant=min(pot_grant_cel, pot_grant_inc),
            ce_per_ton=min(pot_grant_cel, pot_grant_inc)
            * crf
            / surplus_bi_act.weighted,
            grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
            grant_dist=abs(pot_grant_cel - pot_grant_inc),
            total_activity=bi_act * project_life,
            percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_red_equip,
            cel=ce_limit,
        )

        # print(
        #     f"{bi_act=}   |   total_activity={hold_alt.total_activity}   |   grant_cel_minus_inc={hold_alt.grant_cel_minus_inc}"
        # )

        n += 1

        if pot_grant_cel < grant:
            low_act = bi_act
        else:
            high_act = bi_act
        bi_act = np.floor(np.add(low_act, high_act) / 2)

    # Decrement activity until pot grant at cel is approximately inc cost
    step_decrease = hold_alt.grant_cel_minus_inc > 0
    if step_decrease:
        step_act = np.array(hold_alt.annual_activity) - 1
    else:
        step_act = np.array(hold_alt.annual_activity) + 1

    last_step = False
    while np.array(step_act).all() >= step:
        # Exception when truth value of an array with more than one element
        # is ambiguous. If all elements are True, then return True.
        try:
            if (
                hold_alt.annual_activity == annual_activity
                and hold_alt.pot_grant_cel <= pot_grant_inc
            ) or hold_alt.grant_cel_minus_inc == 0:
                break
        except ValueError as e:
            if (
                (hold_alt.annual_activity == annual_activity).all()
                and hold_alt.pot_grant_cel <= pot_grant_inc
            ) or hold_alt.grant_cel_minus_inc == 0:
                break

        surplus_step_act = calc_surplus_emissions(
            red_equip=red_equip,
            base_equip=base_equip,
            project_life=project_life,
            year_1=year_1,
            load_factor=load_factor,
            annual_activity=step_act,
            percent_op=percent_op,
            verbose=False,
        )

        pot_grant_cel = round_down_hundred(ce_limit * surplus_step_act.weighted / crf)

        hold_alt = ProjectAlts(
            base=base_equip,
            red=red_equip,
            per_nox_red=surplus_step_act.per_nox_red,
            nox=surplus_step_act.nox,
            rog=surplus_step_act.rog,
            pm=surplus_step_act.pm,
            weighted=surplus_step_act.weighted,
            rate=rate,
            annual_activity=step_act,
            project_life=project_life,
            pot_grant_cel=pot_grant_cel,
            pot_grant_inc=pot_grant_inc,
            grant=min(pot_grant_cel, pot_grant_inc),
            ce_per_ton=min(pot_grant_cel, pot_grant_inc)
            * crf
            / surplus_step_act.weighted,
            grant_cel_minus_inc=pot_grant_cel - pot_grant_inc,
            grant_dist=abs(pot_grant_cel - pot_grant_inc),
            total_activity=step_act * project_life,
            percent_cost=min(pot_grant_cel, pot_grant_inc) / cost_red_equip,
            cel=ce_limit,
        )

        # print(
        #     f"{step_act=}   |   total_activity={hold_alt.total_activity}   |   grant_cel_minus_inc={hold_alt.grant_cel_minus_inc}"
        # )

        if last_step:
            break
        elif step_decrease:
            if hold_alt.grant_cel_minus_inc < 0:
                step_act += step
                last_step = True
            else:
                step_act -= step
        else:
            if hold_alt.grant_cel_minus_inc >= 0:
                break
            else:
                step_act += step

    return hold_alt


def calc_annual_usage(data: str, date_col: str = "date", meter_col: str = "value"):
    """Return a dataframe of annualized values based on the most recent
    meter value.

    Args:
        data (str): Path to csv or xlsx file containing date_col and meter_col.
        date_col (str, optional): Name for column containing dates. Dates should be in YYYY-MM-DD format. Defaults to "date".
        meter_col (str, optional): Name for column containing meter values. Defaults to "value".

    Raises:
        Exception: data is not a path to a .csv or .xlsx file.

    Returns:
        DataFrame: Dataframe of annualized values based on the most recent meter value.
    """

    if os.path.isfile(data):
        if os.path.splitext(data)[1] == ".csv":
            df = pd.read_csv(data)
        elif os.path.splitext(data)[1] == ".xlsx":
            df = pd.read_excel(data)
        else:
            raise Exception(f"{data}")
    else:
        raise Exception(f"{data} is not a path to a .csv or .xlsx file.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    date_max = df[date_col].max()

    df["days_to_max"] = date_max - df["date"]
    df["days_to_max"] = df["days_to_max"].dt.days
    df["years_to_max"] = df["days_to_max"] / 365.2425
    df["value_to_max"] = (
        df.loc[df["date"] == date_max, meter_col].iloc[0] - df[meter_col]
    )
    df["annual_to_max"] = df["value_to_max"] / df["years_to_max"]

    df = df.set_index("date")

    return df
