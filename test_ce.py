from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

import ce

emi_df = pd.read_csv(Path(__file__).parents[0] / "output/emission_factors.csv")


def test_crf():
    THREEPLACES = Decimal(".001")

    # Table D-24
    assert Decimal(str(ce.calc_crf(i=0.01, n=1))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("1.010")
    assert Decimal(str(ce.calc_crf(i=0.01, n=2))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.508")
    assert Decimal(str(ce.calc_crf(i=0.01, n=3))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.340")
    assert Decimal(str(ce.calc_crf(i=0.01, n=4))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.256")
    assert Decimal(str(ce.calc_crf(i=0.01, n=5))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.206")
    assert Decimal(str(ce.calc_crf(i=0.01, n=6))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.173")
    assert Decimal(str(ce.calc_crf(i=0.01, n=7))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.149")
    assert Decimal(str(ce.calc_crf(i=0.01, n=8))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.131")
    assert Decimal(str(ce.calc_crf(i=0.01, n=9))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.117")
    assert Decimal(str(ce.calc_crf(i=0.01, n=10))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.106")

    # Table D-25
    assert Decimal(str(ce.calc_crf(i=0.02, n=1))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("1.020")
    assert Decimal(str(ce.calc_crf(i=0.02, n=2))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.515")
    assert Decimal(str(ce.calc_crf(i=0.02, n=3))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.347")
    assert Decimal(str(ce.calc_crf(i=0.02, n=4))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.263")
    assert Decimal(str(ce.calc_crf(i=0.02, n=5))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.212")
    assert Decimal(str(ce.calc_crf(i=0.02, n=6))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.179")
    assert Decimal(str(ce.calc_crf(i=0.02, n=7))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.155")
    assert Decimal(str(ce.calc_crf(i=0.02, n=8))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.137")
    assert Decimal(str(ce.calc_crf(i=0.02, n=9))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.123")
    assert Decimal(str(ce.calc_crf(i=0.02, n=10))).quantize(
        THREEPLACES, rounding=ROUND_HALF_UP
    ) == Decimal("0.111")


def test_pot_grant_cel_2step():
    # Example 6
    # Difference is due to example's capital recovery factor's lower precision
    pot_grant_cel = ce.calc_pot_grant_cel_2s(
        cel_base=30000,
        cel_at=100000,
        surplus_base=0.2812,
        surplus_at=0.0401,
        project_life_s1=3,
        project_life=10,
    )

    assert Decimal(str(pot_grant_cel)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("62790.08237635295").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


# Test 2017 guideslines example calculations (2018-09-18) for section ii
def test_example_1():
    # Example 1 repower
    engine_type = "ci"
    hp = 300
    standard = "t0"
    emy = 1988
    project_life = 3
    year_1 = 2017
    load_factor = 0.48
    annual_activity = 1500
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 300
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("4.868488095238096").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_2():
    # Example 2 retrofit
    engine_type = "ci"
    hp = 160
    standard = "t2"
    emy = 2004
    project_life = 5
    year_1 = 2017
    load_factor = 0.36
    annual_activity = 850
    percent_op = 1

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    # Level 3 DPF, 85 percent PM reduction
    # PM weighted
    surplus = base.calc_annual_emissions(
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        baseline=True,
    )
    surplus = surplus.pm * 0.85 * 20

    assert Decimal(str(surplus)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.22092444444444445").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_3():
    # Example 3 replacement
    engine_type = "ci"
    hp = 170
    standard = "t0"
    emy = 1985
    project_life = 10
    year_1 = 2018
    load_factor = 0.70
    annual_activity = 1000
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 340
    red_standard = "t4f"
    red_engine_my = 2018

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("3.672839506172839").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_4():
    # Example 4 two for one
    engine_type = "ci"
    emy = 2005
    hp = 240
    standard = "t2"
    project_life = 3
    year_1 = 2017
    load_factor = 0.36
    annual_activity = 750
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 210
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    base2 = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_my=2004,
        engine_type="ci",
        hp=180,
        standard="t2",
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=[base, base2],
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=[annual_activity, 350],
        percent_op=[percent_op, 1],
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.6803831190476193").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_5():
    # Example 5 replacement
    engine_type = "ci"
    emy = 2006
    hp = 310
    standard = "t3"
    project_life = 5
    year_1 = 2017
    load_factor = 0.54
    annual_activity = 350
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 350
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.27529604166666666").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_6():
    # Example 6 lsi replacement with ze
    engine_type = "lsi-a"
    emy = 2003
    hp = 91
    standard = "uc"
    project_life_s1 = 3
    project_life = 10
    year_1 = 2018
    load_factor = 0.30
    annual_activity = 750
    percent_op = 1

    red_engine_type = "ze"
    red_hp = 70
    red_standard = "ze"
    red_engine_my = 2018

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions_2s(
        red_equip=red,
        base_equip=base,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        project_life_s1=project_life_s1,
        project_life=project_life,
    )
    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.12351952256944444").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_7():
    # Example 7 replacement with ze
    engine_type = "ci"
    emy = 1998
    hp = 59
    standard = "t1"
    project_life_s1 = 4
    project_life = 4
    year_1 = 2018
    load_factor = 0.34
    annual_activity = 700
    percent_op = 1

    red_engine_type = "ze"
    red_hp = 40
    red_standard = "ze"
    red_engine_my = 2018

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions_2s(
        red_equip=red,
        base_equip=base,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        project_life_s1=project_life_s1,
        project_life=project_life,
    )
    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.4710385185185186").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_8():
    # Example 8 portable generator
    engine_type = "ci"
    emy = 2006
    hp = 150
    standard = "t3"
    project_life = 5
    year_1 = 2017
    load_factor = 0.74
    annual_activity = 500
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 150
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.34358672288359793").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_9():
    # Example 9 repower
    engine_type = "ci"
    emy = 2006
    hp = 503
    standard = "t3"
    project_life = 5
    year_1 = 2017
    load_factor = 0.73
    annual_activity = 700
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 500
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        verbose=True,
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("1.4108305289351852").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_10():
    # Example 10 repower to ze
    engine_type = "ci"
    emy = 2006
    hp = 120
    standard = "t3"
    project_life_s1 = 7
    project_life = 10
    year_1 = 2017
    load_factor = 0.65
    annual_activity = 1000
    percent_op = 1

    red_engine_type = "ze"
    red_hp = 100
    red_standard = "ze"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions_2s(
        red_equip=red,
        base_equip=base,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        project_life_s1=project_life_s1,
        project_life=project_life,
    )
    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.4564315476190476").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )


def test_example_min():
    # Example 9 repower
    engine_type = "ci"
    emy = 2006
    hp = 503
    standard = "t3"
    project_life = 5
    year_1 = 2017
    load_factor = 0.73
    annual_activity = 700
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 500
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id="Baseline",
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced",
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=base,
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=312,  # Minimum activity
        percent_op=percent_op,
        verbose=True,
    )

    min_act = ce.min_annual_act(
        red_equip=red,
        base_equip=base,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=annual_activity,
        percent_op=percent_op,
        ce_limit=30000,
        cost_red_equip=92000,
        max_percent=0.85,
        rate=0.01,
        project_life=5,
        tol=1000,
        step=1,
    )

    # Test minimization to lowest activity
    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal(str(min_act.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )

    # Test grant at cel at min activity
    assert min_act.pot_grant_cel == 78300

    # Test min resulted in shortest distance between grant at cel and inc cost
    assert min_act.grant_cel_minus_inc == 100


def test_example_2for1_min():
    # Example 4 two for one; adapted to test optimization
    # Reduce the cost of reduced technology to test reducing
    # pot grant at cel to incremental cost
    engine_type = "ci"
    emy = 2005
    hp = 240
    standard = "t2"
    project_life = 3
    year_1 = 2017
    load_factor = 0.36
    annual_activity = 750
    percent_op = 1

    red_engine_type = "ci"
    red_hp = 210
    red_standard = "t4f"
    red_engine_my = 2017

    base = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_my=emy,
        engine_type=engine_type,
        hp=hp,
        standard=standard,
        emissions_table=emi_df,
    )

    base2 = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_my=2004,
        engine_type="ci",
        hp=180,
        standard="t2",
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id=None,
        engine_id="1",
        engine_type=red_engine_type,
        engine_my=red_engine_my,
        hp=red_hp,
        standard=red_standard,
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions(
        red_equip=red,
        base_equip=[base, base2],
        project_life=project_life,
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=[533, 248],  # Minimum activity
        percent_op=[percent_op, 1],
        verbose=True,
    )

    min_act = ce.min_annual_act(
        red_equip=red,
        base_equip=[base, base2],
        year_1=year_1,
        load_factor=load_factor,
        annual_activity=[annual_activity, 350],
        percent_op=[percent_op, 1],
        ce_limit=30000,
        cost_red_equip=50000,  # Changed from example
        max_percent=0.8,
        rate=0.01,
        project_life=3,
        tol=100,
        step=1,
    )

    # Test minimization to lowest activity
    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal(str(min_act.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )

    # Test minimization resulted grant at cel equal to inc cost
    assert min_act.pot_grant_cel == 40000

    # Test minimization resulted in 0 distance between grant at cel and inc cost
    assert min_act.grant_cel_minus_inc == 0


def test_example_2for1_2step():
    # Example 4 reworked to two step
    base = ce.OffRoadEquipment(
        unit_id="Baseline Equipment 1",
        engine_id="1",
        engine_my=2005,
        engine_type="ci",
        hp=240,
        standard="t2",
        emissions_table=emi_df,
    )

    base2 = ce.OffRoadEquipment(
        unit_id="Baseline Equipment 2",
        engine_id="1",
        engine_my=2004,
        engine_type="ci",
        hp=180,
        standard="t2",
        emissions_table=emi_df,
    )

    red = ce.OffRoadEquipment(
        unit_id="Reduced Equipment ZE",
        engine_id="1",
        engine_type="ze",
        engine_my=2017,
        hp=210,
        standard="ze",
        emissions_table=emi_df,
    )

    surplus = ce.calc_surplus_emissions_2s(
        red_equip=red,
        base_equip=[base, base2],
        year_1=2017,
        load_factor=0.36,
        annual_activity=[750, 350],
        percent_op=[1, 1],
        project_life_s1=3,
        project_life=10,
        verbose=True,
    )

    # This is close to example 4's value. Difference is due to reduced
    # equipment being 180 hp instead of 210 hp.
    assert Decimal(str(surplus.s1.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.687244892857143").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )

    assert Decimal(str(surplus.weighted)).quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    ) == Decimal("0.25357561071428575").quantize(
        Decimal("1.0000"), rounding=ROUND_HALF_UP
    )
