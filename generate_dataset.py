"""
WattWise - Realistic Synthetic Electricity Consumption Dataset Generator

Generates 20,000+ household records using physics-based appliance power calculations.
Previous month units are simulated from an INDEPENDENT prior household state to avoid target leakage.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_ROWS = 20000
DAYS_IN_MONTH = 30
OUTPUT_PATH = os.path.join("dataset", "electricity_data.csv")

# --- Appliance power ratings (kW unless noted) ---
AC_POWER_KW = (1.2, 2.0)
FAN_POWER_W = (55, 80)
LED_POWER_W = (8, 15)
BULB_POWER_W = (18, 25)
TV_POWER_W = (70, 180)
LAPTOP_POWER_W = (45, 90)
COOLER_POWER_W = (180, 300)
GEYSER_POWER_KW = (1.8, 2.2)
REFRIGERATOR_KWH_DAY = (1.0, 1.8)
WASHING_MACHINE_KWH_CYCLE = (0.8, 1.5)

HOUSE_TYPES = ["1BHK", "2BHK", "3BHK", "4BHK"]
OCCUPATIONS = ["Student", "Employed", "Self-Employed", "Retired", "Homemaker"]
INCOME_LEVELS = ["Low", "Medium", "High", "Very High"]
SEASONS = ["Summer", "Monsoon", "Winter", "Spring"]
WORK_FROM_HOME = ["Yes", "No", "Partial"]

INCOME_MAP = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
HOUSE_MAP = {"1BHK": 1, "2BHK": 2, "3BHK": 3, "4BHK": 4}
SEASON_MAP = {"Summer": 0, "Monsoon": 1, "Winter": 2, "Spring": 3}
WFH_MAP = {"No": 0, "Partial": 1, "Yes": 2}
OCCUPATION_MAP = {
    "Student": 0,
    "Employed": 1,
    "Self-Employed": 2,
    "Retired": 3,
    "Homemaker": 4,
}


def _rand_range(low, high, size=1):
    return np.random.uniform(low, high, size)


def _income_ac_prob(income: str) -> float:
    base = {"Low": 0.08, "Medium": 0.35, "High": 0.65, "Very High": 0.85}
    return base[income]


def _sample_ac_count(income: str, house_type: str, family: int) -> int:
    prob = _income_ac_prob(income)
    house_bonus = (HOUSE_MAP[house_type] - 1) * 0.08
    family_bonus = max(0, family - 2) * 0.04
    p = min(0.95, prob + house_bonus + family_bonus)
    max_ac = min(4, max(1, HOUSE_MAP[house_type]))
    count = 0
    for _ in range(max_ac):
        if np.random.random() < p:
            count += 1
        p *= 0.55
    return count


def _sample_appliance_counts(house_type: str, income: str, family: int) -> dict:
    rooms = HOUSE_MAP[house_type]
    income_idx = INCOME_MAP[income]

    fan_count = int(np.clip(
        rooms * 1.5 + family * 0.4 + np.random.poisson(1),
        1, 10
    ))
    light_count = int(np.clip(
        rooms * 3 + family * 0.8 + np.random.poisson(2),
        4, 20
    ))
    led_pct = float(np.clip(
        20 + income_idx * 18 + np.random.normal(0, 8),
        5, 100
    ))

    cooler = 1 if (income_idx >= 1 and np.random.random() < 0.45) else 0
    geyser = 1 if np.random.random() < (0.5 + income_idx * 0.12) else 0
    refrigerator = 1
    washing_machine = 1 if (family >= 2 and income_idx >= 0) else int(np.random.random() < 0.3)

    return {
        "FanCount": fan_count,
        "LightCount": light_count,
        "LEDPercentage": round(led_pct, 1),
        "Cooler": cooler,
        "Geyser": geyser,
        "Refrigerator": refrigerator,
        "WashingMachine": washing_machine,
    }


def _sample_lifestyle(occupation: str, family: int, wfh: str, income: str) -> dict:
    occ = OCCUPATION_MAP[occupation]
    wfh_idx = WFH_MAP[wfh]

    if occupation == "Student":
        laptop_hours = float(np.clip(np.random.normal(5 + wfh_idx, 1.5), 1, 12))
        tv_hours = float(np.clip(np.random.normal(2 + family * 0.3, 1.2), 0.5, 8))
    elif occupation == "Employed":
        laptop_hours = float(np.clip(np.random.normal(3 + wfh_idx * 3, 1.5), 0.5, 14))
        tv_hours = float(np.clip(np.random.normal(1.5 + family * 0.4, 1.0), 0.5, 6))
    elif occupation == "Self-Employed":
        laptop_hours = float(np.clip(np.random.normal(4 + wfh_idx * 2.5, 1.8), 1, 14))
        tv_hours = float(np.clip(np.random.normal(2 + family * 0.35, 1.2), 0.5, 7))
    elif occupation == "Retired":
        laptop_hours = float(np.clip(np.random.normal(1.5, 1.0), 0, 6))
        tv_hours = float(np.clip(np.random.normal(4 + family * 0.2, 1.5), 1, 10))
    else:  # Homemaker
        laptop_hours = float(np.clip(np.random.normal(1 + wfh_idx * 0.5, 0.8), 0, 5))
        tv_hours = float(np.clip(np.random.normal(2.5 + family * 0.35, 1.2), 0.5, 8))

    return {
        "LaptopHours": round(laptop_hours, 1),
        "TVHours": round(tv_hours, 1),
    }


def _sample_climate(season: str) -> tuple:
    season_climate = {
        "Summer": (32, 42, 45, 75),
        "Monsoon": (26, 34, 70, 92),
        "Winter": (8, 22, 35, 65),
        "Spring": (22, 32, 40, 60),
    }
    t_lo, t_hi, h_lo, h_hi = season_climate[season]
    temp = float(np.clip(np.random.normal((t_lo + t_hi) / 2, (t_hi - t_lo) / 4), t_lo, t_hi))
    humidity = float(np.clip(np.random.normal((h_lo + h_hi) / 2, (h_hi - h_lo) / 5), h_lo, h_hi))
    return round(temp, 1), round(humidity, 1)


def _ac_hours_from_climate(temp: float, humidity: float, season: str, ac_count: int) -> float:
    if ac_count == 0:
        return 0.0
    comfort = 26.0
    heat_stress = max(0, temp - comfort)
    humidity_factor = 1.0 + max(0, humidity - 60) * 0.005
    if season == "Summer":
        base = 6 + heat_stress * 0.35
    elif season == "Monsoon":
        base = 3 + heat_stress * 0.25
    elif season == "Winter":
        base = max(0, heat_stress * 0.15)
    else:
        base = 2 + heat_stress * 0.2
    hours = base * humidity_factor * (0.85 + 0.05 * ac_count)
    return round(float(np.clip(hours, 0, 16)), 1)


def _compute_monthly_units(
    family: int,
    ac_count: int,
    ac_hours: float,
    fan_count: int,
    light_count: int,
    led_pct: float,
    cooler: int,
    geyser: int,
    refrigerator: int,
    washing_machine: int,
    laptop_hours: float,
    tv_hours: float,
    temp: float,
    humidity: float,
    season: str,
) -> float:
    """Physics-based monthly kWh calculation."""
    noise = np.random.normal(0, 8)

    # AC load
    ac_kw = _rand_range(*AC_POWER_KW)[0] if ac_count > 0 else 0
    ac_kwh = ac_count * ac_kw * ac_hours * DAYS_IN_MONTH

    # Fan load (higher in non-AC rooms and summer)
    fan_hours = 8 + max(0, temp - 28) * 0.15
    if season == "Winter":
        fan_hours *= 0.4
    fan_w = _rand_range(*FAN_POWER_W)[0]
    fan_kwh = fan_count * (fan_w / 1000) * fan_hours * DAYS_IN_MONTH

    # Lighting
    led_frac = led_pct / 100.0
    bulb_frac = 1.0 - led_frac
    light_hours = 5 + family * 0.3
    led_w = _rand_range(*LED_POWER_W)[0]
    bulb_w = _rand_range(*BULB_POWER_W)[0]
    avg_light_w = led_frac * led_w + bulb_frac * bulb_w
    light_kwh = light_count * (avg_light_w / 1000) * light_hours * DAYS_IN_MONTH

    # Cooler (summer/monsoon dominant)
    cooler_kwh = 0.0
    if cooler:
        if season in ("Summer", "Monsoon"):
            cooler_hours = 4 + max(0, temp - 30) * 0.2
        else:
            cooler_hours = 0.5
        cooler_w = _rand_range(*COOLER_POWER_W)[0]
        cooler_kwh = (cooler_w / 1000) * cooler_hours * DAYS_IN_MONTH

    # Geyser (winter dominant)
    geyser_kwh = 0.0
    if geyser:
        if season == "Winter":
            geyser_cycles = family * 0.9
        elif season == "Spring":
            geyser_cycles = family * 0.4
        else:
            geyser_cycles = family * 0.15
        geyser_kw = _rand_range(*GEYSER_POWER_KW)[0]
        geyser_kwh = geyser_cycles * geyser_kw * (20 / 60) * DAYS_IN_MONTH / 30

    # Refrigerator (continuous)
    ref_kwh_day = _rand_range(*REFRIGERATOR_KWH_DAY)[0]
    ref_kwh = refrigerator * ref_kwh_day * DAYS_IN_MONTH

    # Washing machine
    wash_kwh = 0.0
    if washing_machine:
        cycles_per_week = 2 + family * 0.35
        cycles_month = cycles_per_week * (DAYS_IN_MONTH / 7)
        wash_kwh = cycles_month * _rand_range(*WASHING_MACHINE_KWH_CYCLE)[0]

    # Laptop & TV
    laptop_w = _rand_range(*LAPTOP_POWER_W)[0]
    tv_w = _rand_range(*TV_POWER_W)[0]
    laptop_kwh = (laptop_w / 1000) * laptop_hours * DAYS_IN_MONTH
    tv_kwh = (tv_w / 1000) * tv_hours * DAYS_IN_MONTH

    # Standby & misc
    misc_kwh = 15 + family * 2

    total = (
        ac_kwh + fan_kwh + light_kwh + cooler_kwh + geyser_kwh
        + ref_kwh + wash_kwh + laptop_kwh + tv_kwh + misc_kwh + noise
    )
    return max(50.0, total)


def compute_estimated_base_load(row: dict) -> float:
    """Deterministic physics estimate for feature engineering (no random noise)."""
    ac_kw = 1.6
    ac_kwh = row["ACCount"] * ac_kw * row["ACHours"] * DAYS_IN_MONTH

    fan_hours = 8 + max(0, row["Temperature"] - 28) * 0.15
    if row["Season"] == "Winter":
        fan_hours *= 0.4
    fan_kwh = row["FanCount"] * 0.0675 * fan_hours * DAYS_IN_MONTH

    led_frac = row["LEDPercentage"] / 100.0
    avg_light_w = led_frac * 11.5 + (1 - led_frac) * 21.5
    light_hours = 5 + row["FamilyMembers"] * 0.3
    light_kwh = row["LightCount"] * (avg_light_w / 1000) * light_hours * DAYS_IN_MONTH

    cooler_kwh = 0.0
    if row["Cooler"]:
        if row["Season"] in ("Summer", "Monsoon"):
            cooler_hours = 4 + max(0, row["Temperature"] - 30) * 0.2
        else:
            cooler_hours = 0.5
        cooler_kwh = 0.24 * cooler_hours * DAYS_IN_MONTH

    geyser_kwh = 0.0
    if row["Geyser"]:
        if row["Season"] == "Winter":
            geyser_cycles = row["FamilyMembers"] * 0.9
        elif row["Season"] == "Spring":
            geyser_cycles = row["FamilyMembers"] * 0.4
        else:
            geyser_cycles = row["FamilyMembers"] * 0.15
        geyser_kwh = geyser_cycles * 2.0 * (20 / 60) * DAYS_IN_MONTH / 30

    ref_kwh = row["Refrigerator"] * 1.4 * DAYS_IN_MONTH

    wash_kwh = 0.0
    if row["WashingMachine"]:
        cycles_month = (2 + row["FamilyMembers"] * 0.35) * (DAYS_IN_MONTH / 7)
        wash_kwh = cycles_month * 1.15

    laptop_kwh = 0.0675 * row["LaptopHours"] * DAYS_IN_MONTH
    tv_kwh = 0.125 * row["TVHours"] * DAYS_IN_MONTH
    misc_kwh = 15 + row["FamilyMembers"] * 2

    return (
        ac_kwh + fan_kwh + light_kwh + cooler_kwh + geyser_kwh
        + ref_kwh + wash_kwh + laptop_kwh + tv_kwh + misc_kwh
    )


def _generate_household_state() -> dict:
    """Generate one coherent household profile."""
    house_type = np.random.choice(HOUSE_TYPES, p=[0.2, 0.35, 0.3, 0.15])
    income = np.random.choice(INCOME_LEVELS, p=[0.25, 0.4, 0.25, 0.1])
    family = int(np.clip(np.random.poisson(3.5) + 1, 1, 8))
    occupation = np.random.choice(OCCUPATIONS, p=[0.15, 0.4, 0.15, 0.1, 0.2])
    wfh = np.random.choice(WORK_FROM_HOME, p=[0.35, 0.45, 0.2])
    season = np.random.choice(SEASONS, p=[0.3, 0.25, 0.25, 0.2])

    temp, humidity = _sample_climate(season)
    ac_count = _sample_ac_count(income, house_type, family)
    ac_hours = _ac_hours_from_climate(temp, humidity, season, ac_count)

    appliances = _sample_appliance_counts(house_type, income, family)
    lifestyle = _sample_lifestyle(occupation, family, wfh, income)

    return {
        "FamilyMembers": family,
        "HouseType": house_type,
        "Occupation": occupation,
        "IncomeLevel": income,
        "WorkFromHome": wfh,
        "Season": season,
        "Temperature": temp,
        "Humidity": humidity,
        "ACCount": ac_count,
        "ACHours": ac_hours,
        **appliances,
        **lifestyle,
    }


def _apply_drift(prev: dict) -> dict:
    """Simulate realistic month-to-month changes for previous month state."""
    curr = prev.copy()

    if np.random.random() < 0.12:
        curr["FamilyMembers"] = int(np.clip(
            curr["FamilyMembers"] + np.random.choice([-1, 1]), 1, 8
        ))

    if np.random.random() < 0.08:
        idx = HOUSE_TYPES.index(curr["HouseType"])
        new_idx = int(np.clip(idx + np.random.choice([-1, 0, 1]), 0, 3))
        curr["HouseType"] = HOUSE_TYPES[new_idx]

    if np.random.random() < 0.15:
        curr["ACCount"] = int(np.clip(
            curr["ACCount"] + np.random.choice([-1, 0, 1]), 0, 4
        ))

    if np.random.random() < 0.25:
        seasons = SEASONS
        idx = seasons.index(curr["Season"])
        curr["Season"] = seasons[(idx - 1) % 4]

    curr["Temperature"], curr["Humidity"] = _sample_climate(curr["Season"])
    curr["ACHours"] = _ac_hours_from_climate(
        curr["Temperature"], curr["Humidity"], curr["Season"], curr["ACCount"]
    )

    if np.random.random() < 0.1:
        curr["IncomeLevel"] = np.random.choice(INCOME_LEVELS)

    appliances = _sample_appliance_counts(
        curr["HouseType"], curr["IncomeLevel"], curr["FamilyMembers"]
    )
    curr.update(appliances)

    lifestyle = _sample_lifestyle(
        curr["Occupation"], curr["FamilyMembers"], curr["WorkFromHome"], curr["IncomeLevel"]
    )
    curr.update(lifestyle)

    return curr


def calculate_bill(units: float) -> float:
    """Indian domestic slab tariff (approximate)."""
    bill = 0.0
    remaining = units
    slabs = [(100, 3.0), (100, 4.5), (100, 6.5), (float("inf"), 8.5)]
    for limit, rate in slabs:
        chunk = min(remaining, limit)
        bill += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break
    fixed_charge = 50
    return round(bill + fixed_charge, 2)


def calculate_carbon(units: float) -> float:
    """kg CO2 — India grid factor ~0.82 kg/kWh."""
    return round(units * 0.82, 2)


def generate_dataset(n: int = NUM_ROWS) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        current = _generate_household_state()
        previous = _apply_drift(current)

        prev_units = _compute_monthly_units(
            previous["FamilyMembers"],
            previous["ACCount"],
            previous["ACHours"],
            previous["FanCount"],
            previous["LightCount"],
            previous["LEDPercentage"],
            previous["Cooler"],
            previous["Geyser"],
            previous["Refrigerator"],
            previous["WashingMachine"],
            previous["LaptopHours"],
            previous["TVHours"],
            previous["Temperature"],
            previous["Humidity"],
            previous["Season"],
        )

        monthly_units = _compute_monthly_units(
            current["FamilyMembers"],
            current["ACCount"],
            current["ACHours"],
            current["FanCount"],
            current["LightCount"],
            current["LEDPercentage"],
            current["Cooler"],
            current["Geyser"],
            current["Refrigerator"],
            current["WashingMachine"],
            current["LaptopHours"],
            current["TVHours"],
            current["Temperature"],
            current["Humidity"],
            current["Season"],
        )

        row = {
            **current,
            "PreviousMonthUnits": round(prev_units, 1),
            "MonthlyUnits": round(monthly_units, 1),
            "ElectricityBill": calculate_bill(monthly_units),
            "CarbonEmission": calculate_carbon(monthly_units),
        }
        row["EstimatedBaseLoad"] = round(compute_estimated_base_load(row), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs("dataset", exist_ok=True)
    print(f"Generating {NUM_ROWS} realistic household records...")
    df = generate_dataset(NUM_ROWS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"MonthlyUnits — mean: {df['MonthlyUnits'].mean():.1f}, std: {df['MonthlyUnits'].std():.1f}")
    print(f"Range: {df['MonthlyUnits'].min():.1f} – {df['MonthlyUnits'].max():.1f} kWh")


if __name__ == "__main__":
    main()
