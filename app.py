"""
WattWise - Flask Application

Smart Household Electricity Consumption & Bill Prediction System.
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")

DAYS_IN_MONTH = 30

VALID_OPTIONS = {
    "HouseType": ["1BHK", "2BHK", "3BHK", "4BHK"],
    "Occupation": ["Student", "Employed", "Self-Employed", "Retired", "Homemaker"],
    "IncomeLevel": ["Low", "Medium", "High", "Very High"],
    "WorkFromHome": ["Yes", "No", "Partial"],
    "Season": ["Summer", "Monsoon", "Winter", "Spring"],
}

NUMERIC_BOUNDS = {
    "FamilyMembers": (1, 8),
    "Temperature": (5, 48),
    "Humidity": (20, 95),
    "ACCount": (0, 4),
    "ACHours": (0, 16),
    "FanCount": (1, 10),
    "LightCount": (4, 20),
    "LEDPercentage": (0, 100),
    "Cooler": (0, 1),
    "Geyser": (0, 1),
    "Refrigerator": (0, 1),
    "WashingMachine": (0, 1),
    "LaptopHours": (0, 16),
    "TVHours": (0, 12),
    "PreviousMonthUnits": (50, 2000),
}

model = None
metadata = None


def load_model():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)


def compute_estimated_base_load(data: dict) -> float:
    """Physics-based base load estimate (matches training pipeline)."""
    ac_kw = 1.6
    ac_kwh = data["ACCount"] * ac_kw * data["ACHours"] * DAYS_IN_MONTH

    fan_hours = 8 + max(0, data["Temperature"] - 28) * 0.15
    if data["Season"] == "Winter":
        fan_hours *= 0.4
    fan_kwh = data["FanCount"] * 0.0675 * fan_hours * DAYS_IN_MONTH

    led_frac = data["LEDPercentage"] / 100.0
    avg_light_w = led_frac * 11.5 + (1 - led_frac) * 21.5
    light_hours = 5 + data["FamilyMembers"] * 0.3
    light_kwh = data["LightCount"] * (avg_light_w / 1000) * light_hours * DAYS_IN_MONTH

    cooler_kwh = 0.0
    if data["Cooler"]:
        if data["Season"] in ("Summer", "Monsoon"):
            cooler_hours = 4 + max(0, data["Temperature"] - 30) * 0.2
        else:
            cooler_hours = 0.5
        cooler_kwh = 0.24 * cooler_hours * DAYS_IN_MONTH

    geyser_kwh = 0.0
    if data["Geyser"]:
        if data["Season"] == "Winter":
            geyser_cycles = data["FamilyMembers"] * 0.9
        elif data["Season"] == "Spring":
            geyser_cycles = data["FamilyMembers"] * 0.4
        else:
            geyser_cycles = data["FamilyMembers"] * 0.15
        geyser_kwh = geyser_cycles * 2.0 * (20 / 60) * DAYS_IN_MONTH / 30

    ref_kwh = data["Refrigerator"] * 1.4 * DAYS_IN_MONTH

    wash_kwh = 0.0
    if data["WashingMachine"]:
        cycles_month = (2 + data["FamilyMembers"] * 0.35) * (DAYS_IN_MONTH / 7)
        wash_kwh = cycles_month * 1.15

    laptop_kwh = 0.0675 * data["LaptopHours"] * DAYS_IN_MONTH
    tv_kwh = 0.125 * data["TVHours"] * DAYS_IN_MONTH
    misc_kwh = 15 + data["FamilyMembers"] * 2

    return round(
        ac_kwh + fan_kwh + light_kwh + cooler_kwh + geyser_kwh
        + ref_kwh + wash_kwh + laptop_kwh + tv_kwh + misc_kwh,
        2,
    )


def calculate_bill(units: float) -> float:
    bill = 0.0
    remaining = units
    slabs = [(100, 3.0), (100, 4.5), (100, 6.5), (float("inf"), 8.5)]
    for limit, rate in slabs:
        chunk = min(remaining, limit)
        bill += chunk * rate
        remaining -= chunk
        if remaining <= 0:
            break
    return round(bill + 50, 2)


def calculate_carbon(units: float) -> float:
    return round(units * 0.82, 2)


def get_consumption_category(units: float) -> str:
    if units < 150:
        return "Low"
    elif units < 300:
        return "Moderate"
    elif units < 500:
        return "High"
    else:
        return "Very High"


def estimate_savings(data: dict, predicted_units: float) -> float:
    """Estimate potential monthly savings from efficiency improvements."""
    savings = 0.0

    if data["LEDPercentage"] < 80:
        bulb_savings = data["LightCount"] * (1 - data["LEDPercentage"] / 100) * 0.015 * 5 * 30
        savings += bulb_savings * 5.0

    if data["ACCount"] > 0 and data["ACHours"] > 6:
        ac_reduction = (data["ACHours"] - 6) * data["ACCount"] * 1.6 * 30 * 0.15
        savings += ac_reduction * 5.0

    if data["Season"] == "Summer" and data["Cooler"] and not data["ACCount"]:
        savings += 30

    if predicted_units > 400:
        savings += 50

    return round(min(savings, predicted_units * 0.25 * 5.0), 2)


def generate_reasons(data: dict, predicted_units: float, base_load: float) -> list:
    reasons = []

    if data["ACCount"] > 0:
        ac_contrib = data["ACCount"] * data["ACHours"] * 1.6 * 30
        pct = min(100, (ac_contrib / max(predicted_units, 1)) * 100)
        reasons.append({
            "factor": "Air Conditioning",
            "impact": "High" if pct > 35 else "Moderate",
            "detail": f"{data['ACCount']} AC unit(s) running ~{data['ACHours']} hrs/day contributes ~{pct:.0f}% of load.",
        })

    if data["Season"] == "Summer" and data["Temperature"] > 32:
        reasons.append({
            "factor": "High Temperature",
            "impact": "High",
            "detail": f"At {data['Temperature']}°C in {data['Season']}, cooling demand is elevated.",
        })
    elif data["Season"] == "Winter" and data["Geyser"]:
        reasons.append({
            "factor": "Winter Heating",
            "impact": "Moderate",
            "detail": "Geyser usage increases significantly during winter months.",
        })

    if data["FamilyMembers"] >= 5:
        reasons.append({
            "factor": "Large Household",
            "impact": "Moderate",
            "detail": f"{data['FamilyMembers']} members increase lighting, fan, and appliance usage.",
        })

    if data["WorkFromHome"] in ("Yes", "Partial"):
        reasons.append({
            "factor": "Work From Home",
            "impact": "Moderate",
            "detail": f"WFH ({data['WorkFromHome']}) adds ~{data['LaptopHours']} hrs/day laptop usage.",
        })

    if data["LEDPercentage"] < 50:
        reasons.append({
            "factor": "Inefficient Lighting",
            "impact": "Low",
            "detail": f"Only {data['LEDPercentage']}% LED — switching to LED can reduce lighting cost by 60%.",
        })

    if data["Refrigerator"]:
        reasons.append({
            "factor": "Refrigerator",
            "impact": "Low",
            "detail": "Continuous 24/7 load (~42 kWh/month) from refrigerator operation.",
        })

    if data["PreviousMonthUnits"] > 0:
        trend = predicted_units - data["PreviousMonthUnits"]
        direction = "higher" if trend > 0 else "lower"
        reasons.append({
            "factor": "Historical Trend",
            "impact": "Low",
            "detail": f"Previous month was {data['PreviousMonthUnits']:.0f} kWh; current profile suggests {direction} usage.",
        })

    reasons.append({
        "factor": "Physics Base Estimate",
        "impact": "Info",
        "detail": f"Estimated base load from appliance physics: {base_load:.0f} kWh/month.",
    })

    return reasons


def build_usage_breakdown(data: dict) -> dict:
    """Component-wise kWh breakdown for chart."""
    ac_kwh = data["ACCount"] * 1.6 * data["ACHours"] * DAYS_IN_MONTH

    fan_hours = 8 + max(0, data["Temperature"] - 28) * 0.15
    if data["Season"] == "Winter":
        fan_hours *= 0.4
    fan_kwh = data["FanCount"] * 0.0675 * fan_hours * DAYS_IN_MONTH

    led_frac = data["LEDPercentage"] / 100.0
    avg_light_w = led_frac * 11.5 + (1 - led_frac) * 21.5
    light_hours = 5 + data["FamilyMembers"] * 0.3
    light_kwh = data["LightCount"] * (avg_light_w / 1000) * light_hours * DAYS_IN_MONTH

    cooler_kwh = 0.0
    if data["Cooler"]:
        if data["Season"] in ("Summer", "Monsoon"):
            cooler_hours = 4 + max(0, data["Temperature"] - 30) * 0.2
        else:
            cooler_hours = 0.5
        cooler_kwh = 0.24 * cooler_hours * DAYS_IN_MONTH

    geyser_kwh = 0.0
    if data["Geyser"]:
        if data["Season"] == "Winter":
            geyser_cycles = data["FamilyMembers"] * 0.9
        elif data["Season"] == "Spring":
            geyser_cycles = data["FamilyMembers"] * 0.4
        else:
            geyser_cycles = data["FamilyMembers"] * 0.15
        geyser_kwh = geyser_cycles * 2.0 * (20 / 60) * DAYS_IN_MONTH / 30

    ref_kwh = data["Refrigerator"] * 1.4 * DAYS_IN_MONTH
    wash_kwh = 0.0
    if data["WashingMachine"]:
        cycles_month = (2 + data["FamilyMembers"] * 0.35) * (DAYS_IN_MONTH / 7)
        wash_kwh = cycles_month * 1.15

    laptop_kwh = 0.0675 * data["LaptopHours"] * DAYS_IN_MONTH
    tv_kwh = 0.125 * data["TVHours"] * DAYS_IN_MONTH
    misc_kwh = 15 + data["FamilyMembers"] * 2

    return {
        "AC": round(ac_kwh, 1),
        "Fans": round(fan_kwh, 1),
        "Lighting": round(light_kwh, 1),
        "Cooler": round(cooler_kwh, 1),
        "Geyser": round(geyser_kwh, 1),
        "Refrigerator": round(ref_kwh, 1),
        "Washing Machine": round(wash_kwh, 1),
        "Laptop": round(laptop_kwh, 1),
        "TV": round(tv_kwh, 1),
        "Miscellaneous": round(misc_kwh, 1),
    }


INTEGER_FIELDS = {
    "FamilyMembers", "ACCount", "FanCount", "LightCount",
    "Cooler", "Geyser", "Refrigerator", "WashingMachine",
}


def validate_input(data: dict) -> tuple:
    errors = []

    for field, options in VALID_OPTIONS.items():
        val = data.get(field, "")
        if val not in options:
            errors.append(f"Invalid {field}: '{val}'. Must be one of {options}.")

    for field, (lo, hi) in NUMERIC_BOUNDS.items():
        try:
            raw = data.get(field, None)
            if raw is None or raw == "":
                errors.append(f"{field} is required.")
                continue
            val = float(raw)
            if val < lo or val > hi:
                errors.append(f"{field} must be between {lo} and {hi}.")
                continue
            data[field] = int(val) if field in INTEGER_FIELDS else val
        except (TypeError, ValueError):
            errors.append(f"{field} must be a valid number.")

    return errors, data


def parse_request_data(req) -> dict:
    return {
        "FamilyMembers": req.form.get("FamilyMembers", req.json.get("FamilyMembers") if req.is_json else None),
        "HouseType": req.form.get("HouseType", req.json.get("HouseType") if req.is_json else None),
        "Occupation": req.form.get("Occupation", req.json.get("Occupation") if req.is_json else None),
        "IncomeLevel": req.form.get("IncomeLevel", req.json.get("IncomeLevel") if req.is_json else None),
        "WorkFromHome": req.form.get("WorkFromHome", req.json.get("WorkFromHome") if req.is_json else None),
        "Season": req.form.get("Season", req.json.get("Season") if req.is_json else None),
        "Temperature": req.form.get("Temperature", req.json.get("Temperature") if req.is_json else None),
        "Humidity": req.form.get("Humidity", req.json.get("Humidity") if req.is_json else None),
        "ACCount": req.form.get("ACCount", req.json.get("ACCount") if req.is_json else None),
        "ACHours": req.form.get("ACHours", req.json.get("ACHours") if req.is_json else None),
        "FanCount": req.form.get("FanCount", req.json.get("FanCount") if req.is_json else None),
        "LightCount": req.form.get("LightCount", req.json.get("LightCount") if req.is_json else None),
        "LEDPercentage": req.form.get("LEDPercentage", req.json.get("LEDPercentage") if req.is_json else None),
        "Cooler": req.form.get("Cooler", req.json.get("Cooler") if req.is_json else None),
        "Geyser": req.form.get("Geyser", req.json.get("Geyser") if req.is_json else None),
        "Refrigerator": req.form.get("Refrigerator", req.json.get("Refrigerator") if req.is_json else None),
        "WashingMachine": req.form.get("WashingMachine", req.json.get("WashingMachine") if req.is_json else None),
        "LaptopHours": req.form.get("LaptopHours", req.json.get("LaptopHours") if req.is_json else None),
        "TVHours": req.form.get("TVHours", req.json.get("TVHours") if req.is_json else None),
        "PreviousMonthUnits": req.form.get("PreviousMonthUnits", req.json.get("PreviousMonthUnits") if req.is_json else None),
    }


@app.route("/")
def index():
    return render_template("index.html", options=VALID_OPTIONS, metadata=metadata or {})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

        raw = parse_request_data(request)
        errors, data = validate_input(raw)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        base_load = compute_estimated_base_load(data)
        data["EstimatedBaseLoad"] = base_load

        import pandas as pd
        feature_cols = metadata["feature_columns"] if metadata else [
            "HouseType", "Occupation", "IncomeLevel", "WorkFromHome", "Season",
            "FamilyMembers", "Temperature", "Humidity",
            "ACCount", "ACHours", "FanCount", "LightCount", "LEDPercentage",
            "Cooler", "Geyser", "Refrigerator", "WashingMachine",
            "LaptopHours", "TVHours", "PreviousMonthUnits", "EstimatedBaseLoad",
        ]
        X = pd.DataFrame([{k: data[k] for k in feature_cols}])

        predicted_units = float(model.predict(X)[0])
        predicted_units = max(50.0, round(predicted_units, 1))

        bill = calculate_bill(predicted_units)
        carbon = calculate_carbon(predicted_units)
        category = get_consumption_category(predicted_units)
        savings = estimate_savings(data, predicted_units)
        reasons = generate_reasons(data, predicted_units, base_load)
        breakdown = build_usage_breakdown(data)

        return jsonify({
            "success": True,
            "monthly_units": predicted_units,
            "electricity_bill": bill,
            "carbon_emission": carbon,
            "monthly_savings": savings,
            "consumption_category": category,
            "estimated_base_load": base_load,
            "reasons": reasons,
            "usage_breakdown": breakdown,
            "model_name": metadata.get("best_model", "Random Forest") if metadata else "Random Forest",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": metadata.get("best_model") if metadata else None,
    })


if __name__ == "__main__":
    try:
        load_model()
        print(f"Model loaded: {metadata.get('best_model', 'Unknown') if metadata else 'Unknown'}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    app.run(debug=True, host="0.0.0.0", port=5000)
