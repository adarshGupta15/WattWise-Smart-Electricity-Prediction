
## Live Demo

🔗 **Live Application:** https://wattwise-smart-electricity-prediction.onrender.com
Built as a **B.Tech CSE Major Project** demonstrating end-to-end ML pipeline engineering, physics-based feature engineering, and modern full-stack development.

---

## Features

- **Realistic Predictions** — Physics-based appliance power calculations (AC, fans, lighting, geyser, etc.)
- **ML Model Comparison** — Automatically selects the best among Linear Regression, Decision Tree, Random Forest, and Gradient Boosting
- **Feature Engineering** — `EstimatedBaseLoad` derived from Power × Hours × Days physics
- **No Target Leakage** — Previous month units simulated from an independent prior household state
- **Interactive Dashboard** — Glassmorphism UI with Chart.js visualizations
- **Prediction Insights** — Detailed reasons explaining consumption drivers
- **Bill & Carbon Calculation** — Indian domestic slab tariff and grid emission factor

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, Flask |
| ML | Scikit-Learn, Pandas, NumPy, Joblib |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Model | Random Forest Regressor (auto-selected best) |

---

## Project Structure

```
smart-electricity-prediction/
│
├── app.py                  # Flask web application & API
├── generate_dataset.py     # Synthetic dataset generator (20,000+ rows)
├── train_model.py          # Model training & comparison pipeline
├── requirements.txt        # Python dependencies
├── README.md
│
├── models/
│   ├── model.pkl           # Trained model (generated)
│   └── model_metadata.json # Model metrics & feature info
│
├── dataset/
│   └── electricity_data.csv # Synthetic dataset (generated)
│
├── templates/
│   └── index.html          # Main dashboard template
│
└── static/
    ├── style.css           # Glassmorphism styles
    └── script.js           # Frontend logic & charts
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

```bash
# Clone or navigate to project directory
cd smart-electricity-prediction

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the three scripts in order:

### Step 1: Generate Dataset

```bash
python generate_dataset.py
```

Generates 20,000 realistic household records with physics-based consumption calculations. Output: `dataset/electricity_data.csv`

### Step 2: Train Model

```bash
python train_model.py
```

Compares four regression models, selects the best by R² score, and saves to `models/model.pkl`.

### Step 3: Run Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## How ML Works

### 1. Dataset Generation

Each row represents a realistic Indian household with correlated features:

- **Income → AC ownership probability**
- **Occupation → Laptop hours, WFH patterns**
- **Family size → TV hours, laundry cycles**
- **Temperature & Season → AC hours, geyser/cooler usage**
- **House type → Fan count, light count**

Monthly units are computed using:

```
kWh = Power(kW) × Hours/day × Days/month
```

Appliance power ratings follow real-world ranges (AC: 1.2–2.0 kW, Fan: 55–80W, Geyser: 1.8–2.2 kW, etc.).

### 2. Target Leakage Prevention

`PreviousMonthUnits` is generated from a **drifted prior household state** — family size, AC count, season, and appliances may differ from the current month. This ensures the model learns from current profile features, not just historical consumption.

### 3. Feature Engineering

`EstimatedBaseLoad` is a deterministic physics calculation added as a feature. The ML model learns corrections over this baseline, improving accuracy while maintaining interpretability.

### 4. Model Selection

Four models are trained and evaluated:

| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline interpretability |
| Decision Tree | Non-linear splits |
| Random Forest | Ensemble robustness |
| Gradient Boosting | Sequential error correction |

The model with the highest R² on the test set is saved automatically.

### 5. Prediction Pipeline

```
User Input → Validation → EstimatedBaseLoad → ML Model → Bill/Carbon/Savings → Dashboard
```

---

## Dataset Explanation

| Column | Description |
|--------|-------------|
| FamilyMembers | 1–8 household members |
| HouseType | 1BHK, 2BHK, 3BHK, 4BHK |
| Occupation | Student, Employed, Self-Employed, Retired, Homemaker |
| IncomeLevel | Low, Medium, High, Very High |
| WorkFromHome | Yes, No, Partial |
| Season | Summer, Monsoon, Winter, Spring |
| Temperature | 5–48°C |
| Humidity | 20–95% |
| ACCount | 0–4 air conditioners |
| ACHours | 0–16 hours/day |
| FanCount | 1–10 ceiling fans |
| LightCount | 4–20 lights |
| LEDPercentage | 0–100% LED adoption |
| Cooler/Geyser/Refrigerator/WashingMachine | Binary appliance flags |
| LaptopHours/TVHours | Daily usage hours |
| PreviousMonthUnits | Prior month kWh (trend only) |
| EstimatedBaseLoad | Physics-based estimate |
| MonthlyUnits | Target variable (kWh) |

---

## Screenshots

> Add screenshots of the dashboard here after running the application.

1. **Input Form** — Household profile entry with all appliance fields
2. **Prediction Dashboard** — Monthly units, bill, carbon, savings cards
3. **Usage Breakdown Chart** — Doughnut chart of appliance-wise consumption
4. **Daily Load Profile** — 24-hour load curve
5. **Prediction Insights** — Factor-wise impact analysis

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard |
| POST | `/api/predict` | Run prediction (JSON body) |
| GET | `/api/health` | Health check & model status |

### Example API Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "FamilyMembers": 4,
    "HouseType": "2BHK",
    "Occupation": "Employed",
    "IncomeLevel": "Medium",
    "WorkFromHome": "Partial",
    "Season": "Summer",
    "Temperature": 35,
    "Humidity": 65,
    "ACCount": 1,
    "ACHours": 6,
    "FanCount": 4,
    "LightCount": 8,
    "LEDPercentage": 60,
    "Cooler": 0,
    "Geyser": 1,
    "Refrigerator": 1,
    "WashingMachine": 1,
    "LaptopHours": 4,
    "TVHours": 3,
    "PreviousMonthUnits": 250
  }'
```

---

## Future Improvements

- [ ] Real smart meter IoT integration via MQTT/API
- [ ] Time-series forecasting with LSTM/Prophet for next 3 months
- [ ] User authentication and consumption history tracking
- [ ] Regional tariff configuration (state-wise DISCOM rates)
- [ ] Solar panel offset modeling
- [ ] Mobile-responsive PWA with offline support
- [ ] SHAP explainability for individual predictions
- [ ] Anomaly detection for unusual consumption spikes
- [ ] Integration with weather APIs for live temperature data
- [ ] Multi-language support (Hindi, regional languages)

---

## License

This project is developed for academic purposes as a B.Tech CSE Major Project.

---

## Author

WattWise — Smart Energy Analytics Project, 2026 by Adarsh Gupta 

