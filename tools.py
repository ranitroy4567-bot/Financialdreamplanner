import os
import joblib
import pandas as pd

# Load ML artifacts
preprocessor = joblib.load("artifacts/preprocessor.pkl")
salary_model = joblib.load("artifacts/salary_model.pkl")

# Load City Base Costs directly from uploaded city_goal_costs.csv
def load_city_costs(costs_csv="city_goal_costs.csv"):
    if os.path.exists(costs_csv):
        df_costs = pd.read_csv(costs_csv)
        df_costs.columns = df_costs.columns.str.strip()
        # Average area costs per city
        city_avg = df_costs.groupby('City')[['Marriage_Cost_Current', 'Car_Cost_Current', 'Home_Cost_Current']].mean().to_dict(orient='index')
        return city_avg
    return {}

CITY_COSTS_DATA = load_city_costs()
DEFAULT_COSTS = {"Marriage_Cost_Current": 800000, "Car_Cost_Current": 1000000, "Home_Cost_Current": 8000000}

def predict_salary(city: str, education: str, job_role: str) -> float:
    """Predicts monthly starting salary directly."""
    df_input = pd.DataFrame([[city, education, job_role]], columns=["City", "Education", "Job Role"])
    encoded_input = preprocessor.transform(df_input)
    predicted_monthly = salary_model.predict(encoded_input)[0]
    return float(predicted_monthly)

def calculate_future_cost(city: str, goal_type: str, years: int) -> float:
    """Calculates future cost using 0.06% annual inflation."""
    city_data = CITY_COSTS_DATA.get(city, DEFAULT_COSTS)
    cost_key = f"{goal_type}_Cost_Current"
    base_cost = city_data.get(cost_key, DEFAULT_COSTS.get(cost_key, 1000000))
    inflation_rate = 6.0 
    future_cost = base_cost * ((1 + inflation_rate) ** years)
    return round(future_cost, 2)

def calculate_monthly_investment(future_cost: float, years: int, annual_return_rate: float = 0.08) -> float:
    months = years * 12
    monthly_rate = annual_return_rate / 12
    if monthly_rate == 0:
        return round(future_cost / months, 2)
    sip = future_cost / ((((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate))
    return round(sip, 2)

def evaluate_feasibility(monthly_salary: float, target_save_pct: float, total_required_investment: float) -> dict:
    capacity = monthly_salary * (target_save_pct / 100)
    shortfall = total_required_investment - capacity
    
    if shortfall <= 0:
        status = "Achievable"
    elif shortfall <= capacity * 0.3:
        status = "Challenging"
    else:
        status = "Highly Challenging"
        
    return {
        "monthly_capacity": round(capacity, 2),
        "total_required": round(total_required_investment, 2),
        "shortfall_or_surplus": round(-shortfall, 2),
        "status": status
    }

def recommend_investment_category(years: int) -> str:
    if years <= 3:
        return "Short-term: Lower-volatility / capital-preservation category"
    elif years <= 7:
        return "Medium-term: Diversified balanced category"
    else:
        return "Long-term: Diversified long-term growth-oriented category"