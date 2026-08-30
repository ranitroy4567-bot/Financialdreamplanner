from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tools import (
    predict_salary, 
    calculate_future_cost, 
    calculate_monthly_investment, 
    evaluate_feasibility, 
    recommend_investment_category
)
from rag_engine import query_rag

app = FastAPI(title="AI-Powered Financial Dream Planner")

class GoalInput(BaseModel):
    marriage_years: int = Field(..., ge=1, description="Years until marriage")
    car_years: int = Field(..., ge=1, description="Years until buying a car")
    home_years: int = Field(..., ge=1, description="Years until buying a home")

class PlanRequest(BaseModel):
    name: str
    age: int
    city: str
    education: str
    job_role: str
    savings_target_pct: float
    goals: GoalInput

@app.post("/generate-plan")
def generate_plan(req: PlanRequest):
    # Input Validation Guardrails[cite: 1]
    if req.savings_target_pct < 0 or req.savings_target_pct > 100:
        raise HTTPException(status_code=400, detail="Savings percentage must be between 0% and 100%")
    
    # 1. Salary Prediction[cite: 1]
    monthly_salary = predict_salary(req.city, req.education, req.job_role)
    
    # 2. Goal-wise Calculations[cite: 1]
    goal_timelines = {
        "Marriage": req.goals.marriage_years,
        "Car": req.goals.car_years,
        "Home": req.goals.home_years
    }
    
    plan_details = {}
    total_required_monthly = 0
    
    for goal, years in goal_timelines.items():
        future_cost = calculate_future_cost(req.city, goal, years)
        required_sip = calculate_monthly_investment(future_cost, years)
        category = recommend_investment_category(years)
        
        total_required_monthly += required_sip
        plan_details[goal] = {
            "future_cost": future_cost,
            "required_monthly_investment": required_sip,
            "recommended_category": category
        }
        
    # 3. Feasibility Analysis[cite: 1]
    feasibility = evaluate_feasibility(monthly_salary, req.savings_target_pct, total_required_monthly)
    
    return {
        "user_profile": {"name": req.name, "estimated_monthly_salary": round(monthly_salary, 2)},
        "goals_breakdown": plan_details,
        "feasibility_analysis": feasibility
    }

@app.get("/rag-query")
def ask_rag(question: str):
    """Answers financial questions using RAG strictly grounded in local context[cite: 1]."""
    answer = query_rag(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)