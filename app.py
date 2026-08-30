import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="AI Financial Planner",
    page_icon="💰",
    layout="wide"
)

# FastAPI backend address
API_URL = "http://127.0.0.1:8000"

st.title("💰 Financial Plan Generator")
st.caption("Personalized salary prediction, goal feasibility analysis, and RAG assistant")

# Create navigation tabs
tab1, tab2 = st.tabs(["🎯 Goal Planner", "🤖 Financial RAG Assistant"])

# =========================================================
# TAB 1: Financial Goal Planner
# =========================================================
with tab1:
    st.subheader("Personal & Financial Details")
    
    with st.form(key="planner_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("Full Name", value="Ranit Roy")
            age = st.number_input("Age", min_value=18, max_value=80, value=25)
            savings_pct = st.slider("Target Monthly Savings (%)", min_value=5.0, max_value=80.0, value=20.0, step=1.0)
            
        with col2:
            city = st.selectbox("Current City", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Pune"])
            education = st.selectbox("Education Level", ["B.Tech", "B.Sc", "M.Tech", "MBA", "B.Com"])
            job_role = st.selectbox("Job Role", ["Software Engineer", "Data Analyst", "Financial Analyst", "Product Manager"])
            
        with col3:
            st.markdown("**Goal Timelines (Years)**")
            marriage_yrs = st.number_input("💍 Marriage", min_value=1, max_value=30, value=5)
            car_yrs = st.number_input("🚗 Car Purchase", min_value=1, max_value=30, value=3)
            home_yrs = st.number_input("🏠 Home Purchase", min_value=1, max_value=40, value=15)
            
        submit_button = st.form_submit_button(label="Generate Plan", use_container_width=True)

    if submit_button:
        # Construct payload matching main.py PlanRequest model exactly
        payload = {
            "name": name,
            "age": int(age),
            "city": city,
            "education": education,
            "job_role": job_role,
            "savings_target_pct": float(savings_pct),
            "goals": {
                "marriage_years": int(marriage_yrs),
                "car_years": int(car_yrs),
                "home_years": int(home_yrs)
            }
        }
        
        with st.spinner("Processing salary model and calculating targets..."):
            try:
                response = requests.post(f"{API_URL}/generate-plan", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    user = data["user_profile"]
                    goals = data["goals_breakdown"]
                    feasibility = data["feasibility_analysis"]
                    
                    st.divider()
                    
                    # Top Metrics Banner
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Predicted Monthly Salary", f"₹{user['estimated_monthly_salary']:,.2f}")
                    m2.metric("Monthly Savings Capacity", f"₹{feasibility['monthly_capacity']:,.2f}")
                    
                    # Read correct total key based on tools module version
                    total_req = feasibility.get("total_required_sip") or feasibility.get("total_required", 0)
                    shortfall = feasibility.get("shortfall_or_surplus", 0)
                    
                    m3.metric("Required Monthly Investment", f"₹{total_req:,.2f}")
                    m4.metric("Net Surplus / Shortfall", f"₹{shortfall:,.2f}")
                    
                    # Status Callout
                    status = feasibility["status"]
                    advice = feasibility.get("actionable_advice", "")
                    
                    if status == "Achievable":
                        st.success(f"**Feasibility Status: {status}**\n\n{advice}")
                    elif status == "Challenging":
                        st.warning(f"**Feasibility Status: {status}**\n\n{advice}")
                    else:
                        st.error(f"**Feasibility Status: {status}**\n\n{advice}")
                    
                    # Individual Goal Cards
                    st.subheader("🎯 Goal Breakdown")
                    g1, g2, g3 = st.columns(3)
                    
                    goal_cards = [
                        ("💍 Marriage", goals.get("Marriage"), g1),
                        ("🚗 Car", goals.get("Car"), g2),
                        ("🏠 Home", goals.get("Home"), g3)
                    ]
                    
                    for title, details, col in goal_cards:
                        if details:
                            with col:
                                st.markdown(f"### {title}")
                                st.write(f"**Future Cost:** ₹{details['future_cost']:,.2f}")
                                
                                if "required_monthly_sip" in details:
                                    st.write(f"**Monthly SIP:** ₹{details['required_monthly_sip']:,.2f}")
                                    st.write(f"**Down Payment:** ₹{details['down_payment_required']:,.2f}")
                                    st.write(f"**Estimated EMI:** ₹{details['future_estimated_emi']:,.2f}")
                                else:
                                    st.write(f"**Monthly Investment:** ₹{details['required_monthly_investment']:,.2f}")
                                    
                                st.info(f"**Strategy:**\n{details['recommended_category']}")
                else:
                    error_msg = response.json().get("detail", "Validation Failed")
                    st.error(f"Backend Error ({response.status_code}): {error_msg}")
                    
            except Exception as e:
                st.error(f"Failed to reach server at `{API_URL}`. Make sure Uvicorn is active. Details: {e}")

# =========================================================
# TAB 2: RAG Knowledge Base Assistant
# =========================================================
with tab2:
    st.subheader("💬 Financial RAG Knowledge Base Assistant")
    st.caption("Ask questions strictly processed against local internal documents.")
    
    user_query = st.text_input("Enter your question:", placeholder="e.g., How much emergency fund should a fresher keep?")
    
    if st.button("Search Knowledge Base", use_container_width=True):
        if user_query.strip():
            with st.spinner("Searching internal KB..."):
                try:
                    res = requests.get(f"{API_URL}/rag-query", params={"question": user_query})
                    if res.status_code == 200:
                        rag_output = res.json()
                        st.markdown("### Answer")
                        st.write(rag_output["answer"])
                    else:
                        st.error(f"Error ({res.status_code}): Unable to fetch RAG response.")
                except Exception as e:
                    st.error(f"API Connection error: {e}")
        else:
            st.warning("Please enter a question before searching.")