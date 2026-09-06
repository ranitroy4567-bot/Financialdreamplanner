# 💰 AI-Powered Financial Dream & Goal Planner

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-121212.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end local financial planning ecosystem designed for freshers and early-career professionals. The application combines **Machine Learning for salary prediction**, **financial timeline modeling**, an **Agentic tool orchestrator**, and a **RAG (Retrieval-Augmented Generation) Assistant** grounded in local financial guidelines.

---

## 📌 Executive Summary

Planning long-term life goals—such as marriage, buying a car, or purchasing a home—can be overwhelming for early-career professionals. The **AI-Powered Financial Dream & Goal Planner** solves this by delivering hyper-personalized, inflation-adjusted financial blueprints. 

By evaluating location, education, job role, and savings behavior, the platform estimates starting salary capabilities, projects compounding future goal costs, and calculates exact SIP (Systematic Investment Plan) requirements and loan EMI commitments.

---

## 🔥 Key Features

* **🤖 ML-Driven Salary Prediction**: Predicts realistic monthly starting salaries using `scikit-learn` regression pipelines based on City, Education, and Job Role.
* **📈 Goal Cost Projections**: Projects future costs using category-specific compound inflation rates (7% Marriage, 5% Car, 8% Home Real Estate).
* **🎯 Down Payment & Loan Breakdown**: Computes required monthly SIP investments to accumulate target down payments alongside future post-goal loan EMIs (e.g., 20% down payment + 80% auto/home loan).
* **⚖️ Feasibility Engine**: Evaluates current monthly savings capacity against total required monthly investments to categorize plans as **Achievable**, **Challenging**, or **Highly Challenging**.
* **💬 Local Financial RAG Assistant**: Interactive Q&A engine powered by `LangChain`, `ChromaDB`, and `HuggingFace` embeddings for strict, grounded internal knowledge retrieval.
* **🌐 Web UI & REST API**: Streamlit frontend paired with a modular FastAPI backend delivering real-time interactive financial planning.

---

## 📐 Architecture & System Flow

![Alt Text]([https://user-images.githubusercontent.com/.../image.png](https://github.com/user-attachments/assets/2d6f4d35-b778-4085-a094-a9e21020818e))

Project Structure

├── artifacts/                  # Model & Preprocessor artifacts
│   ├── preprocessor.pkl        # OneHotEncoder ColumnTransformer pipeline
│   └── salary_model.pkl        # Best trained regression model (RandomForest/Linear)
├── kb/                         # Local Knowledge Base documents for RAG
│   ├── financial_guidelines.txt
│   ├── investment_categories.txt
│   └── goal_planning_rules.txt
├── app.py                      # Streamlit interactive UI
├── main.py                     # FastAPI server application & endpoints
├── train.py                    # Model training, evaluation & artifact export script
├── tools.py                    # Financial math, inflation & feasibility business logic
├── rag_engine.py               # Document loading, vector storage & RAG query engine
├── city_goal_costs.csv         # Reference dataset for base goal costs per city
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

🧮 Financial Formulas & Business Logic

### 1. Future Goal Cost Projection
$$\text{Future Cost} = \text{Base Cost} \times (1 + r)^n$$
*Where $r$ = Annual inflation rate by goal category (Marriage: 7%, Car: 5%, Home: 8%), $n$ = Years until goal.*

### 2. Systematic Investment Plan (SIP) Calculation
$$\text{Monthly SIP} = \frac{\text{Target Down Payment}}{\left( \frac{(1 + i)^m - 1}{i} \right) \times (1 + i)}$$
*Where $i = \frac{\text{Annual Return}}{12}$ (assumed at 10% p.a.), $m = n \times 12$ months.*

### 3. Loan Equated Monthly Installment (EMI)
$$\text{EMI} = P \times r \times \frac{(1 + r)^n}{(1 + r)^n - 1}$$
*Where $P$ = Loan principal (e.g., 80% of vehicle/home cost), $r$ = Monthly interest rate, $n$ = Loan tenure in months.*

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9, 3.10, or 3.11 installed.
* Virtual environment configured (`venv` or `conda`).

### Installation

1. **Clone the repository**:
   ```bash
 git clone [https://github.com/ranitroy4567-bot/Financialdreamplanner.git](https://github.com/ranitroy4567-bot/Financialdreamplanner.git)
   cd Financialdreamplanner
