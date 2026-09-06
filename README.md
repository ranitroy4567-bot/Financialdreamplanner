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




