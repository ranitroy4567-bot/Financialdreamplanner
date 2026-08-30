import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_pipeline(csv_path="salary_data.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'. Please place salary_data.csv in the directory.")

    # 1. Load Data & Clean Headers
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove 'Unnamed: 5' trailing columns
    
    # 2. Map Column Names from dataset
    rename_mapping = {
        "Job_Role": "Job Role",
        "JobRole": "Job Role",
        "Monthly_Salary": "Salary"
    }
    df = df.rename(columns=rename_mapping)
    
    # Enforce Fresher Rule: Drop Experience and Age if present
    cols_to_drop = [c for c in ["Experience", "Age"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    X = df[["City", "Education", "Job Role"]]
    y = df["Salary"]  # Represents Monthly Salary
    
    # 3. Preprocessing Setup
    categorical_features = ["City", "Education", "Job Role"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    # 4. Model Training & Comparison
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float("inf")
    best_name = ""
    
    print("--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X_train_encoded, y_train)
        preds = model.predict(X_test_encoded)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name} -> MAE: ₹{mae:.2f}, R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            
    print(f"\nSelected Model: {best_name} (R2: {best_r2:.4f})")
    
    # 5. Save Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
    joblib.dump(best_model, "artifacts/salary_model.pkl")
    print("Saved preprocessor and best model to 'artifacts/' folder.")

if __name__ == "__main__":
    train_and_save_pipeline()
