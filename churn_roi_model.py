import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("⏳ 1. churn ROI model is running...")
df = pd.read_csv('data.csv')


df = df.drop('customerID', axis=1)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']


print("🧠 2. building and training the machine learning model...")
# 切分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\n" + "="*50)
print(" 📊 Model Evaluation & Business ROI Report")
print("="*50)

print("\n[Algorithm-Level Evaluation Metrics]")
print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


cost_per_offer = 20    

value_per_saved = 100  


baseline_cost = len(y_test) * cost_per_offer

baseline_profit = (tp + fn) * value_per_saved - baseline_cost


model_cost = (tp + fp) * cost_per_offer
model_profit = (tp * value_per_saved) - model_cost

print(f"\n[💰 Business Decision & Profit Calculation (Based on {len(y_test)} Test Set Users)]")
print(f"Blind Marketing Strategy (No Model) Profit: ${baseline_profit:,.2f}")
print(f"Model-Based Strategy Profit: ${model_profit:,.2f}")
print("-" * 40)
print(f"🚀 Model-Driven ROI Uplift: ${model_profit - baseline_profit:,.2f}")
print("="*50)