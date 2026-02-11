# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, choose input features and target price, then split into training and testing sets.

2.Create a scaled linear regression pipeline, train it on the data, and make predictions.

3.Create a polynomial (degree 2) regression pipeline with scaling, train it, and make predictions.

4.Calculate MSE, MAE, and R² for both models and plot actual vs predicted prices to compare.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Mahashree S
RegisterNumber:  212225230163
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())

# select features & target
X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']

# split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 1. Linear Regression (with scaling)
lr= Pipeline([
    ('scalar',StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train,y_train)
y_pred_linear = lr.predict(X_test)

# 2. Polynomial Regression (degree=2)
poly_model = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scalar',StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict(X_test)

# Evaluate models
print('Name: MAHASHREE S ')
print('Reg. No:212225230163 ')
print("Linear Regression")

print('MSE=',mean_squared_error(y_test,y_pred_linear))
print('MAE=',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)
print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_poly):.2f}")
print(f"R²: {r2_score(y_test, y_pred_poly):.2f}")
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test,y_pred_poly, label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()], 'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs polynomial Regression")
plt.legend()
plt.show()
```

## Output:

# load data
<img width="800" height="482" alt="image" src="https://github.com/user-attachments/assets/a3f016df-573b-4778-a9d8-c423bc58e76d" />

# Evaluate models
<img width="290" height="73" alt="image" src="https://github.com/user-attachments/assets/57b21968-38b1-401c-8952-6f70c1928c41" />
<img width="1188" height="757" alt="image" src="https://github.com/user-attachments/assets/2f2e0c83-cb7b-4487-a5cb-736c467b8e18" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
