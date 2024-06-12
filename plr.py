import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('poly.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Polynomial Regression model
poly_degree = 2
poly_reg = PolynomialFeatures(degree=poly_degree)
x_poly = poly_reg.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)

# Streamlit UI
st.title('Salary Prediction with Polynomial Regression')

# Sidebar for user input
st.sidebar.header('Input Parameters')
level = st.sidebar.number_input('Enter Level:', min_value=1, max_value=10, value=1)

# Predict function
def predict_salary(level):
    level_arr = np.array([[level]])
    level_poly = poly_reg.transform(level_arr)
    predicted_salary = poly_reg_model.predict(level_poly)
    return predicted_salary[0]

# Display prediction
predicted_salary = predict_salary(level)
st.write(f"Predicted Salary for Level {level}: ${predicted_salary:.2f}")