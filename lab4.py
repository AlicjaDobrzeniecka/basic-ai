import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1].values
X = X.reshape(-1,1)
Y = dataset.iloc[:, 2].values

print(X)