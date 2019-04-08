import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1].values
X = X.reshape(10,1)
Y = dataset.iloc[:, 2].values

regressor = LinearRegression()
regressor.fit(X, Y)

plt.scatter(X, Y, color = 'red')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color
= 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


arr=np.array([6])
arr=arr.reshape(1,-1)
print(regressor.predict(arr))
print(lin_reg_2.predict(poly_reg.fit_transform(arr)))

''' Polynomial przewidział lepiej wartość dla 6 (oryginalnie jest to 150000),
według regresji liniowej to 289939, a polynomiala 134484. Polynomial
lepiej oddaje relację jaka zachodzi w danych, czyli funkcję rosnącą wykładniczo'''

poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color
= 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lin_reg_2.predict(poly_reg.fit_transform(arr)))

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color
= 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lin_reg_2.predict(poly_reg.fit_transform(arr)))

'''
Najlepiej wartość przewidział model polynomial z parametrem 4 ([143275.05827508]).
Stało się to z tego powodu, że nastąpiło overfitting i model uwzględnił każdą daną
bez koniecznego uogólnienia lub ta funkcja rzeczywiście jest poprzez niego
najlepiej odwzorowana i nie jest to overfitting.
'''