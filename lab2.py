import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X,
Y, test_size = 0.3)

X_train_one = np.concatenate(X_train)
training_set = pd.DataFrame(np.hstack((X_train_one[:,None], y_train[:,None])))

X_test_one = np.concatenate(X_test)
test_set = pd.DataFrame(np.hstack((X_test_one[:,None], y_test[:,None])))

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Wysokość wynagrodzenia vs doświadczenie')
plt.xlabel('Lata doświadczenia')
plt.ylabel('Wysokość wynagrodzenia')
plt.show()

# =============================================================================
# Ile wynosi przewidywane przez model wynagrodzenie przy
# stażu pracy wynoszącym 4 lata, a ile wynosi rzeczywiste wynagrodzenie dla tego
# samego stażu pracy (1 pkt)?
# =============================================================================

# Wynagrodzenie rzeczywiste dla 4.1 to 57081
# Wynagrodzenie według modelu dla 4.1 to 64287.6
