import pandas as pd
import numpy as np
from linear_regression import LinearRegression

age_height_df = pd.read_csv('age_height.csv')
X = age_height_df['Age']
Y = age_height_df['Height(cm)']

# age_height_df.info()
X = np.reshape(X,(-1, 1))
X_train = X[0:80]
X_test = X[80:]

Y_train = Y[0:80]
Y_test = Y[80:]
# print(X)
model = LinearRegression()

model.fit(X_train, Y_train)

prediction = model.predict(X_test)
mse = model.mean_squared_error(Y_test, prediction)
score = model.model_score(Y_test, prediction)
print(f"Mean Squared Error: {np.sqrt(mse)}")
print(f"Model Score: {score}")
print(prediction)