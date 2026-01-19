import pandas as pd
import numpy as np
from linear_regression import LinearRegression

age_height_df = pd.read_csv('age_height.csv')
X = age_height_df['Age']
Y = age_height_df['Height(cm)']

# age_height_df.info()
X = np.reshape(X,(-1, 1))
# print(X)
model = LinearRegression()

model.fit(X, Y)

test = [30, 40, 21, 34]
test = np.reshape(test, (-1, 1))

prediction = model.predict(test)
print(prediction)