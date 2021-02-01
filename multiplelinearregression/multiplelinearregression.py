import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.style.use('ggplot')

# Read the csv file.
df=pd.read_csv('housing.csv')

# Print the first 5 rows of the data set.
print(df.head())

# Shape of the dataframe.
print(df.shape)

# Data type of each column.
print(df.dtypes)

# Number of null values.
print(df.info())

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print(df.corr())

mlr= LinearRegression()
X = df[['SquareFeetArea','Bedrooms']]
y = df['Price']
# Fit linear regression.
mlr.fit(X, y)
y_pred = mlr.predict(X)

plt.plot(X, y_pred, color='red')
plt.show()

# Get the slope and intercept of the line best fit.
print("Intercept",mlr.intercept_)


print("Coefficient",mlr.coef_)

print("R2 Score",r2_score(y, y_pred))


