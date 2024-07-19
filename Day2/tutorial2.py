import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


df =pd.read_csv('Day2\home.csv')

df.columns = df.columns.str.strip()

reg = LinearRegression()

X = df[['area']]
y = df['price']

reg.fit(X, y)

plt.scatter(df['area'], df['price'], color='blue')

plt.plot(df['area'], reg.predict(X), color='red')

area_to_predict = pd.DataFrame({'area': [5000]})
predicted_price = reg.predict(area_to_predict)
print(f"Predicted price for area: {predicted_price[0]}")

plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

print(reg.coef_)
print(reg.intercept_)


df1=pd.read_csv('Day2\area.csv')
print(df1)
p=reg.predict(df1)
df1['prices']=p
df1.to_csv('Day2\area.csv',index=False)
print(df1)