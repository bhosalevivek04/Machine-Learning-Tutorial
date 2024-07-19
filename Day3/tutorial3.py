import pandas as pd
import numpy as np
import math
from sklearn import linear_model

df=pd.read_csv('Day3\homeprices.csv')
print(df)


median_bedroom=math.floor(df.bedroom.median())
# print(median_bedroom)

df.bedroom=df.bedroom.fillna(median_bedroom)
# print(df) 

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)

print(reg.coef_)
print(reg.intercept_)

a=reg.predict([[3000,3,40]])
print(a)
print(reg.predict([[2500,4,5]]))
