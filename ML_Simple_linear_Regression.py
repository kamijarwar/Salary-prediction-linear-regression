#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


# In[2]:


x = np.array([[1], [2], [3], [4], [5],[6],[7],[8]])
y = np.array([30,35, 45,50, 60, 68, 79, 87])
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print (f"solpe (slope/weight):{model.coef_[0]:.2f}")
print(f"Intercept (Bias): {model.intercept_:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared ($R^2$) Score: {r2:.2f}")


# In[ ]:




