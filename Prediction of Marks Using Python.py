#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading data from remote link
url = "https://raw.githubusercontent.com/itsaman08/Prediction-Of-Marks-Using-Python/main/studentscores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")

# a is no. of hours studied
a = 7
s_data.head(a)

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='-')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2, random_state=0)

### **Training the Algorithm**

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data\n",
plt.scatter(X, y)
plt.plot(X, line);
plt.title("Comparison")
plt.show()

### **Making Predictions**
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


score_pred=np.array([7])
score_pred=score_pred.reshape(-1,1)
predict=regressor.predict(score_pred)
print("No of hours={}".format(7))
print("Predicted Score={}".format(predict[0]))

### **Evaluating the model**

from sklearn import metrics
print('Mean Absolute Error:' , metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




