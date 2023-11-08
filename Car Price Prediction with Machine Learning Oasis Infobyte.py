#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# ![car%20price%20prediction_.jpg](attachment:car%20price%20prediction_.jpg)

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading the data from csv
df = pd.read_csv('car data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.isnull()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.Fuel_Type.value_counts()


# In[11]:


df.Selling_type.value_counts()


# In[12]:


df.Transmission.value_counts()


# In[13]:


# encolding values
df.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)

df.replace({'Selling_type':{'Dealer':0, 'Individual':1}}, inplace=True)

df.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace=True)


# In[14]:


df.head()


# In[15]:


X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)


# In[18]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[19]:


pred = lr.predict(X_train)


# In[24]:


from sklearn.metrics import r2_score
rsq = r2_score(y_train, pred)
print("R square Error: ", rsq)


# In[26]:


plt.scatter(y_train, pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')


# In[29]:


# plotting the actual and predicted value
c = [i for i in range(1, len(y_train)+1, 1)]
plt.plot(c, y_train, color='b', linestyle='-', label='Actual Values')
plt.plot(c, pred, color='r', linestyle='-', label='Predicted Values')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Prediction Using X_train')
plt.legend()
plt.show()


# In[30]:


pred_2 = lr.predict(X_test)

rsq = r2_score(y_test, pred_2)
print('R Square Error y_test: ',rsq)


# In[32]:


c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='b',linestyle='-',label="Actual Values")
plt.plot(c,pred_2,color='r',linestyle='-',label="Predicted Values")
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Prediction using x_test')
plt.legend()
plt.show()


# You can find this project on <a href='https://github.com/Vyas-Rishabh/Car-Price-Prediction-with-Machine-Learning-OIBSIP'><b>GitHub.</b></a>
