#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading the file

# In[2]:


data = pd.read_csv('C://Users//DELL//Downloads//Financial Risk of Loan Approval//Loan.csv')
data


# ### Checking Null Values

# In[3]:


data.isna().sum()


# ### Checking for correalted columns & removing it

# In[4]:


a = data.corr()
plt.figure(figsize=(30,30))
sns.heatmap(a,annot=True,cmap='cubehelix')


# In[5]:


# Correlated Columns
corr_col=a[(a>0.70) & (a!=1.0)]
corr_col.columns[corr_col.any()]


# ### I could see the correlation between:
# 1. Age and Experience
# 2. AnnualIncome and MonthlyIncome
# 3. LoanAmount and MonthlyLoanPayment
# 4. TotalAssets and NetWorth
# 5. BaseInterestRate and InterestRate

# ### So I am gonna remove Age, MonthlyIncome, NetWorth, InterestRate,and MonthlyLoanPayment along with Application date which is no need

# In[6]:


data = data.drop(columns=['ApplicationDate','Age','MonthlyLoanPayment','InterestRate','NetWorth','MonthlyIncome'])
data


# ### EDA

# In[7]:


str_data = data.select_dtypes('object')
#Seperating string columns

num_data = data.select_dtypes(['int','float'])
#Seperating numeric columns


# ### Count of each character columns

# In[8]:


for i in str_data.columns:
    sns.countplot(data=data,x = data[i],hue='LoanApproved')
    plt.show()    


# ### Line plot of numeric variables with Risk score as x and classifying the plot based on Loan Approval

# In[9]:


for i in num_data.columns:
    sns.lineplot(data=data,x='RiskScore',y= data[i],hue='LoanApproved')
    plt.show() 


# ### Label Encoding

# In[10]:


for i in str_data.columns:
    data[i] = data[i].astype('category').cat.codes


# In[11]:


data


# ### Splitting data into train and test data

# In[12]:


x = data.drop(columns=['RiskScore'])
x


# In[13]:


y = data['RiskScore']
y


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Regression Model

# In[15]:


from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
rc = RandomForestRegressor()
rc = rc.fit(x_train,y_train)
rc_pred = rc.predict(x_test)
rc_acc = r2_score(y_test,rc_pred) * 100
print('Predicton of Test data :',rc_acc)


# In[16]:


rc_pred1 = rc.predict(x_train)
rc_acc1 = r2_score(y_train,rc_pred1) * 100
print('Predicton of Train data :',rc_acc1)


# In[17]:


plt.scatter(y_test,rc_pred)
plt.plot(y_test,y_test,color='red')
plt.title('Orginal data vs Predicted data')
plt.show()


# In[18]:


a =x.iloc[[14254,14255],:]
a


# In[19]:



x_new = np.array(a)
x_new


# In[20]:


rc.predict(x_new)

