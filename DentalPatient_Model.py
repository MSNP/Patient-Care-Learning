#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import sklearn
import numpy as np


# In[5]:


dental_patient_df = pd.read_csv('DentalPaient_training.csv',sep=',')


# In[7]:


dental_patient_df.head()


# In[8]:


dental_patient_df.describe()


# In[18]:


x_features=['Age','Gender','Complaint','Symptom']


# In[49]:


y_features=['Symptom']


# In[50]:


y_dental_encode_df=pd.get_dummies(dental_patient_df[y_features],drop_first=True)


# In[52]:


y_dental_encode_df.head()


# In[ ]:





# In[14]:


category=['Age','Gender','Complaint']


# In[20]:


dental_encode_df=pd.get_dummies(dental_patient_df[x_features],drop_first=True)


# In[21]:


X=dental_encode_df


# In[23]:


X.head()


# In[44]:


import statsmodels.api as sm


# In[45]:


X=sm.add_constant(dental_encode_df)


# In[54]:


Y=sm.add_constant(y_dental_encode_df)


# In[33]:


import sklearn.model_selection as train_test_split


# In[56]:


train_X, test_X, train_y, test_y =sklearn.model_selection.train_test_split(X ,Y,train_size = 0.8,random_state = 42)


# In[58]:


lpl_model_1= sm.OLS(train_y,train_X).fit()


# In[64]:


from sklearn import linear_model


# In[65]:


regr=linear_model.LinearRegression()


# In[67]:


regr.fit(train_X,train_y)


# In[68]:


predict=regr.predict(test_X)


# In[71]:


predict.view()


# In[76]:


print(train_y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


lpl_model_l= sm.OLS(train_y,train_X).fit()
lpl_model_l.summary2()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




