#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('kyphosis.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[9]:


fig=sns.pairplot(df,hue='Kyphosis')
fig.savefig('pairplots.jpg')


# In[11]:


from sklearn.model_selection import train_test_split


# In[18]:


x=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)


# In[21]:


#fit and predict using decision tree


# In[22]:


from sklearn.tree import DecisionTreeClassifier


# In[23]:


dt=DecisionTreeClassifier()


# In[24]:


dt.fit(x_train,y_train)


# In[25]:


pred=dt.predict(x_test)


# In[27]:


#evaluate model


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[29]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# In[30]:


#now using random forest


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


rfc=RandomForestClassifier(n_estimators=250)


# In[33]:


rfc.fit(x_train,y_train)


# In[34]:


pred2=rfc.predict(x_test)


# In[35]:


pred2


# In[36]:


#evaluate model


# In[37]:


print(confusion_matrix(y_test,pred2))
print()
print(classification_report(y_test,pred2))


# In[38]:


#better precision by random forest


# In[ ]:




