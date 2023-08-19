#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("TravelInsurancePrediction.csv")
data.head()


# In[2]:


data.drop(columns=["Unnamed: 0"], inplace=True)


# In[3]:


data.isnull().sum()


# In[4]:


data.info()


# In[5]:


data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})


# In[10]:


import plotly.express as px
from IPython.display import IFrame
figure = px.histogram(data, x = "Employment Type", 
                      color = "TravelInsurance")
figure.write_html("employment type and travel insurance.html")
IFrame('employment type and travel insurance.html', width=800, height=600)


# In[11]:


figure = px.histogram(data, x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Income")
figure.write_html("income and travel insurance.html")
IFrame('income and travel insurance.html', width=800, height=600)


# In[27]:


import numpy as np
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[29]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,predictions))


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


# In[31]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=6, n_estimators=100, oob_score=True)
rf.fit(xtrain, ytrain)


# In[32]:


rf.oob_score_


# In[33]:


y_pred = rf.predict(xtest)
print(accuracy_score(ytest, y_pred))


# In[34]:


from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# In[35]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = [1,2,3,4,5]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,3,4, 5,6,7,8,9, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,3, 4,5,6,7,8,9,10]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
               
pprint(random_grid)


# In[36]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(xtrain, ytrain)


# In[37]:


rf_random.best_params_


# In[38]:


y_pred = rf_random.predict(xtest)


# In[39]:


accuracy_score(ytest, y_pred)


# In[40]:


from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 9, 10, 11, 12],
    'max_features': [1 ,2, 3],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [2,3,4,5,6],
    'n_estimators': [1200,1250,1300,1350,1400]
}# Create a based model
rf = RandomForestClassifier()# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[41]:


# Fit the grid search to the data
grid_search.fit(xtrain, ytrain)
grid_search.best_params_


# In[43]:


rf = RandomForestClassifier(bootstrap=True, max_depth=9, max_features=3, min_samples_leaf=2, min_samples_split=4, n_estimators=1200)
rf.fit(xtrain,ytrain)
y_pred = rf.predict(xtest)
accuracy_score(ytest, y_pred)


# In[13]:





# In[ ]:




