```python
import pandas as pd
data = pd.read_csv("TravelInsurancePrediction.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Employment Type</th>
      <th>GraduateOrNot</th>
      <th>AnnualIncome</th>
      <th>FamilyMembers</th>
      <th>ChronicDiseases</th>
      <th>FrequentFlyer</th>
      <th>EverTravelledAbroad</th>
      <th>TravelInsurance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>31</td>
      <td>Government Sector</td>
      <td>Yes</td>
      <td>400000</td>
      <td>6</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>31</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>1250000</td>
      <td>7</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>34</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>500000</td>
      <td>4</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>28</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>700000</td>
      <td>3</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>28</td>
      <td>Private Sector/Self Employed</td>
      <td>Yes</td>
      <td>700000</td>
      <td>8</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(columns=["Unnamed: 0"], inplace=True)
```


```python
data.isnull().sum()
```




    Age                    0
    Employment Type        0
    GraduateOrNot          0
    AnnualIncome           0
    FamilyMembers          0
    ChronicDiseases        0
    FrequentFlyer          0
    EverTravelledAbroad    0
    TravelInsurance        0
    dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1987 entries, 0 to 1986
    Data columns (total 9 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   Age                  1987 non-null   int64 
     1   Employment Type      1987 non-null   object
     2   GraduateOrNot        1987 non-null   object
     3   AnnualIncome         1987 non-null   int64 
     4   FamilyMembers        1987 non-null   int64 
     5   ChronicDiseases      1987 non-null   int64 
     6   FrequentFlyer        1987 non-null   object
     7   EverTravelledAbroad  1987 non-null   object
     8   TravelInsurance      1987 non-null   int64 
    dtypes: int64(5), object(4)
    memory usage: 139.8+ KB
    


```python
data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})
```


```python
import plotly.express as px
figure = px.histogram(data, x = "Employment Type", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Employment Type")
figure.show(renderer='iframe_connected')
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>




```python
import plotly.express as px
data = data
figure = px.histogram(data, x = "Age", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Age")
figure.show(renderer='iframe_connected')
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_25.html"
    frameborder="0"
    allowfullscreen
></iframe>




```python
import plotly.express as px
data = data
figure = px.histogram(data, x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Income")
figure.show(renderer='iframe_connected')
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_26.html"
    frameborder="0"
    allowfullscreen
></iframe>




```python
import numpy as np
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])
```


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
```


```python
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,predictions))
```

    0.8090452261306532
    


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
```


```python
rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=6, n_estimators=100, oob_score=True)
rf.fit(xtrain, ytrain)
```

    C:\Users\Nicholas Bagwandeen\AppData\Local\Temp\ipykernel_35044\3342771140.py:2: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    




    RandomForestClassifier(max_depth=6, n_jobs=-1, oob_score=True, random_state=42)




```python
rf.oob_score_
```




    0.8316554809843401




```python
y_pred = rf.predict(xtest)
print(accuracy_score(ytest, y_pred))
```

    0.8592964824120602
    


```python
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
```


```python
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
```

    {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
     'max_features': [1, 2, 3, 4, 5],
     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
     'n_estimators': [100, 311, 522, 733, 944, 1155, 1366, 1577, 1788, 2000]}
    


```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(xtrain, ytrain)
```

    Fitting 3 folds for each of 500 candidates, totalling 1500 fits
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:926: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    




    RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=500,
                       n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [10, 20, 30, 40, 50, 60,
                                                          70, 80, 90, 100, 110,
                                                          None],
                                            'max_features': [1, 2, 3, 4, 5],
                                            'min_samples_leaf': [1, 2, 3, 4, 5, 6,
                                                                 7, 8, 9, 10],
                                            'min_samples_split': [2, 3, 4, 5, 6, 7,
                                                                  8, 9, 10],
                                            'n_estimators': [100, 311, 522, 733,
                                                             944, 1155, 1366, 1577,
                                                             1788, 2000]},
                       random_state=42, verbose=2)




```python
rf_random.best_params_
```




    {'n_estimators': 1366,
     'min_samples_split': 4,
     'min_samples_leaf': 2,
     'max_features': 2,
     'max_depth': 10,
     'bootstrap': True}




```python
y_pred = rf_random.predict(xtest)
```


```python
accuracy_score(ytest, y_pred)
```




    0.8542713567839196




```python
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
```


```python
# Fit the grid search to the data
grid_search.fit(xtrain, ytrain)
grid_search.best_params_
```

    Fitting 3 folds for each of 1875 candidates, totalling 5625 fits
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:926: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    




    {'bootstrap': True,
     'max_depth': 9,
     'max_features': 3,
     'min_samples_leaf': 2,
     'min_samples_split': 4,
     'n_estimators': 1200}




```python
rf = RandomForestClassifier(bootstrap=True, max_depth=9, max_features=3, min_samples_leaf=2, min_samples_split=4, n_estimators=1200)
rf.fit(xtrain,ytrain)
y_pred = rf.predict(xtest)
accuracy_score(ytest, y_pred)
```

    C:\Users\Nicholas Bagwandeen\AppData\Local\Temp\ipykernel_35044\2290841162.py:2: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    
    




    0.8542713567839196




```python

```
