---
layout: post
title: "Fraud Detection Using SMOTE"
subtitle: "Predicting Fraud Using Synthetic Minority Ovrsampling Technique "
date: 2023-09-06
background: '/img/posts/Fraud-Detection/Anomalyimages.jpg'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---


# Anomaly Detection Practice On Bank Fraud Using SMOTE

<br>
## Table of Contents

- [Import Packages](#import_python_packages)
- [Read CSV And Explore Column Names](#loading_data)
- [Ratios & Value_Counts For Targets](#ratio)
- [Explore Data In Graph Form](#graph)
- [Compare Original Dataset to SMOTE Data set](#smote)
- [Explanation Of Confusion Matrix](#confusion)
- [Model Section](#models)


## Import Packages


```python
# basic python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# models
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# model prep and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# python cmd
import os

# warnings ignore
import warnings
# set warnings to ignore
warnings.filterwarnings('ignore')

```

## Upload File In GoogleColab


```python
from google.colab import files
import io

uploaded = files.upload()
```


## Read CSV And Explore Column Names


```python
# read in csv
df = pd.read_csv('creditcard_sampledata_3.csv')

#explore data columns
print(df.head())
print(df.info())
```

       Unnamed: 0        V1        V2        V3        V4        V5        V6  \
    0      258647  1.725265 -1.337256 -1.012687 -0.361656 -1.431611 -1.098681   
    1       69263  0.683254 -1.681875  0.533349 -0.326064 -1.455603  0.101832   
    2       96552  1.067973 -0.656667  1.029738  0.253899 -1.172715  0.073232   
    3      281898  0.119513  0.729275 -1.678879 -1.551408  3.128914  3.210632   
    4       86917  1.271253  0.275694  0.159568  1.003096 -0.128535 -0.608730   
    
             V7        V8        V9  ...       V21       V22       V23       V24  \
    0 -0.842274 -0.026594 -0.032409  ...  0.414524  0.793434  0.028887  0.419421   
    1 -0.520590  0.114036 -0.601760  ...  0.116898 -0.304605 -0.125547  0.244848   
    2 -0.745771  0.249803  1.383057  ... -0.189315 -0.426743  0.079539  0.129692   
    3  0.356276  0.920374 -0.160589  ... -0.335825 -0.906171  0.108350  0.593062   
    4  0.088777 -0.145336  0.156047  ...  0.031958  0.123503 -0.174528 -0.147535   
    
            V25       V26       V27       V28  Amount  Class  
    0 -0.367529 -0.155634 -0.015768  0.010790  189.00      0  
    1  0.069163 -0.460712 -0.017068  0.063542  315.17      0  
    2  0.002778  0.970498 -0.035056  0.017313   59.98      0  
    3 -0.424303  0.164201  0.245881  0.071029    0.89      0  
    4  0.735909 -0.262270  0.015577  0.015955    6.53      0  
    
    [5 rows x 31 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5050 entries, 0 to 5049
    Data columns (total 31 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Unnamed: 0  5050 non-null   int64  
     1   V1          5050 non-null   float64
     2   V2          5050 non-null   float64
     3   V3          5050 non-null   float64
     4   V4          5050 non-null   float64
     5   V5          5050 non-null   float64
     6   V6          5050 non-null   float64
     7   V7          5050 non-null   float64
     8   V8          5050 non-null   float64
     9   V9          5050 non-null   float64
     10  V10         5050 non-null   float64
     11  V11         5050 non-null   float64
     12  V12         5050 non-null   float64
     13  V13         5050 non-null   float64
     14  V14         5050 non-null   float64
     15  V15         5050 non-null   float64
     16  V16         5050 non-null   float64
     17  V17         5050 non-null   float64
     18  V18         5050 non-null   float64
     19  V19         5050 non-null   float64
     20  V20         5050 non-null   float64
     21  V21         5050 non-null   float64
     22  V22         5050 non-null   float64
     23  V23         5050 non-null   float64
     24  V24         5050 non-null   float64
     25  V25         5050 non-null   float64
     26  V26         5050 non-null   float64
     27  V27         5050 non-null   float64
     28  V28         5050 non-null   float64
     29  Amount      5050 non-null   float64
     30  Class       5050 non-null   int64  
    dtypes: float64(29), int64(2)
    memory usage: 1.2 MB
    None
    

## Ratios & Value_Counts For Targets


```python
# count fraud anomolies
occ = df['Class'].value_counts()
occ
```




    0    5000
    1      50
    Name: Class, dtype: int64




```python
# ratio of class column of fraud anomalies
# 0 = no fraud / 1 = Fraud
occ/len(df)
```




    0    0.990099
    1    0.009901
    Name: Class, dtype: float64



# Explore Data In Graph Form


```python
# function to prep data & segregate the targets from the features
def prep_data(df):
    X = df.iloc[:, 1:30]
    X = np.array(X).astype(float)
    y = df.iloc[:, 30]
    y=np.array(y).astype(float)
    return X,y
```


```python
# how to search function code
import inspect
lines = inspect.getsource(prep_data)
print(lines)
```

    def prep_data(df):
        X = df.iloc[:, 1:30]
        X = np.array(X).astype(float)
        y = df.iloc[:, 30]
        y=np.array(y).astype(float)
        return X,y
    
    


```python
# create scatterplot function
def plot_data(X, y):
	plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
	plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
	plt.legend()
	return plt.show()
```


```python
# run data thru the prep data function to segrate the targets from the features
X, y = prep_data(df)

# plot scatterplot
plot_data(X, y)
```


    
![png](\img\posts\Fraud-Detection\output_15_0.png)
    



```python
# anomaly data resampling methods --> over and under resampling
# Undersampling majority class('non-fraud') / take random draws from non-fraud
# drawback - throwing away data
#
# Oversampling minority class - take random draws from the fraud class and copy them to increase the amount of data
# drawback copying data and training model on duplicate data
```


```python
# oversampling using imlearn.over_sampling
# SMOTE = synthetic Minority Ovrsampling Technique / uses knn to create new fraud cases
```


```python
# initialize and define resampling method SMOTE
# Synthetic Minority Ovrsampling Technique
method = SMOTE()
```


```python
# create resampled features and targets
X_resampled, y_resampled = method.fit_resample(X, y)
```


```python
#plot resampled data to compare against prior graph
plot_data(X_resampled, y_resampled)
```


    
![png](img\posts\Fraud-Detection\output_20_0.png)
    


# Compare Original Dataset to SMOTE Data set


```python
# compare ploot function
def compare_plot(X,y,X_resampled,y_resampled, method):
    # Start a plot figure
    f, (ax1, ax2) = plt.subplots(1, 2)
    # sub-plot number 1, this is our normal data
    c0 = ax1.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",alpha=0.5)
    c1 = ax1.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1",alpha=0.5, c='r')
    ax1.set_title('Original Data')
    # sub-plot number 2, this is our oversampled data
    ax2.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=.5)
    ax2.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=.5,c='r')
    ax2.set_title(method)
    # some settings and ready to go
    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    #plt.tight_layout(pad=3)
    return plt.show()
```


```python
# Print the value_counts on the original labels y
print(pd.value_counts(pd.Series(y)))

# Print the value_counts
print(pd.value_counts(pd.Series(y_resampled)))

# Run compare_plot
compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
```

    0.0    5000
    1.0      50
    dtype: int64
    0.0    5000
    1.0    5000
    dtype: int64
    


    
![png](\img\posts\Fraud-Detection\output_23_1.png)
    


## Reminder / Explanation Of Confusion Matrix

![png](\img\posts\Fraud-Detection\ConfusionMatrixPic.jpg)

## Model Section

## Not Using SMOTE


```python
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)

# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Obtain model predictions
predicted = model.predict(X_test)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

```

    Classification report:
                   precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      1505
             1.0       0.89      0.80      0.84        10
    
        accuracy                           1.00      1515
       macro avg       0.94      0.90      0.92      1515
    weighted avg       1.00      1.00      1.00      1515
    
    Confusion matrix:
     [[1504    1]
     [   2    8]]
    

# Using SMOTE


```python
# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.30, random_state=0)

# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Obtain model predictions
predicted = model.predict(X_test)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Classification report:
                   precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      1511
             1.0       1.00      1.00      1.00      1489
    
        accuracy                           1.00      3000
       macro avg       1.00      1.00      1.00      3000
    weighted avg       1.00      1.00      1.00      3000
    
    Confusion matrix:
     [[1504    7]
     [   3 1486]]
    


```python
# This is the pipeline module we need for this from imblearn
from imblearn.pipeline import Pipeline

# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE()
model = LogisticRegression()

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
```


```python
# Split your data X and y, into a training and a test set and fit the pipeline onto the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data
pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Classifcation report:
                   precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      1505
             1.0       0.54      0.70      0.61        10
    
        accuracy                           0.99      1515
       macro avg       0.77      0.85      0.80      1515
    weighted avg       0.99      0.99      0.99      1515
    
    Confusion matrix:
     [[1499    6]
     [   3    7]]
    
