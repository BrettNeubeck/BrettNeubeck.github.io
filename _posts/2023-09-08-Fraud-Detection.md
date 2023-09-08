---
layout: post
title: "Fraud Detection Using SMOTE"
subtitle: "Predicting Fraud Using Synthetic Minority Oversampling Technique "
date: 2023-09-08
background: '/img/posts/Fraud-Detection-09-08/Anomalyimages.jpg'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---

### Table of Contents

- [Summary](#summary)
- [Import Packages](#import_python_packages)
- [Read CSV & Explore Column Names](#loading_data)
- [Ratios & Value_Counts for Targets](#ratio)
- [Explore Data in Graph Form](#graph)
- [Compare Original Dataset to SMOTE Dataset](#smote)
- [Explanation of Confusion Matrix](#confusion)
- [Model Section](#model)
- [Conclusion](#conclusion)

<br>
### Summary
<a id='summary'></a>

Unbalanced datasets pose a common challenge in machine learning where the number of instances in one class significantly outweighs the other(s). This imbalance can lead to biased model performance, where the algorithm tends to favor the majority class, as it seeks to minimize overall error. To address this issue, techniques like minority oversampling and majority undersampling are often employed to rebalance the dataset and improve the model's ability to accurately classify minority class instances.

**Minority Oversampling:**
In minority oversampling, the goal is to increase the representation of the minority class by generating synthetic instances. One prominent method for this purpose is the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE works by creating synthetic samples for the minority class by interpolating between existing data points. It selects a minority instance, identifies its k nearest neighbors, and generates new instances along the line segments connecting the chosen instance to its neighbors. This effectively augments the minority class and helps balance the dataset.

**Majority Undersampling:**
On the other hand, majority undersampling involves reducing the number of instances in the majority class to match the minority class. This can be done randomly or through more sophisticated techniques that carefully select instances to maintain the dataset's representativeness.

**Challenges with Measurement:**
Evaluating models trained on imbalanced datasets poses its own set of challenges. Traditional metrics like ROC AUC (Receiver Operating Characteristic Area Under the Curve) may not be the most appropriate choice in such cases because they can be overly optimistic. ROC AUC evaluates a model's ability to discriminate between classes across different thresholds, which can lead to inflated performance scores when the dataset is imbalanced.

**Brier Score Loss:**
Instead of ROC AUC, the Brier Score Loss is a more suitable metric for evaluating models on imbalanced datasets. The Brier Score measures the mean squared difference between the predicted probabilities and the actual outcomes. It rewards models for assigning high probabilities to the true positive instances, making it sensitive to class imbalance.

**Confusion Matrix:**
In addition to the Brier Score, the confusion matrix is a valuable tool for assessing model performance on unbalanced datasets. It breaks down the model's predictions into categories like true positives, true negatives, false positives, and false negatives, providing a clearer picture of how well the model is performing for each class. Metrics like precision, recall, and F1-score can be derived from the confusion matrix, offering a more balanced view of the model's accuracy across classes.

In summary, unbalanced datasets are a common challenge in machine learning, and addressing this issue through techniques like SMOTE, majority undersampling, and appropriate evaluation metrics like the Brier Score Loss and the confusion matrix is crucial for building models that perform well on imbalanced data, especially when ROC AUC is not suitable due to class imbalance.

### Import Packages
<a id='import_python_packages'></a>


```python
# basic python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# models
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# pipeline from imblearn
from imblearn.pipeline import Pipeline

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

### Upload File In GoogleColab


```python
from google.colab import files
import io

uploaded = files.upload()
```
    
### Read CSV & Explore Column Names
<a id='loading_data'></a>

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
    

### Ratios & Value_Counts for Targets
<a id='ratio'></a>

```python
# count fraud anomalies
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



### Explore Data in Graph Form
<a id='graph'></a>

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
# create scatterplot function taking first two feature columns for x & y axis on graph
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


    
![png](\img\posts\Fraud-Detection-09-08\output_17_0.png)
    



```python
# oversampling using imlearn.over_sampling smote
# SMOTE = Synthetic Minority Ovrsampling TEchnique / uses knn to create new fraud cases
# initialize and define resampling method SMOTE

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


    
![png](\img\posts\Fraud-Detection-09-08\output_20_0.png)
    


### Compare Original Dataset to SMOTE Dataset
<a id='smote'></a>


```python
# compare plot function
def compare_plot(X,y,X_resampled,y_resampled, method):
    # start a plot figure
    f, (ax1, ax2) = plt.subplots(1, 2)
    # sub-plot number 1, normal data
    c0 = ax1.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0",alpha=0.5)
    c1 = ax1.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1",alpha=0.5, c='r')
    ax1.set_title('Original Data')
    # sub-plot number 2, oversampled data
    ax2.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=.5)
    ax2.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=.5,c='r')
    ax2.set_title(method)

    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    #plt.tight_layout(pad=3)
    return plt.show()
```


```python
# print original value_counts
print(pd.value_counts(pd.Series(y)))

# print new value_counts
print(pd.value_counts(pd.Series(y_resampled)))

# print compare plot
compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
```

    0.0    5000
    1.0      50
    dtype: int64
    0.0    5000
    1.0    5000
    dtype: int64
    


    
![png](\img\posts\Fraud-Detection-09-08\output_24_1.png)
    


### Reminder / Explanation of Confusion Matrix
<a id='confusion'></a>

![png](\img\posts\Fraud-Detection-09-08\ConfusionMatrixPic.jpg)

### Model Section
<a id='model'></a>

<br>
#### Not Using SMOTE


```python
# plot impact of brier for single forecasts
from sklearn.metrics import brier_score_loss

# segragate labels and targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)

# fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# predict model
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)
# keep only the class 1 predictions
probs=probs[:,1]
#calculate bier score
loss = brier_score_loss(y_test, probs)
print('Brier Score Loss:\n', loss)
# results
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

```

    Brier Score Loss:
     0.0018169696384545437
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
    


```python
from sklearn.metrics import brier_score_loss
```


```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
```


    
![png](\img\posts\Fraud-Detection-09-08\output_31_0.png)
    


#### Using SMOTE


```python
# segragate labels and targets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.30, random_state=0)

# fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# predict model
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)
# keep only the class 1 predictions
probs=probs[:,1]
#calculate bier score
loss = brier_score_loss(y_test, probs)
print('Brier Score Loss:\n', loss)

# results
print('Classification report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Brier Score Loss:
     0.002725477458352906
    Classification report:
                   precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      1511
             1.0       1.00      1.00      1.00      1489
    
        accuracy                           1.00      3000
       macro avg       1.00      1.00      1.00      3000
    weighted avg       1.00      1.00      1.00      3000
    
    Confusion matrix:
     [[1506    5]
     [   4 1485]]
    


```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
```


    
![png](\img\posts\Fraud-Detection-09-08\output_34_0.png)
    



```python
# define resampling method & model for the pipeline
resampling = SMOTE()
model = LogisticRegression()

# define the pipeline
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])
```


```python
# segragate labels and targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)

# fit & predict pipeline
pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
probs = model.predict_proba(X_test)
# keep only the class 1 predictions
probs=probs[:,1]
#calculate bier score
loss = brier_score_loss(y_test, probs)
print('Brier Score Loss:\n', loss)
print()
# results
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
```

    Brier Score Loss:
     0.002822190433164936
    
    Classifcation report:
                   precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      1505
             1.0       0.73      0.80      0.76        10
    
        accuracy                           1.00      1515
       macro avg       0.86      0.90      0.88      1515
    weighted avg       1.00      1.00      1.00      1515
    
    Confusion matrix:
     [[1502    3]
     [   2    8]]
    


```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
```


    
![png](\img\posts\Fraud-Detection-09-08\output_37_0.png)
    


### Conclusion

Based on the results obtained from the classification model with a pipeline using SMOTE resampling and logistic regression, we can draw the following conclusions:

1. **Brier Score Loss**: The Brier Score Loss is a measure of the accuracy of probability predictions, and in this case, it is exceptionally low, indicating that the model's predicted probabilities are very close to the actual outcomes. With a Brier Score Loss of approximately 0.0024, the model's probability predictions are highly accurate.

2. **Classification Report**:
   - **Precision**: The precision for class 1 (positive class) is 0.73, which means that when the model predicts a positive outcome, it is correct about 73% of the time.
   - **Recall**: The recall for class 1 is 0.80, indicating that the model correctly identifies 80% of the actual positive instances.
   - **F1-Score**: The F1-score, which balances precision and recall, is 0.76 for class 1. It provides a single metric to evaluate the model's overall performance.
   - **Accuracy**: The overall accuracy of the model is 100%, which might be a bit misleading due to the class imbalance. It's crucial to consider other metrics, especially when dealing with imbalanced datasets.

3. **Confusion Matrix**: The confusion matrix provides a more detailed view of the model's performance:
   - True Positives (TP): 8
   - True Negatives (TN): 1502
   - False Positives (FP): 3
   - False Negatives (FN): 2

   The model correctly identifies most of the positive cases (TP), with very few false positives and false negatives. This suggests that the model is effective at distinguishing the positive class from the negative class.

In summary, the model shows strong performance in terms of Brier Score Loss, precision, recall, and F1-score for the positive class. However, it's important to note that the dataset appears to be highly imbalanced, with a significantly larger number of negative class instances.
