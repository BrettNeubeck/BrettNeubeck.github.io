---
layout: post
title: "Bank Fraud Anomaly Detection"
subtitle: "Anomaly Detection On UCI Bank Data Using SVM & Neural Networks"
date: 2023-08-16
background: '/img/posts/Bank-Fraud-Anomaly-Detection/BankFraud.jpg'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---

```python
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
```


```python
from google.colab import files
import io

uploaded = files.upload()
```    


```python
df_bank = pd.read_csv('bank-additional-full_normalised.csv')
```


```python
df_bank.head()
```





  <div id="df-b80d1587-fcb0-4ef8-845d-6c8bfd305424">
    <div class="colab-df-container">
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
      <th>age</th>
      <th>job=housemaid</th>
      <th>job=services</th>
      <th>job=admin.</th>
      <th>job=blue-collar</th>
      <th>job=technician</th>
      <th>job=retired</th>
      <th>job=management</th>
      <th>job=unemployed</th>
      <th>job=self-employed</th>
      <th>...</th>
      <th>previous</th>
      <th>poutcome=nonexistent</th>
      <th>poutcome=failure</th>
      <th>poutcome=success</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.209877</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0.882307</td>
      <td>0.376569</td>
      <td>0.980730</td>
      <td>1.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.296296</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.000000</td>
      <td>0.484412</td>
      <td>0.615063</td>
      <td>0.981183</td>
      <td>1.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.246914</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.937500</td>
      <td>0.698753</td>
      <td>0.602510</td>
      <td>0.957379</td>
      <td>0.859735</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.160494</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.142857</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.333333</td>
      <td>0.269680</td>
      <td>0.192469</td>
      <td>0.150759</td>
      <td>0.512287</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.530864</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.333333</td>
      <td>0.340608</td>
      <td>0.154812</td>
      <td>0.174790</td>
      <td>0.512287</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b80d1587-fcb0-4ef8-845d-6c8bfd305424')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b80d1587-fcb0-4ef8-845d-6c8bfd305424 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b80d1587-fcb0-4ef8-845d-6c8bfd305424');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# SVM Model


```python
data = df_bank
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41188 entries, 0 to 41187
    Data columns (total 63 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   age                            41188 non-null  float64
     1   job=housemaid                  41188 non-null  int64  
     2   job=services                   41188 non-null  int64  
     3   job=admin.                     41188 non-null  int64  
     4   job=blue-collar                41188 non-null  int64  
     5   job=technician                 41188 non-null  int64  
     6   job=retired                    41188 non-null  int64  
     7   job=management                 41188 non-null  int64  
     8   job=unemployed                 41188 non-null  int64  
     9   job=self-employed              41188 non-null  int64  
     10  job=unknown                    41188 non-null  int64  
     11  job=entrepreneur               41188 non-null  int64  
     12  job=student                    41188 non-null  int64  
     13  marital=married                41188 non-null  int64  
     14  marital=single                 41188 non-null  int64  
     15  marital=divorced               41188 non-null  int64  
     16  marital=unknown                41188 non-null  int64  
     17  education=basic.4y             41188 non-null  int64  
     18  education=high.school          41188 non-null  int64  
     19  education=basic.6y             41188 non-null  int64  
     20  education=basic.9y             41188 non-null  int64  
     21  education=professional.course  41188 non-null  int64  
     22  education=unknown              41188 non-null  int64  
     23  education=university.degree    41188 non-null  int64  
     24  education=illiterate           41188 non-null  int64  
     25  default=0                      41188 non-null  int64  
     26  default=unknown                41188 non-null  int64  
     27  default=1                      41188 non-null  int64  
     28  housing=0                      41188 non-null  int64  
     29  housing=1                      41188 non-null  int64  
     30  housing=unknown                41188 non-null  int64  
     31  loan=0                         41188 non-null  int64  
     32  loan=1                         41188 non-null  int64  
     33  loan=unknown                   41188 non-null  int64  
     34  contact=cellular               41188 non-null  int64  
     35  month=may                      41188 non-null  int64  
     36  month=jun                      41188 non-null  int64  
     37  month=jul                      41188 non-null  int64  
     38  month=aug                      41188 non-null  int64  
     39  month=oct                      41188 non-null  int64  
     40  month=nov                      41188 non-null  int64  
     41  month=dec                      41188 non-null  int64  
     42  month=mar                      41188 non-null  int64  
     43  month=apr                      41188 non-null  int64  
     44  month=sep                      41188 non-null  int64  
     45  day_of_week=mon                41188 non-null  int64  
     46  day_of_week=tue                41188 non-null  int64  
     47  day_of_week=wed                41188 non-null  int64  
     48  day_of_week=thu                41188 non-null  int64  
     49  day_of_week=fri                41188 non-null  int64  
     50  duration                       41188 non-null  float64
     51  campaign                       41188 non-null  float64
     52  pdays                          41188 non-null  float64
     53  previous                       41188 non-null  float64
     54  poutcome=nonexistent           41188 non-null  int64  
     55  poutcome=failure               41188 non-null  int64  
     56  poutcome=success               41188 non-null  int64  
     57  emp.var.rate                   41188 non-null  float64
     58  cons.price.idx                 41188 non-null  float64
     59  cons.conf.idx                  41188 non-null  float64
     60  euribor3m                      41188 non-null  float64
     61  nr.employed                    41188 non-null  float64
     62  category                       41188 non-null  int64  
    dtypes: float64(10), int64(53)
    memory usage: 19.8 MB
    


```python
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
outlier_fraction = len(data[data['category']==1])/float(len(data[data['category']==0]))
clfsvm = OneClassSVM(kernel="rbf", nu=outlier_fraction)
X = data.loc[:,data.columns!='category']
Y = data['category']
y_pred = clfsvm.fit_predict(X)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y).sum()
print("{}: {}".format("No. of Anomalous Points with One-Class SVM ",n_errors))
print("Accuracy Score :")
print(accuracy_score(Y,y_pred))
print("Classification Report :")
print(classification_report(Y,y_pred))
```

    No. of Anomalous Points with One-Class SVM : 7252
    Accuracy Score :
    0.8239292997960571
    Classification Report :
                  precision    recall  f1-score   support
    
               0       0.91      0.89      0.90     36548
               1       0.25      0.28      0.27      4640
    
        accuracy                           0.82     41188
       macro avg       0.58      0.59      0.58     41188
    weighted avg       0.83      0.82      0.83     41188
    
    


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
plt.style.use("ggplot")
sns.FacetGrid(data, hue="category").map(plt.scatter, "nr.employed", "age", edgecolor="k").add_legend()
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_8_0.png)
    



```python
data.shape
```




    (41188, 63)




```python
from scipy import spatial
sample_data = data.head(41180) 
samples = data.loc[41181:41188]
```


```javascript
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return False;
}
```


    <IPython.core.display.Javascript object>



```python
frame = []
for i in range(41181, 41188): 
    t1 = samples.loc[i]
    cls = samples.loc[i]["category"]
    for j in range(41180):
        t2 = sample_data.loc[j]
        class_label = data.loc[j]["category"]
        similarity = 1 - spatial.distance.cosine(t1, t2)
        if (class_label == 1):
            frame.append([class_label, similarity, j])
        
    df = pd.DataFrame(frame, columns=['category', 'Similarity', 'Transaction ID'])
    df_sorted = df.sort_values("Similarity", ascending=False)
    print("Top 5 suspected-fraud transactions having highest similarity with transaction ID = "+str(i)+ ":")
    print(df_sorted.iloc[:5])
    print("\n")
    frame = []
```

    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41181:
          category  Similarity  Transaction ID
    4417       1.0    0.933260           39073
    218        1.0    0.933256            1837
    3959       1.0    0.865343           35210
    626        1.0    0.865231            5502
    514        1.0    0.865045            4597
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41182:
          category  Similarity  Transaction ID
    1531       1.0    0.901827           13014
    3355       1.0    0.898451           29863
    3898       1.0    0.836246           34601
    2518       1.0    0.835903           22287
    4289       1.0    0.835808           37988
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41183:
          category  Similarity  Transaction ID
    163        1.0    0.876614            1366
    3052       1.0    0.794651           27036
    162        1.0    0.794483            1353
    349        1.0    0.793905            3079
    176        1.0    0.793708            1473
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41184:
          category  Similarity  Transaction ID
    4572       1.0    0.859526           40607
    3707       1.0    0.859427           32960
    3564       1.0    0.858049           31552
    4536       1.0    0.858018           40266
    2369       1.0    0.854684           20918
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41185:
          category  Similarity  Transaction ID
    3333       1.0    0.892698           29636
    4084       1.0    0.892566           36207
    985        1.0    0.892001            8525
    3469       1.0    0.891721           30819
    21         1.0    0.891262             153
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41186:
          category  Similarity  Transaction ID
    1893       1.0    0.896340           16376
    3690       1.0    0.896141           32785
    4619       1.0    0.895991           40949
    3078       1.0    0.895785           27266
    45         1.0    0.895512             332
    
    
    Top 5 suspected-fraud transactions having highest similarity with transaction ID = 41187:
          category  Similarity  Transaction ID
    3800       1.0    0.964525           33717
    3119       1.0    0.893301           27606
    1254       1.0    0.892938           10897
    4181       1.0    0.892748           37092
    1063       1.0    0.892707            9180
    
    
    


```python
data.shape
```




    (41188, 63)




```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)  #Considered only 3 components to put into 3 dimensions
to_model_cols = data.columns[0:63]
outliers = data.loc[data['category']==1]
outlier_index=list(outliers.index)
scaler = StandardScaler()
X = scaler.fit_transform(data[to_model_cols])
X_reduce = pca.fit_transform(X)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3_using_PCA")
# Plotting compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=3, lw=1, label="inliers",c="green")
# Plot x for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
           s=60, lw=2, marker="x", c="red", label="outliers")
ax.legend()
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_14_0.png)
    



```python
from sklearn.manifold import TSNE
standardized_data = StandardScaler().fit_transform(data)
data = standardized_data
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=250)
data = tsne.fit_transform(data)
```

    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 41188 samples in 0.007s...
    [t-SNE] Computed neighbors for 41188 samples in 6.399s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 41188
    [t-SNE] Computed conditional probabilities for sample 2000 / 41188
    [t-SNE] Computed conditional probabilities for sample 3000 / 41188
    [t-SNE] Computed conditional probabilities for sample 4000 / 41188
    [t-SNE] Computed conditional probabilities for sample 5000 / 41188
    [t-SNE] Computed conditional probabilities for sample 6000 / 41188
    [t-SNE] Computed conditional probabilities for sample 7000 / 41188
    [t-SNE] Computed conditional probabilities for sample 8000 / 41188
    [t-SNE] Computed conditional probabilities for sample 9000 / 41188
    [t-SNE] Computed conditional probabilities for sample 10000 / 41188
    [t-SNE] Computed conditional probabilities for sample 11000 / 41188
    [t-SNE] Computed conditional probabilities for sample 12000 / 41188
    [t-SNE] Computed conditional probabilities for sample 13000 / 41188
    [t-SNE] Computed conditional probabilities for sample 14000 / 41188
    [t-SNE] Computed conditional probabilities for sample 15000 / 41188
    [t-SNE] Computed conditional probabilities for sample 16000 / 41188
    [t-SNE] Computed conditional probabilities for sample 17000 / 41188
    [t-SNE] Computed conditional probabilities for sample 18000 / 41188
    [t-SNE] Computed conditional probabilities for sample 19000 / 41188
    [t-SNE] Computed conditional probabilities for sample 20000 / 41188
    [t-SNE] Computed conditional probabilities for sample 21000 / 41188
    [t-SNE] Computed conditional probabilities for sample 22000 / 41188
    [t-SNE] Computed conditional probabilities for sample 23000 / 41188
    [t-SNE] Computed conditional probabilities for sample 24000 / 41188
    [t-SNE] Computed conditional probabilities for sample 25000 / 41188
    [t-SNE] Computed conditional probabilities for sample 26000 / 41188
    [t-SNE] Computed conditional probabilities for sample 27000 / 41188
    [t-SNE] Computed conditional probabilities for sample 28000 / 41188
    [t-SNE] Computed conditional probabilities for sample 29000 / 41188
    [t-SNE] Computed conditional probabilities for sample 30000 / 41188
    [t-SNE] Computed conditional probabilities for sample 31000 / 41188
    [t-SNE] Computed conditional probabilities for sample 32000 / 41188
    [t-SNE] Computed conditional probabilities for sample 33000 / 41188
    [t-SNE] Computed conditional probabilities for sample 34000 / 41188
    [t-SNE] Computed conditional probabilities for sample 35000 / 41188
    [t-SNE] Computed conditional probabilities for sample 36000 / 41188
    [t-SNE] Computed conditional probabilities for sample 37000 / 41188
    [t-SNE] Computed conditional probabilities for sample 38000 / 41188
    [t-SNE] Computed conditional probabilities for sample 39000 / 41188
    [t-SNE] Computed conditional probabilities for sample 40000 / 41188
    [t-SNE] Computed conditional probabilities for sample 41000 / 41188
    [t-SNE] Computed conditional probabilities for sample 41188 / 41188
    [t-SNE] Mean sigma: 2.229839
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 84.140144
    [t-SNE] KL divergence after 251 iterations: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000
    


```python
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3_using_tSNE")
# Plotting the compressed data points
ax.scatter(data[:, 0], data[:, 1], zs=data[:, 2], s=3, lw=1, label="inliers",c="green")
# Plot x(s) for the ground truth outliers
out_index = [i for i in outlier_index if i <= 41188]
ax.scatter(data[out_index,0],data[out_index,1], data[out_index,2], lw=2, s=60, 
           marker="x", c="red", label="outliers")
ax.legend()
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_16_0.png)
    



```python
df_bank.columns
```




    Index(['age', 'job=housemaid', 'job=services', 'job=admin.', 'job=blue-collar',
           'job=technician', 'job=retired', 'job=management', 'job=unemployed',
           'job=self-employed', 'job=unknown', 'job=entrepreneur', 'job=student',
           'marital=married', 'marital=single', 'marital=divorced',
           'marital=unknown', 'education=basic.4y', 'education=high.school',
           'education=basic.6y', 'education=basic.9y',
           'education=professional.course', 'education=unknown',
           'education=university.degree', 'education=illiterate', 'default=0',
           'default=unknown', 'default=1', 'housing=0', 'housing=1',
           'housing=unknown', 'loan=0', 'loan=1', 'loan=unknown',
           'contact=cellular', 'month=may', 'month=jun', 'month=jul', 'month=aug',
           'month=oct', 'month=nov', 'month=dec', 'month=mar', 'month=apr',
           'month=sep', 'day_of_week=mon', 'day_of_week=tue', 'day_of_week=wed',
           'day_of_week=thu', 'day_of_week=fri', 'duration', 'campaign', 'pdays',
           'previous', 'poutcome=nonexistent', 'poutcome=failure',
           'poutcome=success', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
           'euribor3m', 'nr.employed', 'category'],
          dtype='object')




```python
df_bank_part1 = df_bank[[ 'default=1', 'housing=0', 'housing=1',
       'housing=unknown', 'loan=0', 'loan=1', 'loan=unknown',
       'contact=cellular', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome=nonexistent', 'poutcome=failure',
       'poutcome=success', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed', 'category']]
```


```python
plt.figure(figsize=(15,15))
sns.heatmap(df_bank_part1.corr().round(2), annot=True, cmap='YlGnBu')
#plt.show()
```




    <Axes: >




    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_19_1.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
```


```python
df_values = df_bank.values
```


```python
# The last element contains the labels
labels = df_values[:, -1]

# The other data points are the features
data = df_values[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)
```


```python
print(sum(labels==1))
print(sum(labels==0))
```

    4640
    36548
    

Rescale the data into the range 0 to 1

Convert it to tensorflow data, 32 bit float

Tensor flow has it's own data types- storage control

Note the scaling uisng only the range of the train data, not test data


```python
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```

Split into normal and anomalous data sets


```python
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
```

Plot a couple of normal and anomalous ECG patterns


```python
#change hard encoded number to reference the amount of columns
plt.grid()
plt.plot(np.arange(62), normal_train_data[0])
plt.title("Normal Data")
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_29_0.png)
    



```python
plt.grid()
plt.plot(np.arange(62), anomalous_train_data[0])
plt.title("Anomalous Data")
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_30_0.png)
    



```python
#decoders last layer needs to be numbver of columns
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(62, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
```


```python
autoencoder.compile(optimizer='adam', loss='mae')
```


```python
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
```

    Epoch 1/20
    8/8 [==============================] - 7s 32ms/step - loss: 0.4746 - val_loss: 0.4734
    Epoch 2/20
    8/8 [==============================] - 0s 11ms/step - loss: 0.4692 - val_loss: 0.4651
    Epoch 3/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.4590 - val_loss: 0.4479
    Epoch 4/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.4375 - val_loss: 0.4138
    Epoch 5/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.3949 - val_loss: 0.3528
    Epoch 6/20
    8/8 [==============================] - 0s 11ms/step - loss: 0.3245 - val_loss: 0.2732
    Epoch 7/20
    8/8 [==============================] - 0s 11ms/step - loss: 0.2456 - val_loss: 0.2091
    Epoch 8/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1917 - val_loss: 0.1733
    Epoch 9/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1639 - val_loss: 0.1551
    Epoch 10/20
    8/8 [==============================] - 0s 11ms/step - loss: 0.1503 - val_loss: 0.1500
    Epoch 11/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1466 - val_loss: 0.1483
    Epoch 12/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1456 - val_loss: 0.1478
    Epoch 13/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1452 - val_loss: 0.1474
    Epoch 14/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1447 - val_loss: 0.1454
    Epoch 15/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1444 - val_loss: 0.1453
    Epoch 16/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1442 - val_loss: 0.1454
    Epoch 17/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1438 - val_loss: 0.1443
    Epoch 18/20
    8/8 [==============================] - 0s 11ms/step - loss: 0.1433 - val_loss: 0.1442
    Epoch 19/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1427 - val_loss: 0.1443
    Epoch 20/20
    8/8 [==============================] - 0s 10ms/step - loss: 0.1422 - val_loss: 0.1426
    


```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f1594f9f2b0>




    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_34_1.png)
    



```python
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(62), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_35_0.png)
    



```python
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(62), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_36_0.png)
    



```python
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
```

    118/118 [==============================] - 0s 1ms/step
    


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_37_1.png)
    



```python
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
```

    Threshold:  0.17724909
    


```python
reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

```

    230/230 [==============================] - 0s 1ms/step
    


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_39_1.png)
    



```python
plt.hist(train_loss[None,:], bins=50,color="blue")

plt.hist(test_loss[None, :], bins=50,color='red')

plt.plot([threshold,threshold],[0,350],linestyle=":")
```




    [<matplotlib.lines.Line2D at 0x7f1120282ef0>]




    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_40_1.png)
    



```python
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))
```


```python
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
```

    Accuracy = 0.2391357125515902
    Precision = 0.10843900306077832
    Recall = 0.8312849162011173
    


```python
from sklearn.metrics import confusion_matrix
```


```python
sns.set_context("poster", font_scale = .75)
cm = confusion_matrix(test_labels, preds)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
```


    
![png](\img\posts\Bank-Fraud-Anomaly-Detection\output_44_0.png)
    

