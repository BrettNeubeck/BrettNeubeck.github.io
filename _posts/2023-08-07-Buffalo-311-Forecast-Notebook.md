---
layout: post
title: "Buffalo 311 Call Forecasting"
subtitle: "Forecasting Buffalo 311 Call Volume Using AMIRA, FBProphet & XGBOOST"
date: 2023-08-07
background: '/img/posts/311-forecasting/Buffalo311Logo.jpg'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---


# Imports


```python
import requests
import pandas as pd
import math
import datetime
import urllib.request
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')
```

## Retrieve Open Buffalo Data
<a id='open_buffalo_api_request'></a>


```python
#hide api ket in text file so its not public
app_token = open('api_key.txt', 'r').read()
#app_token
```


```python
#hide api token
#use request to pull 311 data from Jan 01, 2020 until today / can slice data in pandas later
limit = 500000
app_token = open('api_key.txt', 'r').read()
uri = f"https://data.buffalony.gov/resource/whkc-e5vr.json?$limit={limit}&$$app_token={app_token}&$where=open_date>'2020-01-10T12:00:00'"
r = requests.get(uri)
print('Status code ',r.status_code)
print('Number of rows returned ',len(r.json()))
print('Endoced URI with params ',r.url)
new_json = r.json()
#new_json
```

    Status code  200
    Number of rows returned  277742
    Endoced URI with params  https://data.buffalony.gov/resource/whkc-e5vr.json?$limit=500000&$$app_token=NnGV0W4ip4YEFBLvBMGAjaByD&$where=open_date%3E'2020-01-10T12:00:00'
    


```python
#make pandas df from  json / check head
df=pd.DataFrame(new_json)
print(df.shape)
df.head(2)
```

    (277742, 33)
    




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
      <th>case_reference</th>
      <th>open_date</th>
      <th>closed_date</th>
      <th>status</th>
      <th>subject</th>
      <th>reason</th>
      <th>type</th>
      <th>object_type</th>
      <th>address_number</th>
      <th>address_line_1</th>
      <th>...</th>
      <th>census_tract_2010</th>
      <th>census_block_group_2010</th>
      <th>census_block_2010</th>
      <th>tractce20</th>
      <th>geoid20_tract</th>
      <th>geoid20_blockgroup</th>
      <th>geoid20_block</th>
      <th>address_line_2</th>
      <th>x_coordinate</th>
      <th>y_coordinate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001109603</td>
      <td>2020-01-10T12:02:00.000</td>
      <td>2020-01-22T13:35:00.000</td>
      <td>Closed</td>
      <td>Dept of Public Works</td>
      <td>Sanitation</td>
      <td>Recycling Tote Deliver (Req_Serv)</td>
      <td>Property</td>
      <td>420</td>
      <td>PARKDALE</td>
      <td>...</td>
      <td>171</td>
      <td>2</td>
      <td>2000</td>
      <td>017100</td>
      <td>36029017100</td>
      <td>360290171002</td>
      <td>360290171002000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001109605</td>
      <td>2020-01-10T12:07:00.000</td>
      <td>2020-01-14T13:28:00.000</td>
      <td>Closed</td>
      <td>Dept of Public Works</td>
      <td>Rodent Control</td>
      <td>Rodents (Req_Serv)</td>
      <td>Property</td>
      <td>92</td>
      <td>PROCTOR</td>
      <td>...</td>
      <td>41</td>
      <td>2</td>
      <td>2002</td>
      <td>004100</td>
      <td>36029004100</td>
      <td>360290041002</td>
      <td>360290041002002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 33 columns</p>
</div>




```python
#save original Buffalo 311 Data to csv
df.to_csv('Buffalo311Data.csv', index = None, header=True)
```

## Format Open Buffalo Data Dates
<a id='format_open_buffalo_data'></a>


```python
#add count column with a value of one to rows to count calls by frequency
df['count'] = 1
#df.head()
```


```python
#format open date to pd datetime to get ready to parse column
df['time'] = pd.to_datetime(df['open_date'])
#df.info() #check to make sure Dtype is datetime64
```


```python
#take a look at teh columns I need
#parsing by hour will provide 24 times more training data than doing it by day
df1 = df[['time', 'count']]
df1.head(3)
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
      <th>time</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-10 12:02:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-10 12:07:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-10 12:08:00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### parsing by hour will provide 24 times more training data than doing it by day
#### I beleive that would equate to needing an extra six years of training data if I did it by day
#### If I am only testing since the start of 2023 and projecting roughly seven months
#### I think it is better to train on granular data rather than daily data going back an extra six years
#### I feel like the more recent the training data, especially after covid the better the results will yield


```python
#use date time to parse by year
df1['time'] = pd.to_datetime(df1['time']).dt.strftime('%Y-%m-%d %H')
df1.head()
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
      <th>time</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-10 12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-10 12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-10 12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-10 12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-10 12</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#group by hour, count and reset index
df2 = df1.groupby(['time']).count().reset_index()
```


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18336 entries, 0 to 18335
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   time    18336 non-null  object
     1   count   18336 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 286.6+ KB
    


```python
#make sure time is pd datetime after groupby. seems to create new object after groupby
df2['time'] = pd.to_datetime(df2['time'])
```


```python
#set time as index
df2 = df2.set_index('time')
```


```python
#df2.head()
```


```python
#make df by day instead of hour for some charts
#ultimatly decided to use hourly because I feel it would help for staffing and letting people go home early 
df_day = df1.copy()
```


```python
df_day['time'] = pd.to_datetime(df_day['time']).dt.strftime('%Y-%m-%d')
df_day = df_day.groupby(['time']).count().reset_index()
df_day['time'] = pd.to_datetime(df_day['time'])
df_day = df_day.set_index('time')

```


```python
#df_day.head()
```


```python
#make graph of calls by day
color_pal = sns.color_palette()
df_day.plot(style='.',
         figsize=(10,5),
         ms=1,
         color = color_pal[0],
         title='311 Calls By Day')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_21_0.png)
    



```python
#make graph of call by hour
color_pal = sns.color_palette()
df2.plot(style='.',
         figsize=(10,5),
         ms=1,
         color = color_pal[0],
         title='311 Calls By Hour')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_22_0.png)
    



```python
#create seasons for boxplot graph showing dayofweek and season
from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                       'Wednesday', 'Thursday',
                                       'Friday', 'Saturday', 'Sunday'],
                           ordered=True)




def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    #df['weekofyear'] = df['date'].dt.weekofyear
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320)%1300
    
    df['season'] = pd.cut(df['date_offset'], [0, 300, 600, 900, 1300],
                         labels=['Spring', 'Summer', 'Fall', 'Winter'])


    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear', 'weekday', 'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(df_day, label='count')

features_and_target = pd.concat([X, y], axis=1)
```


```python
features_and_target.head()
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
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>weekday</th>
      <th>season</th>
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>Friday</td>
      <td>Winter</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2020-01-11</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>11</td>
      <td>11</td>
      <td>2</td>
      <td>Saturday</td>
      <td>Winter</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2020-01-12</th>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>12</td>
      <td>12</td>
      <td>2</td>
      <td>Sunday</td>
      <td>Winter</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2020-01-13</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>13</td>
      <td>13</td>
      <td>3</td>
      <td>Monday</td>
      <td>Winter</td>
      <td>368</td>
    </tr>
    <tr>
      <th>2020-01-14</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>14</td>
      <td>14</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>Winter</td>
      <td>316</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=features_and_target.dropna(),
           x='weekday',
           y='count',
           hue='season',
           ax = ax,
           linewidth=1,
           palette='YlGnBu')

ax.set_title('Number of 311 Calls by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('311 Call Count')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_25_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data=features_and_target.dropna(),
           x='weekofyear',
           y='count',
           hue='season',
           ax = ax,
           linewidth=1,
           palette='YlGnBu')

ax.set_title('Number Of 311 Calls By Week Of Year')
ax.set_xlabel('Week')
ax.set_ylabel('311 Call Count')
plt.xticks(rotation = 90) # rotates x-axis by 45 degrees
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_26_0.png)
    


### Group data by hour and see what the call volume is and group data by week


```python
split_date = '1-jan-2023'
df_train = df2.loc[df2.index <= split_date].copy()
df_test = df2.loc[df2.index > split_date].copy()

# plot train and test so you can see where the split is

df_test \
    .rename(columns={'count': 'TEST SET'}) \
    .join(df_train.rename(columns={'count': 'TRAINING SET'}),
         how='outer') \
    .plot(figsize=(15,5), title='Buffalo 311 Calls By Hour', style='.')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_28_0.png)
    


Train Prophet model
datetime column named ds
target = y


```python
#format data for prophet model using ds and y
df_train_prophet = df_train.reset_index()\
    .rename(columns={'time': 'ds',
                    'count': 'y'})
```


```python
df_train_prophet.head()
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-10 12:00:00</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-10 13:00:00</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-10 14:00:00</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-10 15:00:00</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-10 16:00:00</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%time
model = Prophet()
model.fit(df_train_prophet)
```

    15:18:59 - cmdstanpy - INFO - Chain [1] start processing
    15:19:07 - cmdstanpy - INFO - Chain [1] done processing
    

    CPU times: total: 1.42 s
    Wall time: 9.67 s
    




    <prophet.forecaster.Prophet at 0x2cc235e5f10>




```python
%%time
# create test dat with same column names
df_test_prophet = df_test.reset_index()\
    .rename(columns={'time': 'ds',
                    'count': 'y'})

df_test_fcst = model.predict(df_test_prophet)
```

    CPU times: total: 2.02 s
    Wall time: 541 ms
    


```python
df_test_fcst.head()
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
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>daily</th>
      <th>...</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 01:00:00</td>
      <td>19.038017</td>
      <td>-18.082230</td>
      <td>16.134619</td>
      <td>19.038017</td>
      <td>19.038017</td>
      <td>-20.199082</td>
      <td>-20.199082</td>
      <td>-20.199082</td>
      <td>-6.409008</td>
      <td>...</td>
      <td>-14.649280</td>
      <td>-14.649280</td>
      <td>-14.649280</td>
      <td>0.859206</td>
      <td>0.859206</td>
      <td>0.859206</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.161066</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 03:00:00</td>
      <td>19.041680</td>
      <td>-16.196708</td>
      <td>16.441544</td>
      <td>19.041680</td>
      <td>19.041680</td>
      <td>-18.544646</td>
      <td>-18.544646</td>
      <td>-18.544646</td>
      <td>-5.126092</td>
      <td>...</td>
      <td>-14.317852</td>
      <td>-14.317852</td>
      <td>-14.317852</td>
      <td>0.899298</td>
      <td>0.899298</td>
      <td>0.899298</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.497034</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 07:00:00</td>
      <td>19.049007</td>
      <td>-16.467086</td>
      <td>17.694639</td>
      <td>19.049007</td>
      <td>19.049007</td>
      <td>-18.550251</td>
      <td>-18.550251</td>
      <td>-18.550251</td>
      <td>-6.533164</td>
      <td>...</td>
      <td>-12.996319</td>
      <td>-12.996319</td>
      <td>-12.996319</td>
      <td>0.979233</td>
      <td>0.979233</td>
      <td>0.979233</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.498756</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01 08:00:00</td>
      <td>19.050838</td>
      <td>-1.699195</td>
      <td>31.412708</td>
      <td>19.050838</td>
      <td>19.050838</td>
      <td>-4.490101</td>
      <td>-4.490101</td>
      <td>-4.490101</td>
      <td>7.049725</td>
      <td>...</td>
      <td>-12.538989</td>
      <td>-12.538989</td>
      <td>-12.538989</td>
      <td>0.999163</td>
      <td>0.999163</td>
      <td>0.999163</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.560738</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01 09:00:00</td>
      <td>19.052670</td>
      <td>7.991571</td>
      <td>42.268996</td>
      <td>19.052670</td>
      <td>19.052670</td>
      <td>6.905000</td>
      <td>6.905000</td>
      <td>6.905000</td>
      <td>17.921719</td>
      <td>...</td>
      <td>-12.035792</td>
      <td>-12.035792</td>
      <td>-12.035792</td>
      <td>1.019073</td>
      <td>1.019073</td>
      <td>1.019073</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.957670</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))
fig = model.plot(df_test_fcst, ax= ax)
plt.xticks(rotation = 45) # rotates x-axis by 45 degrees
ax.set_title('Prophet Forecast')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_35_0.png)
    



```python
fig = model.plot_components(df_test_fcst)
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_36_0.png)
    


### Compare Forecast to Actuals


```python
#plot teh forecast with the actual
f, ax = plt.subplots(figsize=(15,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst, ax= ax)
```


    
![png](\img\posts\311-Forecasting-notebook\output_38_0.png)
    


### Zoom into actual vs predicted
RED DOTS ARE ACTUAL/ BLUE LINES ARE PREDICTIONS
/nseems like there is a mismatch between actual and predicted probably due to 2023 blizzard.


```python
#zoomed in plot of forecast vs actual
import datetime

f, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst, ax= ax)
# ax.set_xbound(lower='01-01-2015',
#              upper='02-01-2015')
ax.set_xlim([datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)])
ax.set_ylim(0,110)
plot = plt.suptitle('January 2023 Forecast vs Actuals')
```


    
![png](\img\posts\311-Forecasting-notebook\output_40_0.png)
    


### Zoom into first week of jan / Blizzard affected call projections
Seems like the 2023 Blizard caused more calls than predicted especially when you compare to the first week of April


```python
#zoomed in plot of forecast vs actual
import datetime

f, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst, ax= ax)
# ax.set_xbound(lower='01-01-2015',
#              upper='02-01-2015')
ax.set_xlim([datetime.date(2023, 1, 1), datetime.date(2023, 1, 7)])
ax.set_ylim(0,100)
plot = plt.suptitle('First Week Of January 2023 Forecast vs Actuals')
```


    
![png](\img\posts\311-Forecasting-notebook\output_42_0.png)
    


### Zoom Into First Week Of April 2023
###### Aprils predcition is looking better than the first week of Jan
###### Outliers in first week of Jan are most likly from the Blizzard


```python
#zoomed in plot of forecast vs actual
import datetime

f, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst, ax= ax)
# ax.set_xbound(lower='01-01-2015',
#              upper='02-01-2015')
ax.set_xlim([datetime.date(2023, 4, 1), datetime.date(2023, 4, 7)])
ax.set_ylim(0,70)
plot = plt.suptitle('First Week Of April 2023 Forecast vs Actuals')
```


    
![png](\img\posts\311-Forecasting-notebook\output_44_0.png)
    


# evaluate the model with Error Metrics


```python
# true value vs predicted value
np.sqrt(mean_squared_error(y_true=df_test['count'],
                          y_pred=df_test_fcst['yhat']))
```




    18.53331967782901




```python
mean_absolute_error(y_true=df_test['count'],
                   y_pred=df_test_fcst['yhat'])
```




    16.444075387619147



# Adding Holidays To Prophet Model


```python
# set up holiday dataframe
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()


holidays = cal.holidays(start=df2.index.min(),
                        end=df2.index.max(),
                        return_name=True)
holiday_df = pd.DataFrame(data=holidays,
                          columns=['holiday'])
holiday_df = holiday_df.reset_index().rename(columns={'index':'ds'})
```


```python
holiday_df.head()
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
      <th>ds</th>
      <th>holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-20</td>
      <td>Birthday of Martin Luther King, Jr.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-02-17</td>
      <td>Washington’s Birthday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-05-25</td>
      <td>Memorial Day</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-07-03</td>
      <td>Independence Day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-09-07</td>
      <td>Labor Day</td>
    </tr>
  </tbody>
</table>
</div>




```python
holiday_df['holiday'].value_counts()
```




    Birthday of Martin Luther King, Jr.     4
    Washington’s Birthday                   4
    Memorial Day                            3
    Independence Day                        3
    Labor Day                               3
    Columbus Day                            3
    Veterans Day                            3
    Thanksgiving Day                        3
    Christmas Day                           3
    New Year's Day                          3
    Juneteenth National Independence Day    2
    Name: holiday, dtype: int64




```python
import time
```


```python
#run Prophet model with holiday df as a parameter for holiday
#%%time

model_with_holidays = Prophet(holidays=holiday_df)
model_with_holidays.fit(df_train_prophet)
```

    15:35:23 - cmdstanpy - INFO - Chain [1] start processing
    15:35:31 - cmdstanpy - INFO - Chain [1] done processing
    




    <prophet.forecaster.Prophet at 0x2cc2cfe3c40>




```python
# predict on training set with model
df_test_fcst_with_hols = model_with_holidays.predict(df=df_test_prophet)
```


```python
df_test_fcst_with_hols.head()
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
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>Birthday of Martin Luther King, Jr.</th>
      <th>Birthday of Martin Luther King, Jr._lower</th>
      <th>Birthday of Martin Luther King, Jr._upper</th>
      <th>Christmas Day</th>
      <th>...</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 01:00:00</td>
      <td>19.962327</td>
      <td>-17.298810</td>
      <td>16.125482</td>
      <td>19.962327</td>
      <td>19.962327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-15.474464</td>
      <td>-15.474464</td>
      <td>-15.474464</td>
      <td>1.114713</td>
      <td>1.114713</td>
      <td>1.114713</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.726449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 03:00:00</td>
      <td>19.966306</td>
      <td>-15.429744</td>
      <td>17.809987</td>
      <td>19.966306</td>
      <td>19.966306</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-15.147649</td>
      <td>-15.147649</td>
      <td>-15.147649</td>
      <td>1.160615</td>
      <td>1.160615</td>
      <td>1.160615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.815920</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 07:00:00</td>
      <td>19.974263</td>
      <td>-15.194841</td>
      <td>17.696200</td>
      <td>19.974263</td>
      <td>19.974263</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-13.741419</td>
      <td>-13.741419</td>
      <td>-13.741419</td>
      <td>1.251959</td>
      <td>1.251959</td>
      <td>1.251959</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.869277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01 08:00:00</td>
      <td>19.976252</td>
      <td>-0.710243</td>
      <td>32.056320</td>
      <td>19.976252</td>
      <td>19.976252</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-13.242802</td>
      <td>-13.242802</td>
      <td>-13.242802</td>
      <td>1.274699</td>
      <td>1.274699</td>
      <td>1.274699</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15.025595</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01 09:00:00</td>
      <td>19.978242</td>
      <td>10.426556</td>
      <td>42.368802</td>
      <td>19.978242</td>
      <td>19.978242</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-12.690424</td>
      <td>-12.690424</td>
      <td>-12.690424</td>
      <td>1.297399</td>
      <td>1.297399</td>
      <td>1.297399</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26.507473</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
model_with_holidays.plot_components(
    df_test_fcst_with_hols)
```




    
![png](\img\posts\311-Forecasting-notebook\output_56_0.png)
    




    
![png](\img\posts\311-Forecasting-notebook\output_56_1.png)
    



```python
#zoomed in plot of forecast vs actual

# red dots show actual vs projected line in blue
import datetime

f, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst_with_hols, ax= ax)
# ax.set_xbound(lower='07-01-2015',
#              upper='07-07-2015')
ax.set_xlim([datetime.date(2023, 1, 1), datetime.date(2023, 5, 1)])
ax.set_ylim(0,110)
plt.xticks(rotation = 45) # rotates x-axis by 45 degrees
plot = plt.suptitle('January to April Forecast vs Actuals')
```


    
![png](\img\posts\311-Forecasting-notebook\output_57_0.png)
    



```python
#zoomed in plot of forecast vs actual
import datetime

f, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_test.index, df_test['count'], color ='r')
fig = model.plot(df_test_fcst_with_hols, ax= ax)
# ax.set_xbound(lower='07-01-2015',
#              upper='07-07-2015')
ax.set_xlim([datetime.date(2023, 4, 1), datetime.date(2023, 4, 7)])
ax.set_ylim(0,80)
plot = plt.suptitle('First Week Of April Forecast vs Actuals With Holidays')
```


    
![png](\img\posts\311-Forecasting-notebook\output_58_0.png)
    



```python
# true value vs predicted value
np.sqrt(mean_squared_error(y_true=df_test['count'],
                          y_pred=df_test_fcst_with_hols['yhat']))
```




    19.14157031066351




```python
mean_absolute_error(y_true=df_test['count'],
                   y_pred=df_test_fcst_with_hols['yhat'])
```




    17.139861907579142



# Predict into the future


```python
future = model.make_future_dataframe(periods=365*24, freq='h', include_history=False)
```


```python
forecast = model_with_holidays.predict(future)
```


```python
forecast.head()
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
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>Birthday of Martin Luther King, Jr.</th>
      <th>Birthday of Martin Luther King, Jr._lower</th>
      <th>Birthday of Martin Luther King, Jr._upper</th>
      <th>Christmas Day</th>
      <th>...</th>
      <th>weekly</th>
      <th>weekly_lower</th>
      <th>weekly_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01 01:00:00</td>
      <td>19.962327</td>
      <td>-16.882693</td>
      <td>15.841012</td>
      <td>19.962327</td>
      <td>19.962327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-15.474464</td>
      <td>-15.474464</td>
      <td>-15.474464</td>
      <td>1.114713</td>
      <td>1.114713</td>
      <td>1.114713</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.726449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01 02:00:00</td>
      <td>19.964316</td>
      <td>-12.946475</td>
      <td>17.548705</td>
      <td>19.964316</td>
      <td>19.964316</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-15.343241</td>
      <td>-15.343241</td>
      <td>-15.343241</td>
      <td>1.137683</td>
      <td>1.137683</td>
      <td>1.137683</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.207288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01 03:00:00</td>
      <td>19.966306</td>
      <td>-14.783082</td>
      <td>17.679284</td>
      <td>19.966306</td>
      <td>19.966306</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-15.147649</td>
      <td>-15.147649</td>
      <td>-15.147649</td>
      <td>1.160615</td>
      <td>1.160615</td>
      <td>1.160615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.815920</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01 04:00:00</td>
      <td>19.968295</td>
      <td>-21.090066</td>
      <td>11.068481</td>
      <td>19.968295</td>
      <td>19.968295</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-14.888335</td>
      <td>-14.888335</td>
      <td>-14.888335</td>
      <td>1.183508</td>
      <td>1.183508</td>
      <td>1.183508</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-4.982981</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01 05:00:00</td>
      <td>19.970284</td>
      <td>-28.096313</td>
      <td>3.966954</td>
      <td>19.970284</td>
      <td>19.970284</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>-14.566408</td>
      <td>-14.566408</td>
      <td>-14.566408</td>
      <td>1.206364</td>
      <td>1.206364</td>
      <td>1.206364</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-10.279008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))
fig = model.plot(forecast, ax= ax)
ax.set_title('Prophet Forecast')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_65_0.png)
    



```python
#zoomed in plot of forecast
import datetime

fig, ax = plt.subplots(figsize=(10,5))
fig = model.plot(forecast, ax= ax)
ax.set_xlim([datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)])
ax.set_ylim(0,100)
plt.xticks(rotation = 45) # rotates x-axis by 45 degrees
ax.set_title('Prophet Forecast')
plt.show()


```


    
![png](\img\posts\311-Forecasting-notebook\output_66_0.png)
    



```python
#zoomed in plot of forecast for upcoming week; first week of may
import datetime

fig, ax = plt.subplots(figsize=(10,5))
fig = model.plot(forecast, ax= ax)
ax.set_xlim([datetime.date(2023, 6, 1), datetime.date(2023, 6, 7)])
ax.set_ylim(0,80)
ax.set_title('Prophet Forecast First Week Of June')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_67_0.png)
    



```python
#zoomed in plot of forecast for upcoming Friday; first firday of june
import datetime

fig, ax = plt.subplots(figsize=(10,5))
fig = model.plot(forecast, ax= ax)
ax.set_xlim([datetime.date(2023, 6, 2), datetime.date(2023, 6, 3)])
ax.set_ylim(0,70)
ax.set_title('Prophet Forecast First Friday Of June 2023 By Hour')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_68_0.png)
    


The graph above shows a peak call time between 9am and 3pm friday June 2nd

### XGBOOST Forecasting


```python
split_date = '1-jan-2023'
df_train2 = df2.loc[df2.index <= split_date].copy()
df_test2 = df2.loc[df2.index > split_date].copy()

fig, ax = plt.subplots(figsize=(15,5))
df_train2.plot(ax=ax, label='Training Data', title='Train/Test Data Split')
df_test2.plot(ax=ax, label='Testing Data')
ax.axvline(split_date, color='black', ls='--')
ax.legend(['Training Data', 'Test Data'])
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_71_0.png)
    



```python
#check out two weeks in Jan 2022 to get a better picture of the data
df_train2.loc[(df_train.index > '01-01-2022') & (df_train2.index < '01-15-2022')].plot(figsize=(15,5), title='Two Weeks in Jan 2022')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_72_0.png)
    


### Create more features to add to the model


```python
def create_features_2(df):
    '''
    create time series features based on index date
    '''
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df3 = create_features_2(df2)
```

### Visualize Data By New Features


```python
df3.head(2)
df3.shape
```




    (18336, 7)




```python
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='hour', y='count')
ax.set_title('Buffalo 311 Call Count By Hour')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_77_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='month', y='count', palette='Blues')
ax.set_title('Buffalo 311 Call Count By Month')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_78_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='quarter', y='count', palette='Blues')
ax.set_title('Buffalo 311 Call Count By Quarter')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_79_0.png)
    



```python
#note that  2023 data is only until 05/04/2023
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='year', y='count', palette = 'Reds')
ax.set_title('Buffalo 311 Call Count By Year')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_80_0.png)
    


##### Seems like 2023 is going to be a blow out year for 311 calls since it is already equal to 2021 call count
##### We are about to be into the 3rd quarter which is historically the largest quarter according to the quarter graph above


```python
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='dayofweek', y='count', palette='Greens')
ax.set_title('Buffalo 311 Call Count By Day Of Week')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_82_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data = df3, x='dayofyear', y='count', palette='Purples')
ax.set_title('Buffalo 311 Call Count By Day Of Year')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_83_0.png)
    


### Create XGBOOST Model


```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
```


```python
df2.head()
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
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10 12:00:00</th>
      <td>21</td>
    </tr>
    <tr>
      <th>2020-01-10 13:00:00</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2020-01-10 14:00:00</th>
      <td>22</td>
    </tr>
    <tr>
      <th>2020-01-10 15:00:00</th>
      <td>30</td>
    </tr>
    <tr>
      <th>2020-01-10 16:00:00</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_train = create_features_2(df_train)
df_test = create_features_2(df_test)
```


```python
print(df_train.columns)
print(df_test.columns)
```

    Index(['count', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear'], dtype='object')
    Index(['count', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear'], dtype='object')
    


```python
features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target = 'count'
```


```python
X_train = df_train[features]
y_train = df_train[target]

X_test = df_test[features]
y_test = df_test[target]
```


```python
#had to lower learning rate to yield better MSE results and have the model not overfit
reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50,
                      learning_rate = 0.001)
reg.fit(X_train, y_train,
       eval_set = [(X_train, y_train), (X_test, y_test)],
       verbose = 75)
```

    [0]	validation_0-rmse:24.42930	validation_1-rmse:21.25184
    [75]	validation_0-rmse:23.02614	validation_1-rmse:19.67356
    [150]	validation_0-rmse:21.74553	validation_1-rmse:18.24009
    [225]	validation_0-rmse:20.57594	validation_1-rmse:17.05880
    [300]	validation_0-rmse:19.51455	validation_1-rmse:15.93974
    [375]	validation_0-rmse:18.54768	validation_1-rmse:14.92595
    [450]	validation_0-rmse:17.66992	validation_1-rmse:14.00971
    [525]	validation_0-rmse:16.87422	validation_1-rmse:13.17353
    [600]	validation_0-rmse:16.15464	validation_1-rmse:12.41610
    [675]	validation_0-rmse:15.50567	validation_1-rmse:11.74820
    [750]	validation_0-rmse:14.91105	validation_1-rmse:11.12767
    [825]	validation_0-rmse:14.37133	validation_1-rmse:10.58090
    [900]	validation_0-rmse:13.87591	validation_1-rmse:10.11855
    [975]	validation_0-rmse:13.41815	validation_1-rmse:9.75119
    [999]	validation_0-rmse:13.27996	validation_1-rmse:9.65540
    




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=50,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.001, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=1000, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=None, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=50,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.001, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=1000, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=None, ...)</pre></div></div></div></div></div>



### Feature Importance


```python
from xgboost import plot_importance
plot_importance(reg)
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_93_0.png)
    


### Forecast on the Test Data


```python
df_test['prediction'] = reg.predict(X_test)
```


```python
df3 = df3.merge(df_test[['prediction']], how='left', left_index=True, right_index=True)
```


```python
ax = df3[['count']].plot(figsize = (15,5))
df3['prediction'].plot(ax=ax, style='.')
plt.legend(['True Data', 'Predictions'])
ax.set_title('Raw Data & Predictions')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_97_0.png)
    


### Zoom into the test vs projections
### Feb projection(blue) vs actual(red)


```python
ax = df_test.loc[(df_test.index > '02-01-2023') & (df_test.index < '03-01-2023')]['count'] \
    .plot(figsize=(15,5), title = ' Feb 2023')
df_test.loc[(df_test.index > '02-01-2023') & (df_test.index <'03-01-2023')]['prediction'] \
    .plot(style='.')
plt.legend(['Prediction','Real Data' ])
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_99_0.png)
    



```python
score = np.sqrt(mean_squared_error(df_test['count'], df_test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')
```

    RMSE Score on Test set: 9.66
    

# Calaculate Error for best and worst days

### Worst Projected Days Happened Close to Blizzard 


```python
df_test['error'] = np.abs(df_test[target] - df_test['prediction'])
```


```python
df_test['date'] = df_test.index.date
```


```python
#worst projected days happended close to the blizzard
df_test.groupby('date')['error'].mean().sort_values(ascending=False).head(5)
```




    date
    2023-01-05    19.826350
    2023-01-03    17.533686
    2023-02-24    17.288375
    2023-01-04    16.783298
    2023-01-09    14.823705
    Name: error, dtype: float64




```python
#best projected days of forecast
df_test.groupby('date')['error'].mean().sort_values(ascending=True).head(5)
```




    date
    2023-01-15    0.749523
    2023-03-25    0.754664
    2023-01-29    0.799345
    2023-04-30    0.824599
    2023-03-18    0.903888
    Name: error, dtype: float64



### Outlier Analysis

#### Decided to keep outliers in for the K_fold xgb model


```python
df2['count'].plot(kind='hist', bins=1000)
```




    <AxesSubplot: ylabel='Frequency'>




    
![png](\img\posts\311-Forecasting-notebook\output_107_1.png)
    



```python
df2.query('count < 10').plot(figsize=(15,5), style='.')
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_108_1.png)
    



```python
df2.query('count > 50').plot(figsize=(15,5), style='.')
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_109_1.png)
    



```python
split_date = '1-jan-2023'
df_train3 = df2.loc[df2.index <= split_date].copy()
df_test3 = df2.loc[df2.index > split_date].copy()

fig, ax = plt.subplots(figsize=(15,5))
df_train3.plot(ax=ax, label='Training Data', title='Train/Test Data Split')
df_test3.plot(ax=ax, label='Testing Data')
ax.axvline(split_date, color='black', ls='--')
ax.legend(['Training Data', 'Test Data'])
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_110_0.png)
    


### Time Series Cross Validation


```python
from sklearn.model_selection import TimeSeriesSplit
```


```python
df2.shape
```




    (18336, 1)




```python
print(df2.head())
print(df2.tail())
```

                         count
    time                      
    2020-01-10 12:00:00     21
    2020-01-10 13:00:00     20
    2020-01-10 14:00:00     22
    2020-01-10 15:00:00     30
    2020-01-10 16:00:00      3
                         count
    time                      
    2023-05-09 19:00:00      5
    2023-05-09 20:00:00      1
    2023-05-09 21:00:00      2
    2023-05-09 22:00:00     17
    2023-05-09 23:00:00      2
    


```python
#tss = TimeSeriesSplit(n_splits=2, test_size=24*365, gap = 24)
tss = TimeSeriesSplit(n_splits=8, test_size=24*60*1, gap = 24)
df4 = df2.sort_index()
```


```python
df4.head()
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
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10 12:00:00</th>
      <td>21</td>
    </tr>
    <tr>
      <th>2020-01-10 13:00:00</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2020-01-10 14:00:00</th>
      <td>22</td>
    </tr>
    <tr>
      <th>2020-01-10 15:00:00</th>
      <td>30</td>
    </tr>
    <tr>
      <th>2020-01-10 16:00:00</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axs = plt.subplots(8, 1, figsize=(15,15),
                            sharex = True)

fold = 0

for train_idx, val_idx in tss.split(df4):
    train = df4.iloc[train_idx]
    test = df4.iloc[val_idx]
    train['count'].plot(ax=axs[fold],
                        label='Training Set', 
                        title=f'Data Train/Test Split Fold {fold}')
    test['count'].plot(ax=axs[fold],
                       label = 'Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold +=1
```


    
![png](\img\posts\311-Forecasting-notebook\output_117_0.png)
    



```python
# make function for new time features 
def create_features4(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df4 = create_features4(df4)
```


```python
df4.head()
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
      <th>count</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10 12:00:00</th>
      <td>21</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-01-10 13:00:00</th>
      <td>20</td>
      <td>13</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-01-10 14:00:00</th>
      <td>22</td>
      <td>14</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-01-10 15:00:00</th>
      <td>30</td>
      <td>15</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-01-10 16:00:00</th>
      <td>3</td>
      <td>16</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Lag Features


```python
# make lag feature function
def add_lags(df):
    target_map = df['count'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('182 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    return df
```


```python
#test lags function
#df4 = add_lags(df)
```


```python
#check df after lags function test
#df4.tail()
```

### Train Using Cross Validation


```python
tss = TimeSeriesSplit(n_splits=8, test_size=24*60*1, gap = 24)
df4 = df2.sort_index()

fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df4):
    train = df4.iloc[train_idx]
    test = df4.iloc[val_idx]
    
    train = create_features4(train)
    test = create_features4(test)
    
    train = add_lags(train)
    test = add_lags(test)
    
    FEATURES4 = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2', 'lag3']
    TARGET4 = 'count'
    
    X_train = train[FEATURES4]
    y_train = train[TARGET4]

    X_test = test[FEATURES4]
    y_test = test[TARGET4]
    
    
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=100000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.1)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=10000)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
    
    
```

    [15:49:23] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:20.28806	validation_1-rmse:25.23168
    [365]	validation_0-rmse:6.34963	validation_1-rmse:9.52946
    [15:49:23] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:21.08501	validation_1-rmse:20.03218
    [495]	validation_0-rmse:6.41239	validation_1-rmse:7.29066
    [15:49:24] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:20.86114	validation_1-rmse:29.95618
    [163]	validation_0-rmse:7.21473	validation_1-rmse:21.61451
    [15:49:25] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.05514	validation_1-rmse:22.50675
    [252]	validation_0-rmse:7.36629	validation_1-rmse:11.83633
    [15:49:25] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.03981	validation_1-rmse:26.97997
    [972]	validation_0-rmse:6.62589	validation_1-rmse:11.07590
    [15:49:26] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.46248	validation_1-rmse:21.76932
    [3115]	validation_0-rmse:5.52317	validation_1-rmse:7.01527
    [15:49:31] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.24770	validation_1-rmse:28.82747
    [837]	validation_0-rmse:6.73216	validation_1-rmse:19.81065
    [15:49:33] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.71290	validation_1-rmse:20.37634
    [239]	validation_0-rmse:8.34257	validation_1-rmse:12.02755
    


```python
print(f'Scores across all folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')
```

    Scores across all folds 12.4270
    Fold scores:[9.432683974684368, 7.263089487308816, 21.508818615589814, 11.799599285719262, 11.072779306904335, 7.013240970100097, 19.336143148373846, 11.989809586405414]
    

### Retrain model on all training data to prepare for forecasting the future


```python
#retrain all data to prepare for forecasting
#still leverage all data for forecast prediction
#switchedn_estimator to 1500 because the model still seemed like it was getting better
#also moved the learning rate down

df4 = create_features4(df4)
df4 = add_lags(df4)

FEATURES4 = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2', 'lag3']
TARGET4 = 'count'

X_all = df4[FEATURES4]
y_all = df4[TARGET4]

reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',
                       #n_estimators=25000,
                       n_estimators=100000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       #learning_rate=0.0001) # LEARNING FASTER SEEMS TO BOOST PERFORMANCE OF THE MODEL
                       learning_rate=0.1)

reg.fit(X_all, y_all,
       eval_set=[(X_all, y_all)],
       verbose=10000)
```

    [15:50:49] WARNING: c:\users\dev-admin\croot2\xgboost-split_1675461376218\work\src\objective\regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
    [0]	validation_0-rmse:22.43720
    [10000]	validation_0-rmse:4.73683
    [20000]	validation_0-rmse:4.05477
    [30000]	validation_0-rmse:3.67679
    [40000]	validation_0-rmse:3.42238
    [50000]	validation_0-rmse:3.23311
    [60000]	validation_0-rmse:3.07159
    [70000]	validation_0-rmse:2.93679
    [80000]	validation_0-rmse:2.82711
    [90000]	validation_0-rmse:2.72784
    [99999]	validation_0-rmse:2.63702
    




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBRegressor(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=50,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=100000, n_jobs=None, num_parallel_tree=None,
             objective=&#x27;reg:linear&#x27;, predictor=None, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=50,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.1, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=3, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=100000, n_jobs=None, num_parallel_tree=None,
             objective=&#x27;reg:linear&#x27;, predictor=None, ...)</pre></div></div></div></div></div>



###  Lowered RMSE when I change the n_estimator to 100K and move the learning rate down

### Forecasting The Future


```python
#find max value and set rage from last date 
FurtueStartDate = df4.index.max()
'''
df_day['time'] = pd.to_datetime(df_day['time']).dt.strftime('%Y-%m-%d')
'''
#FurtueStartDate = FurtueStartDate.dt.strftime('%Y-%m-%d')
FurtueStartDate
```




    Timestamp('2023-05-09 23:00:00')




```python
#can't project longer than the smallest lag
future = pd.date_range('2023-05-04', '11/02/2023' , freq='1h')
future_df = pd.DataFrame(index=future)
```


```python
future_df['is_future'] = True
df4['is_future'] = False
df_and_future = pd.concat([df4, future_df])
```


```python
df_and_future = create_features4(df_and_future)
df_and_future = add_lags(df_and_future)
```


```python
df_and_future
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
      <th>count</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>lag1</th>
      <th>lag2</th>
      <th>lag3</th>
      <th>is_future</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10 12:00:00</th>
      <td>21.0</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-01-10 13:00:00</th>
      <td>20.0</td>
      <td>13</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-01-10 14:00:00</th>
      <td>22.0</td>
      <td>14</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-01-10 15:00:00</th>
      <td>30.0</td>
      <td>15</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2020-01-10 16:00:00</th>
      <td>3.0</td>
      <td>16</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2020</td>
      <td>10</td>
      <td>10</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-11-01 20:00:00</th>
      <td>NaN</td>
      <td>20</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 21:00:00</th>
      <td>NaN</td>
      <td>21</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 22:00:00</th>
      <td>NaN</td>
      <td>22</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 23:00:00</th>
      <td>NaN</td>
      <td>23</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-02 00:00:00</th>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>306</td>
      <td>2</td>
      <td>44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>22705 rows × 13 columns</p>
</div>




```python
#may need to go in and fill in lag Nan's with ffill
future_with_features = df_and_future.query('is_future').copy()
future_with_features
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
      <th>count</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>lag1</th>
      <th>lag2</th>
      <th>lag3</th>
      <th>is_future</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-05-04 00:00:00</th>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2023</td>
      <td>124</td>
      <td>4</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-05-04 01:00:00</th>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2023</td>
      <td>124</td>
      <td>4</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-05-04 02:00:00</th>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2023</td>
      <td>124</td>
      <td>4</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-05-04 03:00:00</th>
      <td>NaN</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2023</td>
      <td>124</td>
      <td>4</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-05-04 04:00:00</th>
      <td>NaN</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>2023</td>
      <td>124</td>
      <td>4</td>
      <td>18</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2023-11-01 20:00:00</th>
      <td>NaN</td>
      <td>20</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 21:00:00</th>
      <td>NaN</td>
      <td>21</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 22:00:00</th>
      <td>NaN</td>
      <td>22</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-01 23:00:00</th>
      <td>NaN</td>
      <td>23</td>
      <td>2</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>305</td>
      <td>1</td>
      <td>44</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2023-11-02 00:00:00</th>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>11</td>
      <td>2023</td>
      <td>306</td>
      <td>2</td>
      <td>44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>4369 rows × 13 columns</p>
</div>



### Run Predict


```python
FEATURES4 = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2', 'lag3']
TARGET4 = 'count'

#run trained model on features in future df
#save in new column called pred
future_with_features['pred'] = reg.predict(future_with_features[FEATURES4])
```


```python
import matplotlib.pyplot as plt

future_with_features['pred'].plot(figsize = (10,5),
                                  #color= 'Blue',
                                  color= color_pal[0],
                                  ms=1, 
                                  lw=1,
                                  title='XGBOOST Future Predictions')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_139_0.png)
    


### Save & Load XGBOOST model


```python
reg.save_model('311XGB_model.json')
```


```python
reg_new = xgb.XGBRegressor()
reg_new.load_model('311XGB_model.json')
```

### Run saved model on future features


```python
#run saved model under reg_new and plot results
future_with_features['pred'] = reg_new.predict(future_with_features[FEATURES4])

future_with_features['pred'].plot(figsize = (10,5),
                                  #color= 'Blue',
                                  color= color_pal[0],
                                  ms=1, 
                                  lw=1,
                                  title='XGBOOST Future Predictions')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_144_0.png)
    



```python
#zoomed in plot of forecast for upcoming week; first week of may
import datetime
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(future_with_features.index, future_with_features['pred'])
#fig = future_with_features['pred'].plot(future_with_features['pred'], ax= ax)
ax.set_xlim([datetime.date(2023, 7, 1), datetime.date(2023, 7, 7)])
ax.set_ylim(0,80)
ax.set_title('XGBOOST Forecast First Week Of July')
plt.show()

```


    
![png](\img\posts\311-Forecasting-notebook\output_145_0.png)
    



```python
#zoomed in plot of forecast for upcoming week; first week of may
import datetime
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(future_with_features.index, future_with_features['pred'])
#fig = future_with_features['pred'].plot(future_with_features['pred'], ax= ax)
ax.set_xlim([datetime.date(2023, 11, 1), datetime.date(2023, 11, 2)])
ax.set_ylim(0,80)
ax.set_title('XGBOOST Forecast November 1st By Hour')
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_146_0.png)
    



```python
from xgboost import plot_importance
plot_importance(reg_new)
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_147_0.png)
    


### AMIRA
Auto Regression, , Integarted(differencing) , Moving Average

##### can not figure out how to do hourly, switched to daily data


```python
!pip3 install pmdarima
```

    Requirement already satisfied: pmdarima in c:\users\brett\anaconda3\envs\school\lib\site-packages (2.0.3)
    Requirement already satisfied: numpy>=1.21.2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.23.5)
    Requirement already satisfied: setuptools!=50.0.0,>=38.6.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (65.6.3)
    Requirement already satisfied: pandas>=0.19 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.5.3)
    Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (0.29.34)
    Requirement already satisfied: urllib3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.26.14)
    Requirement already satisfied: joblib>=0.11 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.2.0)
    Requirement already satisfied: statsmodels>=0.13.2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (0.13.5)
    Requirement already satisfied: scikit-learn>=0.22 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.2.1)
    Requirement already satisfied: scipy>=1.3.2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pmdarima) (1.10.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=0.19->pmdarima) (2022.7)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=0.19->pmdarima) (2.8.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-learn>=0.22->pmdarima) (3.1.0)
    Requirement already satisfied: packaging>=21.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels>=0.13.2->pmdarima) (22.0)
    Requirement already satisfied: patsy>=0.5.2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels>=0.13.2->pmdarima) (0.5.3)
    Requirement already satisfied: six in c:\users\brett\anaconda3\envs\school\lib\site-packages (from patsy>=0.5.2->statsmodels>=0.13.2->pmdarima) (1.16.0)
    


```python
df_day['count'].plot(figsize=(12,5))
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_151_1.png)
    


### Check For Stationarity


```python
!pip install statsmodels
```

    Requirement already satisfied: statsmodels in c:\users\brett\anaconda3\envs\school\lib\site-packages (0.13.5)
    Requirement already satisfied: numpy>=1.17 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels) (1.23.5)
    Requirement already satisfied: scipy>=1.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels) (1.10.0)
    Requirement already satisfied: pandas>=0.25 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels) (1.5.3)
    Requirement already satisfied: packaging>=21.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels) (22.0)
    Requirement already satisfied: patsy>=0.5.2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels) (0.5.3)
    Requirement already satisfied: pytz>=2020.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=0.25->statsmodels) (2022.7)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=0.25->statsmodels) (2.8.2)
    Requirement already satisfied: six in c:\users\brett\anaconda3\envs\school\lib\site-packages (from patsy>=0.5.2->statsmodels) (1.16.0)
    


```python
#import statsmodels.api as sm
import statsmodels.api
import statsmodels.formula.api as smf
```


```python
from statsmodels.tsa.stattools import adfuller
```


```python
def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print('1. ADF : ', dftest[0])
    print('2. P-Value : ', dftest[1])
    print('3. Num Of Lags : ', dftest[2])
    print('4. Num Of Observations Used For ADF Regression and Critical Values Calculation : ', dftest[3])
    print('5. Critical Values : ')
    for key, val in dftest[4].items():
        print('\t', key, ':',val)
```


```python
# P-Value is probability and should be as low as possible
#the smaller the better P-Value
#small value means teh data set is stationary
ad_test(df_day['count'])
```

    1. ADF :  -4.7528316341012635
    2. P-Value :  6.688647512497074e-05
    3. Num Of Lags :  22
    4. Num Of Observations Used For ADF Regression and Critical Values Calculation :  1189
    5. Critical Values : 
    	 1% : -3.435861752677197
    	 5% : -2.8639738850277796
    	 10% : -2.568065847341873
    


```python
from pmdarima import auto_arima
# Ignore  warnings
import warnings
warnings.filterwarnings('ignore')
```

### make predictions on future values

### Predict the future


```python
df_day.head()
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
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-10</th>
      <td>98</td>
    </tr>
    <tr>
      <th>2020-01-11</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2020-01-12</th>
      <td>13</td>
    </tr>
    <tr>
      <th>2020-01-13</th>
      <td>368</td>
    </tr>
    <tr>
      <th>2020-01-14</th>
      <td>316</td>
    </tr>
  </tbody>
</table>
</div>




```python
split_date = '1-jan-2023'
df_train_day = df_day.loc[df_day.index <= split_date].copy()
df_test_day = df_day.loc[df_day.index > split_date].copy()

fig, ax = plt.subplots(figsize=(15,5))
df_train_day.plot(ax=ax, label='Training Data', title='Train/Test Data Split BY Day')
df_test_day.plot(ax=ax, label='Testing Data')
ax.axvline(split_date, color='black', ls='--')
ax.legend(['Training Data', 'Test Data'])
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_162_0.png)
    



```python
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
```


```python
import statsmodels.api
import statsmodels.formula.api as smf
```


```python
stepwise_fit = auto_arima(df_day['count'],
                          trace=True,
                          supress_warnings=True)

stepwise_fit.summary()  
```

    Performing stepwise search to minimize aic
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=15485.928, Time=1.29 sec
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=16082.799, Time=0.03 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=16076.953, Time=0.05 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=15868.409, Time=0.27 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=16080.802, Time=0.02 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=15577.259, Time=0.55 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=15544.333, Time=0.56 sec
     ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=1.19 sec
     ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=15334.875, Time=1.49 sec
     ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=15546.470, Time=1.10 sec
     ARIMA(3,1,3)(0,0,0)[0] intercept   : AIC=15311.143, Time=1.53 sec
     ARIMA(4,1,3)(0,0,0)[0] intercept   : AIC=15164.604, Time=1.69 sec
     ARIMA(4,1,2)(0,0,0)[0] intercept   : AIC=15301.116, Time=1.93 sec
     ARIMA(5,1,3)(0,0,0)[0] intercept   : AIC=15108.060, Time=1.90 sec
     ARIMA(5,1,2)(0,0,0)[0] intercept   : AIC=15151.929, Time=1.64 sec
     ARIMA(5,1,4)(0,0,0)[0] intercept   : AIC=15056.058, Time=2.09 sec
     ARIMA(4,1,4)(0,0,0)[0] intercept   : AIC=15151.651, Time=1.83 sec
     ARIMA(5,1,5)(0,0,0)[0] intercept   : AIC=14925.176, Time=2.65 sec
     ARIMA(4,1,5)(0,0,0)[0] intercept   : AIC=14951.563, Time=2.32 sec
     ARIMA(5,1,5)(0,0,0)[0]             : AIC=14923.674, Time=1.89 sec
     ARIMA(4,1,5)(0,0,0)[0]             : AIC=14949.732, Time=1.61 sec
     ARIMA(5,1,4)(0,0,0)[0]             : AIC=15054.277, Time=1.25 sec
     ARIMA(4,1,4)(0,0,0)[0]             : AIC=15149.548, Time=0.89 sec
    
    Best model:  ARIMA(5,1,5)(0,0,0)[0]          
    Total fit time: 29.778 seconds
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>   <td>1212</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(5, 1, 5)</td> <th>  Log Likelihood     </th> <td>-7450.837</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 10 May 2023</td> <th>  AIC                </th> <td>14923.674</td>
</tr>
<tr>
  <th>Time:</th>                <td>16:08:20</td>     <th>  BIC                </th> <td>14979.765</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>14944.793</td>
</tr>
<tr>
  <th></th>                      <td> - 1212</td>     <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.9527</td> <td>    0.061</td> <td>   15.629</td> <td> 0.000</td> <td>    0.833</td> <td>    1.072</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -1.5558</td> <td>    0.046</td> <td>  -33.884</td> <td> 0.000</td> <td>   -1.646</td> <td>   -1.466</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>    1.0378</td> <td>    0.081</td> <td>   12.746</td> <td> 0.000</td> <td>    0.878</td> <td>    1.197</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>   -1.1024</td> <td>    0.049</td> <td>  -22.362</td> <td> 0.000</td> <td>   -1.199</td> <td>   -1.006</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>    0.1677</td> <td>    0.053</td> <td>    3.145</td> <td> 0.002</td> <td>    0.063</td> <td>    0.272</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -1.5405</td> <td>    0.053</td> <td>  -28.943</td> <td> 0.000</td> <td>   -1.645</td> <td>   -1.436</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    1.8972</td> <td>    0.057</td> <td>   33.325</td> <td> 0.000</td> <td>    1.786</td> <td>    2.009</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>   -1.8915</td> <td>    0.057</td> <td>  -33.031</td> <td> 0.000</td> <td>   -2.004</td> <td>   -1.779</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>    1.4121</td> <td>    0.062</td> <td>   22.934</td> <td> 0.000</td> <td>    1.291</td> <td>    1.533</td>
</tr>
<tr>
  <th>ma.L5</th>  <td>   -0.6891</td> <td>    0.036</td> <td>  -19.358</td> <td> 0.000</td> <td>   -0.759</td> <td>   -0.619</td>
</tr>
<tr>
  <th>sigma2</th> <td> 1.568e+04</td> <td>  237.142</td> <td>   66.111</td> <td> 0.000</td> <td> 1.52e+04</td> <td> 1.61e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.01</td> <th>  Jarque-Bera (JB):  </th> <td>210261.66</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.91</td> <th>  Prob(JB):          </th>   <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>1.25</td> <th>  Skew:              </th>   <td>4.43</td>   
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.03</td> <th>  Kurtosis:          </th>   <td>66.94</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
#use order from best order in the step above

model_day=sm.tsa.ARIMA(df_train_day['count'], order=(5,1,5))
model_day=model_day.fit()
model_day.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>count</td>      <th>  No. Observations:  </th>   <td>1084</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(5, 1, 5)</td>  <th>  Log Likelihood     </th> <td>-6684.911</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 10 May 2023</td> <th>  AIC                </th> <td>13391.822</td>
</tr>
<tr>
  <th>Time:</th>                <td>16:08:47</td>     <th>  BIC                </th> <td>13446.684</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>13412.592</td>
</tr>
<tr>
  <th></th>                      <td> - 1084</td>     <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.9872</td> <td>    0.059</td> <td>   16.838</td> <td> 0.000</td> <td>    0.872</td> <td>    1.102</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -1.5865</td> <td>    0.045</td> <td>  -35.588</td> <td> 0.000</td> <td>   -1.674</td> <td>   -1.499</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>    1.0828</td> <td>    0.080</td> <td>   13.602</td> <td> 0.000</td> <td>    0.927</td> <td>    1.239</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>   -1.1353</td> <td>    0.047</td> <td>  -23.966</td> <td> 0.000</td> <td>   -1.228</td> <td>   -1.042</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>    0.1974</td> <td>    0.053</td> <td>    3.742</td> <td> 0.000</td> <td>    0.094</td> <td>    0.301</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -1.5795</td> <td>    0.052</td> <td>  -30.481</td> <td> 0.000</td> <td>   -1.681</td> <td>   -1.478</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    1.9451</td> <td>    0.057</td> <td>   33.879</td> <td> 0.000</td> <td>    1.833</td> <td>    2.058</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>   -1.9274</td> <td>    0.059</td> <td>  -32.557</td> <td> 0.000</td> <td>   -2.043</td> <td>   -1.811</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>    1.4612</td> <td>    0.060</td> <td>   24.198</td> <td> 0.000</td> <td>    1.343</td> <td>    1.580</td>
</tr>
<tr>
  <th>ma.L5</th>  <td>   -0.7163</td> <td>    0.038</td> <td>  -18.685</td> <td> 0.000</td> <td>   -0.791</td> <td>   -0.641</td>
</tr>
<tr>
  <th>sigma2</th> <td> 1.612e+04</td> <td>  250.719</td> <td>   64.298</td> <td> 0.000</td> <td> 1.56e+04</td> <td> 1.66e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.00</td> <th>  Jarque-Bera (JB):  </th> <td>195507.26</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.96</td> <th>  Prob(JB):          </th>   <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>2.45</td> <th>  Skew:              </th>   <td>4.61</td>   
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>   <td>68.17</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



Make Predictions On Test Set


```python
start=len(df_train_day)
end=len(df_train_day)+len(df_test_day)-1
pred_day=model_day.predict(start=start, end=end, typ='levels')
#print(pred) # dates did not appear do use next line of code
pred_day.index=df_day.index[start:end+1]
print(pred_day)
```

    time
    2023-01-02    479.428567
    2023-01-03    674.749061
    2023-01-04    604.407490
    2023-01-05    557.103754
    2023-01-06    398.781992
                     ...    
    2023-05-05    469.493291
    2023-05-06    312.357834
    2023-05-07    315.081241
    2023-05-08    506.594428
    2023-05-09    507.856525
    Name: predicted_mean, Length: 128, dtype: float64
    


```python
pred_day.plot(legend=True)
df_test_day['count'].plot(legend=True)
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_169_1.png)
    



```python
df_day['count'].mean()
```




    229.16006600660066




```python
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_day=sqrt(mean_squared_error(pred_day, df_test_day['count']))
rmse_day
```




    224.44111330608987



Make predictions


```python
#Fit model on all df_day
model_day = sm.tsa.ARIMA(df_day['count'], order=(5,0,5))
model_day=model_day.fit()
df_day.tail() # predict into the future after training
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
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-05-05</th>
      <td>286</td>
    </tr>
    <tr>
      <th>2023-05-06</th>
      <td>51</td>
    </tr>
    <tr>
      <th>2023-05-07</th>
      <td>51</td>
    </tr>
    <tr>
      <th>2023-05-08</th>
      <td>482</td>
    </tr>
    <tr>
      <th>2023-05-09</th>
      <td>387</td>
    </tr>
  </tbody>
</table>
</div>




```python
index_future_dates=pd.date_range(start='2023-05-09', end='2023-12-31', freq='D')
pred_day=model_day.predict(start=len(df_day), end=len(df_day)+(236), type='levels').rename('ARIMA Predictions BY Day')
pred_day.index=index_future_dates
print(pred_day)
```

    2023-05-09    297.317413
    2023-05-10    336.257542
    2023-05-11    253.195036
    2023-05-12     48.160946
    2023-05-13    112.191123
                     ...    
    2023-12-27    262.186749
    2023-12-28    227.061904
    2023-12-29    157.228323
    2023-12-30    188.266458
    2023-12-31    265.360301
    Freq: D, Name: ARIMA Predictions BY Day, Length: 237, dtype: float64
    


```python
pred_day.info()
```

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 237 entries, 2023-05-09 to 2023-12-31
    Freq: D
    Series name: ARIMA Predictions BY Day
    Non-Null Count  Dtype  
    --------------  -----  
    237 non-null    float64
    dtypes: float64(1)
    memory usage: 3.7 KB
    


```python
pred_day.plot(figsize=(12,5),legend=True)
```




    <AxesSubplot: >




    
![png](\img\posts\311-Forecasting-notebook\output_176_1.png)
    


# Running AMIRA on hourly data then projecting to daily
# Test data looks better than training on daily


```python
df5 = df2.copy()
#df5 = df5.drop(['is_future', 'lag1', 'lag2', 'lag3'], axis=1)
df5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 18336 entries, 2020-01-10 12:00:00 to 2023-05-09 23:00:00
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   count   18336 non-null  int64
    dtypes: int64(1)
    memory usage: 286.5 KB
    


```python
df5['count'].plot(figsize=(12,5))
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_179_1.png)
    



```python
# P-Value is probability and should be as low as possible
#the smaller the better P-Value
#small value means teh data set is stationary
ad_test(df5['count'])
```

    1. ADF :  -22.210656144053754
    2. P-Value :  0.0
    3. Num Of Lags :  31
    4. Num Of Observations Used For ADF Regression and Critical Values Calculation :  18304
    5. Critical Values : 
    	 1% : -3.430707310823011
    	 5% : -2.8616979180198356
    	 10% : -2.5668540555869606
    


```python
stepwise_fit5 = auto_arima(df5['count'],
                          trace=True,
                          supress_warnings=True)

stepwise_fit5.summary()    
```

    Performing stepwise search to minimize aic
     ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=139245.469, Time=12.43 sec
     ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=160319.042, Time=0.23 sec
     ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=140155.291, Time=0.29 sec
     ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=148101.958, Time=2.04 sec
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=169221.769, Time=0.13 sec
     ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=139761.479, Time=5.38 sec
     ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=139280.154, Time=8.03 sec
     ARIMA(3,0,2)(0,0,0)[0] intercept   : AIC=139282.702, Time=11.24 sec
     ARIMA(2,0,3)(0,0,0)[0] intercept   : AIC=138849.798, Time=19.91 sec
     ARIMA(1,0,3)(0,0,0)[0] intercept   : AIC=139740.324, Time=6.93 sec
     ARIMA(3,0,3)(0,0,0)[0] intercept   : AIC=139249.468, Time=16.14 sec
     ARIMA(2,0,4)(0,0,0)[0] intercept   : AIC=138392.690, Time=22.20 sec
     ARIMA(1,0,4)(0,0,0)[0] intercept   : AIC=139725.762, Time=9.09 sec
     ARIMA(3,0,4)(0,0,0)[0] intercept   : AIC=138687.604, Time=24.18 sec
     ARIMA(2,0,5)(0,0,0)[0] intercept   : AIC=138328.398, Time=14.43 sec
     ARIMA(1,0,5)(0,0,0)[0] intercept   : AIC=139462.275, Time=10.38 sec
     ARIMA(3,0,5)(0,0,0)[0] intercept   : AIC=138326.439, Time=19.29 sec
     ARIMA(4,0,5)(0,0,0)[0] intercept   : AIC=137966.103, Time=26.64 sec
     ARIMA(4,0,4)(0,0,0)[0] intercept   : AIC=138459.123, Time=27.90 sec
     ARIMA(5,0,5)(0,0,0)[0] intercept   : AIC=137865.368, Time=29.40 sec
     ARIMA(5,0,4)(0,0,0)[0] intercept   : AIC=139131.798, Time=27.98 sec
     ARIMA(5,0,5)(0,0,0)[0]             : AIC=140472.475, Time=14.99 sec
    
    Best model:  ARIMA(5,0,5)(0,0,0)[0] intercept
    Total fit time: 309.258 seconds
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>18336</td>  
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(5, 0, 5)</td> <th>  Log Likelihood     </th> <td>-68920.684</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 10 May 2023</td> <th>  AIC                </th> <td>137865.368</td>
</tr>
<tr>
  <th>Time:</th>                <td>16:34:35</td>     <th>  BIC                </th> <td>137959.167</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>137896.186</td>
</tr>
<tr>
  <th></th>                     <td> - 18336</td>     <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>    0.6856</td> <td>    0.063</td> <td>   10.810</td> <td> 0.000</td> <td>    0.561</td> <td>    0.810</td>
</tr>
<tr>
  <th>ar.L1</th>     <td>    3.3027</td> <td>    0.031</td> <td>  108.077</td> <td> 0.000</td> <td>    3.243</td> <td>    3.363</td>
</tr>
<tr>
  <th>ar.L2</th>     <td>   -5.0795</td> <td>    0.077</td> <td>  -65.888</td> <td> 0.000</td> <td>   -5.231</td> <td>   -4.928</td>
</tr>
<tr>
  <th>ar.L3</th>     <td>    4.5577</td> <td>    0.093</td> <td>   49.226</td> <td> 0.000</td> <td>    4.376</td> <td>    4.739</td>
</tr>
<tr>
  <th>ar.L4</th>     <td>   -2.3567</td> <td>    0.063</td> <td>  -37.415</td> <td> 0.000</td> <td>   -2.480</td> <td>   -2.233</td>
</tr>
<tr>
  <th>ar.L5</th>     <td>    0.5307</td> <td>    0.021</td> <td>   25.760</td> <td> 0.000</td> <td>    0.490</td> <td>    0.571</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>   -2.4724</td> <td>    0.030</td> <td>  -81.359</td> <td> 0.000</td> <td>   -2.532</td> <td>   -2.413</td>
</tr>
<tr>
  <th>ma.L2</th>     <td>    2.9964</td> <td>    0.053</td> <td>   56.326</td> <td> 0.000</td> <td>    2.892</td> <td>    3.101</td>
</tr>
<tr>
  <th>ma.L3</th>     <td>   -2.0045</td> <td>    0.054</td> <td>  -37.183</td> <td> 0.000</td> <td>   -2.110</td> <td>   -1.899</td>
</tr>
<tr>
  <th>ma.L4</th>     <td>    0.6717</td> <td>    0.031</td> <td>   21.894</td> <td> 0.000</td> <td>    0.612</td> <td>    0.732</td>
</tr>
<tr>
  <th>ma.L5</th>     <td>   -0.0364</td> <td>    0.011</td> <td>   -3.297</td> <td> 0.001</td> <td>   -0.058</td> <td>   -0.015</td>
</tr>
<tr>
  <th>sigma2</th>    <td>  106.6402</td> <td>    0.304</td> <td>  350.752</td> <td> 0.000</td> <td>  106.044</td> <td>  107.236</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.13</td> <th>  Jarque-Bera (JB):  </th> <td>1126273.03</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.72</td> <th>  Prob(JB):          </th>    <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>1.19</td> <th>  Skew:              </th>    <td>1.96</td>   
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>    <td>41.19</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
split_date = '1-jan-2023'
df_train_day = df_day.loc[df_day.index <= split_date].copy()
df_test_day = df_day.loc[df_day.index > split_date].copy()

fig, ax = plt.subplots(figsize=(15,5))
df_train_day.plot(ax=ax, label='Training Data', title='Train/Test Data Split BY Day')
df_test_day.plot(ax=ax, label='Testing Data')
ax.axvline(split_date, color='black', ls='--')
ax.legend(['Training Data', 'Test Data'])
plt.show()
```


    
![png](\img\posts\311-Forecasting-notebook\output_182_0.png)
    



```python
#from statsmodels.tsa.arima_model import ARIMA
model_day=sm.tsa.ARIMA(df_train_day['count'], order=(5,0,5))
model_day=model_day.fit()
model_day.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>count</td>      <th>  No. Observations:  </th>   <td>1084</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(5, 0, 5)</td>  <th>  Log Likelihood     </th> <td>-6698.985</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 10 May 2023</td> <th>  AIC                </th> <td>13421.970</td>
</tr>
<tr>
  <th>Time:</th>                <td>16:44:37</td>     <th>  BIC                </th> <td>13481.831</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>13444.632</td>
</tr>
<tr>
  <th></th>                      <td> - 1084</td>     <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>  <td>  230.3111</td> <td>   39.164</td> <td>    5.881</td> <td> 0.000</td> <td>  153.551</td> <td>  307.072</td>
</tr>
<tr>
  <th>ar.L1</th>  <td>    1.7576</td> <td>    0.022</td> <td>   81.197</td> <td> 0.000</td> <td>    1.715</td> <td>    1.800</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -2.1975</td> <td>    0.024</td> <td>  -92.527</td> <td> 0.000</td> <td>   -2.244</td> <td>   -2.151</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>    2.1699</td> <td>    0.034</td> <td>   64.358</td> <td> 0.000</td> <td>    2.104</td> <td>    2.236</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>   -1.7470</td> <td>    0.028</td> <td>  -62.794</td> <td> 0.000</td> <td>   -1.802</td> <td>   -1.693</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>    0.9487</td> <td>    0.023</td> <td>   41.549</td> <td> 0.000</td> <td>    0.904</td> <td>    0.993</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -1.4176</td> <td>    0.028</td> <td>  -49.868</td> <td> 0.000</td> <td>   -1.473</td> <td>   -1.362</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    1.7757</td> <td>    0.042</td> <td>   42.510</td> <td> 0.000</td> <td>    1.694</td> <td>    1.858</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>   -1.6703</td> <td>    0.059</td> <td>  -28.253</td> <td> 0.000</td> <td>   -1.786</td> <td>   -1.554</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>    1.2575</td> <td>    0.047</td> <td>   26.648</td> <td> 0.000</td> <td>    1.165</td> <td>    1.350</td>
</tr>
<tr>
  <th>ma.L5</th>  <td>   -0.5664</td> <td>    0.038</td> <td>  -15.068</td> <td> 0.000</td> <td>   -0.640</td> <td>   -0.493</td>
</tr>
<tr>
  <th>sigma2</th> <td> 1.678e+04</td> <td>  251.650</td> <td>   66.696</td> <td> 0.000</td> <td> 1.63e+04</td> <td> 1.73e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>4.49</td> <th>  Jarque-Bera (JB):  </th> <td>193826.67</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.03</td> <th>  Prob(JB):          </th>   <td>0.00</td>   
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>2.63</td> <th>  Skew:              </th>   <td>4.64</td>   
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>   <td>67.85</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



### Make Predictions On Daily Test Set


```python
start=len(df_train_day)
end=len(df_train_day)+len(df_test_day)-1
pred_day=model_day.predict(start=start, end=end, typ='levels')
#print(pred) # dates did not appear do use next line of code
pred_day.index=df_day.index[start:end+1]
print(pred_day)
```

    time
    2023-01-02    526.434333
    2023-01-03    772.138844
    2023-01-04    714.534101
    2023-01-05    611.437116
    2023-01-06    378.830014
                     ...    
    2023-05-05    190.894294
    2023-05-06    -15.404300
    2023-05-07     26.984342
    2023-05-08    295.538955
    2023-05-09    421.600123
    Name: predicted_mean, Length: 128, dtype: float64
    


```python
pred_day.plot(legend=True)
df_test_day['count'].plot(legend=True)
```




    <AxesSubplot: xlabel='time'>




    
![png](\img\posts\311-Forecasting-notebook\output_186_1.png)
    



```python
df_day['count'].mean()
```




    229.16006600660066




```python
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_day=sqrt(mean_squared_error(pred_day, df_test_day['count']))
rmse_day
```




    157.9142119549536




```python
#Fit model on all df5
model_day = sm.tsa.ARIMA(df_day['count'], order=(5,0,5))
model_day=model_day.fit()
df_day.tail() # predict into the future after training
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
      <th>count</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-05-05</th>
      <td>286</td>
    </tr>
    <tr>
      <th>2023-05-06</th>
      <td>51</td>
    </tr>
    <tr>
      <th>2023-05-07</th>
      <td>51</td>
    </tr>
    <tr>
      <th>2023-05-08</th>
      <td>482</td>
    </tr>
    <tr>
      <th>2023-05-09</th>
      <td>387</td>
    </tr>
  </tbody>
</table>
</div>




```python
index_future_dates=pd.date_range(start='2023-05-09', end='2023-12-31', freq='D')
pred_day=model_day.predict(start=len(df_day), end=len(df_day)+(236), type='levels').rename('ARIMA Predictions BY Day')
pred_day.index=index_future_dates
print(pred_day)
```

    2023-05-09    297.317413
    2023-05-10    336.257542
    2023-05-11    253.195036
    2023-05-12     48.160946
    2023-05-13    112.191123
                     ...    
    2023-12-27    262.186749
    2023-12-28    227.061904
    2023-12-29    157.228323
    2023-12-30    188.266458
    2023-12-31    265.360301
    Freq: D, Name: ARIMA Predictions BY Day, Length: 237, dtype: float64
    


```python
pred_day.info()
```

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 237 entries, 2023-05-09 to 2023-12-31
    Freq: D
    Series name: ARIMA Predictions BY Day
    Non-Null Count  Dtype  
    --------------  -----  
    237 non-null    float64
    dtypes: float64(1)
    memory usage: 3.7 KB
    


```python
pred_day.plot(figsize=(12,5),legend=True)
```




    <AxesSubplot: >




    
![png](\img\posts\311-Forecasting-notebook\output_192_1.png)
    


### the zigzag pattern on the above projection is caused by low call volume on weekends


```python

```
