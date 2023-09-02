---
layout: post
title: "Hotel Cancellation EDA & Predictions"
subtitle: "Predicting Hotel Cancellations SK-Learn, Neural Networks & XGBOOST"
date: 2023-07-02
background: '/img/posts/Hotel-Cancellations/Cancelled.png'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('TrainData.csv')
#booking status is y
```


```python
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
      <th>no_of_adults</th>
      <th>no_of_children</th>
      <th>no_of_weekend_nights</th>
      <th>no_of_week_nights</th>
      <th>type_of_meal_plan</th>
      <th>required_car_parking_space</th>
      <th>room_type_reserved</th>
      <th>lead_time</th>
      <th>arrival_year</th>
      <th>arrival_month</th>
      <th>arrival_date</th>
      <th>market_segment_type</th>
      <th>repeated_guest</th>
      <th>no_of_previous_cancellations</th>
      <th>no_of_previous_bookings_not_canceled</th>
      <th>avg_price_per_room</th>
      <th>no_of_special_requests</th>
      <th>booking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>2017</td>
      <td>12</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>110.80</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>2018</td>
      <td>4</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>145.00</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>349</td>
      <td>2018</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96.67</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>2018</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120.00</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2018</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69.50</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
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
      <th>no_of_adults</th>
      <th>no_of_children</th>
      <th>no_of_weekend_nights</th>
      <th>no_of_week_nights</th>
      <th>type_of_meal_plan</th>
      <th>required_car_parking_space</th>
      <th>room_type_reserved</th>
      <th>lead_time</th>
      <th>arrival_year</th>
      <th>arrival_month</th>
      <th>arrival_date</th>
      <th>market_segment_type</th>
      <th>repeated_guest</th>
      <th>no_of_previous_cancellations</th>
      <th>no_of_previous_bookings_not_canceled</th>
      <th>avg_price_per_room</th>
      <th>no_of_special_requests</th>
      <th>booking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
      <td>18137.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.846777</td>
      <td>0.107515</td>
      <td>0.811104</td>
      <td>2.208965</td>
      <td>0.318465</td>
      <td>0.031648</td>
      <td>0.336770</td>
      <td>85.377405</td>
      <td>2017.820698</td>
      <td>7.432762</td>
      <td>15.660804</td>
      <td>0.806197</td>
      <td>0.025087</td>
      <td>0.022440</td>
      <td>0.151403</td>
      <td>103.478868</td>
      <td>0.617522</td>
      <td>0.327618</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.516020</td>
      <td>0.408901</td>
      <td>0.873470</td>
      <td>1.426365</td>
      <td>0.629140</td>
      <td>0.175066</td>
      <td>0.772865</td>
      <td>86.611736</td>
      <td>0.383616</td>
      <td>3.076999</td>
      <td>8.772788</td>
      <td>0.645972</td>
      <td>0.156393</td>
      <td>0.370078</td>
      <td>1.714135</td>
      <td>35.474103</td>
      <td>0.787941</td>
      <td>0.469357</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2017.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>2018.000000</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>80.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>57.000000</td>
      <td>2018.000000</td>
      <td>8.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.450000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>127.000000</td>
      <td>2018.000000</td>
      <td>10.000000</td>
      <td>23.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.270000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>443.000000</td>
      <td>2018.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>58.000000</td>
      <td>540.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
plt.figure(figsize = (14,7))
sns.heatmap(data.corr().round(2), annot =True, cmap='YlGnBu')

#Negative corr (arrival year & month) market segment and lead time
#corr( repeat guest & no previous not canceled / Market segment & repeat guest) num of children / room type
```




    <AxesSubplot: >




    
![png](\img\posts\Hotel-Cancellations\output_4_1.png)
    



```python
#check for null data & check data types
#change dates to date format
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18137 entries, 0 to 18136
    Data columns (total 18 columns):
     #   Column                                Non-Null Count  Dtype  
    ---  ------                                --------------  -----  
     0   no_of_adults                          18137 non-null  int64  
     1   no_of_children                        18137 non-null  int64  
     2   no_of_weekend_nights                  18137 non-null  int64  
     3   no_of_week_nights                     18137 non-null  int64  
     4   type_of_meal_plan                     18137 non-null  int64  
     5   required_car_parking_space            18137 non-null  int64  
     6   room_type_reserved                    18137 non-null  int64  
     7   lead_time                             18137 non-null  int64  
     8   arrival_year                          18137 non-null  int64  
     9   arrival_month                         18137 non-null  int64  
     10  arrival_date                          18137 non-null  int64  
     11  market_segment_type                   18137 non-null  int64  
     12  repeated_guest                        18137 non-null  int64  
     13  no_of_previous_cancellations          18137 non-null  int64  
     14  no_of_previous_bookings_not_canceled  18137 non-null  int64  
     15  avg_price_per_room                    18137 non-null  float64
     16  no_of_special_requests                18137 non-null  int64  
     17  booking_status                        18137 non-null  int64  
    dtypes: float64(1), int64(17)
    memory usage: 2.5 MB
    


```python
#check for nulls
data.isnull().sum()/len(data)
```




    no_of_adults                            0.0
    no_of_children                          0.0
    no_of_weekend_nights                    0.0
    no_of_week_nights                       0.0
    type_of_meal_plan                       0.0
    required_car_parking_space              0.0
    room_type_reserved                      0.0
    lead_time                               0.0
    arrival_year                            0.0
    arrival_month                           0.0
    arrival_date                            0.0
    market_segment_type                     0.0
    repeated_guest                          0.0
    no_of_previous_cancellations            0.0
    no_of_previous_bookings_not_canceled    0.0
    avg_price_per_room                      0.0
    no_of_special_requests                  0.0
    booking_status                          0.0
    dtype: float64




```python
#check for outliers but only on contiuous numbers
plt.title("Lead_Time", fontdict = {'fontsize': 20})
sns.boxplot(x=data["lead_time"], palette = 'YlGnBu')
```




    <AxesSubplot: title={'center': 'Lead_Time'}, xlabel='lead_time'>




    
![png](\img\posts\Hotel-Cancellations\output_7_1.png)
    



```python
#check for outliers but only on contiuous numbers
plt.title("Avg Price Per Room", fontdict = {'fontsize': 20})
sns.boxplot(x=data["avg_price_per_room"],palette = 'YlGnBu')
```




    <AxesSubplot: title={'center': 'Avg Price Per Room'}, xlabel='avg_price_per_room'>




    
![png](\img\posts\Hotel-Cancellations\output_8_1.png)
    



```python
sns.jointplot(x='lead_time', y='avg_price_per_room', data = data,palette = 'YlGnBu', kind='kde', fill=True)
```




    <seaborn.axisgrid.JointGrid at 0x24a23e46df0>




    
![png](\img\posts\Hotel-Cancellations\output_9_1.png)
    



```python
sns.jointplot(x='lead_time', y='avg_price_per_room', data = data, cmap = 'YlGnBu', kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x24a23d0ffa0>




    
![png](\img\posts\Hotel-Cancellations\output_10_1.png)
    



```python
# sns.pairplot(data)
```


```python
# penguins = sns.load_dataset("penguins")
# #sns.pairplot(penguins)
# penguins.info()
```


```python
plt.figure(figsize = (20, 25))
plt.subplot(5,2,3)
sns.kdeplot(x='lead_time', hue='booking_status', palette = 'Set2', fill=True, data=data)

plt.subplot(5,2,4)
sns.kdeplot(x='arrival_month', hue='booking_status', palette = 'Set2', fill=True, data=data)

plt.subplot(5,2,1)
sns.kdeplot(x='arrival_date', hue='booking_status', palette = 'Set2', fill=True, data=data)

plt.subplot(5,2,2)
sns.kdeplot(x = 'booking_status', hue = 'repeated_guest', palette = 'Set2', fill=True, data = data)
```




    <AxesSubplot: xlabel='booking_status', ylabel='Density'>




    
![png](\img\posts\Hotel-Cancellations\output_13_1.png)
    



```python
data = data.rename(columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_date': 'day'})
```


```python
data['date'] = pd.to_datetime(data[['year', 'month', 'day']],format='%Y-%m-%d', errors='coerce')
```


```python
data
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
      <th>no_of_adults</th>
      <th>no_of_children</th>
      <th>no_of_weekend_nights</th>
      <th>no_of_week_nights</th>
      <th>type_of_meal_plan</th>
      <th>required_car_parking_space</th>
      <th>room_type_reserved</th>
      <th>lead_time</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>market_segment_type</th>
      <th>repeated_guest</th>
      <th>no_of_previous_cancellations</th>
      <th>no_of_previous_bookings_not_canceled</th>
      <th>avg_price_per_room</th>
      <th>no_of_special_requests</th>
      <th>booking_status</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>2017</td>
      <td>12</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>110.80</td>
      <td>2</td>
      <td>0</td>
      <td>2017-12-28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>2018</td>
      <td>4</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>145.00</td>
      <td>0</td>
      <td>1</td>
      <td>2018-04-14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>349</td>
      <td>2018</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>96.67</td>
      <td>0</td>
      <td>1</td>
      <td>2018-10-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>2018</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120.00</td>
      <td>0</td>
      <td>1</td>
      <td>2018-06-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>2018</td>
      <td>1</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69.50</td>
      <td>1</td>
      <td>0</td>
      <td>2018-01-20</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18132</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>103</td>
      <td>2018</td>
      <td>4</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>115.00</td>
      <td>0</td>
      <td>1</td>
      <td>2018-04-19</td>
    </tr>
    <tr>
      <th>18133</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>129</td>
      <td>2018</td>
      <td>8</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88.01</td>
      <td>1</td>
      <td>0</td>
      <td>2018-08-10</td>
    </tr>
    <tr>
      <th>18134</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>90</td>
      <td>2018</td>
      <td>7</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>105.30</td>
      <td>0</td>
      <td>1</td>
      <td>2018-07-13</td>
    </tr>
    <tr>
      <th>18135</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>2018</td>
      <td>11</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>123.33</td>
      <td>1</td>
      <td>0</td>
      <td>2018-11-10</td>
    </tr>
    <tr>
      <th>18136</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>2018</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>65.00</td>
      <td>0</td>
      <td>0</td>
      <td>2018-04-09</td>
    </tr>
  </tbody>
</table>
<p>18137 rows × 19 columns</p>
</div>




```python
#make each reservation a 1 to count
data['reservations'] = 1
# # Use GroupBy() to compute the sum
data1 = data.groupby('date').sum()
data1
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
      <th>no_of_adults</th>
      <th>no_of_children</th>
      <th>no_of_weekend_nights</th>
      <th>no_of_week_nights</th>
      <th>type_of_meal_plan</th>
      <th>required_car_parking_space</th>
      <th>room_type_reserved</th>
      <th>lead_time</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>market_segment_type</th>
      <th>repeated_guest</th>
      <th>no_of_previous_cancellations</th>
      <th>no_of_previous_bookings_not_canceled</th>
      <th>avg_price_per_room</th>
      <th>no_of_special_requests</th>
      <th>booking_status</th>
      <th>reservations</th>
    </tr>
    <tr>
      <th>date</th>
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
      <th>2017-07-01</th>
      <td>61</td>
      <td>0</td>
      <td>4</td>
      <td>76</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>8001</td>
      <td>68578</td>
      <td>238</td>
      <td>34</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3134.00</td>
      <td>3</td>
      <td>25</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2017-07-02</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79</td>
      <td>2017</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>76.50</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-03</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>80</td>
      <td>2017</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>76.50</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-04</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>106</td>
      <td>2017</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>68.00</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-05</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>71</td>
      <td>2017</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>55.80</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>164</td>
      <td>8</td>
      <td>38</td>
      <td>268</td>
      <td>9</td>
      <td>0</td>
      <td>31</td>
      <td>7408</td>
      <td>157404</td>
      <td>936</td>
      <td>2106</td>
      <td>33</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>6705.57</td>
      <td>52</td>
      <td>10</td>
      <td>78</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>73</td>
      <td>8</td>
      <td>19</td>
      <td>140</td>
      <td>11</td>
      <td>1</td>
      <td>22</td>
      <td>4043</td>
      <td>74666</td>
      <td>444</td>
      <td>1036</td>
      <td>39</td>
      <td>1</td>
      <td>6</td>
      <td>58</td>
      <td>4205.69</td>
      <td>35</td>
      <td>10</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2018-12-29</th>
      <td>108</td>
      <td>13</td>
      <td>66</td>
      <td>171</td>
      <td>13</td>
      <td>0</td>
      <td>29</td>
      <td>9223</td>
      <td>119062</td>
      <td>708</td>
      <td>1711</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6100.70</td>
      <td>48</td>
      <td>14</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2018-12-30</th>
      <td>98</td>
      <td>10</td>
      <td>44</td>
      <td>100</td>
      <td>6</td>
      <td>1</td>
      <td>23</td>
      <td>5506</td>
      <td>94846</td>
      <td>564</td>
      <td>1410</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6257.65</td>
      <td>39</td>
      <td>10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2018-12-31</th>
      <td>51</td>
      <td>4</td>
      <td>39</td>
      <td>42</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>3194</td>
      <td>48432</td>
      <td>288</td>
      <td>744</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2947.58</td>
      <td>10</td>
      <td>7</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
<p>548 rows × 19 columns</p>
</div>




```python
from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                       'Wednesday',
                                       'Thursday', 'Friday',
                                       'Saturday', 'Sunday'],
                           ordered=True)
```


```python
data.index
```




    RangeIndex(start=0, stop=18137, step=1)




```python

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = data1.copy()
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

X, y = create_features(data1, label='reservations')

features_and_target = pd.concat([X, y], axis=1)
```


```python
features_and_target.head(20)
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
      <th>reservations</th>
    </tr>
    <tr>
      <th>date</th>
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
      <th>2017-07-01</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>182</td>
      <td>1</td>
      <td>26</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2017-07-02</th>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>183</td>
      <td>2</td>
      <td>26</td>
      <td>Sunday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-03</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>184</td>
      <td>3</td>
      <td>27</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-04</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>185</td>
      <td>4</td>
      <td>27</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-05</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>186</td>
      <td>5</td>
      <td>27</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-06</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>187</td>
      <td>6</td>
      <td>27</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2017-07-07</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>188</td>
      <td>7</td>
      <td>27</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2017-07-08</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>189</td>
      <td>8</td>
      <td>27</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2017-07-10</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>191</td>
      <td>10</td>
      <td>28</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2017-07-11</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>192</td>
      <td>11</td>
      <td>28</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2017-07-12</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>193</td>
      <td>12</td>
      <td>28</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-13</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>194</td>
      <td>13</td>
      <td>28</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2017-07-14</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>195</td>
      <td>14</td>
      <td>28</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-15</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>196</td>
      <td>15</td>
      <td>28</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2017-07-16</th>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>197</td>
      <td>16</td>
      <td>28</td>
      <td>Sunday</td>
      <td>Summer</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2017-07-17</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>198</td>
      <td>17</td>
      <td>29</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2017-07-18</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>199</td>
      <td>18</td>
      <td>29</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2017-07-19</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>200</td>
      <td>19</td>
      <td>29</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-20</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>201</td>
      <td>20</td>
      <td>29</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-21</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>202</td>
      <td>21</td>
      <td>29</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (20,25))


plt.subplot(3,2,1)
plt.gca().set_title('Bookings By Month')
sns.countplot(x = 'month', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])


plt.subplot(3,2,2)
plt.gca().set_title('Variable market_segment_type')
sns.countplot(x = 'market_segment_type', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])
#Need to label market segments(Offline, Online, corporate, aviation, complementary)


plt.subplot(3,2,3)
plt.gca().set_title('Variable booking_status')
sns.countplot(x = 'booking_status', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(3,2,4)
plt.gca().set_title('Variable no_of_children')
sns.countplot(x = 'no_of_children', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])
```




    <AxesSubplot: title={'center': 'Variable no_of_children'}, xlabel='no_of_children', ylabel='count'>




    
![png](\img\posts\Hotel-Cancellations\output_22_1.png)
    



```python
plt.figure(figsize = (20, 25))
plt.suptitle("Booking Status By Month",fontweight="bold", fontsize=20)

plt.subplot(5,2,1)
sns.countplot(x = 'booking_status', hue = 'month', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])
```




    <AxesSubplot: xlabel='booking_status', ylabel='count'>




    
![png](\img\posts\Hotel-Cancellations\output_23_1.png)
    



```python
#change calceled from 0 and 1

plt.figure(figsize = (20, 25))
plt.suptitle("Analysis Of Variable booking_status",fontweight="bold", fontsize=20)

plt.subplot(5,2,1)
sns.countplot(x = 'booking_status', hue = 'no_of_adults', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,2)
sns.countplot(x = 'booking_status', hue = 'no_of_children', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,3)
sns.countplot(x = 'booking_status', hue = 'no_of_weekend_nights',palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,4)
sns.countplot(x = 'booking_status', hue = 'market_segment_type', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,5)
sns.countplot(x = 'booking_status', hue = 'type_of_meal_plan', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,6)
sns.countplot(x = 'booking_status', hue = 'required_car_parking_space', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,7)
sns.countplot(x = 'booking_status', hue = 'room_type_reserved', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])

plt.subplot(5,2,8)
sns.countplot(x = 'booking_status', hue = 'year', palette = 'YlGnBu', data = data.loc[data['year'] == 2018])
```




    <AxesSubplot: xlabel='booking_status', ylabel='count'>




    
![png](\img\posts\Hotel-Cancellations\output_24_1.png)
    



```python
sns.scatterplot(data=data, x="lead_time", y="avg_price_per_room", palette = 'YlGnBu', hue = 'booking_status')
```




    <AxesSubplot: xlabel='lead_time', ylabel='avg_price_per_room'>




    
![png](\img\posts\Hotel-Cancellations\output_25_1.png)
    



```python
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data = features_and_target.dropna(),
            x='weekday',
            y='reservations',
            hue='season',
            ax=ax,
            linewidth=1,
            palette='YlGnBu')

ax.set_title('Number of Reservations by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('reservations')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\Hotel-Cancellations\output_26_0.png)
    



```python

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = data1.copy()
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

X, y = create_features(data1, label='booking_status')

features_and_target = pd.concat([X, y], axis=1)
```


```python
features_and_target.head(20)
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
      <th>booking_status</th>
    </tr>
    <tr>
      <th>date</th>
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
      <th>2017-07-01</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>182</td>
      <td>1</td>
      <td>26</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2017-07-02</th>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>183</td>
      <td>2</td>
      <td>26</td>
      <td>Sunday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-03</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>184</td>
      <td>3</td>
      <td>27</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-04</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>185</td>
      <td>4</td>
      <td>27</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-05</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>186</td>
      <td>5</td>
      <td>27</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-06</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>187</td>
      <td>6</td>
      <td>27</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2017-07-07</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>188</td>
      <td>7</td>
      <td>27</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2017-07-08</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>189</td>
      <td>8</td>
      <td>27</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-10</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>191</td>
      <td>10</td>
      <td>28</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-11</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>192</td>
      <td>11</td>
      <td>28</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2017-07-12</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>193</td>
      <td>12</td>
      <td>28</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-13</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>194</td>
      <td>13</td>
      <td>28</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2017-07-14</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>195</td>
      <td>14</td>
      <td>28</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-15</th>
      <td>5</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>196</td>
      <td>15</td>
      <td>28</td>
      <td>Saturday</td>
      <td>Summer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2017-07-16</th>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>197</td>
      <td>16</td>
      <td>28</td>
      <td>Sunday</td>
      <td>Summer</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2017-07-17</th>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>198</td>
      <td>17</td>
      <td>29</td>
      <td>Monday</td>
      <td>Summer</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2017-07-18</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>199</td>
      <td>18</td>
      <td>29</td>
      <td>Tuesday</td>
      <td>Summer</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2017-07-19</th>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>200</td>
      <td>19</td>
      <td>29</td>
      <td>Wednesday</td>
      <td>Summer</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2017-07-20</th>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>201</td>
      <td>20</td>
      <td>29</td>
      <td>Thursday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2017-07-21</th>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>2017</td>
      <td>202</td>
      <td>21</td>
      <td>29</td>
      <td>Friday</td>
      <td>Summer</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data = features_and_target.dropna(),
            x='weekday',
            y='booking_status',
            hue='season',
            ax=ax,
            linewidth=1,
            palette='YlGnBu')

ax.set_title('Number of Cancelations by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Cancelations')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\Hotel-Cancellations\output_29_0.png)
    



```python
data_2018 = features_and_target.loc[features_and_target['year'] == 2018]
data_2018.head()
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
      <th>booking_status</th>
    </tr>
    <tr>
      <th>date</th>
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
      <th>2018-01-01</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Monday</td>
      <td>Winter</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>Tuesday</td>
      <td>Winter</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>Wednesday</td>
      <td>Winter</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>Friday</td>
      <td>Winter</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(data = data_2018.dropna(),
            x='weekday',
            y='booking_status',
            hue='season',
            ax=ax,
            linewidth=1,
            palette='YlGnBu')

ax.set_title('Number of Cancelations by Day of Week 2018 Only')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Cancelations')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\Hotel-Cancellations\output_31_0.png)
    



```python
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(data = data_2018.dropna(),
            x='weekofyear',
            y='booking_status',
            hue='season',
            ax=ax,
            linewidth=1,
            palette='YlGnBu')

ax.set_title('Number of Cancelations by Day of Week 2018 Only')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Cancelations')
ax.legend(bbox_to_anchor=(1,1))
plt.show()
```


    
![png](\img\posts\Hotel-Cancellations\output_32_0.png)
    



```python
date_data = features_and_target.copy()
```


```python
data1 = data.merge(date_data, on = 'date')
data1.head()
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
      <th>no_of_adults</th>
      <th>no_of_children</th>
      <th>no_of_weekend_nights</th>
      <th>no_of_week_nights</th>
      <th>type_of_meal_plan</th>
      <th>required_car_parking_space</th>
      <th>room_type_reserved</th>
      <th>lead_time</th>
      <th>year_x</th>
      <th>month_x</th>
      <th>...</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month_y</th>
      <th>year_y</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>weekday</th>
      <th>season</th>
      <th>booking_status_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>118</td>
      <td>2017</td>
      <td>12</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>12</td>
      <td>2017</td>
      <td>362</td>
      <td>28</td>
      <td>52</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>2017</td>
      <td>12</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>12</td>
      <td>2017</td>
      <td>362</td>
      <td>28</td>
      <td>52</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>42</td>
      <td>2017</td>
      <td>12</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>12</td>
      <td>2017</td>
      <td>362</td>
      <td>28</td>
      <td>52</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>2017</td>
      <td>12</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>12</td>
      <td>2017</td>
      <td>362</td>
      <td>28</td>
      <td>52</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>35</td>
      <td>2017</td>
      <td>12</td>
      <td>...</td>
      <td>3</td>
      <td>4</td>
      <td>12</td>
      <td>2017</td>
      <td>362</td>
      <td>28</td>
      <td>52</td>
      <td>Thursday</td>
      <td>Winter</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
data1.describe().T
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
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no_of_adults</th>
      <td>18116.0</td>
      <td>1.847262</td>
      <td>0.51574</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>no_of_children</th>
      <td>18116.0</td>
      <td>0.107474</td>
      <td>0.408828</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>no_of_weekend_nights</th>
      <td>18116.0</td>
      <td>0.810775</td>
      <td>0.873802</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>no_of_week_nights</th>
      <td>18116.0</td>
      <td>2.208931</td>
      <td>1.426151</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>type_of_meal_plan</th>
      <td>18116.0</td>
      <td>0.318613</td>
      <td>0.629172</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>required_car_parking_space</th>
      <td>18116.0</td>
      <td>0.031629</td>
      <td>0.175016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>room_type_reserved</th>
      <td>18116.0</td>
      <td>0.336553</td>
      <td>0.772457</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>18116.0</td>
      <td>85.433429</td>
      <td>86.63484</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>57.0</td>
      <td>127.0</td>
      <td>443.0</td>
    </tr>
    <tr>
      <th>year_x</th>
      <td>18116.0</td>
      <td>2017.82049</td>
      <td>0.383789</td>
      <td>2017.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>month_x</th>
      <td>18116.0</td>
      <td>7.439059</td>
      <td>3.073214</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>day</th>
      <td>18116.0</td>
      <td>15.645341</td>
      <td>8.7661</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>market_segment_type</th>
      <td>18116.0</td>
      <td>0.805917</td>
      <td>0.645484</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>repeated_guest</th>
      <td>18116.0</td>
      <td>0.024895</td>
      <td>0.15581</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>no_of_previous_cancellations</th>
      <td>18116.0</td>
      <td>0.022411</td>
      <td>0.370221</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>no_of_previous_bookings_not_canceled</th>
      <td>18116.0</td>
      <td>0.150364</td>
      <td>1.711651</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>avg_price_per_room</th>
      <td>18116.0</td>
      <td>103.498054</td>
      <td>35.461471</td>
      <td>0.0</td>
      <td>80.3</td>
      <td>99.45</td>
      <td>120.285</td>
      <td>540.0</td>
    </tr>
    <tr>
      <th>no_of_special_requests</th>
      <td>18116.0</td>
      <td>0.617852</td>
      <td>0.788105</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>booking_status_x</th>
      <td>18116.0</td>
      <td>0.327832</td>
      <td>0.469436</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>reservations</th>
      <td>18116.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>dayofweek</th>
      <td>18116.0</td>
      <td>3.090197</td>
      <td>2.062916</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>quarter</th>
      <td>18116.0</td>
      <td>2.801391</td>
      <td>1.033189</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>month_y</th>
      <td>18116.0</td>
      <td>7.439059</td>
      <td>3.073214</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>year_y</th>
      <td>18116.0</td>
      <td>2017.82049</td>
      <td>0.383789</td>
      <td>2017.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>dayofyear</th>
      <td>18116.0</td>
      <td>210.667311</td>
      <td>93.777748</td>
      <td>1.0</td>
      <td>135.0</td>
      <td>226.0</td>
      <td>286.0</td>
      <td>365.0</td>
    </tr>
    <tr>
      <th>dayofmonth</th>
      <td>18116.0</td>
      <td>15.645341</td>
      <td>8.7661</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>weekofyear</th>
      <td>18116.0</td>
      <td>30.416483</td>
      <td>13.389291</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>33.0</td>
      <td>41.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>booking_status_y</th>
      <td>18116.0</td>
      <td>17.444138</td>
      <td>15.874724</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>23.0</td>
      <td>85.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data1['booking_status_x']
```




    0        0
    1        0
    2        0
    3        0
    4        0
            ..
    18111    1
    18112    0
    18113    1
    18114    0
    18115    1
    Name: booking_status_x, Length: 18116, dtype: int64




```python
#inspect correlation between label and features
print(data1.corr()["booking_status_x"].sort_values(ascending=False))
```

    booking_status_x                        1.000000
    lead_time                               0.434283
    booking_status_y                        0.355568
    year_y                                  0.183568
    year_x                                  0.183568
    avg_price_per_room                      0.145339
    no_of_week_nights                       0.096321
    no_of_adults                            0.093965
    type_of_meal_plan                       0.076771
    no_of_weekend_nights                    0.061341
    no_of_children                          0.035009
    dayofweek                               0.028634
    room_type_reserved                      0.021954
    day                                     0.012104
    dayofmonth                              0.012104
    quarter                                 0.000175
    weekofyear                             -0.011193
    dayofyear                              -0.011705
    month_y                                -0.012305
    month_x                                -0.012305
    no_of_previous_cancellations           -0.032113
    market_segment_type                    -0.045607
    no_of_previous_bookings_not_canceled   -0.060390
    required_car_parking_space             -0.092620
    repeated_guest                         -0.107060
    no_of_special_requests                 -0.248649
    reservations                                 NaN
    Name: booking_status_x, dtype: float64
    

    C:\Users\Brett\AppData\Local\Temp\ipykernel_21676\86188331.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      print(data1.corr()["booking_status_x"].sort_values(ascending=False))
    


```python
data1 = data1.drop(['reservations', 'booking_status_y'], axis = 1)
```


```python
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18116 entries, 0 to 18115
    Data columns (total 28 columns):
     #   Column                                Non-Null Count  Dtype         
    ---  ------                                --------------  -----         
     0   no_of_adults                          18116 non-null  int64         
     1   no_of_children                        18116 non-null  int64         
     2   no_of_weekend_nights                  18116 non-null  int64         
     3   no_of_week_nights                     18116 non-null  int64         
     4   type_of_meal_plan                     18116 non-null  int64         
     5   required_car_parking_space            18116 non-null  int64         
     6   room_type_reserved                    18116 non-null  int64         
     7   lead_time                             18116 non-null  int64         
     8   year_x                                18116 non-null  int64         
     9   month_x                               18116 non-null  int64         
     10  day                                   18116 non-null  int64         
     11  market_segment_type                   18116 non-null  int64         
     12  repeated_guest                        18116 non-null  int64         
     13  no_of_previous_cancellations          18116 non-null  int64         
     14  no_of_previous_bookings_not_canceled  18116 non-null  int64         
     15  avg_price_per_room                    18116 non-null  float64       
     16  no_of_special_requests                18116 non-null  int64         
     17  booking_status_x                      18116 non-null  int64         
     18  date                                  18116 non-null  datetime64[ns]
     19  dayofweek                             18116 non-null  int64         
     20  quarter                               18116 non-null  int64         
     21  month_y                               18116 non-null  int64         
     22  year_y                                18116 non-null  int64         
     23  dayofyear                             18116 non-null  int64         
     24  dayofmonth                            18116 non-null  int64         
     25  weekofyear                            18116 non-null  UInt32        
     26  weekday                               18116 non-null  category      
     27  season                                18052 non-null  category      
    dtypes: UInt32(1), category(2), datetime64[ns](1), float64(1), int64(23)
    memory usage: 3.7 MB
    


```python
X = data1[['avg_price_per_room','no_of_special_requests','market_segment_type','lead_time','required_car_parking_space']]
y = data1['booking_status_x']
print(X.head(4))
print(y.head(4))
```

       avg_price_per_room  no_of_special_requests  market_segment_type  lead_time  \
    0               110.8                       2                    1        118   
    1               107.0                       2                    1         21   
    2                91.5                       0                    1         42   
    3                92.5                       1                    1         61   
    
       required_car_parking_space  
    0                           0  
    1                           0  
    2                           1  
    3                           0  
    0    0
    1    0
    2    0
    3    0
    Name: booking_status_x, dtype: int64
    


```python
#random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
```


```python
#random forest model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(accuracy_score(y_test,rfc_pred)*100)
k=accuracy_score(y_test,rfc_pred)*100
print('\n')
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))
```

    85.22539098436063
    
    
    [[3313  339]
     [ 464 1319]]
    
    
                  precision    recall  f1-score   support
    
               0       0.88      0.91      0.89      3652
               1       0.80      0.74      0.77      1783
    
        accuracy                           0.85      5435
       macro avg       0.84      0.82      0.83      5435
    weighted avg       0.85      0.85      0.85      5435
    
    


```python
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

scale= StandardScaler()
scale.fit(X)
scaled_features = scale.transform(X)
df_feat = pd.DataFrame(X)
#X = df[['avg_price_per_room','no_of_special_requests','market_segment_type','arrival_month','lead_time']]
y = data1['booking_status_x']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.80,random_state=101)
knn = KNeighborsClassifier(n_neighbors=1)
pred = knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('KNN Accuracy score is: ',accuracy_score(y_test,pred)*100)
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

    KNN Accuracy score is:  74.27723728696613
    
    
    [[7752 2017]
     [1711 3013]]
    
    
                  precision    recall  f1-score   support
    
               0       0.82      0.79      0.81      9769
               1       0.60      0.64      0.62      4724
    
        accuracy                           0.74     14493
       macro avg       0.71      0.72      0.71     14493
    weighted avg       0.75      0.74      0.74     14493
    
    


```python
plt.figure(figsize=(14,6))
error = []
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

sns.set_style('whitegrid')
plt.plot(range(1,20),error,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```




    Text(0, 0.5, 'Error Rate')




    
![png](\img\posts\Hotel-Cancellations\output_44_1.png)
    



```python
knn = KNeighborsClassifier(n_neighbors=2)
pred = knn.fit(X_train,y_train)
pred = knn.predict(X_test)
#print('LogisticRegression score is: ',np.round(model.score(y_test,pred)*100,decimals=2))
print('\n')
print('Best KNN Accuracy score is: ',accuracy_score(y_test,pred)*100)
print('\n')
m=accuracy_score(y_test,pred)*100
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

    
    
    Best KNN Accuracy score is:  77.95487476712896
    
    
    [[9052  717]
     [2478 2246]]
    
    
                  precision    recall  f1-score   support
    
               0       0.79      0.93      0.85      9769
               1       0.76      0.48      0.58      4724
    
        accuracy                           0.78     14493
       macro avg       0.77      0.70      0.72     14493
    weighted avg       0.78      0.78      0.76     14493
    
    


```python
#logistical Model

from sklearn.linear_model import LogisticRegression

y = data1['booking_status_x']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40 ,random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print('\n')
print('Accuracy score is: ',accuracy_score(y_test,predictions)*100)
p=accuracy_score(y_test,predictions)*100

print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
```

    
    
    Accuracy score is:  78.52904650200084
    
    
    [[4359  522]
     [1034 1332]]
    
    
                  precision    recall  f1-score   support
    
               0       0.81      0.89      0.85      4881
               1       0.72      0.56      0.63      2366
    
        accuracy                           0.79      7247
       macro avg       0.76      0.73      0.74      7247
    weighted avg       0.78      0.79      0.78      7247
    
    


```python
sns.set_context("poster", font_scale = .75)
cm = confusion_matrix(y_test, predictions)
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


    
![png](\img\posts\Hotel-Cancellations\output_47_0.png)
    



```python
import plotly.express as px
from plotly import graph_objects
from textwrap import wrap
import chart_studio.plotly as py

named_colorscales = px.colors.named_colorscales()
print("\n".join(wrap("".join('{:<12}'.format(c) for c in named_colorscales), 96)))
```

    aggrnyl     agsunset    blackbody   bluered     blues       blugrn      bluyl       brwnyl
    bugn        bupu        burg        burgyl      cividis     darkmint    electric    emrld
    gnbu        greens      greys       hot         inferno     jet         magenta     magma
    mint        orrd        oranges     oryel       peach       pinkyl      plasma      plotly3
    pubu        pubugn      purd        purp        purples     purpor      rainbow     rdbu
    rdpu        redor       reds        sunset      sunsetdark  teal        tealgrn     turbo
    viridis     ylgn        ylgnbu      ylorbr      ylorrd      algae       amp         deep
    dense       gray        haline      ice         matter      solar       speed       tempo
    thermal     turbid      armyrose    brbg        earth       fall        geyser      prgn
    piyg        picnic      portland    puor        rdgy        rdylbu      rdylgn      spectral
    tealrose    temps       tropic      balance     curl        delta       oxy         edge
    hsv         icefire     phase       twilight    mrybm       mygbm
    


```python
!pip install chart-studio
```

    Requirement already satisfied: chart-studio in c:\users\brett\anaconda3\envs\school\lib\site-packages (1.1.0)
    Requirement already satisfied: plotly in c:\users\brett\anaconda3\envs\school\lib\site-packages (from chart-studio) (5.13.0)
    Requirement already satisfied: requests in c:\users\brett\anaconda3\envs\school\lib\site-packages (from chart-studio) (2.28.1)
    Requirement already satisfied: six in c:\users\brett\anaconda3\envs\school\lib\site-packages (from chart-studio) (1.16.0)
    Requirement already satisfied: retrying>=1.3.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from chart-studio) (1.3.4)
    Requirement already satisfied: tenacity>=6.2.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from plotly->chart-studio) (8.2.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from requests->chart-studio) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from requests->chart-studio) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from requests->chart-studio) (3.4)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from requests->chart-studio) (2.0.4)
    


```python

```

### xgboost


```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
```


```python

xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)
xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
b1 = accuracy*100
print('accuracy: %f' %(accuracy))
```

    accuracy: 0.844901
    


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
#inspect correlation between label and features


print(data1.corr()["booking_status_x"].sort_values(ascending=False))
```

    booking_status_x                        1.000000
    lead_time                               0.434283
    year_y                                  0.183568
    year_x                                  0.183568
    avg_price_per_room                      0.145339
    no_of_week_nights                       0.096321
    no_of_adults                            0.093965
    type_of_meal_plan                       0.076771
    no_of_weekend_nights                    0.061341
    no_of_children                          0.035009
    dayofweek                               0.028634
    room_type_reserved                      0.021954
    day                                     0.012104
    dayofmonth                              0.012104
    quarter                                 0.000175
    weekofyear                             -0.011193
    dayofyear                              -0.011705
    month_y                                -0.012305
    month_x                                -0.012305
    no_of_previous_cancellations           -0.032113
    market_segment_type                    -0.045607
    no_of_previous_bookings_not_canceled   -0.060390
    required_car_parking_space             -0.092620
    repeated_guest                         -0.107060
    no_of_special_requests                 -0.248649
    Name: booking_status_x, dtype: float64
    


```python
X = data1[['avg_price_per_room',
           'no_of_special_requests',
           'market_segment_type',
           'lead_time',
           'required_car_parking_space',
          'repeated_guest',
          'no_of_week_nights',
          'no_of_adults']]
y = data1['booking_status_x']
print(X.head(4))
print(y.head(4))
```

       avg_price_per_room  no_of_special_requests  market_segment_type  lead_time  \
    0               110.8                       2                    1        118   
    1               107.0                       2                    1         21   
    2                91.5                       0                    1         42   
    3                92.5                       1                    1         61   
    
       required_car_parking_space  repeated_guest  no_of_week_nights  no_of_adults  
    0                           0               0                  4             2  
    1                           0               0                  1             2  
    2                           1               0                  1             2  
    3                           0               0                  3             2  
    0    0
    1    0
    2    0
    3    0
    Name: booking_status_x, dtype: int64
    


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 123)
```

### Build Pipeline


```python
!pip install category-encoders
```

    Requirement already satisfied: category-encoders in c:\users\brett\anaconda3\envs\school\lib\site-packages (2.6.0)
    Requirement already satisfied: scipy>=1.0.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (1.10.0)
    Requirement already satisfied: patsy>=0.5.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (0.5.3)
    Requirement already satisfied: pandas>=1.0.5 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (1.5.3)
    Requirement already satisfied: numpy>=1.14.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (1.23.5)
    Requirement already satisfied: statsmodels>=0.9.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (0.13.5)
    Requirement already satisfied: scikit-learn>=0.20.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from category-encoders) (1.2.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=1.0.5->category-encoders) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pandas>=1.0.5->category-encoders) (2022.7)
    Requirement already satisfied: six in c:\users\brett\anaconda3\envs\school\lib\site-packages (from patsy>=0.5.1->category-encoders) (1.16.0)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-learn>=0.20.0->category-encoders) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-learn>=0.20.0->category-encoders) (3.1.0)
    Requirement already satisfied: packaging>=21.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from statsmodels>=0.9.0->category-encoders) (22.0)
    


```python
conda install -c anaconda py-xgboost
```

    Retrieving notices: ...working... done
    Collecting package metadata (current_repodata.json): ...working... done
    Solving environment: ...working... done
    
    # All requested packages already installed.
    
    
    Note: you may need to restart the kernel to use updated packages.
    

    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 23.1.0
      latest version: 23.7.3
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    Or to minimize the number of packages updated during conda update use
    
         conda install conda=23.7.3
    
    
    


```python
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
#from xgboost import XGBclassifier
import xgboost as xgb
#from xgboost import XGBclassifier

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', xgb.XGBClassifier(random_state = 123))
]
pipe = Pipeline(steps = estimators)
pipe
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder()),
                (&#x27;clf&#x27;,
                 XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=None, colsample_bynode=None,
                               colsample_bytree=None,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=None, gpu_id=None,
                               grow_policy=None, importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None, missing=nan,
                               monotone_constraints=None, n_estimators=100,
                               n_jobs=None, num_parallel_tree=None,
                               predictor=None, random_state=123, ...))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder()),
                (&#x27;clf&#x27;,
                 XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=None, colsample_bynode=None,
                               colsample_bytree=None,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=None, gpu_id=None,
                               grow_policy=None, importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None, missing=nan,
                               monotone_constraints=None, n_estimators=100,
                               n_jobs=None, num_parallel_tree=None,
                               predictor=None, random_state=123, ...))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">TargetEncoder</label><div class="sk-toggleable__content"><pre>TargetEncoder()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=123, ...)</pre></div></div></div></div></div></div></div>



### Set up hyperparameter tuning


```python
!pip install scikit-optimize
```

    Requirement already satisfied: scikit-optimize in c:\users\brett\anaconda3\envs\school\lib\site-packages (0.9.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-optimize) (1.2.0)
    Requirement already satisfied: pyaml>=16.9 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-optimize) (21.10.1)
    Requirement already satisfied: scikit-learn>=0.20.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-optimize) (1.2.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-optimize) (1.23.5)
    Requirement already satisfied: scipy>=0.19.1 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-optimize) (1.10.0)
    Requirement already satisfied: PyYAML in c:\users\brett\anaconda3\envs\school\lib\site-packages (from pyaml>=16.9->scikit-optimize) (6.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\brett\anaconda3\envs\school\lib\site-packages (from scikit-learn>=0.20.0->scikit-optimize) (3.1.0)
    


```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=10, scoring='roc_auc', random_state=8) 
```


```python
#train xgboost model
opt.fit(X_train, y_train)
```

    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    Warning: No categorical columns found. Calling 'transform' will only return input data.
    




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>BayesSearchCV(cv=3,
              estimator=Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder()),
                                        (&#x27;clf&#x27;,
                                         XGBClassifier(base_score=None,
                                                       booster=None,
                                                       callbacks=None,
                                                       colsample_bylevel=None,
                                                       colsample_bynode=None,
                                                       colsample_bytree=None,
                                                       early_stopping_rounds=None,
                                                       enable_categorical=False,
                                                       eval_metric=None,
                                                       feature_types=None,
                                                       gamma=None, gpu_id=None,
                                                       grow_policy=None,
                                                       importance_type=N...
                             &#x27;clf__learning_rate&#x27;: Real(low=0.001, high=1.0, prior=&#x27;log-uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__max_depth&#x27;: Integer(low=2, high=8, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__reg_alpha&#x27;: Real(low=0.0, high=10.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__reg_lambda&#x27;: Real(low=0.0, high=10.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__subsample&#x27;: Real(low=0.5, high=1.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">BayesSearchCV</label><div class="sk-toggleable__content"><pre>BayesSearchCV(cv=3,
              estimator=Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder()),
                                        (&#x27;clf&#x27;,
                                         XGBClassifier(base_score=None,
                                                       booster=None,
                                                       callbacks=None,
                                                       colsample_bylevel=None,
                                                       colsample_bynode=None,
                                                       colsample_bytree=None,
                                                       early_stopping_rounds=None,
                                                       enable_categorical=False,
                                                       eval_metric=None,
                                                       feature_types=None,
                                                       gamma=None, gpu_id=None,
                                                       grow_policy=None,
                                                       importance_type=N...
                             &#x27;clf__learning_rate&#x27;: Real(low=0.001, high=1.0, prior=&#x27;log-uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__max_depth&#x27;: Integer(low=2, high=8, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__reg_alpha&#x27;: Real(low=0.0, high=10.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__reg_lambda&#x27;: Real(low=0.0, high=10.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;),
                             &#x27;clf__subsample&#x27;: Real(low=0.5, high=1.0, prior=&#x27;uniform&#x27;, transform=&#x27;normalize&#x27;)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder()),
                (&#x27;clf&#x27;,
                 XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=None, colsample_bynode=None,
                               colsample_bytree=None,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=None, gpu_id=None,
                               grow_policy=None, importance_type=None,
                               interaction_constraints=None, learning_rate=None,
                               max_bin=None, max_cat_threshold=None,
                               max_cat_to_onehot=None, max_delta_step=None,
                               max_depth=None, max_leaves=None,
                               min_child_weight=None, missing=nan,
                               monotone_constraints=None, n_estimators=100,
                               n_jobs=None, num_parallel_tree=None,
                               predictor=None, random_state=123, ...))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">TargetEncoder</label><div class="sk-toggleable__content"><pre>TargetEncoder()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=123, ...)</pre></div></div></div></div></div></div></div></div></div></div></div></div>



### Evaluate the model / make predictions


```python
opt.best_estimator_
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder(cols=[])),
                (&#x27;clf&#x27;,
                 XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=0.9425384185492701,
                               colsample_bynode=0.9095956806239844,
                               colsample_bytree=0.706128679361455,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=1.6598135411398998,
                               gpu_id=None, g...icy=None,
                               importance_type=None,
                               interaction_constraints=None,
                               learning_rate=0.7929828265552742, max_bin=None,
                               max_cat_threshold=None, max_cat_to_onehot=None,
                               max_delta_step=None, max_depth=7,
                               max_leaves=None, min_child_weight=None,
                               missing=nan, monotone_constraints=None,
                               n_estimators=100, n_jobs=None,
                               num_parallel_tree=None, predictor=None,
                               random_state=123, ...))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;encoder&#x27;, TargetEncoder(cols=[])),
                (&#x27;clf&#x27;,
                 XGBClassifier(base_score=None, booster=None, callbacks=None,
                               colsample_bylevel=0.9425384185492701,
                               colsample_bynode=0.9095956806239844,
                               colsample_bytree=0.706128679361455,
                               early_stopping_rounds=None,
                               enable_categorical=False, eval_metric=None,
                               feature_types=None, gamma=1.6598135411398998,
                               gpu_id=None, g...icy=None,
                               importance_type=None,
                               interaction_constraints=None,
                               learning_rate=0.7929828265552742, max_bin=None,
                               max_cat_threshold=None, max_cat_to_onehot=None,
                               max_delta_step=None, max_depth=7,
                               max_leaves=None, min_child_weight=None,
                               missing=nan, monotone_constraints=None,
                               n_estimators=100, n_jobs=None,
                               num_parallel_tree=None, predictor=None,
                               random_state=123, ...))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">TargetEncoder</label><div class="sk-toggleable__content"><pre>TargetEncoder(cols=[])</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=0.9425384185492701,
              colsample_bynode=0.9095956806239844,
              colsample_bytree=0.706128679361455, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=1.6598135411398998, gpu_id=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.7929828265552742, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=7, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=123, ...)</pre></div></div></div></div></div></div></div>




```python
opt.best_score_
```




    0.9117809899504422




```python
opt.score(X_test, y_test)
```




    0.9192097148275309




```python
b2 = opt.best_score_ * 100
```


```python
opt.predict(X_test)
```




    array([0, 0, 0, ..., 0, 0, 0])




```python
opt.predict_proba(X_test)
```




    array([[0.9834009 , 0.01659912],
           [0.86485183, 0.13514817],
           [0.8582463 , 0.14175366],
           ...,
           [0.99780715, 0.00219283],
           [0.95073146, 0.04926857],
           [0.55550265, 0.44449738]], dtype=float32)



### Feature importance


```python
opt.best_estimator_.steps
```




    [('encoder', TargetEncoder(cols=[])),
     ('clf',
      XGBClassifier(base_score=None, booster=None, callbacks=None,
                    colsample_bylevel=0.9425384185492701,
                    colsample_bynode=0.9095956806239844,
                    colsample_bytree=0.706128679361455, early_stopping_rounds=None,
                    enable_categorical=False, eval_metric=None, feature_types=None,
                    gamma=1.6598135411398998, gpu_id=None, grow_policy=None,
                    importance_type=None, interaction_constraints=None,
                    learning_rate=0.7929828265552742, max_bin=None,
                    max_cat_threshold=None, max_cat_to_onehot=None,
                    max_delta_step=None, max_depth=7, max_leaves=None,
                    min_child_weight=None, missing=nan, monotone_constraints=None,
                    n_estimators=100, n_jobs=None, num_parallel_tree=None,
                    predictor=None, random_state=123, ...))]




```python
from xgboost import plot_importance

xgboost_step = opt.best_estimator_.steps[1]
xgboost_model = xgboost_step[1]
plot_importance(xgboost_model)
```




    <AxesSubplot: title={'center': 'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](\img\posts\Hotel-Cancellations\output_76_1.png)
    



```python
# Set the values for number of folds and stopping iterations
n_folds = 5
early_stopping = 10

#create params
params = {'objective': 'binary:logistic',
         'seed':99,
         'eval_metric':'error'}

# Create the DTrain matrix for XGBoost
DTrain = xgb.DMatrix(X_train, label = y_train)

# Create the data frame of cross validations
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, nfold = n_folds,
            early_stopping_rounds = early_stopping)

# Print the cross validations data frame
print(cv_df)
```

       train-error-mean  train-error-std  test-error-mean  test-error-std
    0          0.163573         0.001738         0.173267        0.006149
    1          0.156224         0.002473         0.164780        0.004829
    2          0.151998         0.000750         0.158777        0.005875
    3          0.149065         0.001235         0.158294        0.005947
    4          0.146132         0.001554         0.154637        0.005209
    


```python
cv = xgb.cv(params, DTrain, num_boost_round = 600, nfold=10,
            shuffle = True)
```


```python
# Print the first five rows of the CV results data frame
print(cv.head())

# Calculate the mean of the test AUC scores
print(np.mean(cv['test-error-mean']).round(2))

# Plot the test AUC scores for each iteration
plt.plot(cv['test-error-mean'])
plt.title('Test Error Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test Error Score')
plt.show()
```

       train-error-mean  train-error-std  test-error-mean  test-error-std
    0          0.164689         0.001560         0.170438        0.007783
    1          0.156339         0.003456         0.163262        0.006057
    2          0.152191         0.001917         0.158846        0.007608
    3          0.148910         0.001316         0.156706        0.008268
    4          0.146732         0.001538         0.155051        0.006846
    0.13
    


    
![png](\img\posts\Hotel-Cancellations\output_79_1.png)
    



```python
b1_preds = xgboost_model.predict(X_test)
```


```python
b1_preds
```




    array([0, 0, 0, ..., 0, 0, 0])




```python
target_names = ['Non-Cancel', 'Cancel']
print(classification_report(y_test, b1_preds, target_names=target_names))
```

                  precision    recall  f1-score   support
    
      Non-Cancel       0.88      0.92      0.90      2436
          Cancel       0.83      0.75      0.78      1188
    
        accuracy                           0.87      3624
       macro avg       0.85      0.83      0.84      3624
    weighted avg       0.86      0.87      0.86      3624
    
    


```python
import plotly.express as px
from textwrap import wrap

label = ['Random Forest','K Nearest Neighbours','Logistics Regression', 'XGBoost_1', 'XGBoost_2']
fig = px.pie(labels=label,values=[k,m,p, b1, b2], width = 700,names=label, height = 700)
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.50, 
                  #marker = dict(colors = ['#636EFA','#EF553B','#00CC96','#AB63FA'], line = dict(color = 'white', width = 2)))
                  #marker = dict(color_continuous_scale = px.colors.sequential.Viridis, line = dict(color = 'white', width = 2))
                  #marker = dict(colors = px.colors.sequential.Plasma_r, line = dict(color = 'white', width = 2)))
                  #marker = dict(colors =px.colors.sequential.Plasma, line = dict(color = 'white', width = 2)))
                  marker = dict(colors = px.colors.sequential.YlGnBu, line = dict(color = 'white', width = 2)))

fig.update_layout(annotations = [dict(text = 'Performance Comparison', 
                                      x = 0.5, y = 0.5, font_size = 20, showarrow = False, 
                                      font_family = 'monospace',
                                      font_color = 'black')],
                  showlegend = False)
```

![png](\img\posts\Hotel-Cancellations\output_83_0.png)