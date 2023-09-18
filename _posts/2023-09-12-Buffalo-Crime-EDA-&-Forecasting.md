---
layout: post
title: "City of Buffalo Crime EDA & Forecasting"
subtitle: "Sourcing Data From Buffalo OpenData API To Predict Crime"
date: 2023-09-12
background: '/img/posts/Buffalo-Crime/BuffaloCrimeCover.jpg'
#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---


### Table of Contents

- [Summary](#summary)
- [Imports](#imports)
- [Open Buffalo API Request](#api)
- [Initial Data Shape & Column Name Review](#review)
- [Check For Null Data](#null)
- [Add Date Columns](#date)
- [Exploratory Data Analysis](#eda)
- [Neighborhood Graphs](#neighGraphs)
- [Geospatial Graphs](#map)
- [Geospatial Heatmaps](#heatmap)
- [Forecast Modeling](#forecasting)
- [Conclusion](#conclusion)

### Summary
<a id='summary'></a>

This project serves as a comprehensive demonstration of crime analysis in Buffalo, utilizing an API linked to Buffalo's open data resources. In recognition of the potential data reliability issues noted on the Buffalo Open Data website prior to 2009, the decision was made to focus exclusively on data spanning from 2009 to the present day.

The primary objectives of this endeavor encompass several key aspects:

1. **Data Acquisition Through APIs**: The project commences by harnessing the power of Application Programming Interfaces (APIs) to efficiently collect and retrieve crime-related data from Buffalo's open data repository. This process ensures access to up-to-date and reliable information, essential for subsequent analysis.

2. **Exploratory Data Analysis (EDA)**: Following data acquisition, an initial exploratory analysis phase ensues. During this stage, the project aims to uncover valuable insights and trends within the crime data. This involves examining patterns by year, neighborhood, and crime type, shedding light on key factors influencing Buffalo's crime landscape.

3. **Forecasting Techniques**: Building upon the EDA findings, the project delves into advanced forecasting techniques to enhance our understanding of future crime trends. Three primary forecasting methods are employed:

   - **Simple Moving Averages**: This technique applies a straightforward moving average approach to predict future crime rates. It involves calculating the average of crime occurrences over a defined period, such as months or weeks, providing a basic yet valuable forecasting tool.

   - **Weighted Moving Averages**: In this approach, a weighted average is employed, assigning different levels of importance to data points based on their proximity to the prediction point. This method accommodates the potential significance of recent crime data in making forecasts.

   - **Exponential Moving Averages**: Recognizing the exponential decay of relevance in historical data, exponential moving averages assign greater weight to recent data points. This technique is particularly useful for capturing short-term fluctuations and trends in crime rates.

Through this multifaceted approach, the project contributes to a data-driven understanding of crime dynamics in Buffalo and to make informed decisions for a safer future.

### Import Packages
<a id='imports'></a>


```python
# import packages

import requests
import pandas as pd
import math
import datetime
import urllib.request
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from folium.plugins import HeatMap
import folium

plt.style.use('seaborn-v0_8-darkgrid')
# warnings ignore
import warnings
# set warnings to ignore
#warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None  # default='warn'
```


```python
# bring api key into googleColab
from google.colab import files
#import io

uploaded = files.upload()
```
    
### Buffalo OpenData API
<a id='api'></a>


```python
# open api key
app_token = open('api_key.txt', 'r').read()
# app_token
```


```python
# hide api token & return BuffaloOpenData crime data
limit = 500000
app_token = open('api_key.txt', 'r').read()

uri = f"https://data.buffalony.gov/resource/d6g9-xbgu.json?$limit={limit}&$$app_token={app_token}&$where=incident_datetime>'2009-01-10T12:00:00'"

# send the HTTP GET request
r = requests.get(uri)

# check the response status code and process the data if it's successful
if r.status_code == 200:
    print('Status code:', r.status_code)
    print('Number of rows returned:', len(r.json()))
    print('Encoded URI with params:', r.url)
    new_json = r.json()
    # Process the new_json data as needed
else:
    print('Failed to fetch data. Status code:', r.status_code)

```

    Status code: 200
    Number of rows returned: 239722
    Encoded URI with params: https://data.buffalony.gov/resource/d6g9-xbgu.json?$limit=500000&$$app_token=NnGV0W4ip4YEFBLvBMGAjaByD&$where=incident_datetime%3E'2009-01-10T12:00:00'
    

### Initial Data Shape & Column Review
<a id='review'></a>


```python
data=pd.DataFrame(new_json)
print(data.shape)
data.head()
```

    (239722, 27)
    





  <div id="df-fb652089-6e2d-4b75-98a2-1e2bd781566d" class="colab-df-container">
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
      <th>case_number</th>
      <th>incident_datetime</th>
      <th>incident_type_primary</th>
      <th>incident_description</th>
      <th>parent_incident_type</th>
      <th>hour_of_day</th>
      <th>day_of_week</th>
      <th>address_1</th>
      <th>city</th>
      <th>state</th>
      <th>...</th>
      <th>census_tract</th>
      <th>census_block</th>
      <th>census_block_group</th>
      <th>neighborhood_1</th>
      <th>police_district</th>
      <th>council_district</th>
      <th>tractce20</th>
      <th>geoid20_tract</th>
      <th>geoid20_blockgroup</th>
      <th>geoid20_block</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>09-0100387</td>
      <td>2009-01-10T12:19:00.000</td>
      <td>BURGLARY</td>
      <td>Buffalo Police are investigating this report o...</td>
      <td>Breaking &amp; Entering</td>
      <td>12</td>
      <td>Saturday</td>
      <td>2700 Block BAILEY</td>
      <td>Buffalo</td>
      <td>NY</td>
      <td>...</td>
      <td>51</td>
      <td>1013</td>
      <td>1</td>
      <td>North Park</td>
      <td>District D</td>
      <td>DELAWARE</td>
      <td>005100</td>
      <td>36029005100</td>
      <td>360290001101</td>
      <td>360290002001013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09-0100389</td>
      <td>2009-01-10T12:21:00.000</td>
      <td>BURGLARY</td>
      <td>Buffalo Police are investigating this report o...</td>
      <td>Breaking &amp; Entering</td>
      <td>12</td>
      <td>Saturday</td>
      <td>800 Block EGGERT RD</td>
      <td>Buffalo</td>
      <td>NY</td>
      <td>...</td>
      <td>41</td>
      <td>1009</td>
      <td>1</td>
      <td>Kenfield</td>
      <td>District E</td>
      <td>UNIVERSITY</td>
      <td>004100</td>
      <td>36029004100</td>
      <td>360290001101</td>
      <td>360290002001009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>09-0270361</td>
      <td>2009-01-10T12:27:00.000</td>
      <td>UUV</td>
      <td>Buffalo Police are investigating this report o...</td>
      <td>Theft of Vehicle</td>
      <td>12</td>
      <td>Saturday</td>
      <td>1600 Block MAIN ST</td>
      <td>Buffalo</td>
      <td>NY</td>
      <td>...</td>
      <td>168.02</td>
      <td>1017</td>
      <td>1</td>
      <td>Masten Park</td>
      <td>District E</td>
      <td>MASTEN</td>
      <td>016802</td>
      <td>36029016802</td>
      <td>360290001101</td>
      <td>360290165001017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09-0100435</td>
      <td>2009-01-10T12:30:00.000</td>
      <td>ASSAULT</td>
      <td>Buffalo Police are investigating this report o...</td>
      <td>Assault</td>
      <td>12</td>
      <td>Saturday</td>
      <td>JEFFERSON AV &amp; E FERRY ST</td>
      <td>Buffalo</td>
      <td>NY</td>
      <td>...</td>
      <td>168.02</td>
      <td>2000</td>
      <td>2</td>
      <td>Masten Park</td>
      <td>District E</td>
      <td>MASTEN</td>
      <td>016802</td>
      <td>36029016802</td>
      <td>360290001102</td>
      <td>360290046012000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09-0100421</td>
      <td>2009-01-10T12:30:00.000</td>
      <td>BURGLARY</td>
      <td>Buffalo Police are investigating this report o...</td>
      <td>Breaking &amp; Entering</td>
      <td>12</td>
      <td>Saturday</td>
      <td>100 Block URBAN ST</td>
      <td>Buffalo</td>
      <td>NY</td>
      <td>...</td>
      <td>35.02</td>
      <td>2000</td>
      <td>2</td>
      <td>MLK Park</td>
      <td>District C</td>
      <td>MASTEN</td>
      <td>003502</td>
      <td>36029003502</td>
      <td>360290001102</td>
      <td>360290046012000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fb652089-6e2d-4b75-98a2-1e2bd781566d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-fb652089-6e2d-4b75-98a2-1e2bd781566d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fb652089-6e2d-4b75-98a2-1e2bd781566d');
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


<div id="df-4572c252-2989-4308-a1ce-134e70f228cb">
  <button class="colab-df-quickchart" onclick="quickchart('df-4572c252-2989-4308-a1ce-134e70f228cb')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4572c252-2989-4308-a1ce-134e70f228cb button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# check data types and swicth to int, floats and strings
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 239722 entries, 0 to 239721
    Data columns (total 27 columns):
     #   Column                   Non-Null Count   Dtype 
    ---  ------                   --------------   ----- 
     0   case_number              239722 non-null  object
     1   incident_datetime        239722 non-null  object
     2   incident_type_primary    239722 non-null  object
     3   incident_description     239722 non-null  object
     4   parent_incident_type     239722 non-null  object
     5   hour_of_day              239722 non-null  object
     6   day_of_week              239722 non-null  object
     7   address_1                239705 non-null  object
     8   city                     239722 non-null  object
     9   state                    239722 non-null  object
     10  location                 235055 non-null  object
     11  latitude                 235055 non-null  object
     12  longitude                235055 non-null  object
     13  created_at               239722 non-null  object
     14  census_tract_2010        237713 non-null  object
     15  census_block_group_2010  237713 non-null  object
     16  census_block_2010        237713 non-null  object
     17  census_tract             237713 non-null  object
     18  census_block             237713 non-null  object
     19  census_block_group       237713 non-null  object
     20  neighborhood_1           237713 non-null  object
     21  police_district          237713 non-null  object
     22  council_district         237713 non-null  object
     23  tractce20                237850 non-null  object
     24  geoid20_tract            237850 non-null  object
     25  geoid20_blockgroup       237850 non-null  object
     26  geoid20_block            237850 non-null  object
    dtypes: object(27)
    memory usage: 49.4+ MB
    

### Check For Null Data
<a id='null'></a>


```python
# check for null
data.isnull().sum()
```




    case_number                   0
    incident_datetime             0
    incident_type_primary         0
    incident_description          0
    parent_incident_type          0
    hour_of_day                   0
    day_of_week                   0
    address_1                    17
    city                          0
    state                         0
    location                   4667
    latitude                   4667
    longitude                  4667
    created_at                    0
    census_tract_2010          2009
    census_block_group_2010    2009
    census_block_2010          2009
    census_tract               2009
    census_block               2009
    census_block_group         2009
    neighborhood_1             2009
    police_district            2009
    council_district           2009
    tractce20                  1872
    geoid20_tract              1872
    geoid20_blockgroup         1872
    geoid20_block              1872
    dtype: int64




```python
# chatgpt code for function displaying null & non-null column ratios

def null_nonnull_ratios(dataframe):
    """
    Calculate the ratios of null and non-null data in a pandas DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame for which you want to calculate null and non-null ratios.

    Returns:
    pd.DataFrame: A DataFrame containing columns for null and non-null ratios for each column.
    """
    total_rows = len(dataframe)
    null_counts = dataframe.isnull().sum()
    nonnull_counts = total_rows - null_counts
    null_ratios = null_counts / total_rows
    nonnull_ratios = nonnull_counts / total_rows
    result_df = pd.DataFrame({'null': null_ratios, 'non-null': nonnull_ratios})
    return result_df

ratios = null_nonnull_ratios(data)
print(ratios)

```

                                 null  non-null
    case_number              0.000000  1.000000
    incident_datetime        0.000000  1.000000
    incident_type_primary    0.000000  1.000000
    incident_description     0.000000  1.000000
    parent_incident_type     0.000000  1.000000
    hour_of_day              0.000000  1.000000
    day_of_week              0.000000  1.000000
    address_1                0.000071  0.999929
    city                     0.000000  1.000000
    state                    0.000000  1.000000
    location                 0.019468  0.980532
    latitude                 0.019468  0.980532
    longitude                0.019468  0.980532
    created_at               0.000000  1.000000
    census_tract_2010        0.008381  0.991619
    census_block_group_2010  0.008381  0.991619
    census_block_2010        0.008381  0.991619
    census_tract             0.008381  0.991619
    census_block             0.008381  0.991619
    census_block_group       0.008381  0.991619
    neighborhood_1           0.008381  0.991619
    police_district          0.008381  0.991619
    council_district         0.008381  0.991619
    tractce20                0.007809  0.992191
    geoid20_tract            0.007809  0.992191
    geoid20_blockgroup       0.007809  0.992191
    geoid20_block            0.007809  0.992191
    

### Add Date Columns
<a id='date'></a>


```python
# make new date columns to groupby for EDA

data.index = pd.DatetimeIndex(data['incident_datetime'])

data['Year'] = data.index.year
data['Month'] = data.index.month
data['dayOfWeek'] = data.index.dayofweek
data['dayOfMonth'] = data.index.day
data['dayOfYear'] = data.index.dayofyear
data['weekOfMonth'] = data.dayOfMonth.apply(lambda d: (d - 1) // 7 + 1)

dayOfYear = list(data.index.dayofyear)

weekOfYear = [math.ceil(i/7) for i in dayOfYear]
data['weekOfYear'] = weekOfYear
```


```python
# code for color slection on graphs / comment out later

import math

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig
```


```python
# available colors for graphs / comment out later
plt.style.use('dark_background')  # set the background to black
plot_colortable(mcolors.CSS4_COLORS)
plt.show()
```


    
<!--![png](\img\posts\Buffalo-Crime\output_18_0.png)-->
    


### Exploratory Data Analysis
<a id='eda'></a>


```python
# yearly analysis on crime count

# plt.style.use('dark_background')  # set the background to black
# once plt.style is set there is no need to include teh code setting in future plots
ax = data.groupby([data.Year]).size().plot(legend=False, color='yellowgreen', kind='barh')

plt.ylabel('Year', color='white')
plt.xlabel('Number of crimes', color='white')
plt.title('Number of crimes by year', color='white')

plt.tick_params(axis='both', colors='white')  # Set tick color
ax.spines['bottom'].set_color('white')  # Set x-axis color
ax.spines['left'].set_color('white')  # Set y-axis color

plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_20_0.png)
    


The graph presented above illustrates a noteworthy annual decline in the total number of crimes since the year 2009. <br><br>

Furthermore, as depicted in the chart below, the year 2022 accounts for a relatively modest 3.95% of the total crimes recorded in the dataset spanning from 2009 to the present day.


```python
# above graph data in chart form
print(f'Percentage of total crimes in dataset(2009-2023) per year:\n\n{data.Year.value_counts(normalize=True)}')
```

    Percentage of total crimes in dataset(2009-2023) per year:
    
    2010    0.090559
    2009    0.088761
    2012    0.085991
    2011    0.085399
    2013    0.077807
    2014    0.073097
    2015    0.072033
    2016    0.068629
    2018    0.064516
    2017    0.064262
    2019    0.057020
    2020    0.050571
    2021    0.049011
    2022    0.039533
    2023    0.032809
    Name: Year, dtype: float64
    


```python
#crimes by day of week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ax = data.groupby([data.dayOfWeek]).size().plot(legend=False, color='yellowgreen', kind='barh')
#ax = data.groupby([data.Year]).size().plot(legend=False, color='yellowgreen', kind='barh')

plt.ylabel('Day of week', color='white')
plt.yticks(np.arange(7), days)
plt.xlabel('Number Of Crimes', color='white')
plt.title('Number Of Crimes By Day Of Week', color='white')

plt.tick_params(axis='both', colors='white')  # Set tick color
ax.spines['bottom'].set_color('white')  # Set x-axis color
ax.spines['left'].set_color('white')  # Set y-axis color

plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_23_0.png)
    


Friday appears to exhibit a slightly higher incidence of crimes when compared to other days, although this difference is not markedly significant.


```python
# crimes by month
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
data.groupby([data.Month]).size().plot(kind='barh', color='yellowgreen')
plt.ylabel('Months Of The Year')
plt.yticks(np.arange(12), months)
plt.xlabel('Number Of Crimes')
plt.title('Number Of Crimes By Month Of The Year')
plt.show()
```


    
![png](img\posts\Buffalo-Crime\output_25_0.png)
    



```python
# define a dictionary to map numeric month values to month names
month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

# map the numeric month values to month names
data['MonthNames'] = data['Month'].map(month_names)

# calculate the counts of each month and normalize the results
month_counts = data['MonthNames'].value_counts(normalize=True)

print(f'Percentage of Crime Per Month:\n\n{month_counts}')

```

    Percentage of Crime Per Month:
    
    August       0.100879
    July         0.100212
    June         0.091752
    May          0.089896
    September    0.088649
    October      0.086217
    April        0.078445
    November     0.076042
    January      0.075567
    December     0.074728
    March        0.074190
    February     0.063423
    Name: MonthNames, dtype: float64
    

The graphical representations above provide a clear depiction of February consistently registering the lowest number of crimes per month.<br><br>

Moreover, the chart underscores a pronounced disparity in crime rates between the sweltering summer months and the frigid winter months.


```python
plt.figure(figsize=(11,5))
data.resample('M').size().plot(legend=False, color='yellowgreen')
plt.title('Number Of Crimes Per Month (2009 - 2023)')
plt.xlabel('Months')
plt.ylabel('Number Of Crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_28_0.png)
    


The chart presented above vividly illustrates a declining trend in annual crime rates.<br><br>

Furthermore, it unveils a distinctive zigzag pattern, with crime receding during the colder seasons and resurging during the hotter months.


```python
data.groupby([data.dayOfMonth]).size().plot(kind='barh',legend=False, color='yellowgreen')
plt.ylabel('Day of the month')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by day of the month')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_30_0.png)
    



```python
print(f'Percentage Of Crime Per Day Of Month:\n\n{data.dayOfMonth.value_counts(normalize=True)}')
```

    Percentage Of Crime Per Day Of Month:
    
    1     0.041590
    20    0.033643
    23    0.033635
    15    0.033626
    10    0.033606
    24    0.033547
    21    0.033159
    22    0.033080
    28    0.032884
    27    0.032880
    3     0.032792
    4     0.032667
    18    0.032596
    16    0.032529
    11    0.032529
    17    0.032521
    14    0.032475
    12    0.032408
    25    0.032287
    13    0.032246
    19    0.032237
    26    0.032033
    7     0.031916
    5     0.031758
    8     0.031566
    9     0.031411
    6     0.031257
    2     0.031015
    30    0.030577
    29    0.030310
    31    0.019218
    Name: dayOfMonth, dtype: float64
    

The data suggests that the first day of each month consistently records the highest incidence of criminal activities.


```python
# crimes plotted per day
plt.figure(figsize=(11,5))
data.resample('D').size().plot(legend=False, color='yellowgreen')
plt.title('Number Of Crimes Per Day (2009 - 2023)')
plt.xlabel('Days')
plt.ylabel('Number Of Crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_33_0.png)
    



```python
# crimes plotted by week of month
data.groupby([data.weekOfMonth]).size().plot(kind='barh',  color='yellowgreen')
plt.ylabel('Week Of The Month')
plt.xlabel('Number Of Crimes')
plt.title('Number Of Crimes By Week Of The Month')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_34_0.png)
    



```python
print(f'Percentage Of Crime Per Week Of Month:\n\n{data.weekOfMonth.value_counts(normalize=True)}')

#data.weekOfMonth.value_counts(normalize=True)
```

    Percentage Of Crime Per Week Of Month:
    
    1    0.232995
    4    0.230346
    3    0.230313
    2    0.226241
    5    0.080105
    Name: weekOfMonth, dtype: float64
    

Based on the insights gleaned from the preceding graph and chart, it becomes evident that the specific week within a month may not significantly impact crime rates. Notably, the observation that the fifth week records fewer incidents can be attributed to its shorter duration.


```python
# week of year
plt.figure(figsize=(8,10))
data.groupby([data.weekOfYear]).size().sort_values().plot(kind='barh', color='yellowgreen')
plt.ylabel('weeks of the year')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by month of the year')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_37_0.png)
    


The graph above serves as an additional perspective, reaffirming the correlation between warmer months and their respective weeks, which consistently exhibit higher crime rates when contrasted with the colder months.


```python
# number of crimes per week
plt.figure(figsize=(11,5))
data.resample('W').size().plot(legend=False,color='yellowgreen')
plt.title('Number Of Crimes Per Week (2009 - 2023)')
plt.xlabel('Weeks')
plt.ylabel('Number Of Crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_39_0.png)
    


The graph displayed above offers yet another illustrative trendline, dissected on a weekly basis, spanning from 2009 to the present day.<br><br>

Now, let's delve into the substantial decline at the outset of 2023 and investigate whether it can indeed be attributed to the blizzard event.


```python
# grab the dec 2022 and jan 2023 data only
blizzard2022 = data[(data['Year'] == 2022) & (data['Month'] == 12)]
blizzard2023 = data[(data['Year'] == 2023) & (data['Month'] == 1)]
```


```python
# concatenate the two DataFrames
blizzard_combined = pd.concat([blizzard2022, blizzard2023], ignore_index=True)
#blizzard_combined
```


```python
# convert the 'incident_datetime' column to a datetime type if it's not already
blizzard_combined['incident_datetime'] = pd.to_datetime(blizzard_combined['incident_datetime'])

# set the 'incident_datetime' column as the index
blizzard_combined.set_index('incident_datetime', inplace=True)

# plot the number of crimes using resample
plt.figure(figsize=(11, 5))
blizzard_combined.resample('W').size().plot(legend=False, color='yellowgreen')
plt.title('Number Of Crimes Around the Blizzard (Dec 2022-Jan 2023)')
plt.xlabel('Weeks')
plt.ylabel('Number Of Crimes')
plt.show()

```


    
![png](\img\posts\Buffalo-Crime\output_43_0.png)
    


My initial hypothesis has been disproven; the decrease in crime can be attributed to February's weather conditions rather than the blizzard event.

### Neighborhood Graphs
<a id='neighGraphs'></a>


```python
# week of year per neigborhood

listOfNeighborhoods = list(data['neighborhood_1'].unique())

for neighborhood in listOfNeighborhoods:
    df = data[data['neighborhood_1'] == neighborhood]

    # Check if df is empty before resampling and plotting
    if not df.empty:
        plt.figure(figsize=(11, 5))
        df.resample('W').size().plot(legend=False, color='yellowgreen')
        plt.title('Number Of Crimes Per Week (2009 - 2023) For Neighborhood {}'.format(neighborhood))
        plt.xlabel('Weeks')
        plt.ylabel('Number Of Crimes')
        plt.show()
    else:
        print(f"No data for neighborhood {neighborhood}")

```


    
![png](\img\posts\Buffalo-Crime\output_46_0.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_1.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_2.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_3.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_4.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_5.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_6.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_7.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_8.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_9.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_10.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_11.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_12.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_13.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_14.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_15.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_16.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_17.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_18.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_19.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_20.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_21.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_22.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_23.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_24.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_25.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_26.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_27.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_28.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_29.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_30.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_31.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_32.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_33.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_34.png)
    



    
![png](\img\posts\Buffalo-Crime\output_46_35.png)
    


    No data for neighborhood nan
    


```python
# bar chart of crimes
plt.figure(figsize=(8,10))
data.groupby([data['incident_type_primary']]).size().sort_values(ascending=True).plot(kind='barh', color='yellowgreen')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_47_0.png)
    



```python
# chart of crimes
print(f'Percentage of Crimes by types:\n\n{data.incident_type_primary.value_counts(normalize=True)}')
```

    Percentage of Crimes by types:
    
    LARCENY/THEFT               0.438012
    ASSAULT                     0.203365
    BURGLARY                    0.180000
    UUV                         0.086375
    ROBBERY                     0.062623
    RAPE                        0.009090
    SEXUAL ABUSE                0.008685
    THEFT OF SERVICES           0.006916
    MURDER                      0.003216
    Assault                     0.000480
    Breaking & Entering         0.000346
    AGGR ASSAULT                0.000321
    CRIM NEGLIGENT HOMICIDE     0.000271
    Theft                       0.000138
    MANSLAUGHTER                0.000046
    AGG ASSAULT ON P/OFFICER    0.000042
    Robbery                     0.000025
    Sexual Assault              0.000021
    Theft of Vehicle            0.000013
    Other Sexual Offense        0.000008
    Homicide                    0.000004
    SODOMY                      0.000004
    Name: incident_type_primary, dtype: float64
    

### Remove Outlier Crimes / Maybe Label As Others Later


```python
print('Current rows:', data.shape[0])
data['incident_type_primary'] = data['incident_type_primary'].astype(str)
data = data[(data['incident_type_primary'] != 'SODOMY') &
            (data['incident_type_primary'] != 'Homicide') &
            (data['incident_type_primary'] != 'Other Sexual Offense') &
            (data['incident_type_primary'] != 'Theft of Vehicle') &
            (data['incident_type_primary'] != 'Sexual Assault') &
            (data['incident_type_primary'] != 'Robbery') &
            (data['incident_type_primary'] != 'AGG ASSAULT ON P/OFFICER') &
            (data['incident_type_primary'] != 'Theft') &
            (data['incident_type_primary'] != 'CRIM NEGLIGENT HOMICIDE') &
            (data['incident_type_primary'] != 'AGGR ASSAULT') &
            (data['incident_type_primary'] != 'Breaking & Entering') &
            (data['incident_type_primary'] != 'Assault') &
            (data['incident_type_primary'] != 'MANSLAUGHTER')]

print('Rows after removing primary type outliers:', data.shape[0])
```

    Current rows: 239722
    Rows after removing primary type outliers: 239310
    


```python
plt.figure(figsize=(8,10))
data.groupby([data['neighborhood_1']]).size().sort_values(ascending=True)[-70:].plot(kind='barh', color='yellowgreen')
plt.title('Number of crimes by locations')
plt.ylabel('neighborhood_1')
plt.xlabel('Number of crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_51_0.png)
    



```python
# Show 2022 vs 2009
# possible show ratio
```


```python
# grab 2009 data and 2022 data to compare crime charts
data2009 = data[(data['Year'] == 2009)]
data2022 = data[(data['Year'] == 2022)]

```


```python
# 2009 crimes by location

plt.figure(figsize=(8,10))
data2009.groupby([data2009['neighborhood_1']]).size().sort_values(ascending=True)[-70:].plot(kind='barh', color='yellowgreen')
plt.title('Number Of Crimes By Locations In 2009')
plt.ylabel('Neighborhood')
plt.xlabel('Number Of Crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_54_0.png)
    



```python
# 2022 crimes by location

plt.figure(figsize=(8,10))
data2022.groupby([data2022['neighborhood_1']]).size().sort_values(ascending=True)[-70:].plot(kind='barh', color='yellowgreen')
plt.title('Number Of Crimes By Locations In 2022')
plt.ylabel('Neighborhood')
plt.xlabel('Number of crimes')
plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_55_0.png)
    



```python
import plotly.graph_objects as go

# Filter data for 2009 and 2022
data2009 = data[data['Year'] == 2009]
data2022 = data[data['Year'] == 2022]

# Create subplots
fig = go.Figure()

# Subplot 1: 2009 crimes by location
fig.add_trace(go.Bar(
    y=data2009.groupby([data2009['neighborhood_1']]).size().sort_values(ascending=True)[-70:].index,
    x=data2009.groupby([data2009['neighborhood_1']]).size().sort_values(ascending=True)[-70:],
    orientation='h',
    marker=dict(color='deepskyblue'),
    name='2009'
))

# Subplot 2: 2022 crimes by location
fig.add_trace(go.Bar(
    y=data2022.groupby([data2022['neighborhood_1']]).size().sort_values(ascending=True)[-70:].index,
    x=data2022.groupby([data2022['neighborhood_1']]).size().sort_values(ascending=True)[-70:],
    orientation='h',
    marker=dict(color='orchid'),
    name='2022'
))

# Update layout for dark theme
fig.update_layout(
    title='Number of Crimes by Locations (2009 and 2022)',
    yaxis_title='Neighborhood',
    xaxis_title='Number of Crimes',
    barmode='group',
    width=1000,
    height=500,
    plot_bgcolor='black',  # Set background color to black
    paper_bgcolor='black',  # Set paper color to black
    font=dict(color='white')  # Set text color to white
)

# Show plot
fig.show()

```




### Buffalo Crime Geospatial Graphs
<a id='graphs'></a>


```python
# make new data frame with map data
buffalo_map = data[['neighborhood_1','incident_type_primary', 'latitude', 'longitude',  'incident_datetime', 'hour_of_day']]
```


```python
buffalo_map['latitude'] = pd.to_numeric(buffalo_map['latitude'])
buffalo_map['longitude'] = pd.to_numeric(buffalo_map['longitude'])
buffalo_map['hour_of_day'] = pd.to_numeric(buffalo_map['hour_of_day'])

```


```python
buffalo_map['incident_datetime'] = pd.to_datetime(buffalo_map['incident_datetime'])
buffalo_map['Year'] = buffalo_map['incident_datetime'].dt.year
buffalo_map['Month'] = buffalo_map['incident_datetime'].dt.month
```


```python
buffalo_map.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 239310 entries, 2009-01-10 12:19:00 to 2023-09-11 11:12:45
    Data columns (total 8 columns):
     #   Column                 Non-Null Count   Dtype         
    ---  ------                 --------------   -----         
     0   neighborhood_1         237303 non-null  object        
     1   incident_type_primary  239310 non-null  object        
     2   latitude               234651 non-null  float64       
     3   longitude              234651 non-null  float64       
     4   incident_datetime      239310 non-null  datetime64[ns]
     5   hour_of_day            239310 non-null  int64         
     6   Year                   239310 non-null  int64         
     7   Month                  239310 non-null  int64         
    dtypes: datetime64[ns](1), float64(2), int64(3), object(2)
    memory usage: 16.4+ MB
    


```python
# buffalo lat and lon mean
mean_latitude = buffalo_map['latitude'].mean()
print(mean_latitude)
mean_longitude = buffalo_map['longitude'].mean()
print(mean_longitude)
```

    42.911893612215586
    -78.84912654111854
    


```python
# remove outliers that are not in the city limits
buffalo_map = buffalo_map[(buffalo_map['longitude'] < -78.80)]
buffalo_map = buffalo_map[(buffalo_map['latitude'] < 43)]
#buffalo_map.sort_values('Latitude', ascending=False)
```


```python
#ignoring unknown neighborhoods
buffalo_map = buffalo_map[buffalo_map['neighborhood_1'] != 'UNKNOWN']
```


```python
# all crimes per neighborhood
sns.lmplot(x = 'longitude',
           y = 'latitude',
           data=buffalo_map[:],
           fit_reg=False,
           hue="neighborhood_1",
           palette='Dark2',
           height=10,
           ci=2,
           scatter_kws={"marker": "D",
                        "s": 10})
ax = plt.gca()
ax.set_title("All Crime Distribution Per Neighborhood")
```




    Text(0.5, 1.0, 'All Crime Distribution Per Neighborhood')




    
![png](\img\posts\Buffalo-Crime\output_65_1.png)
    



```python
# show most common crime per neighborhood
# preprocessing to group most common crime per neighborhood
sdf = buffalo_map.groupby(['neighborhood_1', 'incident_type_primary']).size().reset_index(name='counts')
idx = sdf.groupby(['neighborhood_1'])['counts'].transform(max) == sdf['counts']
sdf = sdf[idx]
other = buffalo_map.groupby('neighborhood_1')[['longitude', 'latitude']].mean()

sdf = sdf.set_index('neighborhood_1').join(other)
sdf = sdf.reset_index().sort_values("counts",ascending=False)
#sns.lmplot(x='longitude', y='latitude',height=10, hue=incident_type_primary', data=sdf,scatter_kws={"s": sdf['counts'].apply(lambda x: x/100.0)}, fit_reg=False)


#  scatter plot
sns.lmplot(x='longitude', y='latitude', height=10, hue='incident_type_primary', data=sdf, fit_reg=False, scatter=True)

# Annotation code...
for r in sdf.reset_index().to_numpy():
    neighborhood_ = "neighborhood_1: {0}, Count: {1}".format(r[1], int(r[3]))

    #neighborhood_ = "neighborhood_1 {0}, Count : {1}".format(int(r[1]), int(r[3]))
    x = r[4]
    y = r[5]
    plt.annotate(
        neighborhood_,
        xy=(x, y), xytext=(-15, 15),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='grey', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.show()
```


    
![png](\img\posts\Buffalo-Crime\output_66_0.png)
    


The graph above distinctly highlights that, across Buffalo neighborhoods, the prevailing type of crime is predominantly larceny or theft. However, a notable exception to this pattern is the Delavan Grider neighborhood, where the dominant crime category is assault.


```python
# buffalo lat and lon mean
mean_latitude = buffalo_map['latitude'].mean()
print(mean_latitude)
mean_longitude = buffalo_map['longitude'].mean()
print(mean_longitude)
```

    42.91184928528912
    -78.84964614694492
    


```python
# interactive map of buffalo showing crime amount per neighborhood

sdf = buffalo_map.groupby(['neighborhood_1', 'incident_type_primary']).size().reset_index(name='counts')
idx = sdf.groupby(['neighborhood_1'])['counts'].transform(max) == sdf['counts']
sdf = sdf[idx]
other = buffalo_map.groupby('neighborhood_1')[['longitude', 'latitude']].mean()

sdf = sdf.set_index('neighborhood_1').join(other)
sdf = sdf.reset_index().sort_values("counts", ascending=False)

# Create a Folium map centered around Buffalo, New York
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=12)

# Create the scatter plot
for _, row in sdf.iterrows():
    district = f"neighborhood_1: {row['neighborhood_1']}, Count: {int(row['counts'])}"
    x = row['latitude']
    y = row['longitude']

    # Add a marker for each point on the map
    folium.Marker([x, y], tooltip=district).add_to(m)

m
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_424ede9f560bda9c662090b531953446 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_424ede9f560bda9c662090b531953446&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_424ede9f560bda9c662090b531953446 = L.map(
                &quot;map_424ede9f560bda9c662090b531953446&quot;,
                {
                    center: [42.91184928528912, -78.84964614694492],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_eef7469fb8cbd83777890dee46be13eb = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            var marker_59ab636abc61238091c342da6c0e4985 = L.marker(
                [42.95101160834455, -78.86595869784657],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_59ab636abc61238091c342da6c0e4985.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: North Park, Count: 7978
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_d74a9b205b546a95808970ecba083642 = L.marker(
                [42.886341271200955, -78.87522888470104],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_d74a9b205b546a95808970ecba083642.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Central, Count: 6567
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_58b24a71304a88f0d571a93e811fcf5c = L.marker(
                [42.939940428272415, -78.8121809692481],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_58b24a71304a88f0d571a93e811fcf5c.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Kensington-Bailey, Count: 5311
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_4f58411f15d12dc44eb61f6540fe97ea = L.marker(
                [42.92197972273949, -78.87666842540865],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_4f58411f15d12dc44eb61f6540fe97ea.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Elmwood Bidwell, Count: 5294
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_f2aa9ce748b7b2f7a99cc209fc2f4d49 = L.marker(
                [42.909727724913495, -78.87599005190312],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_f2aa9ce748b7b2f7a99cc209fc2f4d49.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Elmwood Bryant, Count: 5135
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_48fc787b0acc567a799e38e7c3375b95 = L.marker(
                [42.8921423858439, -78.83872317612656],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_48fc787b0acc567a799e38e7c3375b95.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Broadway Fillmore, Count: 4578
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_a92811e8383b901e21a6ad3706c68574 = L.marker(
                [42.92101586408641, -78.89218721872187],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_a92811e8383b901e21a6ad3706c68574.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Upper West Side, Count: 3752
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_2cf55ab07e534c4dea9e045fa49949ac = L.marker(
                [42.94831392294221, -78.82304553415061],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_2cf55ab07e534c4dea9e045fa49949ac.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: University Heights, Count: 3677
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_be72b4c68070eb2ead3e2f87d2e0ac71 = L.marker(
                [42.90645529327611, -78.89158416785885],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_be72b4c68070eb2ead3e2f87d2e0ac71.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: West Side, Count: 3371
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_62340bbfa0917d8ac48415837c2c127e = L.marker(
                [42.95433849897541, -78.90204969262295],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_62340bbfa0917d8ac48415837c2c127e.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Riverside, Count: 3252
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_ec48b17b944b17d57d4285acb213b723 = L.marker(
                [42.906462549277265, -78.81746752393467],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_ec48b17b944b17d57d4285acb213b723.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Genesee-Moselle, Count: 3248
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_9b960b2523cd7600dd58baa15c8dd616 = L.marker(
                [42.8932316392701, -78.88431910722365],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_9b960b2523cd7600dd58baa15c8dd616.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Lower West Side, Count: 3243
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_2e9dd2d65526480243d7c9a7979629fc = L.marker(
                [42.89896108878334, -78.87564852027768],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_2e9dd2d65526480243d7c9a7979629fc.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Allentown, Count: 3169
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_f16700985f2f8182e96ae474afd2a134 = L.marker(
                [42.89046777188329, -78.81075079575596],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_f16700985f2f8182e96ae474afd2a134.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Lovejoy, Count: 3035
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_72325dd51c26a4055a14cbd5d7043f09 = L.marker(
                [42.850342291838416, -78.82823792250618],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_72325dd51c26a4055a14cbd5d7043f09.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Hopkins-Tifft, Count: 2954
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_25ce395b356ddb0744abe091361f7728 = L.marker(
                [42.927916816708674, -78.80935734005523],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_25ce395b356ddb0744abe091361f7728.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Kenfield, Count: 2792
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_78dc3d00cdf0b3068ba93b346308c733 = L.marker(
                [42.9382101784534, -78.85788984357788],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_78dc3d00cdf0b3068ba93b346308c733.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Parkside, Count: 2787
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_8fe8846ba05cb00ccb4ddbc0a2e6cf94 = L.marker(
                [42.91095332976732, -78.85516394757957],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_8fe8846ba05cb00ccb4ddbc0a2e6cf94.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Masten Park, Count: 2782
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_76e5e298b78848b9bb2e4b53c6f9c092 = L.marker(
                [42.915845592301736, -78.80674659882756],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_76e5e298b78848b9bb2e4b53c6f9c092.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Schiller Park, Count: 2704
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_766027da519e1eea12e0f69c829307e2 = L.marker(
                [42.84436028396766, -78.81627607966871],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_766027da519e1eea12e0f69c829307e2.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: South Park, Count: 2555
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_630c927057f126be829f005427f30dc2 = L.marker(
                [42.85602868174297, -78.80937451737452],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_630c927057f126be829f005427f30dc2.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Seneca-Cazenovia, Count: 2356
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_19f153af62421c13388d520748b30c64 = L.marker(
                [42.95078811659193, -78.88553228699551],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_19f153af62421c13388d520748b30c64.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: West Hertel, Count: 2310
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_e3a481f82c3d116df5bc851bdd17e45f = L.marker(
                [42.89928937230909, -78.86201722184455],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_e3a481f82c3d116df5bc851bdd17e45f.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Fruit Belt, Count: 2016
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_9eb07417e23e0115f0a453f6b50253f4 = L.marker(
                [42.933806248877715, -78.83540061052254],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_9eb07417e23e0115f0a453f6b50253f4.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Fillmore-Leroy, Count: 1970
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_8bfe2530d253ac1a38de71bb31b4789f = L.marker(
                [42.92092078239609, -78.82950676446617],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_8bfe2530d253ac1a38de71bb31b4789f.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Delavan Grider, Count: 1907
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_fa926fd3f43b9ee7c58dcf82447292d3 = L.marker(
                [42.94015955521076, -78.90260693043703],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_fa926fd3f43b9ee7c58dcf82447292d3.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Black Rock, Count: 1650
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_fe280bdbbfb255eaef51affe311bf7e0 = L.marker(
                [42.910315126903555, -78.83467208121827],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_fe280bdbbfb255eaef51affe311bf7e0.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: MLK Park, Count: 1646
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_31024f59edfe2eba9c09e8a4745b5523 = L.marker(
                [42.94008435960591, -78.88795104679804],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_31024f59edfe2eba9c09e8a4745b5523.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Grant-Amherst, Count: 1363
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_66a7bd0c51e26489cbf1aa9397fe9a20 = L.marker(
                [42.94666409408306, -78.83691069459758],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_66a7bd0c51e26489cbf1aa9397fe9a20.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Central Park, Count: 1348
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_20a7c20ea88f2f8197f6b62e6df656db = L.marker(
                [42.87464917632703, -78.86075015253203],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_20a7c20ea88f2f8197f6b62e6df656db.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Ellicott, Count: 1340
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_72d2ee6733f44187e47c178c27b1b954 = L.marker(
                [42.88600026968716, -78.85876429341964],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_72d2ee6733f44187e47c178c27b1b954.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Pratt-Willert, Count: 1336
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_bd2b7951734de4080d12bf39b47bafb1 = L.marker(
                [42.920222422216504, -78.84842170223708],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_bd2b7951734de4080d12bf39b47bafb1.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Hamlin Park, Count: 1324
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_a12b445820cff222eaa72f35252d0bb8 = L.marker(
                [42.8717363048656, -78.80812929567881],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_a12b445820cff222eaa72f35252d0bb8.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Kaisertown, Count: 1181
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_b22d8ed280210a0d262bda45d8544c18 = L.marker(
                [42.8712664756447, -78.83104461727385],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_b22d8ed280210a0d262bda45d8544c18.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: Seneca Babcock, Count: 976
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );


            var marker_ba6fe14d6a48690a5366fc5ed9cd318b = L.marker(
                [42.868005232177886, -78.8587083060824],
                {}
            ).addTo(map_424ede9f560bda9c662090b531953446);


            marker_ba6fe14d6a48690a5366fc5ed9cd318b.bindTooltip(
                `&lt;div&gt;
                     neighborhood_1: First Ward, Count: 586
                 &lt;/div&gt;`,
                {&quot;sticky&quot;: true}
            );

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# Buffalo Crime Heatmap
<a id='heatmap'></a>


```python
"""
This function generates a folium map with Buffalo location and given zoom value.
"""

def generateBaseMap(default_location=[mean_latitude, mean_longitude], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
```


```python
buffalo_map.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 231842 entries, 2009-01-10 12:19:00 to 2023-09-11 11:12:45
    Data columns (total 8 columns):
     #   Column                 Non-Null Count   Dtype         
    ---  ------                 --------------   -----         
     0   neighborhood_1         231842 non-null  object        
     1   incident_type_primary  231842 non-null  object        
     2   latitude               231842 non-null  float64       
     3   longitude              231842 non-null  float64       
     4   incident_datetime      231842 non-null  datetime64[ns]
     5   hour_of_day            231842 non-null  int64         
     6   Year                   231842 non-null  int64         
     7   Month                  231842 non-null  int64         
    dtypes: datetime64[ns](1), float64(2), int64(3), object(2)
    memory usage: 15.9+ MB
    


```python
buffalo_map.head()
```





  <div id="df-c8df9fa6-031a-42d4-a85d-b8cb20bd7b3f" class="colab-df-container">
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
      <th>neighborhood_1</th>
      <th>incident_type_primary</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>incident_datetime</th>
      <th>hour_of_day</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
    <tr>
      <th>incident_datetime</th>
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
      <th>2009-01-10 12:19:00</th>
      <td>North Park</td>
      <td>BURGLARY</td>
      <td>42.955</td>
      <td>-78.857</td>
      <td>2009-01-10 12:19:00</td>
      <td>12</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2009-01-10 12:21:00</th>
      <td>Kenfield</td>
      <td>BURGLARY</td>
      <td>42.928</td>
      <td>-78.818</td>
      <td>2009-01-10 12:21:00</td>
      <td>12</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2009-01-10 12:27:00</th>
      <td>Masten Park</td>
      <td>UUV</td>
      <td>42.917</td>
      <td>-78.863</td>
      <td>2009-01-10 12:27:00</td>
      <td>12</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2009-01-10 12:30:00</th>
      <td>Masten Park</td>
      <td>ASSAULT</td>
      <td>42.915</td>
      <td>-78.854</td>
      <td>2009-01-10 12:30:00</td>
      <td>12</td>
      <td>2009</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2009-01-10 12:30:00</th>
      <td>MLK Park</td>
      <td>BURGLARY</td>
      <td>42.910</td>
      <td>-78.835</td>
      <td>2009-01-10 12:30:00</td>
      <td>12</td>
      <td>2009</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c8df9fa6-031a-42d4-a85d-b8cb20bd7b3f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
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

    .colab-df-buttons div {
      margin-bottom: 4px;
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
        document.querySelector('#df-c8df9fa6-031a-42d4-a85d-b8cb20bd7b3f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c8df9fa6-031a-42d4-a85d-b8cb20bd7b3f');
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


<div id="df-5da15c2d-52c2-4b3c-a9dc-242e67e4a93d">
  <button class="colab-df-quickchart" onclick="quickchart('df-5da15c2d-52c2-4b3c-a9dc-242e67e4a93d')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5da15c2d-52c2-4b3c-a9dc-242e67e4a93d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
# make night & day column
buffalo_map['dayType'] = buffalo_map['hour_of_day'].apply(lambda x: 'Day' if (x >= 6 and x < 18) else 'Night')
```


```python
# grab summer 2023 data
summer_2023 = buffalo_map.loc[(buffalo_map['Year'] == 2023) & (buffalo_map['Month'] > 5) & (buffalo_map['Month'] < 9)]
# grab summer 2009 data
summer_2009 = buffalo_map.loc[(buffalo_map['Year'] == 2009) & (buffalo_map['Month'] > 5) & (buffalo_map['Month'] < 9)]
```


```python
print(type(summer_2023))
print(type(summer_2009))
print(summer_2023.shape)
print(summer_2009.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    (2835, 9)
    (5811, 9)
    


```python
# make day and night data dfor summer 2023 & summer 2009

summer_2023_day = summer_2023[summer_2023['dayType'] == 'Day']
summer_2023_night = summer_2023[summer_2023['dayType'] == 'Night']
summer_2009_day = summer_2009[summer_2009['dayType'] == 'Day']
summer_2009_night = summer_2009[summer_2009['dayType'] == 'Night']
```


```python
# Heatmap --> 2023 Summer Days
base_map = generateBaseMap()
HeatMap(data=summer_2023_day[['latitude', 'longitude']].\
        groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=12).add_to(base_map)

base_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_f9fb3bcf0425c4d19ba0072c0598bf71 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js&quot;&gt;&lt;/script&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_f9fb3bcf0425c4d19ba0072c0598bf71&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_f9fb3bcf0425c4d19ba0072c0598bf71 = L.map(
                &quot;map_f9fb3bcf0425c4d19ba0072c0598bf71&quot;,
                {
                    center: [42.91184928528912, -78.84964614694492],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );
            L.control.scale().addTo(map_f9fb3bcf0425c4d19ba0072c0598bf71);





            var tile_layer_d3ee093dc9f0720fd963b0aa68541ebd = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_f9fb3bcf0425c4d19ba0072c0598bf71);


            var heat_map_5ce385514e762fd26fb5bba736b9809d = L.heatLayer(
                [[42.832, -78.817], [42.833, -78.807], [42.833, -78.806], [42.834, -78.824], [42.834, -78.816], [42.835, -78.807], [42.836, -78.802], [42.837, -78.827], [42.837, -78.823], [42.837, -78.812], [42.838, -78.819], [42.838, -78.811], [42.839, -78.824], [42.839, -78.819], [42.839, -78.817], [42.839, -78.808], [42.841, -78.834], [42.842, -78.822], [42.842, -78.81], [42.842, -78.809], [42.844, -78.818], [42.845, -78.818], [42.846, -78.824], [42.846, -78.823], [42.847, -78.827], [42.847, -78.824], [42.847, -78.82], [42.847, -78.801], [42.848, -78.824], [42.848, -78.812], [42.849, -78.827], [42.85, -78.825], [42.85, -78.818], [42.85, -78.811], [42.851, -78.823], [42.851, -78.822], [42.851, -78.817], [42.851, -78.81], [42.851, -78.806], [42.851, -78.804], [42.852, -78.826], [42.852, -78.823], [42.852, -78.822], [42.852, -78.821], [42.852, -78.805], [42.853, -78.831], [42.853, -78.82], [42.853, -78.803], [42.853, -78.802], [42.854, -78.83], [42.854, -78.828], [42.854, -78.825], [42.854, -78.814], [42.854, -78.808], [42.854, -78.807], [42.855, -78.833], [42.855, -78.832], [42.855, -78.831], [42.855, -78.829], [42.855, -78.806], [42.856, -78.827], [42.856, -78.823], [42.856, -78.81], [42.856, -78.808], [42.857, -78.833], [42.857, -78.832], [42.857, -78.83], [42.857, -78.821], [42.857, -78.819], [42.857, -78.806], [42.858, -78.831], [42.858, -78.813], [42.858, -78.805], [42.859, -78.828], [42.859, -78.807], [42.86, -78.871], [42.86, -78.825], [42.86, -78.813], [42.861, -78.814], [42.861, -78.801], [42.862, -78.864], [42.862, -78.82], [42.862, -78.815], [42.862, -78.812], [42.862, -78.811], [42.863, -78.819], [42.863, -78.815], [42.864, -78.816], [42.865, -78.859], [42.865, -78.827], [42.865, -78.823], [42.866, -78.826], [42.868, -78.81], [42.868, -78.801], [42.869, -78.829], [42.869, -78.808], [42.87, -78.87], [42.87, -78.868], [42.87, -78.855], [42.87, -78.85], [42.87, -78.832], [42.87, -78.831], [42.87, -78.807], [42.871, -78.883], [42.871, -78.845], [42.871, -78.834], [42.871, -78.833], [42.871, -78.815], [42.871, -78.807], [42.872, -78.868], [42.872, -78.864], [42.872, -78.853], [42.872, -78.845], [42.872, -78.815], [42.872, -78.811], [42.872, -78.806], [42.872, -78.805], [42.872, -78.801], [42.873, -78.822], [42.873, -78.82], [42.873, -78.811], [42.873, -78.806], [42.873, -78.801], [42.874, -78.872], [42.874, -78.871], [42.874, -78.866], [42.874, -78.864], [42.874, -78.848], [42.874, -78.824], [42.874, -78.802], [42.875, -78.874], [42.875, -78.868], [42.875, -78.851], [42.875, -78.85], [42.875, -78.826], [42.875, -78.806], [42.876, -78.868], [42.876, -78.852], [42.876, -78.85], [42.876, -78.806], [42.877, -78.849], [42.877, -78.841], [42.877, -78.833], [42.878, -78.882], [42.878, -78.881], [42.878, -78.875], [42.878, -78.87], [42.878, -78.837], [42.878, -78.824], [42.878, -78.815], [42.878, -78.814], [42.878, -78.801], [42.879, -78.884], [42.879, -78.882], [42.879, -78.88], [42.879, -78.878], [42.879, -78.874], [42.879, -78.861], [42.879, -78.859], [42.879, -78.842], [42.879, -78.836], [42.879, -78.835], [42.879, -78.802], [42.88, -78.88], [42.88, -78.876], [42.88, -78.869], [42.88, -78.859], [42.88, -78.834], [42.88, -78.814], [42.88, -78.802], [42.881, -78.879], [42.881, -78.877], [42.881, -78.868], [42.881, -78.859], [42.881, -78.801], [42.882, -78.881], [42.882, -78.877], [42.882, -78.876], [42.882, -78.868], [42.882, -78.866], [42.882, -78.856], [42.883, -78.878], [42.883, -78.874], [42.883, -78.87], [42.883, -78.86], [42.883, -78.859], [42.883, -78.853], [42.883, -78.851], [42.883, -78.844], [42.884, -78.88], [42.884, -78.873], [42.884, -78.872], [42.884, -78.854], [42.884, -78.833], [42.884, -78.829], [42.885, -78.888], [42.885, -78.877], [42.885, -78.876], [42.885, -78.873], [42.885, -78.872], [42.885, -78.87], [42.885, -78.86], [42.885, -78.855], [42.885, -78.852], [42.885, -78.838], [42.885, -78.832], [42.885, -78.827], [42.885, -78.816], [42.885, -78.815], [42.885, -78.814], [42.885, -78.811], [42.885, -78.807], [42.885, -78.804], [42.886, -78.877], [42.886, -78.875], [42.886, -78.874], [42.886, -78.873], [42.886, -78.871], [42.886, -78.87], [42.886, -78.853], [42.886, -78.851], [42.886, -78.814], [42.886, -78.805], [42.887, -78.882], [42.887, -78.88], [42.887, -78.878], [42.887, -78.875], [42.887, -78.874], [42.887, -78.873], [42.887, -78.872], [42.887, -78.854], [42.887, -78.851], [42.887, -78.837], [42.887, -78.814], [42.887, -78.812], [42.888, -78.882], [42.888, -78.878], [42.888, -78.876], [42.888, -78.874], [42.888, -78.872], [42.888, -78.871], [42.888, -78.861], [42.888, -78.809], [42.888, -78.805], [42.889, -78.883], [42.889, -78.882], [42.889, -78.877], [42.889, -78.876], [42.889, -78.875], [42.889, -78.872], [42.889, -78.871], [42.889, -78.87], [42.889, -78.865], [42.889, -78.86], [42.889, -78.854], [42.889, -78.839], [42.889, -78.835], [42.889, -78.816], [42.889, -78.815], [42.889, -78.811], [42.889, -78.808], [42.889, -78.807], [42.889, -78.806], [42.89, -78.887], [42.89, -78.878], [42.89, -78.876], [42.89, -78.875], [42.89, -78.873], [42.89, -78.868], [42.89, -78.852], [42.89, -78.845], [42.89, -78.84], [42.89, -78.832], [42.89, -78.806], [42.89, -78.803], [42.891, -78.885], [42.891, -78.878], [42.891, -78.877], [42.891, -78.876], [42.891, -78.875], [42.891, -78.873], [42.891, -78.866], [42.891, -78.859], [42.891, -78.857], [42.891, -78.853], [42.891, -78.852], [42.891, -78.851], [42.891, -78.847], [42.891, -78.833], [42.891, -78.831], [42.891, -78.811], [42.891, -78.808], [42.891, -78.807], [42.891, -78.805], [42.892, -78.884], [42.892, -78.883], [42.892, -78.882], [42.892, -78.881], [42.892, -78.879], [42.892, -78.878], [42.892, -78.876], [42.892, -78.87], [42.892, -78.861], [42.892, -78.853], [42.892, -78.851], [42.892, -78.848], [42.892, -78.84], [42.892, -78.837], [42.892, -78.835], [42.892, -78.834], [42.892, -78.814], [42.892, -78.813], [42.893, -78.881], [42.893, -78.878], [42.893, -78.874], [42.893, -78.873], [42.893, -78.87], [42.893, -78.868], [42.893, -78.844], [42.893, -78.841], [42.893, -78.84], [42.893, -78.831], [42.893, -78.821], [42.893, -78.82], [42.893, -78.814], [42.893, -78.809], [42.893, -78.807], [42.893, -78.805], [42.894, -78.886], [42.894, -78.88], [42.894, -78.877], [42.894, -78.876], [42.894, -78.874], [42.894, -78.866], [42.894, -78.849], [42.894, -78.837], [42.894, -78.836], [42.894, -78.835], [42.894, -78.834], [42.894, -78.833], [42.894, -78.832], [42.894, -78.831], [42.894, -78.821], [42.894, -78.805], [42.894, -78.804], [42.895, -78.89], [42.895, -78.884], [42.895, -78.882], [42.895, -78.881], [42.895, -78.878], [42.895, -78.876], [42.895, -78.871], [42.895, -78.865], [42.895, -78.838], [42.895, -78.837], [42.895, -78.833], [42.895, -78.822], [42.896, -78.886], [42.896, -78.885], [42.896, -78.882], [42.896, -78.876], [42.896, -78.875], [42.896, -78.872], [42.896, -78.869], [42.896, -78.866], [42.896, -78.865], [42.896, -78.86], [42.896, -78.835], [42.896, -78.834], [42.896, -78.832], [42.896, -78.824], [42.896, -78.822], [42.896, -78.82], [42.896, -78.819], [42.896, -78.803], [42.897, -78.9], [42.897, -78.888], [42.897, -78.882], [42.897, -78.877], [42.897, -78.876], [42.897, -78.875], [42.897, -78.872], [42.897, -78.871], [42.897, -78.869], [42.897, -78.868], [42.897, -78.866], [42.897, -78.863], [42.897, -78.851], [42.897, -78.85], [42.897, -78.842], [42.897, -78.84], [42.897, -78.839], [42.897, -78.838], [42.897, -78.837], [42.897, -78.836], [42.897, -78.835], [42.897, -78.831], [42.897, -78.823], [42.897, -78.821], [42.897, -78.817], [42.897, -78.815], [42.897, -78.813], [42.897, -78.809], [42.898, -78.893], [42.898, -78.89], [42.898, -78.879], [42.898, -78.876], [42.898, -78.874], [42.898, -78.872], [42.898, -78.871], [42.898, -78.869], [42.898, -78.865], [42.898, -78.863], [42.898, -78.862], [42.898, -78.836], [42.898, -78.807], [42.898, -78.802], [42.899, -78.892], [42.899, -78.887], [42.899, -78.885], [42.899, -78.878], [42.899, -78.877], [42.899, -78.874], [42.899, -78.872], [42.899, -78.87], [42.899, -78.854], [42.899, -78.846], [42.899, -78.841], [42.899, -78.84], [42.899, -78.834], [42.899, -78.832], [42.899, -78.822], [42.899, -78.814], [42.9, -78.883], [42.9, -78.881], [42.9, -78.88], [42.9, -78.879], [42.9, -78.878], [42.9, -78.877], [42.9, -78.874], [42.9, -78.873], [42.9, -78.87], [42.9, -78.867], [42.9, -78.854], [42.9, -78.851], [42.9, -78.827], [42.9, -78.815], [42.901, -78.885], [42.901, -78.884], [42.901, -78.878], [42.901, -78.877], [42.901, -78.874], [42.901, -78.871], [42.901, -78.865], [42.901, -78.862], [42.901, -78.84], [42.901, -78.838], [42.901, -78.822], [42.902, -78.891], [42.902, -78.889], [42.902, -78.888], [42.902, -78.883], [42.902, -78.881], [42.902, -78.879], [42.902, -78.878], [42.902, -78.877], [42.902, -78.873], [42.902, -78.872], [42.902, -78.871], [42.902, -78.863], [42.902, -78.845], [42.902, -78.839], [42.902, -78.821], [42.902, -78.817], [42.902, -78.814], [42.903, -78.892], [42.903, -78.89], [42.903, -78.883], [42.903, -78.876], [42.903, -78.872], [42.903, -78.869], [42.903, -78.847], [42.903, -78.844], [42.903, -78.84], [42.903, -78.839], [42.903, -78.833], [42.903, -78.831], [42.903, -78.828], [42.903, -78.825], [42.903, -78.823], [42.903, -78.816], [42.903, -78.814], [42.904, -78.887], [42.904, -78.885], [42.904, -78.881], [42.904, -78.877], [42.904, -78.876], [42.904, -78.871], [42.904, -78.869], [42.904, -78.865], [42.904, -78.86], [42.904, -78.835], [42.904, -78.822], [42.904, -78.821], [42.904, -78.815], [42.904, -78.809], [42.905, -78.897], [42.905, -78.895], [42.905, -78.89], [42.905, -78.888], [42.905, -78.886], [42.905, -78.884], [42.905, -78.879], [42.905, -78.877], [42.905, -78.871], [42.905, -78.86], [42.906, -78.889], [42.906, -78.887], [42.906, -78.885], [42.906, -78.884], [42.906, -78.878], [42.906, -78.877], [42.906, -78.86], [42.906, -78.856], [42.906, -78.824], [42.906, -78.821], [42.906, -78.816], [42.906, -78.812], [42.906, -78.807], [42.906, -78.806], [42.906, -78.805], [42.906, -78.803], [42.906, -78.802], [42.907, -78.899], [42.907, -78.897], [42.907, -78.896], [42.907, -78.894], [42.907, -78.886], [42.907, -78.883], [42.907, -78.878], [42.907, -78.877], [42.907, -78.872], [42.907, -78.869], [42.907, -78.867], [42.907, -78.866], [42.907, -78.863], [42.907, -78.859], [42.907, -78.854], [42.907, -78.853], [42.907, -78.851], [42.907, -78.832], [42.907, -78.828], [42.907, -78.814], [42.908, -78.899], [42.908, -78.897], [42.908, -78.896], [42.908, -78.895], [42.908, -78.892], [42.908, -78.891], [42.908, -78.89], [42.908, -78.883], [42.908, -78.881], [42.908, -78.878], [42.908, -78.873], [42.908, -78.871], [42.908, -78.87], [42.908, -78.867], [42.908, -78.854], [42.908, -78.851], [42.908, -78.847], [42.908, -78.842], [42.908, -78.838], [42.908, -78.815], [42.908, -78.806], [42.909, -78.896], [42.909, -78.891], [42.909, -78.89], [42.909, -78.889], [42.909, -78.887], [42.909, -78.882], [42.909, -78.878], [42.909, -78.877], [42.909, -78.875], [42.909, -78.872], [42.909, -78.871], [42.909, -78.869], [42.909, -78.867], [42.909, -78.85], [42.909, -78.843], [42.909, -78.814], [42.909, -78.812], [42.909, -78.807], [42.909, -78.805], [42.909, -78.803], [42.909, -78.801], [42.91, -78.9], [42.91, -78.893], [42.91, -78.877], [42.91, -78.866], [42.91, -78.865], [42.91, -78.864], [42.91, -78.863], [42.91, -78.854], [42.91, -78.847], [42.91, -78.836], [42.91, -78.832], [42.91, -78.825], [42.91, -78.821], [42.91, -78.811], [42.91, -78.81], [42.911, -78.896], [42.911, -78.894], [42.911, -78.888], [42.911, -78.884], [42.911, -78.881], [42.911, -78.88], [42.911, -78.877], [42.911, -78.871], [42.911, -78.868], [42.911, -78.867], [42.911, -78.866], [42.911, -78.854], [42.911, -78.852], [42.911, -78.841], [42.911, -78.838], [42.911, -78.835], [42.911, -78.834], [42.911, -78.832], [42.911, -78.823], [42.911, -78.821], [42.911, -78.82], [42.911, -78.816], [42.911, -78.811], [42.912, -78.897], [42.912, -78.895], [42.912, -78.893], [42.912, -78.891], [42.912, -78.89], [42.912, -78.889], [42.912, -78.886], [42.912, -78.885], [42.912, -78.884], [42.912, -78.883], [42.912, -78.882], [42.912, -78.881], [42.912, -78.879], [42.912, -78.878], [42.912, -78.876], [42.912, -78.871], [42.912, -78.869], [42.912, -78.868], [42.912, -78.866], [42.912, -78.864], [42.912, -78.852], [42.912, -78.851], [42.912, -78.835], [42.912, -78.824], [42.912, -78.81], [42.912, -78.809], [42.913, -78.897], [42.913, -78.894], [42.913, -78.892], [42.913, -78.886], [42.913, -78.883], [42.913, -78.882], [42.913, -78.878], [42.913, -78.877], [42.913, -78.869], [42.913, -78.866], [42.913, -78.864], [42.913, -78.863], [42.913, -78.855], [42.913, -78.853], [42.913, -78.852], [42.913, -78.851], [42.913, -78.847], [42.913, -78.844], [42.913, -78.837], [42.913, -78.824], [42.913, -78.812], [42.913, -78.803], [42.914, -78.895], [42.914, -78.893], [42.914, -78.89], [42.914, -78.889], [42.914, -78.883], [42.914, -78.882], [42.914, -78.88], [42.914, -78.875], [42.914, -78.874], [42.914, -78.873], [42.914, -78.866], [42.914, -78.865], [42.914, -78.856], [42.914, -78.854], [42.914, -78.847], [42.914, -78.838], [42.914, -78.824], [42.914, -78.81], [42.914, -78.808], [42.914, -78.807], [42.914, -78.804], [42.914, -78.802], [42.915, -78.894], [42.915, -78.893], [42.915, -78.892], [42.915, -78.891], [42.915, -78.89], [42.915, -78.888], [42.915, -78.885], [42.915, -78.884], [42.915, -78.877], [42.915, -78.874], [42.915, -78.871], [42.915, -78.867], [42.915, -78.854], [42.915, -78.849], [42.915, -78.848], [42.915, -78.847], [42.915, -78.84], [42.915, -78.839], [42.915, -78.834], [42.915, -78.832], [42.915, -78.816], [42.915, -78.812], [42.915, -78.81], [42.915, -78.809], [42.915, -78.807], [42.916, -78.895], [42.916, -78.893], [42.916, -78.888], [42.916, -78.887], [42.916, -78.885], [42.916, -78.878], [42.916, -78.877], [42.916, -78.872], [42.916, -78.869], [42.916, -78.866], [42.916, -78.854], [42.916, -78.851], [42.916, -78.85], [42.916, -78.849], [42.916, -78.847], [42.916, -78.84], [42.916, -78.827], [42.916, -78.824], [42.916, -78.823], [42.917, -78.892], [42.917, -78.891], [42.917, -78.887], [42.917, -78.885], [42.917, -78.883], [42.917, -78.882], [42.917, -78.881], [42.917, -78.879], [42.917, -78.878], [42.917, -78.865], [42.917, -78.854], [42.917, -78.851], [42.917, -78.84], [42.917, -78.825], [42.917, -78.819], [42.917, -78.816], [42.917, -78.813], [42.917, -78.809], [42.918, -78.894], [42.918, -78.883], [42.918, -78.882], [42.918, -78.881], [42.918, -78.877], [42.918, -78.875], [42.918, -78.869], [42.918, -78.868], [42.918, -78.866], [42.918, -78.865], [42.918, -78.863], [42.918, -78.861], [42.918, -78.858], [42.918, -78.857], [42.918, -78.848], [42.918, -78.835], [42.918, -78.829], [42.918, -78.814], [42.918, -78.812], [42.918, -78.802], [42.919, -78.898], [42.919, -78.89], [42.919, -78.888], [42.919, -78.887], [42.919, -78.885], [42.919, -78.877], [42.919, -78.868], [42.919, -78.865], [42.919, -78.863], [42.919, -78.86], [42.919, -78.848], [42.919, -78.847], [42.919, -78.845], [42.919, -78.843], [42.919, -78.829], [42.919, -78.82], [42.919, -78.813], [42.919, -78.809], [42.92, -78.897], [42.92, -78.891], [42.92, -78.89], [42.92, -78.889], [42.92, -78.883], [42.92, -78.88], [42.92, -78.879], [42.92, -78.87], [42.92, -78.869], [42.92, -78.865], [42.92, -78.863], [42.92, -78.86], [42.92, -78.859], [42.92, -78.855], [42.92, -78.853], [42.92, -78.827], [42.92, -78.825], [42.92, -78.812], [42.92, -78.806], [42.921, -78.896], [42.921, -78.891], [42.921, -78.89], [42.921, -78.889], [42.921, -78.884], [42.921, -78.883], [42.921, -78.873], [42.921, -78.866], [42.921, -78.863], [42.921, -78.859], [42.921, -78.849], [42.921, -78.848], [42.921, -78.827], [42.921, -78.826], [42.921, -78.822], [42.921, -78.815], [42.921, -78.814], [42.921, -78.808], [42.921, -78.806], [42.922, -78.896], [42.922, -78.888], [42.922, -78.881], [42.922, -78.879], [42.922, -78.876], [42.922, -78.874], [42.922, -78.872], [42.922, -78.867], [42.922, -78.866], [42.922, -78.858], [42.922, -78.85], [42.922, -78.849], [42.922, -78.84], [42.922, -78.839], [42.922, -78.829], [42.922, -78.827], [42.922, -78.814], [42.922, -78.81], [42.922, -78.809], [42.922, -78.808], [42.922, -78.806], [42.923, -78.887], [42.923, -78.882], [42.923, -78.877], [42.923, -78.874], [42.923, -78.871], [42.923, -78.868], [42.923, -78.85], [42.923, -78.849], [42.923, -78.846], [42.923, -78.836], [42.923, -78.828], [42.923, -78.821], [42.923, -78.817], [42.923, -78.815], [42.923, -78.809], [42.923, -78.806], [42.923, -78.805], [42.923, -78.804], [42.924, -78.896], [42.924, -78.895], [42.924, -78.887], [42.924, -78.885], [42.924, -78.883], [42.924, -78.881], [42.924, -78.878], [42.924, -78.875], [42.924, -78.874], [42.924, -78.849], [42.924, -78.83], [42.924, -78.826], [42.924, -78.822], [42.924, -78.82], [42.924, -78.807], [42.925, -78.895], [42.925, -78.892], [42.925, -78.885], [42.925, -78.879], [42.925, -78.874], [42.925, -78.868], [42.925, -78.854], [42.925, -78.849], [42.925, -78.82], [42.925, -78.811], [42.925, -78.804], [42.926, -78.894], [42.926, -78.893], [42.926, -78.891], [42.926, -78.89], [42.926, -78.889], [42.926, -78.881], [42.926, -78.88], [42.926, -78.877], [42.926, -78.852], [42.926, -78.813], [42.926, -78.811], [42.926, -78.802], [42.927, -78.89], [42.927, -78.889], [42.927, -78.885], [42.927, -78.883], [42.927, -78.879], [42.927, -78.877], [42.927, -78.875], [42.927, -78.849], [42.927, -78.825], [42.927, -78.819], [42.927, -78.818], [42.927, -78.811], [42.927, -78.807], [42.927, -78.804], [42.928, -78.89], [42.928, -78.885], [42.928, -78.884], [42.928, -78.877], [42.928, -78.876], [42.928, -78.875], [42.928, -78.852], [42.928, -78.828], [42.928, -78.827], [42.928, -78.826], [42.928, -78.825], [42.928, -78.824], [42.928, -78.822], [42.928, -78.816], [42.928, -78.815], [42.929, -78.894], [42.929, -78.852], [42.929, -78.85], [42.929, -78.837], [42.929, -78.816], [42.929, -78.815], [42.929, -78.813], [42.929, -78.802], [42.93, -78.872], [42.93, -78.849], [42.93, -78.84], [42.93, -78.835], [42.93, -78.817], [42.93, -78.814], [42.93, -78.813], [42.93, -78.81], [42.93, -78.809], [42.93, -78.808], [42.93, -78.807], [42.93, -78.806], [42.931, -78.839], [42.931, -78.838], [42.931, -78.835], [42.931, -78.834], [42.931, -78.827], [42.931, -78.813], [42.931, -78.81], [42.931, -78.804], [42.932, -78.897], [42.932, -78.892], [42.932, -78.888], [42.932, -78.826], [42.932, -78.824], [42.932, -78.818], [42.932, -78.811], [42.933, -78.847], [42.933, -78.839], [42.933, -78.831], [42.933, -78.828], [42.933, -78.827], [42.933, -78.82], [42.933, -78.819], [42.933, -78.818], [42.933, -78.814], [42.933, -78.81], [42.933, -78.808], [42.934, -78.9], [42.934, -78.851], [42.934, -78.826], [42.934, -78.811], [42.934, -78.81], [42.934, -78.807], [42.935, -78.844], [42.935, -78.842], [42.935, -78.839], [42.935, -78.831], [42.935, -78.815], [42.935, -78.814], [42.935, -78.813], [42.935, -78.812], [42.936, -78.902], [42.936, -78.842], [42.936, -78.834], [42.936, -78.819], [42.936, -78.814], [42.936, -78.812], [42.936, -78.81], [42.936, -78.809], [42.936, -78.808], [42.936, -78.807], [42.936, -78.803], [42.937, -78.899], [42.937, -78.893], [42.937, -78.891], [42.937, -78.879], [42.937, -78.835], [42.937, -78.828], [42.937, -78.822], [42.937, -78.815], [42.937, -78.814], [42.938, -78.905], [42.938, -78.904], [42.938, -78.903], [42.938, -78.902], [42.938, -78.888], [42.938, -78.878], [42.938, -78.84], [42.938, -78.827], [42.938, -78.819], [42.938, -78.816], [42.938, -78.814], [42.938, -78.813], [42.938, -78.811], [42.938, -78.808], [42.938, -78.802], [42.939, -78.905], [42.939, -78.903], [42.939, -78.894], [42.939, -78.877], [42.939, -78.868], [42.939, -78.85], [42.939, -78.838], [42.939, -78.835], [42.939, -78.834], [42.939, -78.832], [42.939, -78.818], [42.939, -78.817], [42.939, -78.816], [42.939, -78.812], [42.94, -78.906], [42.94, -78.905], [42.94, -78.902], [42.94, -78.892], [42.94, -78.88], [42.94, -78.877], [42.94, -78.876], [42.94, -78.851], [42.94, -78.844], [42.94, -78.84], [42.94, -78.823], [42.94, -78.821], [42.94, -78.819], [42.94, -78.817], [42.94, -78.816], [42.94, -78.814], [42.94, -78.809], [42.94, -78.806], [42.94, -78.801], [42.941, -78.902], [42.941, -78.882], [42.941, -78.878], [42.941, -78.873], [42.941, -78.871], [42.941, -78.87], [42.941, -78.847], [42.941, -78.844], [42.941, -78.837], [42.941, -78.82], [42.941, -78.814], [42.941, -78.808], [42.942, -78.905], [42.942, -78.884], [42.942, -78.88], [42.942, -78.878], [42.942, -78.868], [42.942, -78.846], [42.942, -78.822], [42.942, -78.82], [42.942, -78.815], [42.942, -78.806], [42.942, -78.803], [42.942, -78.802], [42.943, -78.906], [42.943, -78.882], [42.943, -78.878], [42.943, -78.87], [42.943, -78.868], [42.943, -78.854], [42.943, -78.835], [42.943, -78.824], [42.943, -78.819], [42.943, -78.814], [42.943, -78.808], [42.944, -78.907], [42.944, -78.903], [42.944, -78.902], [42.944, -78.901], [42.944, -78.868], [42.944, -78.867], [42.944, -78.863], [42.944, -78.855], [42.944, -78.834], [42.944, -78.822], [42.944, -78.819], [42.944, -78.816], [42.944, -78.806], [42.944, -78.804], [42.945, -78.904], [42.945, -78.902], [42.945, -78.895], [42.945, -78.889], [42.945, -78.87], [42.945, -78.861], [42.945, -78.824], [42.945, -78.814], [42.945, -78.813], [42.945, -78.811], [42.946, -78.887], [42.946, -78.886], [42.946, -78.869], [42.946, -78.868], [42.946, -78.866], [42.946, -78.864], [42.946, -78.857], [42.946, -78.854], [42.946, -78.85], [42.946, -78.849], [42.946, -78.837], [42.946, -78.826], [42.946, -78.823], [42.946, -78.815], [42.946, -78.814], [42.946, -78.813], [42.946, -78.81], [42.946, -78.808], [42.946, -78.807], [42.946, -78.805], [42.946, -78.804], [42.947, -78.891], [42.947, -78.877], [42.947, -78.872], [42.947, -78.859], [42.947, -78.855], [42.947, -78.846], [42.947, -78.842], [42.947, -78.829], [42.947, -78.826], [42.947, -78.824], [42.947, -78.823], [42.947, -78.822], [42.947, -78.821], [42.947, -78.82], [42.947, -78.819], [42.947, -78.814], [42.947, -78.811], [42.947, -78.801], [42.948, -78.902], [42.948, -78.901], [42.948, -78.899], [42.948, -78.89], [42.948, -78.888], [42.948, -78.885], [42.948, -78.884], [42.948, -78.883], [42.948, -78.882], [42.948, -78.879], [42.948, -78.875], [42.948, -78.873], [42.948, -78.871], [42.948, -78.87], [42.948, -78.861], [42.948, -78.859], [42.948, -78.856], [42.948, -78.854], [42.948, -78.853], [42.948, -78.851], [42.948, -78.842], [42.948, -78.826], [42.948, -78.822], [42.948, -78.819], [42.948, -78.806], [42.948, -78.805], [42.948, -78.801], [42.949, -78.905], [42.949, -78.889], [42.949, -78.888], [42.949, -78.877], [42.949, -78.871], [42.949, -78.864], [42.949, -78.86], [42.949, -78.847], [42.949, -78.846], [42.949, -78.83], [42.949, -78.829], [42.949, -78.824], [42.949, -78.82], [42.949, -78.815], [42.95, -78.904], [42.95, -78.903], [42.95, -78.899], [42.95, -78.887], [42.95, -78.884], [42.95, -78.882], [42.95, -78.877], [42.95, -78.869], [42.95, -78.868], [42.95, -78.863], [42.95, -78.83], [42.95, -78.829], [42.95, -78.828], [42.95, -78.827], [42.95, -78.825], [42.95, -78.824], [42.95, -78.823], [42.951, -78.908], [42.951, -78.906], [42.951, -78.905], [42.951, -78.897], [42.951, -78.885], [42.951, -78.884], [42.951, -78.878], [42.951, -78.872], [42.951, -78.869], [42.951, -78.858], [42.951, -78.854], [42.951, -78.83], [42.951, -78.826], [42.952, -78.908], [42.952, -78.905], [42.952, -78.903], [42.952, -78.902], [42.952, -78.876], [42.952, -78.873], [42.952, -78.871], [42.952, -78.869], [42.952, -78.868], [42.952, -78.85], [42.952, -78.829], [42.952, -78.827], [42.952, -78.826], [42.953, -78.909], [42.953, -78.901], [42.953, -78.9], [42.953, -78.899], [42.953, -78.874], [42.953, -78.869], [42.953, -78.862], [42.953, -78.859], [42.953, -78.858], [42.953, -78.831], [42.954, -78.904], [42.954, -78.886], [42.954, -78.879], [42.954, -78.878], [42.954, -78.869], [42.954, -78.852], [42.954, -78.826], [42.955, -78.897], [42.955, -78.878], [42.955, -78.874], [42.955, -78.87], [42.955, -78.868], [42.955, -78.861], [42.955, -78.859], [42.955, -78.834], [42.955, -78.833], [42.956, -78.898], [42.956, -78.897], [42.956, -78.89], [42.956, -78.87], [42.956, -78.868], [42.956, -78.863], [42.956, -78.859], [42.956, -78.858], [42.956, -78.854], [42.956, -78.843], [42.956, -78.828], [42.956, -78.824], [42.957, -78.897], [42.957, -78.896], [42.957, -78.895], [42.957, -78.89], [42.957, -78.866], [42.957, -78.859], [42.957, -78.855], [42.957, -78.829], [42.957, -78.826], [42.957, -78.82], [42.957, -78.819], [42.958, -78.907], [42.958, -78.905], [42.958, -78.897], [42.958, -78.895], [42.958, -78.886], [42.958, -78.885], [42.958, -78.884], [42.958, -78.88], [42.958, -78.87], [42.958, -78.869], [42.958, -78.862], [42.958, -78.858], [42.958, -78.834], [42.958, -78.831], [42.959, -78.907], [42.959, -78.896], [42.959, -78.859], [42.96, -78.906], [42.96, -78.902], [42.961, -78.898], [42.963, -78.895], [42.964, -78.901]],
                {&quot;blur&quot;: 15, &quot;maxZoom&quot;: 12, &quot;minOpacity&quot;: 0.5, &quot;radius&quot;: 8}
            ).addTo(map_f9fb3bcf0425c4d19ba0072c0598bf71);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
# Heatmap --> 2023 Summer Nights
base_map = generateBaseMap()
HeatMap(data=summer_2023_night[['latitude', 'longitude']].\
        groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=12).add_to(base_map)

base_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_e71e236b601167eaf3c46ae272b005c6 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js&quot;&gt;&lt;/script&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_e71e236b601167eaf3c46ae272b005c6&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_e71e236b601167eaf3c46ae272b005c6 = L.map(
                &quot;map_e71e236b601167eaf3c46ae272b005c6&quot;,
                {
                    center: [42.91184928528912, -78.84964614694492],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );
            L.control.scale().addTo(map_e71e236b601167eaf3c46ae272b005c6);





            var tile_layer_0d6811b27203f57ce7a252c36e96c1da = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_e71e236b601167eaf3c46ae272b005c6);


            var heat_map_3bab6595809fe90beb0c5403af215b8e = L.heatLayer(
                [[42.833, -78.816], [42.834, -78.816], [42.835, -78.82], [42.835, -78.805], [42.836, -78.822], [42.837, -78.817], [42.837, -78.811], [42.838, -78.808], [42.839, -78.857], [42.839, -78.806], [42.842, -78.825], [42.844, -78.831], [42.845, -78.825], [42.845, -78.821], [42.846, -78.824], [42.846, -78.819], [42.847, -78.855], [42.849, -78.863], [42.849, -78.816], [42.849, -78.803], [42.85, -78.825], [42.85, -78.815], [42.85, -78.813], [42.85, -78.804], [42.85, -78.803], [42.85, -78.802], [42.851, -78.824], [42.851, -78.806], [42.851, -78.805], [42.852, -78.819], [42.854, -78.81], [42.854, -78.808], [42.855, -78.829], [42.855, -78.825], [42.855, -78.821], [42.855, -78.811], [42.855, -78.81], [42.856, -78.827], [42.856, -78.826], [42.856, -78.81], [42.856, -78.803], [42.857, -78.827], [42.857, -78.826], [42.857, -78.813], [42.857, -78.812], [42.858, -78.831], [42.858, -78.829], [42.858, -78.826], [42.858, -78.823], [42.858, -78.813], [42.86, -78.822], [42.86, -78.813], [42.861, -78.813], [42.861, -78.812], [42.863, -78.811], [42.864, -78.816], [42.864, -78.815], [42.865, -78.865], [42.865, -78.86], [42.866, -78.861], [42.866, -78.824], [42.867, -78.831], [42.867, -78.808], [42.868, -78.862], [42.868, -78.807], [42.869, -78.868], [42.869, -78.862], [42.869, -78.85], [42.869, -78.847], [42.87, -78.847], [42.87, -78.839], [42.87, -78.832], [42.87, -78.805], [42.871, -78.844], [42.871, -78.843], [42.871, -78.808], [42.871, -78.807], [42.872, -78.815], [42.872, -78.811], [42.872, -78.809], [42.872, -78.804], [42.873, -78.87], [42.873, -78.865], [42.873, -78.811], [42.873, -78.804], [42.874, -78.874], [42.874, -78.872], [42.874, -78.871], [42.875, -78.874], [42.875, -78.85], [42.875, -78.827], [42.875, -78.806], [42.876, -78.871], [42.876, -78.868], [42.876, -78.849], [42.876, -78.828], [42.877, -78.877], [42.877, -78.861], [42.877, -78.851], [42.877, -78.802], [42.878, -78.88], [42.878, -78.874], [42.878, -78.857], [42.878, -78.837], [42.878, -78.808], [42.878, -78.801], [42.879, -78.885], [42.879, -78.878], [42.879, -78.874], [42.879, -78.835], [42.879, -78.834], [42.88, -78.88], [42.88, -78.878], [42.88, -78.866], [42.88, -78.834], [42.881, -78.878], [42.881, -78.876], [42.881, -78.871], [42.881, -78.853], [42.882, -78.878], [42.882, -78.876], [42.882, -78.873], [42.882, -78.868], [42.882, -78.867], [42.882, -78.866], [42.882, -78.848], [42.882, -78.834], [42.882, -78.833], [42.883, -78.874], [42.883, -78.851], [42.884, -78.874], [42.884, -78.863], [42.884, -78.854], [42.884, -78.824], [42.885, -78.868], [42.885, -78.855], [42.885, -78.839], [42.885, -78.814], [42.886, -78.874], [42.886, -78.872], [42.886, -78.853], [42.886, -78.815], [42.886, -78.814], [42.886, -78.807], [42.887, -78.871], [42.887, -78.867], [42.887, -78.854], [42.887, -78.851], [42.888, -78.882], [42.888, -78.877], [42.888, -78.876], [42.888, -78.875], [42.888, -78.872], [42.888, -78.842], [42.888, -78.814], [42.889, -78.883], [42.889, -78.882], [42.889, -78.876], [42.889, -78.871], [42.889, -78.856], [42.889, -78.855], [42.889, -78.854], [42.889, -78.837], [42.889, -78.813], [42.889, -78.808], [42.89, -78.876], [42.89, -78.874], [42.89, -78.873], [42.89, -78.872], [42.89, -78.87], [42.89, -78.837], [42.89, -78.835], [42.89, -78.806], [42.891, -78.883], [42.891, -78.88], [42.891, -78.879], [42.891, -78.878], [42.891, -78.876], [42.891, -78.875], [42.891, -78.87], [42.891, -78.866], [42.891, -78.857], [42.891, -78.853], [42.891, -78.839], [42.891, -78.831], [42.891, -78.809], [42.892, -78.883], [42.892, -78.877], [42.892, -78.874], [42.892, -78.873], [42.892, -78.859], [42.892, -78.854], [42.892, -78.851], [42.892, -78.85], [42.892, -78.842], [42.892, -78.834], [42.892, -78.833], [42.892, -78.832], [42.892, -78.825], [42.892, -78.807], [42.893, -78.892], [42.893, -78.891], [42.893, -78.889], [42.893, -78.881], [42.893, -78.878], [42.893, -78.877], [42.893, -78.876], [42.893, -78.874], [42.893, -78.871], [42.893, -78.855], [42.893, -78.846], [42.893, -78.841], [42.893, -78.838], [42.893, -78.836], [42.893, -78.834], [42.894, -78.887], [42.894, -78.858], [42.894, -78.835], [42.894, -78.833], [42.894, -78.832], [42.895, -78.889], [42.895, -78.883], [42.895, -78.881], [42.895, -78.877], [42.895, -78.875], [42.895, -78.837], [42.895, -78.832], [42.895, -78.83], [42.895, -78.829], [42.895, -78.826], [42.895, -78.819], [42.895, -78.818], [42.895, -78.809], [42.895, -78.806], [42.896, -78.886], [42.896, -78.865], [42.896, -78.857], [42.896, -78.854], [42.896, -78.841], [42.896, -78.84], [42.896, -78.836], [42.896, -78.835], [42.896, -78.832], [42.896, -78.822], [42.896, -78.802], [42.897, -78.877], [42.897, -78.875], [42.897, -78.872], [42.897, -78.871], [42.897, -78.869], [42.897, -78.866], [42.897, -78.846], [42.897, -78.842], [42.897, -78.84], [42.897, -78.838], [42.897, -78.832], [42.897, -78.816], [42.897, -78.809], [42.898, -78.894], [42.898, -78.889], [42.898, -78.888], [42.898, -78.876], [42.898, -78.873], [42.898, -78.836], [42.898, -78.819], [42.898, -78.818], [42.898, -78.817], [42.898, -78.807], [42.899, -78.888], [42.899, -78.885], [42.899, -78.879], [42.899, -78.878], [42.899, -78.877], [42.899, -78.875], [42.899, -78.873], [42.899, -78.871], [42.899, -78.87], [42.899, -78.84], [42.899, -78.833], [42.899, -78.819], [42.899, -78.818], [42.9, -78.894], [42.9, -78.88], [42.9, -78.878], [42.9, -78.873], [42.9, -78.865], [42.9, -78.854], [42.9, -78.847], [42.9, -78.846], [42.9, -78.841], [42.901, -78.875], [42.901, -78.871], [42.901, -78.865], [42.901, -78.862], [42.901, -78.861], [42.901, -78.825], [42.902, -78.883], [42.902, -78.879], [42.902, -78.873], [42.902, -78.867], [42.902, -78.863], [42.902, -78.858], [42.902, -78.851], [42.902, -78.838], [42.902, -78.827], [42.903, -78.892], [42.903, -78.887], [42.903, -78.883], [42.903, -78.872], [42.903, -78.869], [42.903, -78.847], [42.903, -78.837], [42.903, -78.833], [42.903, -78.83], [42.903, -78.812], [42.904, -78.892], [42.904, -78.883], [42.904, -78.876], [42.904, -78.871], [42.904, -78.866], [42.904, -78.862], [42.904, -78.86], [42.904, -78.854], [42.904, -78.847], [42.904, -78.845], [42.904, -78.835], [42.904, -78.831], [42.904, -78.814], [42.905, -78.89], [42.905, -78.883], [42.905, -78.882], [42.905, -78.877], [42.905, -78.87], [42.905, -78.862], [42.905, -78.845], [42.905, -78.833], [42.905, -78.827], [42.905, -78.814], [42.906, -78.877], [42.906, -78.803], [42.907, -78.895], [42.907, -78.894], [42.907, -78.885], [42.907, -78.858], [42.907, -78.854], [42.907, -78.851], [42.907, -78.803], [42.907, -78.802], [42.908, -78.898], [42.908, -78.897], [42.908, -78.894], [42.908, -78.893], [42.908, -78.881], [42.908, -78.878], [42.908, -78.866], [42.908, -78.851], [42.908, -78.832], [42.908, -78.827], [42.908, -78.812], [42.908, -78.811], [42.909, -78.901], [42.909, -78.898], [42.909, -78.887], [42.909, -78.848], [42.909, -78.84], [42.909, -78.822], [42.909, -78.82], [42.909, -78.818], [42.909, -78.814], [42.91, -78.892], [42.91, -78.854], [42.91, -78.832], [42.911, -78.899], [42.911, -78.887], [42.911, -78.87], [42.911, -78.866], [42.911, -78.855], [42.911, -78.854], [42.911, -78.821], [42.911, -78.808], [42.912, -78.896], [42.912, -78.89], [42.912, -78.887], [42.912, -78.882], [42.912, -78.881], [42.912, -78.877], [42.912, -78.875], [42.912, -78.871], [42.912, -78.869], [42.912, -78.867], [42.912, -78.866], [42.912, -78.865], [42.912, -78.854], [42.912, -78.828], [42.912, -78.822], [42.912, -78.813], [42.912, -78.809], [42.912, -78.806], [42.913, -78.893], [42.913, -78.892], [42.913, -78.889], [42.913, -78.878], [42.913, -78.877], [42.913, -78.875], [42.913, -78.87], [42.913, -78.867], [42.913, -78.861], [42.913, -78.851], [42.913, -78.85], [42.913, -78.841], [42.913, -78.836], [42.913, -78.827], [42.913, -78.822], [42.913, -78.82], [42.913, -78.807], [42.913, -78.801], [42.914, -78.89], [42.914, -78.889], [42.914, -78.864], [42.914, -78.853], [42.914, -78.849], [42.914, -78.841], [42.914, -78.838], [42.914, -78.829], [42.914, -78.824], [42.914, -78.801], [42.915, -78.897], [42.915, -78.896], [42.915, -78.893], [42.915, -78.892], [42.915, -78.891], [42.915, -78.888], [42.915, -78.887], [42.915, -78.877], [42.915, -78.857], [42.915, -78.851], [42.915, -78.844], [42.915, -78.815], [42.915, -78.808], [42.916, -78.898], [42.916, -78.892], [42.916, -78.889], [42.916, -78.877], [42.916, -78.84], [42.916, -78.824], [42.916, -78.82], [42.916, -78.816], [42.917, -78.899], [42.917, -78.881], [42.917, -78.88], [42.917, -78.877], [42.917, -78.84], [42.917, -78.816], [42.917, -78.804], [42.917, -78.803], [42.918, -78.895], [42.918, -78.893], [42.918, -78.885], [42.918, -78.863], [42.918, -78.829], [42.918, -78.802], [42.918, -78.801], [42.919, -78.897], [42.919, -78.896], [42.919, -78.892], [42.919, -78.89], [42.919, -78.861], [42.919, -78.855], [42.919, -78.829], [42.919, -78.827], [42.919, -78.823], [42.919, -78.809], [42.92, -78.896], [42.92, -78.894], [42.92, -78.886], [42.92, -78.877], [42.92, -78.872], [42.92, -78.86], [42.92, -78.859], [42.92, -78.848], [42.92, -78.845], [42.92, -78.835], [42.92, -78.814], [42.92, -78.812], [42.921, -78.891], [42.921, -78.888], [42.921, -78.887], [42.921, -78.886], [42.921, -78.884], [42.921, -78.873], [42.921, -78.871], [42.921, -78.87], [42.921, -78.862], [42.921, -78.847], [42.921, -78.803], [42.921, -78.802], [42.922, -78.89], [42.922, -78.883], [42.922, -78.882], [42.922, -78.877], [42.922, -78.873], [42.922, -78.872], [42.922, -78.839], [42.922, -78.814], [42.922, -78.813], [42.922, -78.81], [42.923, -78.892], [42.923, -78.882], [42.923, -78.877], [42.923, -78.823], [42.923, -78.814], [42.923, -78.809], [42.923, -78.807], [42.923, -78.806], [42.923, -78.803], [42.923, -78.802], [42.924, -78.888], [42.924, -78.886], [42.924, -78.885], [42.924, -78.875], [42.924, -78.873], [42.924, -78.871], [42.924, -78.852], [42.924, -78.826], [42.924, -78.822], [42.924, -78.817], [42.925, -78.892], [42.925, -78.89], [42.925, -78.885], [42.925, -78.881], [42.925, -78.814], [42.925, -78.805], [42.926, -78.897], [42.926, -78.89], [42.926, -78.884], [42.926, -78.882], [42.926, -78.877], [42.926, -78.876], [42.926, -78.845], [42.926, -78.826], [42.926, -78.818], [42.926, -78.815], [42.927, -78.892], [42.927, -78.887], [42.927, -78.886], [42.927, -78.884], [42.927, -78.846], [42.927, -78.823], [42.927, -78.818], [42.928, -78.877], [42.928, -78.874], [42.928, -78.828], [42.928, -78.819], [42.929, -78.894], [42.929, -78.89], [42.929, -78.85], [42.929, -78.814], [42.93, -78.889], [42.93, -78.835], [42.93, -78.817], [42.93, -78.808], [42.93, -78.807], [42.93, -78.804], [42.93, -78.803], [42.93, -78.802], [42.931, -78.899], [42.931, -78.851], [42.931, -78.829], [42.931, -78.817], [42.931, -78.814], [42.931, -78.807], [42.931, -78.806], [42.931, -78.804], [42.931, -78.802], [42.932, -78.846], [42.932, -78.829], [42.932, -78.828], [42.932, -78.824], [42.932, -78.803], [42.932, -78.802], [42.932, -78.801], [42.933, -78.901], [42.933, -78.875], [42.933, -78.847], [42.933, -78.839], [42.933, -78.828], [42.933, -78.819], [42.934, -78.842], [42.934, -78.814], [42.934, -78.813], [42.934, -78.81], [42.935, -78.902], [42.935, -78.835], [42.935, -78.819], [42.936, -78.851], [42.936, -78.835], [42.936, -78.831], [42.936, -78.819], [42.936, -78.818], [42.936, -78.814], [42.936, -78.807], [42.937, -78.903], [42.937, -78.899], [42.937, -78.851], [42.937, -78.835], [42.937, -78.816], [42.938, -78.9], [42.938, -78.84], [42.938, -78.828], [42.938, -78.822], [42.938, -78.816], [42.938, -78.812], [42.938, -78.81], [42.939, -78.904], [42.939, -78.903], [42.939, -78.894], [42.939, -78.839], [42.939, -78.835], [42.939, -78.832], [42.939, -78.815], [42.939, -78.811], [42.939, -78.81], [42.94, -78.907], [42.94, -78.895], [42.94, -78.881], [42.94, -78.868], [42.94, -78.835], [42.94, -78.823], [42.94, -78.816], [42.94, -78.81], [42.941, -78.893], [42.941, -78.892], [42.941, -78.89], [42.941, -78.878], [42.941, -78.876], [42.941, -78.869], [42.941, -78.856], [42.941, -78.842], [42.941, -78.822], [42.941, -78.816], [42.941, -78.811], [42.941, -78.807], [42.941, -78.801], [42.942, -78.905], [42.942, -78.904], [42.942, -78.836], [42.942, -78.832], [42.942, -78.804], [42.942, -78.803], [42.942, -78.802], [42.943, -78.907], [42.943, -78.901], [42.943, -78.823], [42.943, -78.822], [42.943, -78.816], [42.943, -78.814], [42.944, -78.907], [42.944, -78.9], [42.944, -78.886], [42.944, -78.867], [42.944, -78.849], [42.944, -78.818], [42.944, -78.814], [42.944, -78.812], [42.944, -78.807], [42.944, -78.805], [42.945, -78.904], [42.945, -78.902], [42.945, -78.869], [42.945, -78.841], [42.945, -78.823], [42.945, -78.819], [42.945, -78.812], [42.945, -78.804], [42.946, -78.869], [42.946, -78.826], [42.946, -78.817], [42.946, -78.812], [42.946, -78.811], [42.946, -78.809], [42.947, -78.889], [42.947, -78.888], [42.947, -78.869], [42.947, -78.859], [42.947, -78.858], [42.947, -78.85], [42.947, -78.848], [42.947, -78.828], [42.947, -78.82], [42.947, -78.818], [42.947, -78.814], [42.948, -78.891], [42.948, -78.886], [42.948, -78.884], [42.948, -78.871], [42.948, -78.866], [42.948, -78.861], [42.948, -78.851], [42.948, -78.847], [42.948, -78.841], [42.948, -78.825], [42.948, -78.814], [42.948, -78.804], [42.949, -78.9], [42.949, -78.899], [42.949, -78.873], [42.949, -78.87], [42.95, -78.908], [42.95, -78.906], [42.95, -78.904], [42.95, -78.901], [42.95, -78.897], [42.95, -78.887], [42.95, -78.884], [42.95, -78.869], [42.95, -78.827], [42.95, -78.826], [42.951, -78.869], [42.951, -78.826], [42.952, -78.909], [42.952, -78.907], [42.952, -78.877], [42.953, -78.905], [42.953, -78.886], [42.953, -78.87], [42.953, -78.869], [42.954, -78.906], [42.954, -78.898], [42.954, -78.879], [42.954, -78.878], [42.954, -78.85], [42.955, -78.902], [42.955, -78.898], [42.955, -78.897], [42.955, -78.879], [42.955, -78.874], [42.955, -78.833], [42.955, -78.823], [42.955, -78.822], [42.956, -78.904], [42.956, -78.847], [42.957, -78.897], [42.957, -78.876], [42.957, -78.865], [42.957, -78.832], [42.957, -78.831], [42.958, -78.881], [42.958, -78.88], [42.958, -78.875], [42.958, -78.874], [42.958, -78.869], [42.958, -78.859], [42.958, -78.856], [42.958, -78.844], [42.959, -78.896], [42.959, -78.894], [42.959, -78.859], [42.96, -78.905], [42.961, -78.903], [42.961, -78.897], [42.961, -78.896], [42.962, -78.9]],
                {&quot;blur&quot;: 15, &quot;maxZoom&quot;: 12, &quot;minOpacity&quot;: 0.5, &quot;radius&quot;: 8}
            ).addTo(map_e71e236b601167eaf3c46ae272b005c6);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



Upon comparing the day and night heatmaps for Summer 2023, it becomes evident that there is a higher incidence of crime during daylight hours compared to nighttime.


```python
# Heatmap --> 2009 Summer Days
base_map = generateBaseMap()
HeatMap(data=summer_2009_day[['latitude', 'longitude']].\
        groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=12).add_to(base_map)

base_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_b2d3610360388637adde166587c3edf2 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js&quot;&gt;&lt;/script&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_b2d3610360388637adde166587c3edf2&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_b2d3610360388637adde166587c3edf2 = L.map(
                &quot;map_b2d3610360388637adde166587c3edf2&quot;,
                {
                    center: [42.91184928528912, -78.84964614694492],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );
            L.control.scale().addTo(map_b2d3610360388637adde166587c3edf2);





            var tile_layer_1e8e35b793ae5b9faca82b367cf21a97 = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_b2d3610360388637adde166587c3edf2);


            var heat_map_7d2086fa7ce4a5c4f4962447b07edf88 = L.heatLayer(
                [[42.826, -78.834], [42.828, -78.83], [42.833, -78.824], [42.833, -78.823], [42.833, -78.822], [42.833, -78.803], [42.834, -78.819], [42.834, -78.816], [42.834, -78.815], [42.834, -78.811], [42.835, -78.826], [42.835, -78.823], [42.835, -78.807], [42.836, -78.826], [42.836, -78.824], [42.836, -78.821], [42.836, -78.818], [42.837, -78.802], [42.838, -78.833], [42.839, -78.815], [42.839, -78.806], [42.84, -78.817], [42.84, -78.808], [42.84, -78.804], [42.84, -78.803], [42.84, -78.801], [42.841, -78.824], [42.841, -78.819], [42.842, -78.827], [42.842, -78.826], [42.842, -78.825], [42.842, -78.822], [42.842, -78.821], [42.842, -78.82], [42.842, -78.809], [42.843, -78.824], [42.844, -78.832], [42.844, -78.824], [42.845, -78.812], [42.846, -78.824], [42.846, -78.823], [42.846, -78.821], [42.846, -78.81], [42.847, -78.825], [42.847, -78.819], [42.847, -78.814], [42.848, -78.832], [42.848, -78.827], [42.848, -78.824], [42.848, -78.823], [42.848, -78.817], [42.848, -78.812], [42.849, -78.863], [42.849, -78.83], [42.849, -78.814], [42.849, -78.813], [42.849, -78.803], [42.85, -78.831], [42.85, -78.83], [42.85, -78.826], [42.85, -78.824], [42.85, -78.815], [42.85, -78.814], [42.85, -78.813], [42.85, -78.804], [42.85, -78.803], [42.851, -78.831], [42.851, -78.83], [42.851, -78.828], [42.851, -78.826], [42.851, -78.825], [42.851, -78.822], [42.851, -78.82], [42.851, -78.806], [42.852, -78.823], [42.852, -78.82], [42.852, -78.817], [42.852, -78.814], [42.852, -78.807], [42.852, -78.805], [42.852, -78.801], [42.853, -78.831], [42.853, -78.82], [42.853, -78.802], [42.854, -78.812], [42.854, -78.811], [42.854, -78.809], [42.854, -78.808], [42.854, -78.806], [42.854, -78.801], [42.855, -78.828], [42.855, -78.827], [42.855, -78.81], [42.855, -78.808], [42.855, -78.806], [42.856, -78.835], [42.856, -78.822], [42.856, -78.809], [42.856, -78.806], [42.857, -78.824], [42.857, -78.816], [42.857, -78.813], [42.857, -78.812], [42.857, -78.811], [42.857, -78.809], [42.857, -78.807], [42.857, -78.805], [42.858, -78.831], [42.858, -78.823], [42.858, -78.814], [42.858, -78.813], [42.858, -78.81], [42.858, -78.804], [42.858, -78.803], [42.859, -78.833], [42.859, -78.812], [42.859, -78.809], [42.86, -78.871], [42.86, -78.82], [42.86, -78.817], [42.86, -78.814], [42.861, -78.816], [42.861, -78.815], [42.862, -78.818], [42.862, -78.816], [42.862, -78.811], [42.863, -78.821], [42.863, -78.815], [42.863, -78.814], [42.865, -78.868], [42.865, -78.865], [42.865, -78.824], [42.866, -78.866], [42.866, -78.858], [42.866, -78.847], [42.866, -78.828], [42.867, -78.863], [42.867, -78.861], [42.867, -78.86], [42.867, -78.857], [42.867, -78.848], [42.867, -78.832], [42.867, -78.823], [42.867, -78.811], [42.867, -78.806], [42.868, -78.864], [42.868, -78.863], [42.868, -78.86], [42.868, -78.85], [42.868, -78.838], [42.868, -78.81], [42.868, -78.809], [42.868, -78.808], [42.869, -78.864], [42.869, -78.861], [42.869, -78.859], [42.869, -78.858], [42.869, -78.851], [42.869, -78.846], [42.869, -78.834], [42.869, -78.833], [42.869, -78.808], [42.87, -78.846], [42.87, -78.834], [42.87, -78.832], [42.87, -78.822], [42.87, -78.805], [42.87, -78.804], [42.871, -78.858], [42.871, -78.852], [42.871, -78.85], [42.871, -78.834], [42.871, -78.816], [42.871, -78.815], [42.871, -78.812], [42.871, -78.807], [42.872, -78.868], [42.872, -78.864], [42.872, -78.86], [42.872, -78.842], [42.872, -78.821], [42.872, -78.82], [42.872, -78.812], [42.872, -78.81], [42.872, -78.809], [42.872, -78.803], [42.872, -78.802], [42.873, -78.862], [42.873, -78.859], [42.873, -78.813], [42.874, -78.871], [42.874, -78.864], [42.874, -78.862], [42.874, -78.826], [42.874, -78.825], [42.874, -78.804], [42.874, -78.801], [42.875, -78.877], [42.875, -78.874], [42.875, -78.867], [42.875, -78.853], [42.875, -78.85], [42.875, -78.804], [42.875, -78.802], [42.876, -78.856], [42.876, -78.83], [42.876, -78.804], [42.877, -78.867], [42.877, -78.851], [42.877, -78.846], [42.877, -78.825], [42.877, -78.804], [42.877, -78.803], [42.878, -78.882], [42.878, -78.881], [42.878, -78.869], [42.878, -78.855], [42.878, -78.848], [42.878, -78.834], [42.878, -78.816], [42.878, -78.815], [42.878, -78.811], [42.878, -78.801], [42.879, -78.885], [42.879, -78.876], [42.879, -78.87], [42.879, -78.858], [42.879, -78.855], [42.879, -78.841], [42.88, -78.88], [42.88, -78.86], [42.88, -78.858], [42.88, -78.853], [42.88, -78.841], [42.881, -78.88], [42.881, -78.869], [42.881, -78.852], [42.881, -78.833], [42.881, -78.807], [42.882, -78.878], [42.882, -78.876], [42.882, -78.875], [42.882, -78.873], [42.882, -78.865], [42.882, -78.859], [42.882, -78.856], [42.882, -78.852], [42.882, -78.84], [42.882, -78.835], [42.883, -78.878], [42.883, -78.875], [42.883, -78.873], [42.883, -78.867], [42.883, -78.866], [42.883, -78.861], [42.883, -78.859], [42.883, -78.849], [42.884, -78.88], [42.884, -78.879], [42.884, -78.878], [42.884, -78.877], [42.884, -78.875], [42.884, -78.873], [42.884, -78.86], [42.884, -78.857], [42.884, -78.856], [42.884, -78.854], [42.884, -78.839], [42.884, -78.833], [42.884, -78.822], [42.885, -78.888], [42.885, -78.879], [42.885, -78.878], [42.885, -78.877], [42.885, -78.875], [42.885, -78.874], [42.885, -78.87], [42.885, -78.869], [42.885, -78.858], [42.885, -78.857], [42.885, -78.855], [42.885, -78.853], [42.885, -78.848], [42.885, -78.845], [42.885, -78.84], [42.885, -78.839], [42.885, -78.835], [42.885, -78.822], [42.885, -78.814], [42.885, -78.81], [42.885, -78.807], [42.886, -78.879], [42.886, -78.878], [42.886, -78.877], [42.886, -78.876], [42.886, -78.874], [42.886, -78.873], [42.886, -78.872], [42.886, -78.871], [42.886, -78.859], [42.886, -78.844], [42.886, -78.841], [42.886, -78.839], [42.886, -78.82], [42.886, -78.815], [42.886, -78.804], [42.887, -78.879], [42.887, -78.878], [42.887, -78.875], [42.887, -78.872], [42.887, -78.863], [42.887, -78.856], [42.887, -78.844], [42.887, -78.842], [42.887, -78.839], [42.887, -78.837], [42.887, -78.815], [42.887, -78.81], [42.887, -78.802], [42.888, -78.882], [42.888, -78.876], [42.888, -78.875], [42.888, -78.874], [42.888, -78.872], [42.888, -78.855], [42.888, -78.85], [42.888, -78.845], [42.888, -78.838], [42.888, -78.837], [42.888, -78.815], [42.888, -78.814], [42.888, -78.81], [42.888, -78.808], [42.888, -78.806], [42.888, -78.805], [42.888, -78.803], [42.889, -78.884], [42.889, -78.883], [42.889, -78.881], [42.889, -78.858], [42.889, -78.804], [42.89, -78.882], [42.89, -78.879], [42.89, -78.877], [42.89, -78.875], [42.89, -78.873], [42.89, -78.872], [42.89, -78.869], [42.89, -78.859], [42.89, -78.846], [42.89, -78.841], [42.89, -78.812], [42.89, -78.81], [42.89, -78.807], [42.89, -78.806], [42.89, -78.805], [42.89, -78.803], [42.89, -78.802], [42.89, -78.801], [42.891, -78.885], [42.891, -78.884], [42.891, -78.881], [42.891, -78.88], [42.891, -78.879], [42.891, -78.878], [42.891, -78.877], [42.891, -78.874], [42.891, -78.867], [42.891, -78.859], [42.891, -78.845], [42.891, -78.836], [42.891, -78.834], [42.891, -78.832], [42.891, -78.826], [42.891, -78.825], [42.891, -78.803], [42.892, -78.891], [42.892, -78.89], [42.892, -78.888], [42.892, -78.884], [42.892, -78.882], [42.892, -78.881], [42.892, -78.88], [42.892, -78.878], [42.892, -78.877], [42.892, -78.875], [42.892, -78.87], [42.892, -78.862], [42.892, -78.84], [42.892, -78.835], [42.892, -78.831], [42.892, -78.824], [42.892, -78.823], [42.892, -78.808], [42.893, -78.892], [42.893, -78.891], [42.893, -78.89], [42.893, -78.888], [42.893, -78.885], [42.893, -78.883], [42.893, -78.881], [42.893, -78.877], [42.893, -78.875], [42.893, -78.873], [42.893, -78.872], [42.893, -78.871], [42.893, -78.87], [42.893, -78.868], [42.893, -78.853], [42.893, -78.839], [42.893, -78.837], [42.893, -78.83], [42.893, -78.828], [42.893, -78.823], [42.893, -78.814], [42.893, -78.806], [42.894, -78.89], [42.894, -78.886], [42.894, -78.883], [42.894, -78.881], [42.894, -78.876], [42.894, -78.871], [42.894, -78.849], [42.894, -78.842], [42.894, -78.84], [42.894, -78.837], [42.894, -78.835], [42.894, -78.834], [42.894, -78.833], [42.894, -78.83], [42.894, -78.828], [42.894, -78.826], [42.894, -78.824], [42.894, -78.822], [42.894, -78.819], [42.895, -78.89], [42.895, -78.887], [42.895, -78.886], [42.895, -78.884], [42.895, -78.877], [42.895, -78.876], [42.895, -78.872], [42.895, -78.871], [42.895, -78.868], [42.895, -78.866], [42.895, -78.862], [42.895, -78.847], [42.895, -78.843], [42.895, -78.842], [42.895, -78.839], [42.895, -78.838], [42.895, -78.836], [42.895, -78.832], [42.895, -78.83], [42.895, -78.829], [42.895, -78.828], [42.895, -78.826], [42.895, -78.823], [42.895, -78.822], [42.895, -78.819], [42.895, -78.818], [42.895, -78.809], [42.895, -78.803], [42.896, -78.888], [42.896, -78.886], [42.896, -78.879], [42.896, -78.875], [42.896, -78.874], [42.896, -78.873], [42.896, -78.866], [42.896, -78.862], [42.896, -78.855], [42.896, -78.843], [42.896, -78.841], [42.896, -78.836], [42.896, -78.832], [42.896, -78.827], [42.896, -78.824], [42.896, -78.821], [42.896, -78.82], [42.896, -78.818], [42.896, -78.817], [42.896, -78.815], [42.897, -78.901], [42.897, -78.894], [42.897, -78.89], [42.897, -78.889], [42.897, -78.888], [42.897, -78.887], [42.897, -78.88], [42.897, -78.878], [42.897, -78.877], [42.897, -78.874], [42.897, -78.872], [42.897, -78.871], [42.897, -78.869], [42.897, -78.866], [42.897, -78.841], [42.897, -78.84], [42.897, -78.838], [42.897, -78.833], [42.897, -78.825], [42.897, -78.822], [42.897, -78.814], [42.897, -78.812], [42.897, -78.808], [42.897, -78.803], [42.898, -78.89], [42.898, -78.885], [42.898, -78.882], [42.898, -78.88], [42.898, -78.876], [42.898, -78.875], [42.898, -78.874], [42.898, -78.873], [42.898, -78.859], [42.898, -78.848], [42.898, -78.839], [42.898, -78.835], [42.898, -78.831], [42.898, -78.82], [42.898, -78.817], [42.898, -78.807], [42.898, -78.801], [42.899, -78.895], [42.899, -78.893], [42.899, -78.892], [42.899, -78.891], [42.899, -78.889], [42.899, -78.887], [42.899, -78.886], [42.899, -78.885], [42.899, -78.883], [42.899, -78.882], [42.899, -78.881], [42.899, -78.879], [42.899, -78.878], [42.899, -78.877], [42.899, -78.873], [42.899, -78.872], [42.899, -78.87], [42.899, -78.861], [42.899, -78.848], [42.899, -78.837], [42.899, -78.835], [42.899, -78.823], [42.899, -78.819], [42.899, -78.818], [42.899, -78.816], [42.899, -78.815], [42.9, -78.892], [42.9, -78.891], [42.9, -78.89], [42.9, -78.888], [42.9, -78.886], [42.9, -78.885], [42.9, -78.884], [42.9, -78.88], [42.9, -78.878], [42.9, -78.875], [42.9, -78.874], [42.9, -78.873], [42.9, -78.872], [42.9, -78.871], [42.9, -78.87], [42.9, -78.869], [42.9, -78.867], [42.9, -78.865], [42.9, -78.863], [42.9, -78.86], [42.9, -78.855], [42.9, -78.846], [42.9, -78.84], [42.9, -78.834], [42.9, -78.833], [42.9, -78.831], [42.9, -78.826], [42.9, -78.823], [42.9, -78.819], [42.9, -78.817], [42.9, -78.815], [42.901, -78.892], [42.901, -78.891], [42.901, -78.89], [42.901, -78.889], [42.901, -78.888], [42.901, -78.877], [42.901, -78.874], [42.901, -78.871], [42.901, -78.867], [42.901, -78.865], [42.901, -78.862], [42.901, -78.861], [42.901, -78.859], [42.901, -78.852], [42.901, -78.844], [42.901, -78.838], [42.901, -78.834], [42.901, -78.832], [42.901, -78.827], [42.902, -78.895], [42.902, -78.891], [42.902, -78.888], [42.902, -78.887], [42.902, -78.881], [42.902, -78.88], [42.902, -78.878], [42.902, -78.877], [42.902, -78.875], [42.902, -78.873], [42.902, -78.871], [42.902, -78.869], [42.902, -78.867], [42.902, -78.863], [42.902, -78.862], [42.902, -78.861], [42.902, -78.858], [42.902, -78.854], [42.902, -78.851], [42.902, -78.839], [42.902, -78.826], [42.902, -78.825], [42.902, -78.824], [42.902, -78.821], [42.902, -78.816], [42.903, -78.895], [42.903, -78.893], [42.903, -78.89], [42.903, -78.889], [42.903, -78.881], [42.903, -78.876], [42.903, -78.869], [42.903, -78.865], [42.903, -78.864], [42.903, -78.861], [42.903, -78.856], [42.903, -78.839], [42.903, -78.836], [42.903, -78.835], [42.903, -78.833], [42.903, -78.821], [42.904, -78.9], [42.904, -78.897], [42.904, -78.884], [42.904, -78.881], [42.904, -78.871], [42.904, -78.869], [42.904, -78.866], [42.904, -78.864], [42.904, -78.851], [42.904, -78.84], [42.904, -78.832], [42.904, -78.819], [42.904, -78.816], [42.904, -78.813], [42.904, -78.809], [42.904, -78.808], [42.904, -78.801], [42.905, -78.89], [42.905, -78.888], [42.905, -78.885], [42.905, -78.883], [42.905, -78.882], [42.905, -78.881], [42.905, -78.879], [42.905, -78.877], [42.905, -78.861], [42.905, -78.846], [42.905, -78.84], [42.905, -78.832], [42.905, -78.83], [42.905, -78.816], [42.905, -78.814], [42.905, -78.812], [42.905, -78.811], [42.905, -78.809], [42.906, -78.894], [42.906, -78.893], [42.906, -78.891], [42.906, -78.888], [42.906, -78.887], [42.906, -78.886], [42.906, -78.882], [42.906, -78.881], [42.906, -78.878], [42.906, -78.86], [42.906, -78.857], [42.906, -78.849], [42.906, -78.848], [42.906, -78.833], [42.906, -78.824], [42.906, -78.821], [42.906, -78.813], [42.906, -78.811], [42.906, -78.81], [42.906, -78.807], [42.906, -78.805], [42.906, -78.803], [42.907, -78.898], [42.907, -78.897], [42.907, -78.896], [42.907, -78.893], [42.907, -78.892], [42.907, -78.89], [42.907, -78.884], [42.907, -78.883], [42.907, -78.88], [42.907, -78.878], [42.907, -78.872], [42.907, -78.871], [42.907, -78.87], [42.907, -78.859], [42.907, -78.837], [42.907, -78.836], [42.907, -78.831], [42.907, -78.828], [42.907, -78.825], [42.907, -78.817], [42.907, -78.813], [42.907, -78.812], [42.907, -78.811], [42.907, -78.805], [42.907, -78.802], [42.908, -78.899], [42.908, -78.898], [42.908, -78.897], [42.908, -78.896], [42.908, -78.895], [42.908, -78.891], [42.908, -78.889], [42.908, -78.888], [42.908, -78.887], [42.908, -78.884], [42.908, -78.881], [42.908, -78.88], [42.908, -78.878], [42.908, -78.872], [42.908, -78.87], [42.908, -78.867], [42.908, -78.864], [42.908, -78.861], [42.908, -78.855], [42.908, -78.848], [42.908, -78.827], [42.908, -78.821], [42.908, -78.814], [42.908, -78.807], [42.909, -78.896], [42.909, -78.894], [42.909, -78.893], [42.909, -78.891], [42.909, -78.889], [42.909, -78.885], [42.909, -78.884], [42.909, -78.883], [42.909, -78.882], [42.909, -78.879], [42.909, -78.878], [42.909, -78.876], [42.909, -78.875], [42.909, -78.873], [42.909, -78.872], [42.909, -78.871], [42.909, -78.869], [42.909, -78.866], [42.909, -78.859], [42.909, -78.85], [42.909, -78.848], [42.909, -78.844], [42.909, -78.834], [42.909, -78.823], [42.909, -78.822], [42.909, -78.817], [42.909, -78.805], [42.909, -78.804], [42.909, -78.803], [42.909, -78.801], [42.91, -78.9], [42.91, -78.899], [42.91, -78.898], [42.91, -78.896], [42.91, -78.895], [42.91, -78.891], [42.91, -78.883], [42.91, -78.881], [42.91, -78.877], [42.91, -78.875], [42.91, -78.873], [42.91, -78.871], [42.91, -78.868], [42.91, -78.867], [42.91, -78.866], [42.91, -78.865], [42.91, -78.855], [42.91, -78.854], [42.91, -78.847], [42.91, -78.841], [42.91, -78.837], [42.91, -78.836], [42.91, -78.827], [42.91, -78.825], [42.91, -78.815], [42.91, -78.814], [42.91, -78.813], [42.91, -78.809], [42.911, -78.899], [42.911, -78.896], [42.911, -78.895], [42.911, -78.893], [42.911, -78.891], [42.911, -78.89], [42.911, -78.888], [42.911, -78.887], [42.911, -78.886], [42.911, -78.882], [42.911, -78.881], [42.911, -78.877], [42.911, -78.872], [42.911, -78.869], [42.911, -78.868], [42.911, -78.866], [42.911, -78.855], [42.911, -78.854], [42.911, -78.853], [42.911, -78.849], [42.911, -78.846], [42.911, -78.845], [42.911, -78.844], [42.911, -78.843], [42.911, -78.834], [42.911, -78.826], [42.911, -78.818], [42.911, -78.815], [42.911, -78.814], [42.911, -78.81], [42.911, -78.806], [42.912, -78.893], [42.912, -78.892], [42.912, -78.89], [42.912, -78.887], [42.912, -78.882], [42.912, -78.877], [42.912, -78.871], [42.912, -78.869], [42.912, -78.868], [42.912, -78.867], [42.912, -78.866], [42.912, -78.865], [42.912, -78.857], [42.912, -78.854], [42.912, -78.851], [42.912, -78.845], [42.912, -78.839], [42.912, -78.834], [42.912, -78.828], [42.912, -78.821], [42.912, -78.818], [42.912, -78.808], [42.913, -78.899], [42.913, -78.898], [42.913, -78.897], [42.913, -78.895], [42.913, -78.887], [42.913, -78.885], [42.913, -78.884], [42.913, -78.883], [42.913, -78.88], [42.913, -78.879], [42.913, -78.877], [42.913, -78.876], [42.913, -78.861], [42.913, -78.857], [42.913, -78.844], [42.913, -78.843], [42.913, -78.837], [42.913, -78.836], [42.913, -78.83], [42.913, -78.828], [42.913, -78.825], [42.913, -78.815], [42.913, -78.814], [42.913, -78.81], [42.913, -78.801], [42.914, -78.901], [42.914, -78.899], [42.914, -78.894], [42.914, -78.893], [42.914, -78.89], [42.914, -78.884], [42.914, -78.883], [42.914, -78.882], [42.914, -78.881], [42.914, -78.88], [42.914, -78.879], [42.914, -78.871], [42.914, -78.87], [42.914, -78.869], [42.914, -78.865], [42.914, -78.86], [42.914, -78.846], [42.914, -78.841], [42.914, -78.838], [42.914, -78.83], [42.914, -78.827], [42.914, -78.826], [42.914, -78.824], [42.914, -78.803], [42.915, -78.897], [42.915, -78.896], [42.915, -78.894], [42.915, -78.893], [42.915, -78.891], [42.915, -78.89], [42.915, -78.888], [42.915, -78.887], [42.915, -78.885], [42.915, -78.882], [42.915, -78.88], [42.915, -78.877], [42.915, -78.874], [42.915, -78.871], [42.915, -78.87], [42.915, -78.865], [42.915, -78.863], [42.915, -78.86], [42.915, -78.859], [42.915, -78.857], [42.915, -78.856], [42.915, -78.85], [42.915, -78.848], [42.915, -78.847], [42.915, -78.829], [42.915, -78.825], [42.915, -78.816], [42.915, -78.815], [42.915, -78.809], [42.915, -78.808], [42.915, -78.807], [42.915, -78.804], [42.915, -78.802], [42.916, -78.894], [42.916, -78.889], [42.916, -78.887], [42.916, -78.886], [42.916, -78.885], [42.916, -78.884], [42.916, -78.873], [42.916, -78.872], [42.916, -78.865], [42.916, -78.864], [42.916, -78.858], [42.916, -78.857], [42.916, -78.855], [42.916, -78.847], [42.916, -78.845], [42.916, -78.84], [42.916, -78.827], [42.916, -78.821], [42.916, -78.815], [42.916, -78.814], [42.916, -78.811], [42.916, -78.81], [42.916, -78.809], [42.916, -78.802], [42.916, -78.801], [42.917, -78.899], [42.917, -78.891], [42.917, -78.89], [42.917, -78.888], [42.917, -78.887], [42.917, -78.885], [42.917, -78.884], [42.917, -78.883], [42.917, -78.878], [42.917, -78.865], [42.917, -78.864], [42.917, -78.863], [42.917, -78.859], [42.917, -78.855], [42.917, -78.854], [42.917, -78.852], [42.917, -78.85], [42.917, -78.826], [42.917, -78.825], [42.917, -78.824], [42.917, -78.807], [42.917, -78.805], [42.917, -78.803], [42.917, -78.801], [42.918, -78.897], [42.918, -78.895], [42.918, -78.891], [42.918, -78.889], [42.918, -78.888], [42.918, -78.887], [42.918, -78.883], [42.918, -78.877], [42.918, -78.868], [42.918, -78.865], [42.918, -78.855], [42.918, -78.854], [42.918, -78.848], [42.918, -78.847], [42.918, -78.829], [42.918, -78.813], [42.918, -78.809], [42.918, -78.805], [42.919, -78.9], [42.919, -78.899], [42.919, -78.898], [42.919, -78.893], [42.919, -78.89], [42.919, -78.869], [42.919, -78.868], [42.919, -78.865], [42.919, -78.855], [42.919, -78.854], [42.919, -78.849], [42.919, -78.847], [42.919, -78.843], [42.919, -78.833], [42.919, -78.828], [42.919, -78.814], [42.919, -78.811], [42.919, -78.808], [42.919, -78.804], [42.919, -78.803], [42.92, -78.896], [42.92, -78.89], [42.92, -78.871], [42.92, -78.864], [42.92, -78.855], [42.92, -78.851], [42.92, -78.834], [42.92, -78.827], [42.92, -78.814], [42.92, -78.811], [42.92, -78.807], [42.921, -78.892], [42.921, -78.891], [42.921, -78.887], [42.921, -78.886], [42.921, -78.881], [42.921, -78.874], [42.921, -78.87], [42.921, -78.869], [42.921, -78.865], [42.921, -78.864], [42.921, -78.862], [42.921, -78.859], [42.921, -78.85], [42.921, -78.849], [42.921, -78.824], [42.921, -78.815], [42.921, -78.808], [42.921, -78.804], [42.921, -78.803], [42.922, -78.899], [42.922, -78.894], [42.922, -78.89], [42.922, -78.888], [42.922, -78.886], [42.922, -78.88], [42.922, -78.878], [42.922, -78.877], [42.922, -78.874], [42.922, -78.872], [42.922, -78.871], [42.922, -78.864], [42.922, -78.862], [42.922, -78.857], [42.922, -78.854], [42.922, -78.853], [42.922, -78.852], [42.922, -78.85], [42.922, -78.849], [42.922, -78.835], [42.922, -78.832], [42.922, -78.829], [42.922, -78.827], [42.922, -78.809], [42.922, -78.806], [42.922, -78.803], [42.923, -78.899], [42.923, -78.897], [42.923, -78.896], [42.923, -78.89], [42.923, -78.886], [42.923, -78.88], [42.923, -78.875], [42.923, -78.874], [42.923, -78.868], [42.923, -78.856], [42.923, -78.841], [42.923, -78.83], [42.923, -78.823], [42.923, -78.821], [42.923, -78.814], [42.923, -78.812], [42.923, -78.811], [42.923, -78.81], [42.923, -78.809], [42.923, -78.807], [42.923, -78.806], [42.923, -78.805], [42.923, -78.804], [42.924, -78.893], [42.924, -78.892], [42.924, -78.891], [42.924, -78.887], [42.924, -78.886], [42.924, -78.883], [42.924, -78.877], [42.924, -78.875], [42.924, -78.87], [42.924, -78.869], [42.924, -78.852], [42.924, -78.828], [42.924, -78.825], [42.924, -78.817], [42.924, -78.815], [42.924, -78.813], [42.924, -78.812], [42.924, -78.807], [42.924, -78.802], [42.925, -78.897], [42.925, -78.896], [42.925, -78.893], [42.925, -78.891], [42.925, -78.889], [42.925, -78.88], [42.925, -78.845], [42.925, -78.829], [42.925, -78.828], [42.925, -78.82], [42.925, -78.817], [42.925, -78.814], [42.925, -78.808], [42.925, -78.807], [42.925, -78.806], [42.925, -78.805], [42.926, -78.898], [42.926, -78.892], [42.926, -78.89], [42.926, -78.888], [42.926, -78.887], [42.926, -78.882], [42.926, -78.877], [42.926, -78.876], [42.926, -78.871], [42.926, -78.837], [42.926, -78.827], [42.926, -78.82], [42.926, -78.817], [42.926, -78.815], [42.926, -78.814], [42.926, -78.813], [42.926, -78.81], [42.926, -78.808], [42.926, -78.802], [42.927, -78.893], [42.927, -78.891], [42.927, -78.89], [42.927, -78.886], [42.927, -78.883], [42.927, -78.877], [42.927, -78.875], [42.927, -78.851], [42.927, -78.848], [42.927, -78.839], [42.927, -78.822], [42.927, -78.819], [42.927, -78.817], [42.927, -78.815], [42.927, -78.814], [42.927, -78.811], [42.927, -78.809], [42.927, -78.805], [42.927, -78.802], [42.928, -78.89], [42.928, -78.889], [42.928, -78.888], [42.928, -78.885], [42.928, -78.884], [42.928, -78.877], [42.928, -78.875], [42.928, -78.868], [42.928, -78.85], [42.928, -78.824], [42.928, -78.821], [42.928, -78.819], [42.928, -78.811], [42.928, -78.81], [42.928, -78.807], [42.929, -78.891], [42.929, -78.89], [42.929, -78.85], [42.929, -78.838], [42.929, -78.827], [42.929, -78.824], [42.929, -78.819], [42.929, -78.814], [42.929, -78.81], [42.929, -78.803], [42.929, -78.802], [42.93, -78.898], [42.93, -78.891], [42.93, -78.874], [42.93, -78.853], [42.93, -78.841], [42.93, -78.824], [42.93, -78.817], [42.93, -78.814], [42.93, -78.813], [42.93, -78.811], [42.93, -78.81], [42.93, -78.809], [42.93, -78.808], [42.93, -78.807], [42.93, -78.806], [42.93, -78.803], [42.93, -78.802], [42.931, -78.899], [42.931, -78.849], [42.931, -78.848], [42.931, -78.844], [42.931, -78.84], [42.931, -78.838], [42.931, -78.837], [42.931, -78.835], [42.931, -78.825], [42.931, -78.824], [42.931, -78.811], [42.931, -78.804], [42.931, -78.801], [42.932, -78.889], [42.932, -78.877], [42.932, -78.845], [42.932, -78.841], [42.932, -78.832], [42.932, -78.829], [42.932, -78.825], [42.932, -78.819], [42.932, -78.814], [42.932, -78.81], [42.932, -78.803], [42.932, -78.802], [42.933, -78.875], [42.933, -78.853], [42.933, -78.851], [42.933, -78.848], [42.933, -78.841], [42.933, -78.839], [42.933, -78.828], [42.933, -78.826], [42.933, -78.819], [42.933, -78.818], [42.933, -78.815], [42.933, -78.814], [42.933, -78.811], [42.933, -78.81], [42.934, -78.841], [42.934, -78.838], [42.934, -78.832], [42.934, -78.831], [42.934, -78.819], [42.934, -78.816], [42.934, -78.815], [42.934, -78.814], [42.934, -78.813], [42.934, -78.812], [42.934, -78.809], [42.934, -78.803], [42.935, -78.869], [42.935, -78.866], [42.935, -78.844], [42.935, -78.834], [42.935, -78.831], [42.935, -78.821], [42.935, -78.815], [42.935, -78.812], [42.935, -78.81], [42.935, -78.803], [42.935, -78.802], [42.936, -78.894], [42.936, -78.851], [42.936, -78.843], [42.936, -78.838], [42.936, -78.837], [42.936, -78.836], [42.936, -78.835], [42.936, -78.834], [42.936, -78.833], [42.936, -78.822], [42.936, -78.816], [42.936, -78.812], [42.936, -78.811], [42.936, -78.807], [42.936, -78.801], [42.937, -78.901], [42.937, -78.9], [42.937, -78.899], [42.937, -78.895], [42.937, -78.893], [42.937, -78.892], [42.937, -78.844], [42.937, -78.842], [42.937, -78.828], [42.937, -78.822], [42.937, -78.819], [42.937, -78.817], [42.937, -78.816], [42.937, -78.814], [42.937, -78.804], [42.937, -78.801], [42.938, -78.902], [42.938, -78.9], [42.938, -78.899], [42.938, -78.89], [42.938, -78.889], [42.938, -78.88], [42.938, -78.875], [42.938, -78.868], [42.938, -78.867], [42.938, -78.851], [42.938, -78.843], [42.938, -78.842], [42.938, -78.84], [42.938, -78.822], [42.938, -78.819], [42.938, -78.817], [42.938, -78.816], [42.938, -78.814], [42.938, -78.813], [42.938, -78.811], [42.938, -78.81], [42.938, -78.809], [42.939, -78.905], [42.939, -78.901], [42.939, -78.9], [42.939, -78.899], [42.939, -78.894], [42.939, -78.891], [42.939, -78.888], [42.939, -78.875], [42.939, -78.869], [42.939, -78.84], [42.939, -78.839], [42.939, -78.838], [42.939, -78.835], [42.939, -78.834], [42.939, -78.833], [42.939, -78.817], [42.939, -78.816], [42.939, -78.815], [42.939, -78.814], [42.939, -78.813], [42.939, -78.81], [42.939, -78.809], [42.939, -78.807], [42.94, -78.906], [42.94, -78.904], [42.94, -78.893], [42.94, -78.889], [42.94, -78.887], [42.94, -78.875], [42.94, -78.867], [42.94, -78.865], [42.94, -78.861], [42.94, -78.851], [42.94, -78.85], [42.94, -78.846], [42.94, -78.845], [42.94, -78.844], [42.94, -78.843], [42.94, -78.839], [42.94, -78.838], [42.94, -78.835], [42.94, -78.831], [42.94, -78.83], [42.94, -78.826], [42.94, -78.824], [42.94, -78.823], [42.94, -78.821], [42.94, -78.816], [42.94, -78.814], [42.94, -78.813], [42.94, -78.81], [42.94, -78.809], [42.941, -78.904], [42.941, -78.894], [42.941, -78.893], [42.941, -78.891], [42.941, -78.89], [42.941, -78.888], [42.941, -78.883], [42.941, -78.879], [42.941, -78.878], [42.941, -78.875], [42.941, -78.871], [42.941, -78.868], [42.941, -78.856], [42.941, -78.852], [42.941, -78.838], [42.941, -78.835], [42.941, -78.822], [42.941, -78.814], [42.941, -78.809], [42.941, -78.806], [42.941, -78.803], [42.942, -78.901], [42.942, -78.9], [42.942, -78.885], [42.942, -78.883], [42.942, -78.878], [42.942, -78.876], [42.942, -78.868], [42.942, -78.867], [42.942, -78.849], [42.942, -78.845], [42.942, -78.828], [42.942, -78.822], [42.942, -78.818], [42.942, -78.815], [42.942, -78.812], [42.942, -78.811], [42.942, -78.807], [42.942, -78.805], [42.942, -78.804], [42.942, -78.802], [42.943, -78.903], [42.943, -78.898], [42.943, -78.887], [42.943, -78.868], [42.943, -78.821], [42.943, -78.811], [42.943, -78.809], [42.943, -78.807], [42.943, -78.804], [42.944, -78.908], [42.944, -78.907], [42.944, -78.901], [42.944, -78.9], [42.944, -78.898], [42.944, -78.868], [42.944, -78.867], [42.944, -78.863], [42.944, -78.859], [42.944, -78.853], [42.944, -78.833], [42.944, -78.822], [42.944, -78.82], [42.944, -78.818], [42.944, -78.817], [42.944, -78.815], [42.944, -78.811], [42.944, -78.81], [42.945, -78.905], [42.945, -78.887], [42.945, -78.886], [42.945, -78.873], [42.945, -78.87], [42.945, -78.859], [42.945, -78.847], [42.945, -78.846], [42.945, -78.835], [42.945, -78.829], [42.945, -78.826], [42.945, -78.819], [42.945, -78.818], [42.945, -78.817], [42.945, -78.814], [42.945, -78.811], [42.946, -78.906], [42.946, -78.892], [42.946, -78.891], [42.946, -78.84], [42.946, -78.832], [42.946, -78.826], [42.946, -78.823], [42.946, -78.818], [42.946, -78.817], [42.946, -78.815], [42.946, -78.813], [42.946, -78.812], [42.946, -78.81], [42.946, -78.806], [42.946, -78.804], [42.946, -78.803], [42.947, -78.906], [42.947, -78.894], [42.947, -78.893], [42.947, -78.891], [42.947, -78.889], [42.947, -78.888], [42.947, -78.887], [42.947, -78.877], [42.947, -78.872], [42.947, -78.869], [42.947, -78.865], [42.947, -78.86], [42.947, -78.858], [42.947, -78.849], [42.947, -78.848], [42.947, -78.847], [42.947, -78.833], [42.947, -78.832], [42.947, -78.83], [42.947, -78.828], [42.947, -78.826], [42.947, -78.819], [42.947, -78.818], [42.947, -78.814], [42.947, -78.813], [42.947, -78.809], [42.947, -78.807], [42.947, -78.806], [42.948, -78.907], [42.948, -78.903], [42.948, -78.9], [42.948, -78.898], [42.948, -78.888], [42.948, -78.884], [42.948, -78.874], [42.948, -78.871], [42.948, -78.866], [42.948, -78.865], [42.948, -78.863], [42.948, -78.859], [42.948, -78.853], [42.948, -78.843], [42.948, -78.835], [42.948, -78.829], [42.948, -78.827], [42.948, -78.826], [42.948, -78.823], [42.948, -78.82], [42.948, -78.817], [42.948, -78.816], [42.948, -78.814], [42.948, -78.813], [42.948, -78.811], [42.948, -78.81], [42.948, -78.809], [42.948, -78.804], [42.949, -78.899], [42.949, -78.898], [42.949, -78.889], [42.949, -78.887], [42.949, -78.882], [42.949, -78.863], [42.949, -78.849], [42.949, -78.828], [42.949, -78.822], [42.949, -78.819], [42.949, -78.818], [42.949, -78.814], [42.949, -78.813], [42.949, -78.811], [42.95, -78.908], [42.95, -78.907], [42.95, -78.906], [42.95, -78.902], [42.95, -78.888], [42.95, -78.884], [42.95, -78.881], [42.95, -78.842], [42.95, -78.841], [42.95, -78.838], [42.95, -78.831], [42.95, -78.83], [42.95, -78.829], [42.95, -78.824], [42.95, -78.823], [42.95, -78.822], [42.951, -78.908], [42.951, -78.907], [42.951, -78.904], [42.951, -78.903], [42.951, -78.902], [42.951, -78.9], [42.951, -78.898], [42.951, -78.873], [42.951, -78.869], [42.951, -78.861], [42.951, -78.86], [42.951, -78.827], [42.951, -78.826], [42.951, -78.813], [42.952, -78.908], [42.952, -78.906], [42.952, -78.905], [42.952, -78.904], [42.952, -78.901], [42.952, -78.877], [42.952, -78.865], [42.952, -78.831], [42.952, -78.828], [42.953, -78.906], [42.953, -78.905], [42.953, -78.901], [42.953, -78.875], [42.953, -78.864], [42.953, -78.859], [42.953, -78.855], [42.953, -78.85], [42.953, -78.848], [42.953, -78.833], [42.953, -78.831], [42.954, -78.906], [42.954, -78.905], [42.954, -78.904], [42.954, -78.903], [42.954, -78.899], [42.954, -78.898], [42.954, -78.897], [42.954, -78.888], [42.954, -78.879], [42.954, -78.878], [42.954, -78.874], [42.954, -78.865], [42.954, -78.863], [42.954, -78.833], [42.954, -78.826], [42.954, -78.823], [42.955, -78.906], [42.955, -78.905], [42.955, -78.902], [42.955, -78.898], [42.955, -78.897], [42.955, -78.896], [42.955, -78.874], [42.955, -78.87], [42.955, -78.861], [42.955, -78.838], [42.955, -78.829], [42.955, -78.827], [42.955, -78.826], [42.956, -78.906], [42.956, -78.886], [42.956, -78.87], [42.956, -78.869], [42.956, -78.863], [42.956, -78.852], [42.956, -78.832], [42.956, -78.828], [42.957, -78.897], [42.957, -78.896], [42.957, -78.886], [42.957, -78.872], [42.957, -78.87], [42.957, -78.866], [42.957, -78.863], [42.957, -78.862], [42.957, -78.861], [42.957, -78.859], [42.957, -78.823], [42.957, -78.82], [42.957, -78.819], [42.958, -78.907], [42.958, -78.903], [42.958, -78.902], [42.958, -78.901], [42.958, -78.897], [42.958, -78.896], [42.958, -78.895], [42.958, -78.886], [42.958, -78.882], [42.958, -78.869], [42.958, -78.862], [42.958, -78.836], [42.958, -78.831], [42.958, -78.818], [42.959, -78.908], [42.959, -78.905], [42.959, -78.904], [42.959, -78.879], [42.959, -78.859], [42.96, -78.907], [42.96, -78.904], [42.96, -78.903], [42.96, -78.897], [42.961, -78.9], [42.962, -78.902], [42.963, -78.901]],
                {&quot;blur&quot;: 15, &quot;maxZoom&quot;: 12, &quot;minOpacity&quot;: 0.5, &quot;radius&quot;: 8}
            ).addTo(map_b2d3610360388637adde166587c3edf2);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
# Heatmap --> 2009 Summer Nights
base_map = generateBaseMap()
HeatMap(data=summer_2009_night[['latitude', 'longitude']].\
        groupby(['latitude', 'longitude']).sum().reset_index().values.tolist(), radius=8, max_zoom=12).add_to(base_map)

base_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_47c8eb93fed5beeb59f73a3290f30084 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js&quot;&gt;&lt;/script&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_47c8eb93fed5beeb59f73a3290f30084&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_47c8eb93fed5beeb59f73a3290f30084 = L.map(
                &quot;map_47c8eb93fed5beeb59f73a3290f30084&quot;,
                {
                    center: [42.91184928528912, -78.84964614694492],
                    crs: L.CRS.EPSG3857,
                    zoom: 12,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );
            L.control.scale().addTo(map_47c8eb93fed5beeb59f73a3290f30084);





            var tile_layer_92cd35b25359ab47c14d05334ea57c5b = L.tileLayer(
                &quot;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;Data by \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://openstreetmap.org\&quot;\u003eOpenStreetMap\u003c/a\u003e, under \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eODbL\u003c/a\u003e.&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_47c8eb93fed5beeb59f73a3290f30084);


            var heat_map_829314c14f9dce9ef24a70bca4466df6 = L.heatLayer(
                [[42.826, -78.834], [42.832, -78.815], [42.833, -78.824], [42.833, -78.822], [42.833, -78.821], [42.834, -78.816], [42.834, -78.813], [42.834, -78.812], [42.835, -78.826], [42.835, -78.825], [42.835, -78.824], [42.835, -78.811], [42.835, -78.81], [42.835, -78.809], [42.835, -78.802], [42.836, -78.82], [42.836, -78.816], [42.836, -78.808], [42.836, -78.807], [42.837, -78.824], [42.837, -78.823], [42.837, -78.82], [42.837, -78.812], [42.837, -78.808], [42.837, -78.805], [42.837, -78.802], [42.838, -78.857], [42.838, -78.832], [42.838, -78.825], [42.839, -78.857], [42.839, -78.819], [42.84, -78.826], [42.84, -78.803], [42.841, -78.824], [42.841, -78.818], [42.841, -78.815], [42.841, -78.809], [42.842, -78.831], [42.842, -78.827], [42.842, -78.826], [42.842, -78.825], [42.842, -78.824], [42.842, -78.811], [42.842, -78.808], [42.843, -78.822], [42.843, -78.82], [42.843, -78.813], [42.843, -78.81], [42.844, -78.832], [42.844, -78.826], [42.844, -78.824], [42.845, -78.824], [42.845, -78.823], [42.845, -78.813], [42.845, -78.808], [42.846, -78.83], [42.846, -78.824], [42.846, -78.821], [42.846, -78.814], [42.847, -78.832], [42.847, -78.811], [42.848, -78.832], [42.848, -78.824], [42.848, -78.823], [42.848, -78.822], [42.848, -78.813], [42.848, -78.812], [42.848, -78.808], [42.848, -78.803], [42.849, -78.863], [42.849, -78.83], [42.849, -78.823], [42.849, -78.817], [42.849, -78.813], [42.849, -78.812], [42.85, -78.833], [42.85, -78.828], [42.85, -78.827], [42.85, -78.826], [42.85, -78.824], [42.85, -78.823], [42.85, -78.821], [42.85, -78.816], [42.85, -78.815], [42.85, -78.814], [42.85, -78.813], [42.85, -78.811], [42.85, -78.805], [42.85, -78.804], [42.85, -78.803], [42.85, -78.801], [42.851, -78.832], [42.851, -78.828], [42.851, -78.827], [42.851, -78.825], [42.851, -78.816], [42.851, -78.813], [42.851, -78.806], [42.851, -78.805], [42.851, -78.803], [42.851, -78.801], [42.852, -78.828], [42.852, -78.827], [42.852, -78.826], [42.852, -78.824], [42.852, -78.823], [42.852, -78.822], [42.852, -78.817], [42.852, -78.814], [42.852, -78.801], [42.853, -78.83], [42.853, -78.826], [42.853, -78.823], [42.853, -78.821], [42.853, -78.819], [42.853, -78.813], [42.853, -78.811], [42.853, -78.807], [42.854, -78.832], [42.854, -78.831], [42.854, -78.83], [42.854, -78.827], [42.854, -78.82], [42.854, -78.812], [42.854, -78.808], [42.854, -78.807], [42.855, -78.833], [42.855, -78.832], [42.855, -78.83], [42.855, -78.828], [42.855, -78.827], [42.855, -78.825], [42.855, -78.824], [42.855, -78.814], [42.855, -78.812], [42.855, -78.81], [42.855, -78.808], [42.855, -78.806], [42.855, -78.801], [42.856, -78.832], [42.856, -78.831], [42.856, -78.829], [42.856, -78.826], [42.856, -78.824], [42.856, -78.821], [42.856, -78.812], [42.856, -78.811], [42.856, -78.81], [42.856, -78.809], [42.856, -78.808], [42.856, -78.807], [42.857, -78.833], [42.857, -78.83], [42.857, -78.829], [42.857, -78.828], [42.857, -78.816], [42.857, -78.814], [42.857, -78.811], [42.857, -78.809], [42.857, -78.808], [42.857, -78.807], [42.858, -78.833], [42.858, -78.831], [42.858, -78.827], [42.858, -78.825], [42.858, -78.823], [42.858, -78.82], [42.858, -78.817], [42.858, -78.813], [42.858, -78.811], [42.859, -78.827], [42.859, -78.825], [42.859, -78.824], [42.859, -78.816], [42.859, -78.815], [42.859, -78.814], [42.859, -78.812], [42.859, -78.81], [42.859, -78.809], [42.86, -78.825], [42.86, -78.822], [42.86, -78.821], [42.86, -78.814], [42.86, -78.812], [42.86, -78.805], [42.861, -78.839], [42.861, -78.821], [42.861, -78.82], [42.861, -78.819], [42.861, -78.818], [42.862, -78.819], [42.862, -78.814], [42.862, -78.812], [42.863, -78.821], [42.863, -78.82], [42.863, -78.818], [42.863, -78.814], [42.863, -78.811], [42.864, -78.826], [42.864, -78.825], [42.865, -78.865], [42.865, -78.837], [42.865, -78.807], [42.866, -78.866], [42.866, -78.865], [42.866, -78.86], [42.866, -78.849], [42.866, -78.822], [42.867, -78.862], [42.867, -78.861], [42.867, -78.857], [42.867, -78.85], [42.867, -78.849], [42.867, -78.807], [42.867, -78.805], [42.868, -78.848], [42.868, -78.846], [42.868, -78.837], [42.868, -78.81], [42.868, -78.809], [42.868, -78.807], [42.868, -78.805], [42.868, -78.804], [42.868, -78.802], [42.869, -78.861], [42.869, -78.859], [42.869, -78.853], [42.869, -78.848], [42.869, -78.83], [42.869, -78.812], [42.869, -78.81], [42.869, -78.809], [42.869, -78.808], [42.869, -78.807], [42.87, -78.86], [42.87, -78.849], [42.87, -78.848], [42.87, -78.839], [42.87, -78.836], [42.87, -78.834], [42.87, -78.809], [42.87, -78.808], [42.871, -78.883], [42.871, -78.867], [42.871, -78.861], [42.871, -78.859], [42.871, -78.856], [42.871, -78.852], [42.871, -78.838], [42.871, -78.834], [42.871, -78.831], [42.871, -78.805], [42.872, -78.865], [42.872, -78.844], [42.872, -78.843], [42.872, -78.834], [42.872, -78.833], [42.872, -78.832], [42.872, -78.82], [42.872, -78.812], [42.872, -78.811], [42.872, -78.806], [42.872, -78.804], [42.872, -78.803], [42.872, -78.802], [42.873, -78.868], [42.873, -78.864], [42.873, -78.859], [42.873, -78.842], [42.873, -78.813], [42.873, -78.81], [42.873, -78.808], [42.873, -78.803], [42.874, -78.871], [42.874, -78.866], [42.874, -78.864], [42.874, -78.851], [42.874, -78.844], [42.874, -78.827], [42.874, -78.817], [42.874, -78.806], [42.874, -78.805], [42.874, -78.802], [42.874, -78.801], [42.875, -78.874], [42.875, -78.844], [42.875, -78.843], [42.875, -78.806], [42.875, -78.805], [42.875, -78.802], [42.876, -78.856], [42.876, -78.844], [42.876, -78.843], [42.876, -78.829], [42.876, -78.828], [42.876, -78.827], [42.876, -78.804], [42.877, -78.845], [42.877, -78.844], [42.877, -78.841], [42.877, -78.827], [42.877, -78.803], [42.878, -78.881], [42.878, -78.88], [42.878, -78.836], [42.878, -78.834], [42.878, -78.801], [42.879, -78.885], [42.879, -78.88], [42.879, -78.878], [42.879, -78.874], [42.879, -78.87], [42.879, -78.866], [42.879, -78.861], [42.879, -78.856], [42.879, -78.855], [42.879, -78.843], [42.879, -78.839], [42.879, -78.838], [42.879, -78.835], [42.88, -78.88], [42.88, -78.878], [42.88, -78.866], [42.88, -78.86], [42.88, -78.846], [42.88, -78.841], [42.88, -78.802], [42.881, -78.874], [42.881, -78.871], [42.881, -78.855], [42.881, -78.843], [42.881, -78.84], [42.881, -78.836], [42.881, -78.832], [42.881, -78.831], [42.882, -78.877], [42.882, -78.876], [42.882, -78.873], [42.882, -78.871], [42.882, -78.869], [42.882, -78.868], [42.882, -78.859], [42.882, -78.857], [42.882, -78.856], [42.882, -78.85], [42.882, -78.849], [42.882, -78.841], [42.882, -78.836], [42.882, -78.833], [42.883, -78.878], [42.883, -78.874], [42.883, -78.873], [42.883, -78.87], [42.883, -78.867], [42.883, -78.866], [42.883, -78.861], [42.883, -78.86], [42.883, -78.859], [42.883, -78.857], [42.883, -78.854], [42.883, -78.844], [42.883, -78.839], [42.883, -78.835], [42.883, -78.832], [42.884, -78.879], [42.884, -78.878], [42.884, -78.874], [42.884, -78.868], [42.884, -78.866], [42.884, -78.86], [42.884, -78.857], [42.884, -78.855], [42.884, -78.854], [42.884, -78.852], [42.884, -78.845], [42.884, -78.84], [42.884, -78.839], [42.884, -78.833], [42.885, -78.887], [42.885, -78.879], [42.885, -78.878], [42.885, -78.877], [42.885, -78.874], [42.885, -78.873], [42.885, -78.868], [42.885, -78.862], [42.885, -78.86], [42.885, -78.858], [42.885, -78.856], [42.885, -78.855], [42.885, -78.854], [42.885, -78.84], [42.885, -78.839], [42.885, -78.814], [42.885, -78.808], [42.885, -78.804], [42.885, -78.803], [42.886, -78.88], [42.886, -78.878], [42.886, -78.875], [42.886, -78.874], [42.886, -78.873], [42.886, -78.872], [42.886, -78.871], [42.886, -78.848], [42.886, -78.844], [42.886, -78.839], [42.886, -78.837], [42.886, -78.82], [42.886, -78.807], [42.886, -78.805], [42.886, -78.802], [42.887, -78.885], [42.887, -78.878], [42.887, -78.873], [42.887, -78.872], [42.887, -78.871], [42.887, -78.868], [42.887, -78.854], [42.887, -78.851], [42.887, -78.842], [42.887, -78.815], [42.887, -78.813], [42.887, -78.81], [42.887, -78.808], [42.887, -78.806], [42.887, -78.804], [42.887, -78.803], [42.888, -78.882], [42.888, -78.88], [42.888, -78.876], [42.888, -78.874], [42.888, -78.872], [42.888, -78.861], [42.888, -78.854], [42.888, -78.847], [42.888, -78.838], [42.888, -78.837], [42.888, -78.809], [42.888, -78.808], [42.888, -78.805], [42.888, -78.804], [42.889, -78.881], [42.889, -78.876], [42.889, -78.875], [42.889, -78.874], [42.889, -78.871], [42.889, -78.87], [42.889, -78.867], [42.889, -78.859], [42.889, -78.854], [42.889, -78.851], [42.889, -78.839], [42.889, -78.811], [42.889, -78.81], [42.889, -78.805], [42.89, -78.882], [42.89, -78.878], [42.89, -78.877], [42.89, -78.876], [42.89, -78.875], [42.89, -78.874], [42.89, -78.873], [42.89, -78.872], [42.89, -78.861], [42.89, -78.854], [42.89, -78.84], [42.89, -78.838], [42.89, -78.836], [42.89, -78.834], [42.89, -78.832], [42.89, -78.815], [42.89, -78.814], [42.89, -78.811], [42.89, -78.808], [42.89, -78.807], [42.89, -78.806], [42.89, -78.805], [42.89, -78.804], [42.89, -78.803], [42.89, -78.802], [42.89, -78.801], [42.891, -78.885], [42.891, -78.884], [42.891, -78.881], [42.891, -78.88], [42.891, -78.879], [42.891, -78.878], [42.891, -78.877], [42.891, -78.876], [42.891, -78.875], [42.891, -78.874], [42.891, -78.873], [42.891, -78.871], [42.891, -78.863], [42.891, -78.861], [42.891, -78.844], [42.891, -78.839], [42.891, -78.836], [42.891, -78.833], [42.891, -78.83], [42.891, -78.825], [42.891, -78.824], [42.891, -78.812], [42.891, -78.81], [42.891, -78.808], [42.891, -78.803], [42.891, -78.801], [42.892, -78.891], [42.892, -78.89], [42.892, -78.884], [42.892, -78.881], [42.892, -78.88], [42.892, -78.878], [42.892, -78.877], [42.892, -78.876], [42.892, -78.874], [42.892, -78.872], [42.892, -78.871], [42.892, -78.862], [42.892, -78.859], [42.892, -78.85], [42.892, -78.848], [42.892, -78.847], [42.892, -78.846], [42.892, -78.842], [42.892, -78.84], [42.892, -78.836], [42.892, -78.834], [42.892, -78.832], [42.892, -78.83], [42.892, -78.823], [42.892, -78.808], [42.892, -78.807], [42.892, -78.804], [42.892, -78.803], [42.892, -78.802], [42.893, -78.892], [42.893, -78.891], [42.893, -78.89], [42.893, -78.881], [42.893, -78.879], [42.893, -78.878], [42.893, -78.876], [42.893, -78.874], [42.893, -78.873], [42.893, -78.872], [42.893, -78.871], [42.893, -78.868], [42.893, -78.851], [42.893, -78.846], [42.893, -78.843], [42.893, -78.842], [42.893, -78.84], [42.893, -78.839], [42.893, -78.837], [42.893, -78.834], [42.893, -78.833], [42.893, -78.831], [42.893, -78.825], [42.893, -78.824], [42.893, -78.823], [42.893, -78.821], [42.893, -78.806], [42.894, -78.892], [42.894, -78.891], [42.894, -78.89], [42.894, -78.884], [42.894, -78.876], [42.894, -78.874], [42.894, -78.871], [42.894, -78.865], [42.894, -78.862], [42.894, -78.86], [42.894, -78.845], [42.894, -78.843], [42.894, -78.842], [42.894, -78.84], [42.894, -78.835], [42.894, -78.834], [42.894, -78.832], [42.894, -78.83], [42.894, -78.827], [42.894, -78.824], [42.894, -78.822], [42.894, -78.82], [42.894, -78.809], [42.895, -78.892], [42.895, -78.884], [42.895, -78.877], [42.895, -78.875], [42.895, -78.871], [42.895, -78.87], [42.895, -78.865], [42.895, -78.854], [42.895, -78.842], [42.895, -78.841], [42.895, -78.84], [42.895, -78.839], [42.895, -78.838], [42.895, -78.835], [42.895, -78.832], [42.895, -78.83], [42.895, -78.829], [42.895, -78.827], [42.895, -78.825], [42.895, -78.824], [42.895, -78.822], [42.895, -78.821], [42.895, -78.816], [42.895, -78.809], [42.895, -78.805], [42.896, -78.891], [42.896, -78.889], [42.896, -78.888], [42.896, -78.885], [42.896, -78.884], [42.896, -78.883], [42.896, -78.882], [42.896, -78.875], [42.896, -78.874], [42.896, -78.873], [42.896, -78.866], [42.896, -78.865], [42.896, -78.863], [42.896, -78.862], [42.896, -78.855], [42.896, -78.854], [42.896, -78.851], [42.896, -78.845], [42.896, -78.842], [42.896, -78.836], [42.896, -78.835], [42.896, -78.833], [42.896, -78.832], [42.896, -78.827], [42.896, -78.825], [42.896, -78.824], [42.896, -78.822], [42.896, -78.821], [42.896, -78.82], [42.896, -78.819], [42.896, -78.818], [42.896, -78.807], [42.896, -78.805], [42.896, -78.804], [42.896, -78.801], [42.897, -78.891], [42.897, -78.888], [42.897, -78.886], [42.897, -78.883], [42.897, -78.88], [42.897, -78.879], [42.897, -78.876], [42.897, -78.875], [42.897, -78.872], [42.897, -78.868], [42.897, -78.867], [42.897, -78.863], [42.897, -78.859], [42.897, -78.849], [42.897, -78.838], [42.897, -78.835], [42.897, -78.83], [42.897, -78.825], [42.897, -78.823], [42.897, -78.822], [42.897, -78.818], [42.897, -78.817], [42.897, -78.816], [42.897, -78.814], [42.897, -78.813], [42.897, -78.812], [42.897, -78.808], [42.897, -78.807], [42.897, -78.805], [42.898, -78.9], [42.898, -78.893], [42.898, -78.891], [42.898, -78.89], [42.898, -78.889], [42.898, -78.888], [42.898, -78.887], [42.898, -78.886], [42.898, -78.885], [42.898, -78.884], [42.898, -78.882], [42.898, -78.881], [42.898, -78.879], [42.898, -78.875], [42.898, -78.874], [42.898, -78.873], [42.898, -78.871], [42.898, -78.845], [42.898, -78.842], [42.898, -78.836], [42.898, -78.834], [42.898, -78.831], [42.898, -78.824], [42.898, -78.82], [42.898, -78.819], [42.898, -78.817], [42.898, -78.815], [42.898, -78.807], [42.898, -78.806], [42.898, -78.803], [42.898, -78.801], [42.899, -78.897], [42.899, -78.892], [42.899, -78.89], [42.899, -78.887], [42.899, -78.88], [42.899, -78.879], [42.899, -78.878], [42.899, -78.877], [42.899, -78.876], [42.899, -78.875], [42.899, -78.874], [42.899, -78.873], [42.899, -78.872], [42.899, -78.871], [42.899, -78.87], [42.899, -78.86], [42.899, -78.858], [42.899, -78.853], [42.899, -78.843], [42.899, -78.841], [42.899, -78.84], [42.899, -78.837], [42.899, -78.834], [42.899, -78.832], [42.899, -78.831], [42.899, -78.822], [42.899, -78.819], [42.899, -78.818], [42.899, -78.816], [42.9, -78.892], [42.9, -78.891], [42.9, -78.89], [42.9, -78.889], [42.9, -78.888], [42.9, -78.887], [42.9, -78.885], [42.9, -78.881], [42.9, -78.879], [42.9, -78.878], [42.9, -78.877], [42.9, -78.876], [42.9, -78.875], [42.9, -78.873], [42.9, -78.87], [42.9, -78.867], [42.9, -78.862], [42.9, -78.858], [42.9, -78.855], [42.9, -78.853], [42.9, -78.851], [42.9, -78.848], [42.9, -78.84], [42.9, -78.836], [42.9, -78.831], [42.9, -78.827], [42.9, -78.818], [42.901, -78.895], [42.901, -78.894], [42.901, -78.891], [42.901, -78.889], [42.901, -78.884], [42.901, -78.881], [42.901, -78.88], [42.901, -78.879], [42.901, -78.877], [42.901, -78.874], [42.901, -78.865], [42.901, -78.86], [42.901, -78.855], [42.901, -78.845], [42.901, -78.844], [42.901, -78.843], [42.901, -78.842], [42.901, -78.84], [42.901, -78.839], [42.901, -78.835], [42.901, -78.834], [42.901, -78.831], [42.901, -78.827], [42.902, -78.891], [42.902, -78.882], [42.902, -78.877], [42.902, -78.876], [42.902, -78.875], [42.902, -78.874], [42.902, -78.873], [42.902, -78.872], [42.902, -78.871], [42.902, -78.869], [42.902, -78.867], [42.902, -78.866], [42.902, -78.863], [42.902, -78.861], [42.902, -78.857], [42.902, -78.854], [42.902, -78.851], [42.902, -78.84], [42.902, -78.836], [42.902, -78.833], [42.902, -78.832], [42.902, -78.83], [42.902, -78.828], [42.902, -78.825], [42.902, -78.824], [42.902, -78.822], [42.902, -78.814], [42.902, -78.812], [42.902, -78.809], [42.903, -78.891], [42.903, -78.89], [42.903, -78.889], [42.903, -78.881], [42.903, -78.88], [42.903, -78.879], [42.903, -78.877], [42.903, -78.872], [42.903, -78.869], [42.903, -78.862], [42.903, -78.844], [42.903, -78.84], [42.903, -78.838], [42.903, -78.837], [42.903, -78.836], [42.903, -78.835], [42.903, -78.833], [42.903, -78.831], [42.903, -78.822], [42.903, -78.816], [42.903, -78.815], [42.903, -78.814], [42.903, -78.812], [42.904, -78.9], [42.904, -78.897], [42.904, -78.888], [42.904, -78.887], [42.904, -78.884], [42.904, -78.883], [42.904, -78.88], [42.904, -78.877], [42.904, -78.869], [42.904, -78.868], [42.904, -78.866], [42.904, -78.861], [42.904, -78.86], [42.904, -78.854], [42.904, -78.847], [42.904, -78.845], [42.904, -78.84], [42.904, -78.835], [42.904, -78.833], [42.904, -78.831], [42.904, -78.825], [42.904, -78.822], [42.904, -78.821], [42.904, -78.819], [42.904, -78.812], [42.904, -78.81], [42.904, -78.801], [42.905, -78.898], [42.905, -78.897], [42.905, -78.896], [42.905, -78.894], [42.905, -78.892], [42.905, -78.888], [42.905, -78.886], [42.905, -78.885], [42.905, -78.882], [42.905, -78.881], [42.905, -78.88], [42.905, -78.879], [42.905, -78.877], [42.905, -78.876], [42.905, -78.87], [42.905, -78.867], [42.905, -78.866], [42.905, -78.863], [42.905, -78.85], [42.905, -78.84], [42.905, -78.836], [42.905, -78.833], [42.905, -78.832], [42.905, -78.83], [42.905, -78.824], [42.905, -78.822], [42.905, -78.814], [42.905, -78.813], [42.905, -78.81], [42.905, -78.809], [42.906, -78.893], [42.906, -78.891], [42.906, -78.887], [42.906, -78.884], [42.906, -78.881], [42.906, -78.878], [42.906, -78.87], [42.906, -78.868], [42.906, -78.863], [42.906, -78.85], [42.906, -78.849], [42.906, -78.845], [42.906, -78.833], [42.906, -78.826], [42.906, -78.825], [42.906, -78.822], [42.906, -78.817], [42.906, -78.816], [42.906, -78.814], [42.906, -78.81], [42.906, -78.809], [42.906, -78.807], [42.906, -78.805], [42.906, -78.804], [42.907, -78.893], [42.907, -78.892], [42.907, -78.89], [42.907, -78.889], [42.907, -78.888], [42.907, -78.886], [42.907, -78.885], [42.907, -78.884], [42.907, -78.883], [42.907, -78.881], [42.907, -78.878], [42.907, -78.875], [42.907, -78.872], [42.907, -78.871], [42.907, -78.87], [42.907, -78.866], [42.907, -78.86], [42.907, -78.859], [42.907, -78.851], [42.907, -78.85], [42.907, -78.848], [42.907, -78.845], [42.907, -78.839], [42.907, -78.838], [42.907, -78.837], [42.907, -78.835], [42.907, -78.832], [42.907, -78.826], [42.907, -78.824], [42.907, -78.823], [42.907, -78.817], [42.907, -78.814], [42.907, -78.811], [42.907, -78.805], [42.907, -78.802], [42.907, -78.801], [42.908, -78.899], [42.908, -78.898], [42.908, -78.895], [42.908, -78.894], [42.908, -78.888], [42.908, -78.887], [42.908, -78.886], [42.908, -78.885], [42.908, -78.884], [42.908, -78.881], [42.908, -78.88], [42.908, -78.879], [42.908, -78.878], [42.908, -78.877], [42.908, -78.874], [42.908, -78.867], [42.908, -78.861], [42.908, -78.853], [42.908, -78.851], [42.908, -78.85], [42.908, -78.848], [42.908, -78.847], [42.908, -78.841], [42.908, -78.839], [42.908, -78.827], [42.908, -78.826], [42.908, -78.823], [42.908, -78.822], [42.908, -78.817], [42.908, -78.816], [42.908, -78.814], [42.908, -78.804], [42.908, -78.803], [42.909, -78.9], [42.909, -78.897], [42.909, -78.896], [42.909, -78.894], [42.909, -78.893], [42.909, -78.89], [42.909, -78.887], [42.909, -78.886], [42.909, -78.882], [42.909, -78.879], [42.909, -78.878], [42.909, -78.875], [42.909, -78.87], [42.909, -78.866], [42.909, -78.865], [42.909, -78.857], [42.909, -78.854], [42.909, -78.848], [42.909, -78.844], [42.909, -78.842], [42.909, -78.834], [42.909, -78.832], [42.909, -78.822], [42.909, -78.817], [42.909, -78.814], [42.909, -78.81], [42.909, -78.804], [42.909, -78.803], [42.909, -78.801], [42.91, -78.899], [42.91, -78.898], [42.91, -78.893], [42.91, -78.892], [42.91, -78.888], [42.91, -78.887], [42.91, -78.881], [42.91, -78.88], [42.91, -78.877], [42.91, -78.876], [42.91, -78.873], [42.91, -78.871], [42.91, -78.869], [42.91, -78.868], [42.91, -78.867], [42.91, -78.866], [42.91, -78.865], [42.91, -78.861], [42.91, -78.856], [42.91, -78.854], [42.91, -78.851], [42.91, -78.843], [42.91, -78.841], [42.91, -78.835], [42.91, -78.834], [42.91, -78.824], [42.91, -78.823], [42.91, -78.821], [42.91, -78.82], [42.91, -78.819], [42.91, -78.817], [42.91, -78.815], [42.91, -78.814], [42.91, -78.813], [42.91, -78.812], [42.91, -78.811], [42.91, -78.81], [42.91, -78.809], [42.91, -78.808], [42.91, -78.806], [42.911, -78.899], [42.911, -78.895], [42.911, -78.894], [42.911, -78.893], [42.911, -78.892], [42.911, -78.891], [42.911, -78.889], [42.911, -78.888], [42.911, -78.885], [42.911, -78.883], [42.911, -78.882], [42.911, -78.881], [42.911, -78.88], [42.911, -78.878], [42.911, -78.877], [42.911, -78.87], [42.911, -78.867], [42.911, -78.866], [42.911, -78.861], [42.911, -78.857], [42.911, -78.855], [42.911, -78.854], [42.911, -78.85], [42.911, -78.849], [42.911, -78.841], [42.911, -78.839], [42.911, -78.837], [42.911, -78.835], [42.911, -78.833], [42.911, -78.828], [42.911, -78.822], [42.911, -78.814], [42.911, -78.81], [42.911, -78.807], [42.911, -78.806], [42.911, -78.802], [42.912, -78.898], [42.912, -78.897], [42.912, -78.896], [42.912, -78.893], [42.912, -78.89], [42.912, -78.885], [42.912, -78.882], [42.912, -78.879], [42.912, -78.878], [42.912, -78.877], [42.912, -78.876], [42.912, -78.875], [42.912, -78.87], [42.912, -78.868], [42.912, -78.867], [42.912, -78.866], [42.912, -78.865], [42.912, -78.862], [42.912, -78.855], [42.912, -78.849], [42.912, -78.848], [42.912, -78.846], [42.912, -78.844], [42.912, -78.842], [42.912, -78.841], [42.912, -78.84], [42.912, -78.836], [42.912, -78.833], [42.912, -78.83], [42.912, -78.829], [42.912, -78.824], [42.912, -78.822], [42.912, -78.821], [42.912, -78.82], [42.912, -78.817], [42.912, -78.816], [42.912, -78.815], [42.912, -78.814], [42.912, -78.812], [42.912, -78.81], [42.912, -78.809], [42.912, -78.806], [42.913, -78.898], [42.913, -78.897], [42.913, -78.893], [42.913, -78.891], [42.913, -78.889], [42.913, -78.882], [42.913, -78.881], [42.913, -78.879], [42.913, -78.878], [42.913, -78.877], [42.913, -78.874], [42.913, -78.87], [42.913, -78.868], [42.913, -78.863], [42.913, -78.846], [42.913, -78.839], [42.913, -78.837], [42.913, -78.835], [42.913, -78.833], [42.913, -78.828], [42.913, -78.824], [42.913, -78.822], [42.913, -78.821], [42.913, -78.817], [42.913, -78.814], [42.913, -78.808], [42.913, -78.807], [42.913, -78.806], [42.913, -78.803], [42.913, -78.802], [42.914, -78.893], [42.914, -78.891], [42.914, -78.89], [42.914, -78.888], [42.914, -78.885], [42.914, -78.88], [42.914, -78.877], [42.914, -78.876], [42.914, -78.875], [42.914, -78.872], [42.914, -78.868], [42.914, -78.865], [42.914, -78.862], [42.914, -78.857], [42.914, -78.847], [42.914, -78.845], [42.914, -78.842], [42.914, -78.84], [42.914, -78.839], [42.914, -78.838], [42.914, -78.834], [42.914, -78.828], [42.914, -78.822], [42.914, -78.821], [42.914, -78.816], [42.914, -78.812], [42.914, -78.811], [42.914, -78.808], [42.914, -78.803], [42.914, -78.802], [42.914, -78.801], [42.915, -78.9], [42.915, -78.899], [42.915, -78.897], [42.915, -78.894], [42.915, -78.893], [42.915, -78.892], [42.915, -78.89], [42.915, -78.888], [42.915, -78.887], [42.915, -78.885], [42.915, -78.883], [42.915, -78.881], [42.915, -78.88], [42.915, -78.879], [42.915, -78.878], [42.915, -78.877], [42.915, -78.874], [42.915, -78.873], [42.915, -78.871], [42.915, -78.868], [42.915, -78.866], [42.915, -78.865], [42.915, -78.864], [42.915, -78.863], [42.915, -78.86], [42.915, -78.854], [42.915, -78.848], [42.915, -78.847], [42.915, -78.845], [42.915, -78.844], [42.915, -78.829], [42.915, -78.823], [42.915, -78.821], [42.915, -78.82], [42.915, -78.818], [42.915, -78.815], [42.915, -78.814], [42.915, -78.812], [42.915, -78.809], [42.915, -78.808], [42.915, -78.807], [42.915, -78.804], [42.915, -78.803], [42.915, -78.802], [42.915, -78.801], [42.916, -78.897], [42.916, -78.895], [42.916, -78.891], [42.916, -78.889], [42.916, -78.888], [42.916, -78.887], [42.916, -78.886], [42.916, -78.885], [42.916, -78.882], [42.916, -78.88], [42.916, -78.878], [42.916, -78.875], [42.916, -78.873], [42.916, -78.872], [42.916, -78.871], [42.916, -78.867], [42.916, -78.864], [42.916, -78.859], [42.916, -78.858], [42.916, -78.856], [42.916, -78.852], [42.916, -78.847], [42.916, -78.845], [42.916, -78.839], [42.916, -78.83], [42.916, -78.825], [42.916, -78.824], [42.916, -78.823], [42.916, -78.822], [42.916, -78.82], [42.916, -78.818], [42.916, -78.815], [42.916, -78.812], [42.916, -78.811], [42.916, -78.809], [42.916, -78.805], [42.916, -78.802], [42.916, -78.801], [42.917, -78.898], [42.917, -78.893], [42.917, -78.891], [42.917, -78.885], [42.917, -78.881], [42.917, -78.88], [42.917, -78.878], [42.917, -78.867], [42.917, -78.866], [42.917, -78.865], [42.917, -78.863], [42.917, -78.854], [42.917, -78.853], [42.917, -78.85], [42.917, -78.848], [42.917, -78.847], [42.917, -78.845], [42.917, -78.843], [42.917, -78.842], [42.917, -78.827], [42.917, -78.826], [42.917, -78.825], [42.917, -78.817], [42.917, -78.816], [42.917, -78.81], [42.917, -78.807], [42.917, -78.804], [42.917, -78.803], [42.917, -78.802], [42.917, -78.801], [42.918, -78.897], [42.918, -78.896], [42.918, -78.894], [42.918, -78.893], [42.918, -78.892], [42.918, -78.891], [42.918, -78.889], [42.918, -78.887], [42.918, -78.885], [42.918, -78.882], [42.918, -78.881], [42.918, -78.879], [42.918, -78.878], [42.918, -78.877], [42.918, -78.876], [42.918, -78.874], [42.918, -78.872], [42.918, -78.87], [42.918, -78.869], [42.918, -78.868], [42.918, -78.863], [42.918, -78.852], [42.918, -78.848], [42.918, -78.847], [42.918, -78.846], [42.918, -78.843], [42.918, -78.84], [42.918, -78.829], [42.918, -78.827], [42.918, -78.826], [42.918, -78.825], [42.918, -78.813], [42.918, -78.812], [42.918, -78.811], [42.918, -78.81], [42.918, -78.809], [42.918, -78.808], [42.918, -78.805], [42.918, -78.803], [42.918, -78.801], [42.919, -78.899], [42.919, -78.897], [42.919, -78.895], [42.919, -78.893], [42.919, -78.888], [42.919, -78.884], [42.919, -78.882], [42.919, -78.881], [42.919, -78.878], [42.919, -78.877], [42.919, -78.876], [42.919, -78.875], [42.919, -78.872], [42.919, -78.87], [42.919, -78.869], [42.919, -78.86], [42.919, -78.857], [42.919, -78.856], [42.919, -78.855], [42.919, -78.854], [42.919, -78.85], [42.919, -78.849], [42.919, -78.839], [42.919, -78.824], [42.919, -78.823], [42.919, -78.813], [42.919, -78.811], [42.919, -78.81], [42.919, -78.809], [42.919, -78.808], [42.919, -78.801], [42.92, -78.895], [42.92, -78.892], [42.92, -78.891], [42.92, -78.888], [42.92, -78.886], [42.92, -78.884], [42.92, -78.883], [42.92, -78.881], [42.92, -78.879], [42.92, -78.877], [42.92, -78.876], [42.92, -78.872], [42.92, -78.869], [42.92, -78.857], [42.92, -78.856], [42.92, -78.847], [42.92, -78.841], [42.92, -78.827], [42.92, -78.825], [42.92, -78.824], [42.92, -78.814], [42.92, -78.808], [42.92, -78.805], [42.92, -78.801], [42.921, -78.894], [42.921, -78.891], [42.921, -78.89], [42.921, -78.886], [42.921, -78.883], [42.921, -78.88], [42.921, -78.879], [42.921, -78.878], [42.921, -78.875], [42.921, -78.874], [42.921, -78.873], [42.921, -78.871], [42.921, -78.869], [42.921, -78.868], [42.921, -78.866], [42.921, -78.865], [42.921, -78.86], [42.921, -78.85], [42.921, -78.833], [42.921, -78.825], [42.921, -78.822], [42.921, -78.815], [42.921, -78.814], [42.921, -78.813], [42.921, -78.808], [42.921, -78.807], [42.921, -78.806], [42.921, -78.804], [42.921, -78.803], [42.921, -78.802], [42.921, -78.801], [42.922, -78.895], [42.922, -78.894], [42.922, -78.891], [42.922, -78.889], [42.922, -78.888], [42.922, -78.887], [42.922, -78.885], [42.922, -78.883], [42.922, -78.882], [42.922, -78.881], [42.922, -78.88], [42.922, -78.879], [42.922, -78.877], [42.922, -78.875], [42.922, -78.874], [42.922, -78.872], [42.922, -78.871], [42.922, -78.87], [42.922, -78.868], [42.922, -78.866], [42.922, -78.865], [42.922, -78.863], [42.922, -78.857], [42.922, -78.853], [42.922, -78.85], [42.922, -78.849], [42.922, -78.845], [42.922, -78.844], [42.922, -78.843], [42.922, -78.84], [42.922, -78.832], [42.922, -78.83], [42.922, -78.829], [42.922, -78.828], [42.922, -78.827], [42.922, -78.824], [42.922, -78.822], [42.922, -78.805], [42.922, -78.803], [42.923, -78.895], [42.923, -78.894], [42.923, -78.893], [42.923, -78.891], [42.923, -78.89], [42.923, -78.888], [42.923, -78.887], [42.923, -78.884], [42.923, -78.877], [42.923, -78.876], [42.923, -78.874], [42.923, -78.868], [42.923, -78.847], [42.923, -78.829], [42.923, -78.828], [42.923, -78.827], [42.923, -78.825], [42.923, -78.824], [42.923, -78.823], [42.923, -78.816], [42.923, -78.815], [42.923, -78.814], [42.923, -78.811], [42.923, -78.81], [42.923, -78.808], [42.923, -78.807], [42.923, -78.806], [42.923, -78.805], [42.923, -78.803], [42.923, -78.802], [42.923, -78.801], [42.924, -78.897], [42.924, -78.896], [42.924, -78.895], [42.924, -78.893], [42.924, -78.891], [42.924, -78.89], [42.924, -78.888], [42.924, -78.886], [42.924, -78.885], [42.924, -78.882], [42.924, -78.88], [42.924, -78.879], [42.924, -78.878], [42.924, -78.877], [42.924, -78.873], [42.924, -78.869], [42.924, -78.846], [42.924, -78.827], [42.924, -78.812], [42.924, -78.81], [42.924, -78.807], [42.924, -78.804], [42.925, -78.896], [42.925, -78.893], [42.925, -78.89], [42.925, -78.888], [42.925, -78.887], [42.925, -78.879], [42.925, -78.877], [42.925, -78.876], [42.925, -78.869], [42.925, -78.868], [42.925, -78.852], [42.925, -78.826], [42.925, -78.822], [42.925, -78.82], [42.925, -78.816], [42.925, -78.814], [42.925, -78.813], [42.925, -78.811], [42.925, -78.809], [42.925, -78.806], [42.926, -78.898], [42.926, -78.896], [42.926, -78.892], [42.926, -78.891], [42.926, -78.886], [42.926, -78.883], [42.926, -78.882], [42.926, -78.88], [42.926, -78.879], [42.926, -78.878], [42.926, -78.877], [42.926, -78.876], [42.926, -78.875], [42.926, -78.851], [42.926, -78.818], [42.926, -78.817], [42.926, -78.815], [42.926, -78.807], [42.926, -78.804], [42.926, -78.801], [42.927, -78.895], [42.927, -78.893], [42.927, -78.892], [42.927, -78.89], [42.927, -78.883], [42.927, -78.881], [42.927, -78.88], [42.927, -78.879], [42.927, -78.878], [42.927, -78.877], [42.927, -78.87], [42.927, -78.868], [42.927, -78.848], [42.927, -78.847], [42.927, -78.846], [42.927, -78.839], [42.927, -78.827], [42.927, -78.826], [42.927, -78.825], [42.927, -78.822], [42.927, -78.818], [42.927, -78.817], [42.927, -78.816], [42.927, -78.815], [42.927, -78.814], [42.927, -78.807], [42.927, -78.805], [42.927, -78.802], [42.928, -78.89], [42.928, -78.889], [42.928, -78.888], [42.928, -78.886], [42.928, -78.885], [42.928, -78.884], [42.928, -78.883], [42.928, -78.879], [42.928, -78.878], [42.928, -78.877], [42.928, -78.875], [42.928, -78.874], [42.928, -78.873], [42.928, -78.872], [42.928, -78.84], [42.928, -78.839], [42.928, -78.826], [42.928, -78.812], [42.928, -78.81], [42.928, -78.807], [42.929, -78.892], [42.929, -78.875], [42.929, -78.855], [42.929, -78.85], [42.929, -78.838], [42.929, -78.825], [42.929, -78.817], [42.929, -78.815], [42.929, -78.814], [42.929, -78.81], [42.929, -78.807], [42.93, -78.891], [42.93, -78.89], [42.93, -78.889], [42.93, -78.874], [42.93, -78.872], [42.93, -78.849], [42.93, -78.84], [42.93, -78.839], [42.93, -78.838], [42.93, -78.824], [42.93, -78.819], [42.93, -78.814], [42.93, -78.813], [42.93, -78.81], [42.93, -78.809], [42.93, -78.808], [42.93, -78.807], [42.93, -78.805], [42.93, -78.804], [42.93, -78.803], [42.931, -78.891], [42.931, -78.89], [42.931, -78.875], [42.931, -78.852], [42.931, -78.848], [42.931, -78.841], [42.931, -78.836], [42.931, -78.827], [42.931, -78.826], [42.931, -78.825], [42.931, -78.824], [42.931, -78.818], [42.931, -78.817], [42.931, -78.807], [42.931, -78.806], [42.931, -78.804], [42.931, -78.803], [42.932, -78.897], [42.932, -78.892], [42.932, -78.89], [42.932, -78.888], [42.932, -78.877], [42.932, -78.853], [42.932, -78.851], [42.932, -78.848], [42.932, -78.845], [42.932, -78.842], [42.932, -78.839], [42.932, -78.838], [42.932, -78.825], [42.932, -78.822], [42.932, -78.82], [42.932, -78.819], [42.932, -78.817], [42.932, -78.812], [42.932, -78.81], [42.932, -78.808], [42.932, -78.803], [42.933, -78.875], [42.933, -78.852], [42.933, -78.85], [42.933, -78.831], [42.933, -78.827], [42.933, -78.826], [42.933, -78.821], [42.933, -78.817], [42.933, -78.816], [42.933, -78.814], [42.933, -78.81], [42.933, -78.808], [42.934, -78.84], [42.934, -78.838], [42.934, -78.837], [42.934, -78.836], [42.934, -78.833], [42.934, -78.82], [42.934, -78.819], [42.934, -78.818], [42.934, -78.812], [42.934, -78.809], [42.935, -78.902], [42.935, -78.901], [42.935, -78.866], [42.935, -78.839], [42.935, -78.838], [42.935, -78.833], [42.935, -78.829], [42.935, -78.827], [42.935, -78.823], [42.935, -78.82], [42.935, -78.815], [42.935, -78.813], [42.935, -78.812], [42.935, -78.81], [42.935, -78.808], [42.935, -78.803], [42.935, -78.802], [42.935, -78.801], [42.936, -78.897], [42.936, -78.878], [42.936, -78.874], [42.936, -78.847], [42.936, -78.839], [42.936, -78.837], [42.936, -78.836], [42.936, -78.831], [42.936, -78.828], [42.936, -78.821], [42.936, -78.82], [42.936, -78.819], [42.936, -78.815], [42.936, -78.812], [42.936, -78.811], [42.937, -78.903], [42.937, -78.899], [42.937, -78.895], [42.937, -78.892], [42.937, -78.849], [42.937, -78.843], [42.937, -78.839], [42.937, -78.838], [42.937, -78.837], [42.937, -78.836], [42.937, -78.82], [42.937, -78.819], [42.937, -78.817], [42.937, -78.816], [42.937, -78.814], [42.937, -78.808], [42.937, -78.806], [42.937, -78.805], [42.937, -78.804], [42.938, -78.904], [42.938, -78.901], [42.938, -78.899], [42.938, -78.893], [42.938, -78.892], [42.938, -78.889], [42.938, -78.888], [42.938, -78.877], [42.938, -78.867], [42.938, -78.85], [42.938, -78.835], [42.938, -78.828], [42.938, -78.822], [42.938, -78.821], [42.938, -78.819], [42.938, -78.817], [42.938, -78.816], [42.938, -78.814], [42.938, -78.811], [42.938, -78.81], [42.938, -78.809], [42.938, -78.807], [42.938, -78.805], [42.938, -78.803], [42.938, -78.801], [42.939, -78.904], [42.939, -78.903], [42.939, -78.901], [42.939, -78.9], [42.939, -78.899], [42.939, -78.898], [42.939, -78.894], [42.939, -78.893], [42.939, -78.891], [42.939, -78.888], [42.939, -78.887], [42.939, -78.886], [42.939, -78.885], [42.939, -78.877], [42.939, -78.873], [42.939, -78.861], [42.939, -78.849], [42.939, -78.842], [42.939, -78.84], [42.939, -78.833], [42.939, -78.832], [42.939, -78.814], [42.939, -78.812], [42.939, -78.811], [42.939, -78.81], [42.939, -78.808], [42.939, -78.807], [42.939, -78.804], [42.939, -78.803], [42.939, -78.802], [42.94, -78.907], [42.94, -78.903], [42.94, -78.901], [42.94, -78.895], [42.94, -78.893], [42.94, -78.889], [42.94, -78.887], [42.94, -78.879], [42.94, -78.874], [42.94, -78.868], [42.94, -78.864], [42.94, -78.851], [42.94, -78.841], [42.94, -78.84], [42.94, -78.838], [42.94, -78.833], [42.94, -78.83], [42.94, -78.829], [42.94, -78.824], [42.94, -78.823], [42.94, -78.822], [42.94, -78.82], [42.94, -78.817], [42.94, -78.816], [42.94, -78.815], [42.94, -78.814], [42.94, -78.811], [42.94, -78.81], [42.94, -78.808], [42.94, -78.801], [42.941, -78.906], [42.941, -78.905], [42.941, -78.904], [42.941, -78.902], [42.941, -78.899], [42.941, -78.894], [42.941, -78.893], [42.941, -78.892], [42.941, -78.89], [42.941, -78.889], [42.941, -78.888], [42.941, -78.886], [42.941, -78.885], [42.941, -78.884], [42.941, -78.883], [42.941, -78.882], [42.941, -78.881], [42.941, -78.879], [42.941, -78.875], [42.941, -78.874], [42.941, -78.867], [42.941, -78.841], [42.941, -78.837], [42.941, -78.836], [42.941, -78.835], [42.941, -78.814], [42.941, -78.812], [42.941, -78.81], [42.941, -78.805], [42.942, -78.907], [42.942, -78.905], [42.942, -78.904], [42.942, -78.902], [42.942, -78.901], [42.942, -78.892], [42.942, -78.891], [42.942, -78.89], [42.942, -78.888], [42.942, -78.884], [42.942, -78.882], [42.942, -78.878], [42.942, -78.876], [42.942, -78.868], [42.942, -78.864], [42.942, -78.854], [42.942, -78.853], [42.942, -78.852], [42.942, -78.845], [42.942, -78.836], [42.942, -78.825], [42.942, -78.822], [42.942, -78.821], [42.942, -78.817], [42.942, -78.813], [42.942, -78.809], [42.942, -78.808], [42.942, -78.806], [42.942, -78.805], [42.942, -78.803], [42.942, -78.802], [42.943, -78.906], [42.943, -78.903], [42.943, -78.902], [42.943, -78.901], [42.943, -78.9], [42.943, -78.89], [42.943, -78.879], [42.943, -78.867], [42.943, -78.864], [42.943, -78.855], [42.943, -78.852], [42.943, -78.851], [42.943, -78.845], [42.943, -78.839], [42.943, -78.835], [42.943, -78.821], [42.943, -78.819], [42.943, -78.818], [42.943, -78.815], [42.943, -78.814], [42.943, -78.809], [42.943, -78.807], [42.943, -78.805], [42.943, -78.804], [42.944, -78.908], [42.944, -78.907], [42.944, -78.904], [42.944, -78.901], [42.944, -78.9], [42.944, -78.898], [42.944, -78.886], [42.944, -78.878], [42.944, -78.868], [42.944, -78.867], [42.944, -78.864], [42.944, -78.863], [42.944, -78.853], [42.944, -78.849], [42.944, -78.846], [42.944, -78.838], [42.944, -78.823], [42.944, -78.822], [42.944, -78.82], [42.944, -78.818], [42.944, -78.816], [42.944, -78.815], [42.944, -78.814], [42.944, -78.812], [42.944, -78.81], [42.944, -78.809], [42.945, -78.903], [42.945, -78.902], [42.945, -78.886], [42.945, -78.87], [42.945, -78.869], [42.945, -78.862], [42.945, -78.861], [42.945, -78.859], [42.945, -78.858], [42.945, -78.855], [42.945, -78.854], [42.945, -78.847], [42.945, -78.824], [42.945, -78.822], [42.945, -78.821], [42.945, -78.817], [42.945, -78.816], [42.945, -78.815], [42.945, -78.812], [42.945, -78.811], [42.945, -78.807], [42.945, -78.805], [42.946, -78.908], [42.946, -78.903], [42.946, -78.892], [42.946, -78.891], [42.946, -78.888], [42.946, -78.868], [42.946, -78.867], [42.946, -78.862], [42.946, -78.859], [42.946, -78.85], [42.946, -78.832], [42.946, -78.826], [42.946, -78.824], [42.946, -78.822], [42.946, -78.814], [42.946, -78.812], [42.946, -78.81], [42.946, -78.808], [42.946, -78.807], [42.946, -78.804], [42.946, -78.801], [42.947, -78.906], [42.947, -78.904], [42.947, -78.903], [42.947, -78.902], [42.947, -78.901], [42.947, -78.899], [42.947, -78.893], [42.947, -78.878], [42.947, -78.877], [42.947, -78.874], [42.947, -78.867], [42.947, -78.863], [42.947, -78.861], [42.947, -78.859], [42.947, -78.858], [42.947, -78.856], [42.947, -78.854], [42.947, -78.852], [42.947, -78.845], [42.947, -78.844], [42.947, -78.836], [42.947, -78.833], [42.947, -78.83], [42.947, -78.827], [42.947, -78.826], [42.947, -78.825], [42.947, -78.816], [42.947, -78.811], [42.947, -78.801], [42.948, -78.907], [42.948, -78.904], [42.948, -78.903], [42.948, -78.902], [42.948, -78.901], [42.948, -78.899], [42.948, -78.894], [42.948, -78.892], [42.948, -78.89], [42.948, -78.889], [42.948, -78.888], [42.948, -78.885], [42.948, -78.884], [42.948, -78.883], [42.948, -78.878], [42.948, -78.875], [42.948, -78.874], [42.948, -78.871], [42.948, -78.869], [42.948, -78.868], [42.948, -78.866], [42.948, -78.865], [42.948, -78.863], [42.948, -78.862], [42.948, -78.853], [42.948, -78.852], [42.948, -78.85], [42.948, -78.842], [42.948, -78.835], [42.948, -78.83], [42.948, -78.829], [42.948, -78.827], [42.948, -78.824], [42.948, -78.823], [42.948, -78.822], [42.948, -78.819], [42.948, -78.81], [42.948, -78.809], [42.948, -78.808], [42.948, -78.807], [42.948, -78.804], [42.949, -78.908], [42.949, -78.904], [42.949, -78.901], [42.949, -78.9], [42.949, -78.897], [42.949, -78.891], [42.949, -78.889], [42.949, -78.888], [42.949, -78.86], [42.949, -78.858], [42.949, -78.857], [42.949, -78.854], [42.949, -78.845], [42.949, -78.842], [42.949, -78.839], [42.949, -78.822], [42.949, -78.821], [42.949, -78.819], [42.949, -78.814], [42.949, -78.811], [42.949, -78.81], [42.95, -78.904], [42.95, -78.903], [42.95, -78.901], [42.95, -78.9], [42.95, -78.889], [42.95, -78.888], [42.95, -78.887], [42.95, -78.886], [42.95, -78.872], [42.95, -78.871], [42.95, -78.86], [42.95, -78.845], [42.95, -78.831], [42.95, -78.83], [42.95, -78.829], [42.95, -78.828], [42.95, -78.827], [42.95, -78.825], [42.95, -78.824], [42.95, -78.823], [42.951, -78.908], [42.951, -78.907], [42.951, -78.902], [42.951, -78.898], [42.951, -78.883], [42.951, -78.873], [42.951, -78.868], [42.951, -78.865], [42.951, -78.863], [42.951, -78.861], [42.951, -78.854], [42.951, -78.843], [42.951, -78.842], [42.951, -78.826], [42.951, -78.813], [42.952, -78.909], [42.952, -78.908], [42.952, -78.907], [42.952, -78.906], [42.952, -78.905], [42.952, -78.904], [42.952, -78.903], [42.952, -78.902], [42.952, -78.877], [42.952, -78.869], [42.952, -78.865], [42.952, -78.853], [42.952, -78.835], [42.952, -78.83], [42.952, -78.829], [42.952, -78.826], [42.952, -78.825], [42.953, -78.909], [42.953, -78.905], [42.953, -78.902], [42.953, -78.901], [42.953, -78.879], [42.953, -78.875], [42.953, -78.873], [42.953, -78.87], [42.953, -78.869], [42.953, -78.867], [42.953, -78.862], [42.953, -78.86], [42.953, -78.853], [42.953, -78.832], [42.953, -78.828], [42.953, -78.826], [42.954, -78.906], [42.954, -78.905], [42.954, -78.904], [42.954, -78.903], [42.954, -78.9], [42.954, -78.897], [42.954, -78.879], [42.954, -78.878], [42.954, -78.866], [42.954, -78.865], [42.954, -78.852], [42.954, -78.825], [42.955, -78.906], [42.955, -78.904], [42.955, -78.903], [42.955, -78.902], [42.955, -78.899], [42.955, -78.898], [42.955, -78.897], [42.955, -78.89], [42.955, -78.874], [42.955, -78.87], [42.955, -78.869], [42.955, -78.862], [42.955, -78.861], [42.955, -78.859], [42.955, -78.827], [42.956, -78.907], [42.956, -78.905], [42.956, -78.904], [42.956, -78.9], [42.956, -78.899], [42.956, -78.898], [42.956, -78.884], [42.956, -78.87], [42.956, -78.869], [42.956, -78.861], [42.956, -78.847], [42.956, -78.844], [42.956, -78.842], [42.956, -78.832], [42.956, -78.831], [42.956, -78.83], [42.956, -78.82], [42.957, -78.912], [42.957, -78.906], [42.957, -78.905], [42.957, -78.904], [42.957, -78.902], [42.957, -78.89], [42.957, -78.889], [42.957, -78.888], [42.957, -78.887], [42.957, -78.88], [42.957, -78.875], [42.957, -78.873], [42.957, -78.866], [42.957, -78.861], [42.957, -78.859], [42.957, -78.834], [42.957, -78.832], [42.957, -78.819], [42.958, -78.907], [42.958, -78.905], [42.958, -78.903], [42.958, -78.9], [42.958, -78.896], [42.958, -78.886], [42.958, -78.884], [42.958, -78.879], [42.958, -78.874], [42.958, -78.872], [42.958, -78.865], [42.958, -78.862], [42.958, -78.857], [42.958, -78.856], [42.958, -78.836], [42.958, -78.832], [42.958, -78.818], [42.959, -78.908], [42.959, -78.906], [42.959, -78.904], [42.959, -78.903], [42.959, -78.902], [42.959, -78.901], [42.959, -78.899], [42.959, -78.897], [42.959, -78.896], [42.959, -78.895], [42.959, -78.87], [42.96, -78.905], [42.96, -78.899], [42.96, -78.898], [42.96, -78.897], [42.96, -78.895], [42.961, -78.902], [42.961, -78.901], [42.961, -78.9], [42.961, -78.899], [42.961, -78.897], [42.962, -78.901], [42.962, -78.898], [42.962, -78.897], [42.962, -78.896], [42.963, -78.903], [42.963, -78.901], [42.963, -78.9], [42.963, -78.897]],
                {&quot;blur&quot;: 15, &quot;maxZoom&quot;: 12, &quot;minOpacity&quot;: 0.5, &quot;radius&quot;: 8}
            ).addTo(map_47c8eb93fed5beeb59f73a3290f30084);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# Crime Forecasting
<a id='forecasting'></a>


```python
import warnings
#warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import mean
from numpy import array
from prettytable import PrettyTable
from tqdm import tqdm_notebook

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

from sklearn.metrics import mean_squared_error
```


```python
data['latitude'] = pd.to_numeric(data['latitude'])
data['longitude'] = pd.to_numeric(data['longitude'])
data['hour_of_day'] = pd.to_numeric(data['hour_of_day'])
#ignoring unknown neighborhoods
data = data[data['neighborhood_1'] != 'UNKNOWN']
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 236726 entries, 2009-01-10 12:19:00 to 2023-09-11 11:12:45
    Data columns (total 35 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   case_number              236726 non-null  object 
     1   incident_datetime        236726 non-null  object 
     2   incident_type_primary    236726 non-null  object 
     3   incident_description     236726 non-null  object 
     4   parent_incident_type     236726 non-null  object 
     5   hour_of_day              236726 non-null  int64  
     6   day_of_week              236726 non-null  object 
     7   address_1                236710 non-null  object 
     8   city                     236726 non-null  object 
     9   state                    236726 non-null  object 
     10  location                 234291 non-null  object 
     11  latitude                 234291 non-null  float64
     12  longitude                234291 non-null  float64
     13  created_at               236726 non-null  object 
     14  census_tract_2010        234719 non-null  object 
     15  census_block_group_2010  234719 non-null  object 
     16  census_block_2010        234719 non-null  object 
     17  census_tract             234719 non-null  object 
     18  census_block             234719 non-null  object 
     19  census_block_group       234719 non-null  object 
     20  neighborhood_1           234719 non-null  object 
     21  police_district          234719 non-null  object 
     22  council_district         234719 non-null  object 
     23  tractce20                234856 non-null  object 
     24  geoid20_tract            234856 non-null  object 
     25  geoid20_blockgroup       234856 non-null  object 
     26  geoid20_block            234856 non-null  object 
     27  Year                     236726 non-null  int64  
     28  Month                    236726 non-null  int64  
     29  dayOfWeek                236726 non-null  int64  
     30  dayOfMonth               236726 non-null  int64  
     31  dayOfYear                236726 non-null  int64  
     32  weekOfMonth              236726 non-null  int64  
     33  weekOfYear               236726 non-null  int64  
     34  MonthNames               236726 non-null  object 
    dtypes: float64(2), int64(8), object(25)
    memory usage: 65.0+ MB
    


```python
# function to split training and test data

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
```


```python
# decide on the training and test set by using dates

data_tr = data.loc['2011-01-01':'2022-12-31']
data_test = data.loc['2023-01-01':'2023-09-01']
```


```python
listOfNeigh = list(data['neighborhood_1'].unique())
```


```python
train_d = []
for neigh in listOfNeigh:
    df = data_tr[data_tr['neighborhood_1'] == neigh]
    df_gr = df.groupby(['Year', 'Month']).count()
    train_d.append(list(df_gr['incident_datetime'].values))
```


```python
test_d = []
for neigh in listOfNeigh:
    df = data_test[data_test['neighborhood_1'] == neigh]
    df_gr = df.groupby(['Month']).count()
    test_d.append(list(df_gr['incident_datetime'].values))
```


```python
data_test['neighborhood_1'].unique()
```




    array(['South Park', 'Hopkins-Tifft', 'Lower West Side', 'Central',
           'Lovejoy', 'North Park', 'Kensington-Bailey', 'Elmwood Bryant',
           'Pratt-Willert', 'Masten Park', 'West Hertel',
           'University Heights', 'Broadway Fillmore', 'Elmwood Bidwell',
           'Genesee-Moselle', 'Upper West Side', 'West Side', 'Hamlin Park',
           'Ellicott', 'Seneca Babcock', 'Kenfield', nan, 'First Ward',
           'Allentown', 'Black Rock', 'Delavan Grider', 'Schiller Park',
           'Riverside', 'Fruit Belt', 'Central Park', 'MLK Park', 'Parkside',
           'Kaisertown', 'Seneca-Cazenovia', 'Grant-Amherst',
           'Fillmore-Leroy'], dtype=object)



# Crime Projection On The Last Eight Months Using Simple Moving Average


```python
# Simple Moving Average
window = 5
predTot = list()
testTot = list()

# get unique neighborhood names
unique_neighborhoods = data_test['neighborhood_1'].unique()

# walk forward over time steps in test
for neighNum, neighborhood in enumerate(unique_neighborhoods):

    history = train_d[neighNum]
    test = test_d[neighNum]

    # check if there is test data for this neighborhood
    if len(test) == 0:
        continue  # skip neighborhoods with no test data

    preds = []
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length - window, length)])
        obs = test[t]
        preds.append(yhat)
        history.append(obs)

    print('Neighborhood: {}'.format(neighborhood))
    print('Actuals: {}'.format(test))
    print('Predictions: {}'.format(preds))

    # plot
    plt.plot(test, color='yellowgreen')
    plt.plot(preds, color='steelblue')

    # Add neighborhood name as annotation
    plt.annotate(neighborhood, (0.02, 0.9), xycoords='axes fraction', fontsize=12, color='black')

    plt.title(f'Simple Moving Average - {neighborhood}')
    plt.xlabel('Months Staring in Jan')
    plt.ylabel('Number Of Crimes')
    plt.legend(['Test Data', 'Predictions'])
    plt.show()

    plt.show()

    testTot = testTot + test
    predTot = predTot + preds

error = mean_squared_error(predTot, testTot) ** .5
print('Test RMSE: %.3f' % error)

```

    Neighborhood: South Park
    Actuals: [67, 50, 72, 65, 63, 58, 45, 55, 1]
    Predictions: [41.2, 43.8, 46.0, 50.2, 59.8, 63.4, 61.6, 60.6, 57.2]
    


    
![png](\img\posts\Buffalo-Crime\output_94_1.png)
    


    Neighborhood: Hopkins-Tifft
    Actuals: [30, 16, 17, 35, 16, 27, 18, 33, 2]
    Predictions: [22.6, 24.2, 21.2, 18.8, 23.0, 22.8, 22.2, 22.6, 25.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_3.png)
    


    Neighborhood: Lower West Side
    Actuals: [28, 16, 24, 25, 31, 30, 35, 27, 1]
    Predictions: [23.6, 23.6, 21.6, 21.2, 24.2, 24.8, 25.2, 29.0, 29.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_5.png)
    


    Neighborhood: Central
    Actuals: [16, 11, 7, 15, 16, 20, 16, 13, 2]
    Predictions: [11.4, 12.4, 11.8, 9.6, 11.4, 13.0, 13.8, 14.8, 16.0]
    


    
![png](\img\posts\Buffalo-Crime\output_94_7.png)
    


    Neighborhood: Lovejoy
    Actuals: [23, 13, 23, 21, 28, 17, 15, 23, 3]
    Predictions: [18.8, 17.8, 15.8, 16.0, 18.4, 21.6, 20.4, 20.8, 20.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_9.png)
    


    Neighborhood: North Park
    Actuals: [18, 19, 21, 24, 19, 28, 34, 33, 1]
    Predictions: [24.8, 20.4, 19.4, 18.2, 19.4, 20.2, 22.2, 25.2, 27.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_11.png)
    


    Neighborhood: Kensington-Bailey
    Actuals: [32, 27, 30, 36, 40, 34, 34, 41]
    Predictions: [28.2, 28.8, 26.6, 27.2, 30.8, 33.0, 33.4, 34.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_13.png)
    


    Neighborhood: Elmwood Bryant
    Actuals: [41, 39, 52, 44, 45, 45, 42, 53]
    Predictions: [40.0, 39.2, 36.8, 38.4, 41.8, 44.2, 45.0, 45.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_15.png)
    


    Neighborhood: Pratt-Willert
    Actuals: [8, 10, 14, 13, 17, 20, 18, 29, 1]
    Predictions: [13.4, 11.4, 9.4, 9.4, 11.0, 12.4, 14.8, 16.4, 19.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_17.png)
    


    Neighborhood: Masten Park
    Actuals: [33, 21, 20, 10, 19, 15, 14, 24]
    Predictions: [16.0, 19.2, 19.0, 18.4, 18.0, 20.6, 17.0, 15.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_19.png)
    


    Neighborhood: West Hertel
    Actuals: [39, 28, 26, 34, 39, 47, 50, 62, 1]
    Predictions: [35.6, 33.6, 30.4, 28.0, 32.0, 33.2, 34.8, 39.2, 46.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_21.png)
    


    Neighborhood: University Heights
    Actuals: [45, 46, 43, 41, 43, 60, 90, 58, 2]
    Predictions: [41.0, 39.4, 40.8, 38.8, 42.0, 43.6, 46.6, 55.4, 58.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_23.png)
    


    Neighborhood: Broadway Fillmore
    Actuals: [56, 21, 27, 39, 53, 60, 51, 48, 1]
    Predictions: [33.0, 34.2, 30.6, 28.4, 33.0, 39.2, 40.0, 46.0, 50.2]
    


    
![png](\img\posts\Buffalo-Crime\output_94_25.png)
    


    Neighborhood: Elmwood Bidwell
    Actuals: [34, 22, 19, 24, 24, 30, 43, 43, 3]
    Predictions: [22.4, 23.8, 22.8, 21.4, 22.8, 24.6, 23.8, 28.0, 32.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_27.png)
    


    Neighborhood: Genesee-Moselle
    Actuals: [32, 26, 29, 29, 34, 26, 33, 38]
    Predictions: [31.0, 29.8, 28.0, 24.8, 27.6, 30.0, 28.8, 30.2]
    


    
![png](\img\posts\Buffalo-Crime\output_94_29.png)
    


    Neighborhood: Upper West Side
    Actuals: [18, 7, 18, 14, 10, 19, 13, 27]
    Predictions: [15.8, 16.4, 13.6, 13.4, 15.0, 13.4, 13.6, 14.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_31.png)
    


    Neighborhood: West Side
    Actuals: [36, 24, 40, 40, 52, 60, 51, 52, 5]
    Predictions: [35.8, 34.0, 29.6, 29.0, 33.2, 38.4, 43.2, 48.6, 51.0]
    


    
![png](\img\posts\Buffalo-Crime\output_94_33.png)
    


    Neighborhood: Hamlin Park
    Actuals: [30, 15, 11, 13, 30, 19, 20, 15, 1]
    Predictions: [17.6, 19.0, 17.6, 16.2, 16.6, 19.8, 17.6, 18.6, 19.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_35.png)
    


    Neighborhood: Ellicott
    Actuals: [27, 15, 17, 16, 21, 20, 35, 21, 1]
    Predictions: [17.6, 19.0, 17.0, 16.6, 18.8, 19.2, 17.8, 21.8, 22.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_37.png)
    


    Neighborhood: Seneca Babcock
    Actuals: [33, 19, 40, 34, 37, 40, 44, 30, 1]
    Predictions: [26.0, 25.2, 23.2, 25.6, 30.6, 32.6, 34.0, 39.0, 37.0]
    


    
![png](\img\posts\Buffalo-Crime\output_94_39.png)
    


    Neighborhood: Kenfield
    Actuals: [13, 14, 16, 14, 20, 16, 17, 14]
    Predictions: [14.2, 13.0, 13.6, 14.2, 14.8, 15.4, 16.0, 16.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_41.png)
    


    Neighborhood: nan
    Actuals: [14, 18, 22, 25, 30, 22, 16, 11]
    Predictions: [18.6, 17.0, 17.4, 18.2, 21.2, 21.8, 23.4, 23.0]
    


    
![png](\img\posts\Buffalo-Crime\output_94_43.png)
    


    Neighborhood: First Ward
    Actuals: [29, 26, 29, 39, 30, 36, 37, 26]
    Predictions: [25.6, 25.2, 24.2, 23.4, 28.2, 30.6, 32.0, 34.2]
    


    
![png](\img\posts\Buffalo-Crime\output_94_45.png)
    


    Neighborhood: Allentown
    Actuals: [33, 27, 26, 37, 39, 30, 32, 41]
    Predictions: [31.2, 33.0, 28.6, 27.4, 30.8, 32.4, 31.8, 32.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_47.png)
    


    Neighborhood: Black Rock
    Actuals: [12, 15, 14, 24, 21, 25, 23, 15]
    Predictions: [13.6, 12.6, 13.2, 11.4, 15.2, 17.2, 19.8, 21.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_49.png)
    


    Neighborhood: Delavan Grider
    Actuals: [16, 5, 13, 16, 24, 18, 18, 20]
    Predictions: [14.8, 15.2, 12.4, 11.2, 13.2, 14.8, 15.2, 17.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_51.png)
    


    Neighborhood: Schiller Park
    Actuals: [12, 12, 11, 20, 9, 13, 11, 14, 1]
    Predictions: [8.4, 7.6, 7.6, 9.0, 12.0, 12.8, 13.0, 12.8, 13.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_53.png)
    


    Neighborhood: Riverside
    Actuals: [17, 12, 14, 6, 7, 15, 12, 7]
    Predictions: [8.2, 9.6, 9.4, 10.8, 11.2, 11.2, 10.8, 10.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_55.png)
    


    Neighborhood: Fruit Belt
    Actuals: [12, 8, 4, 7, 13, 7, 8, 10]
    Predictions: [7.6, 7.4, 8.2, 6.4, 7.0, 8.8, 7.8, 7.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_57.png)
    


    Neighborhood: Central Park
    Actuals: [19, 16, 9, 17, 14, 18, 22, 17, 1]
    Predictions: [14.0, 14.6, 14.4, 13.2, 14.4, 15.0, 14.8, 16.0, 17.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_59.png)
    


    Neighborhood: MLK Park
    Actuals: [10, 9, 7, 14, 14, 7, 4, 7]
    Predictions: [8.4, 8.8, 8.4, 8.2, 10.2, 10.8, 10.2, 9.2]
    


    
![png](\img\posts\Buffalo-Crime\output_94_61.png)
    


    Neighborhood: Parkside
    Actuals: [41, 17, 30, 32, 28, 24, 22, 27, 2]
    Predictions: [26.4, 27.6, 24.6, 25.4, 28.8, 29.6, 26.2, 27.2, 26.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_62.png)
    


    Neighborhood: Kaisertown
    Actuals: [5, 2, 4, 2, 5, 5, 2, 4]
    Predictions: [3.4, 3.2, 3.0, 3.4, 3.4, 3.6, 3.6, 3.6]
    


    
![png](\img\posts\Buffalo-Crime\output_94_65.png)
    


    Neighborhood: Seneca-Cazenovia
    Actuals: [25, 13, 20, 21, 22, 15, 22, 19, 1]
    Predictions: [17.0, 18.8, 15.8, 14.8, 17.0, 20.2, 18.2, 20.0, 19.8]
    


    
![png](\img\posts\Buffalo-Crime\output_94_67.png)
    


    Neighborhood: Grant-Amherst
    Actuals: [11, 9, 7, 8, 7, 8, 12, 6]
    Predictions: [5.6, 6.8, 6.8, 7.0, 7.8, 8.4, 7.8, 8.4]
    


    
![png](\img\posts\Buffalo-Crime\output_94_69.png)
    


    Test RMSE: 11.191
    

# Crime Projection On The Last Eight Months Using Weighted Moving Average


```python
# Weighted Moving Average
window = 5
predTot = list()
testTot = list()

# get unique neighborhood names
unique_neighborhoods = data_test['neighborhood_1'].unique()

# walk forward over time steps in test
#for neighNum in range(len(train_d)):
for neighNum, neighborhood in enumerate(unique_neighborhoods):

    history = train_d[neighNum]
    test = test_d[neighNum]

    # Check if there is test data for this neighborhood
    if len(test) == 0:
        continue  # Skip neighborhoods with no test data

    preds = []
    for t in range(len(test)):
        length = len(history)
        yhat = np.average([history[i] for i in range(length - window, length)], weights=[1,2,3,4,5])
        obs = test[t]
        preds.append(yhat)
        history.append(obs)

    #print('Neighborhood: {}'.format(neighNum+1))
    print('Neighborhood: {}'.format(neighborhood))
    print('Actuals: {}'.format(test))
    print('Predictions: {}'.format(preds))

    # plot
    plt.plot(test, color='yellowgreen')
    plt.plot(preds, color='steelblue')

    # Add neighborhood name as annotation
    plt.annotate(neighborhood, (0.02, 0.9), xycoords='axes fraction', fontsize=12, color='black')

    plt.title(f'Weighted Moving Average - {neighborhood}')
    plt.xlabel('Months Staring in Jan')
    plt.ylabel('Number Of Crimes')
    plt.legend(['Test Data', 'Predictions'])


    plt.show()

    testTot = testTot + test
    predTot = predTot + preds
error = mean_squared_error(predTot, testTot) ** .5
print('Test RMSE: %.3f' % error)
```

    Neighborhood: South Park
    Actuals: [67, 50, 72, 65, 63, 58, 45, 55, 1]
    Predictions: [35.93333333333333, 43.46666666666667, 45.06666666666667, 54.53333333333333, 59.86666666666667, 63.86666666666667, 62.06666666666667, 56.53333333333333, 54.666666666666664]
    


    
![png](\img\posts\Buffalo-Crime\output_96_1.png)
    


    Neighborhood: Hopkins-Tifft
    Actuals: [30, 16, 17, 35, 16, 27, 18, 33, 2]
    Predictions: [17.733333333333334, 21.333333333333332, 19.333333333333332, 18.4, 23.533333333333335, 22.2, 23.6, 22.2, 25.666666666666668]
    


    
![png](\img\posts\Buffalo-Crime\output_96_3.png)
    


    Neighborhood: Lower West Side
    Actuals: [28, 16, 24, 25, 31, 30, 35, 27, 1]
    Predictions: [20.6, 21.666666666666668, 18.933333333333334, 19.8, 21.733333333333334, 25.8, 27.533333333333335, 30.8, 30.133333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_5.png)
    


    Neighborhood: Central
    Actuals: [16, 11, 7, 15, 16, 20, 16, 13, 2]
    Predictions: [11.066666666666666, 11.933333333333334, 11.133333333333333, 9.6, 11.333333333333334, 13.266666666666667, 15.6, 16.333333333333332, 15.733333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_7.png)
    


    Neighborhood: Lovejoy
    Actuals: [23, 13, 23, 21, 28, 17, 15, 23, 3]
    Predictions: [14.266666666666667, 16.2, 15.133333333333333, 17.666666666666668, 19.0, 22.8, 21.266666666666666, 19.466666666666665, 20.2]
    


    
![png](\img\posts\Buffalo-Crime\output_96_9.png)
    


    Neighborhood: North Park
    Actuals: [18, 19, 21, 24, 19, 28, 34, 33, 1]
    Predictions: [20.933333333333334, 19.266666666666666, 18.0, 18.0, 19.866666666666667, 20.666666666666668, 23.266666666666666, 27.2, 29.8]
    


    
![png](\img\posts\Buffalo-Crime\output_96_11.png)
    


    Neighborhood: Kensington-Bailey
    Actuals: [32, 27, 30, 36, 40, 34, 34, 41]
    Predictions: [37.266666666666666, 35.6, 32.53333333333333, 31.333333333333332, 32.4, 34.666666666666664, 35.0, 35.2]
    


    
![png](\img\posts\Buffalo-Crime\output_96_13.png)
    


    Neighborhood: Elmwood Bryant
    Actuals: [41, 39, 52, 44, 45, 45, 42, 53]
    Predictions: [46.8, 45.2, 43.13333333333333, 45.8, 45.333333333333336, 45.06666666666667, 45.333333333333336, 44.333333333333336]
    


    
![png](\img\posts\Buffalo-Crime\output_96_15.png)
    


    Neighborhood: Pratt-Willert
    Actuals: [8, 10, 14, 13, 17, 20, 18, 29, 1]
    Predictions: [15.466666666666667, 12.466666666666667, 10.733333333333333, 11.0, 11.2, 13.8, 16.333333333333332, 17.4, 21.6]
    


    
![png](\img\posts\Buffalo-Crime\output_96_17.png)
    


    Neighborhood: Masten Park
    Actuals: [33, 21, 20, 10, 19, 15, 14, 24]
    Predictions: [17.933333333333334, 23.466666666666665, 23.466666666666665, 23.0, 18.866666666666667, 18.0, 16.133333333333333, 15.133333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_19.png)
    


    Neighborhood: West Hertel
    Actuals: [39, 28, 26, 34, 39, 47, 50, 62, 1]
    Predictions: [35.733333333333334, 35.46666666666667, 31.533333333333335, 28.2, 29.133333333333333, 33.6, 38.2, 43.266666666666666, 50.86666666666667]
    


    
![png](\img\posts\Buffalo-Crime\output_96_21.png)
    


    Neighborhood: University Heights
    Actuals: [45, 46, 43, 41, 43, 60, 90, 58, 2]
    Predictions: [45.0, 43.13333333333333, 41.46666666666667, 39.733333333333334, 40.46666666666667, 43.0, 48.46666666666667, 62.93333333333333, 63.8]
    


    
![png](\img\posts\Buffalo-Crime\output_96_23.png)
    


    Neighborhood: Broadway Fillmore
    Actuals: [56, 21, 27, 39, 53, 60, 51, 48, 1]
    Predictions: [34.86666666666667, 39.333333333333336, 31.933333333333334, 29.133333333333333, 31.933333333333334, 40.0, 46.93333333333333, 50.6, 51.266666666666666]
    


    
![png](\img\posts\Buffalo-Crime\output_96_25.png)
    


    Neighborhood: Elmwood Bidwell
    Actuals: [34, 22, 19, 24, 24, 30, 43, 43, 3]
    Predictions: [26.666666666666668, 28.466666666666665, 25.6, 22.266666666666666, 22.2, 23.4, 25.2, 31.6, 36.6]
    


    
![png](\img\posts\Buffalo-Crime\output_96_27.png)
    


    Neighborhood: Genesee-Moselle
    Actuals: [32, 26, 29, 29, 34, 26, 33, 38]
    Predictions: [33.13333333333333, 33.13333333333333, 30.933333333333334, 30.266666666666666, 29.4, 30.466666666666665, 29.133333333333333, 30.533333333333335]
    


    
![png](\img\posts\Buffalo-Crime\output_96_29.png)
    


    Neighborhood: Upper West Side
    Actuals: [18, 7, 18, 14, 10, 19, 13, 27]
    Predictions: [18.533333333333335, 19.0, 15.533333333333333, 15.933333333333334, 15.066666666666666, 12.8, 14.666666666666666, 14.466666666666667]
    


    
![png](\img\posts\Buffalo-Crime\output_96_31.png)
    


    Neighborhood: West Side
    Actuals: [36, 24, 40, 40, 52, 60, 51, 52, 5]
    Predictions: [37.2, 34.53333333333333, 28.933333333333334, 31.066666666666666, 33.93333333333333, 41.6, 48.8, 51.4, 52.53333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_33.png)
    


    Neighborhood: Hamlin Park
    Actuals: [30, 15, 11, 13, 30, 19, 20, 15, 1]
    Predictions: [12.866666666666667, 17.2, 16.533333333333335, 14.8, 14.333333333333334, 19.666666666666668, 19.4, 20.2, 19.0]
    


    
![png](\img\posts\Buffalo-Crime\output_96_35.png)
    


    Neighborhood: Ellicott
    Actuals: [27, 15, 17, 16, 21, 20, 35, 21, 1]
    Predictions: [17.0, 19.466666666666665, 17.533333333333335, 16.6, 16.533333333333335, 18.466666666666665, 18.733333333333334, 24.466666666666665, 24.2]
    


    
![png](\img\posts\Buffalo-Crime\output_96_37.png)
    


    Neighborhood: Seneca Babcock
    Actuals: [33, 19, 40, 34, 37, 40, 44, 30, 1]
    Predictions: [24.933333333333334, 25.8, 22.266666666666666, 27.133333333333333, 30.266666666666666, 34.13333333333333, 36.6, 39.93333333333333, 36.93333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_39.png)
    


    Neighborhood: Kenfield
    Actuals: [13, 14, 16, 14, 20, 16, 17, 14]
    Predictions: [16.0, 14.933333333333334, 14.266666666666667, 14.666666666666666, 14.4, 16.333333333333332, 16.533333333333335, 16.866666666666667]
    


    
![png](\img\posts\Buffalo-Crime\output_96_41.png)
    


    Neighborhood: nan
    Actuals: [14, 18, 22, 25, 30, 22, 16, 11]
    Predictions: [18.0, 15.733333333333333, 15.533333333333333, 17.466666666666665, 20.4, 24.4, 24.466666666666665, 22.0]
    


    
![png](\img\posts\Buffalo-Crime\output_96_43.png)
    


    Neighborhood: First Ward
    Actuals: [29, 26, 29, 39, 30, 36, 37, 26]
    Predictions: [32.333333333333336, 30.8, 28.933333333333334, 28.333333333333332, 31.533333333333335, 31.6, 33.4, 35.06666666666667]
    


    
![png](\img\posts\Buffalo-Crime\output_96_45.png)
    


    Neighborhood: Allentown
    Actuals: [33, 27, 26, 37, 39, 30, 32, 41]
    Predictions: [35.86666666666667, 34.93333333333333, 32.266666666666666, 30.066666666666666, 31.8, 33.86666666666667, 33.06666666666667, 33.13333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_47.png)
    


    Neighborhood: Black Rock
    Actuals: [12, 15, 14, 24, 21, 25, 23, 15]
    Predictions: [20.533333333333335, 17.333333333333332, 15.933333333333334, 14.6, 17.333333333333332, 19.0, 21.6, 22.666666666666668]
    


    
![png](\img\posts\Buffalo-Crime\output_96_49.png)
    


    Neighborhood: Delavan Grider
    Actuals: [16, 5, 13, 16, 24, 18, 18, 20]
    Predictions: [19.333333333333332, 18.266666666666666, 13.533333333333333, 12.733333333333333, 13.266666666666667, 16.6, 17.666666666666668, 18.6]
    


    
![png](\img\posts\Buffalo-Crime\output_96_51.png)
    


    Neighborhood: Schiller Park
    Actuals: [12, 12, 11, 20, 9, 13, 11, 14, 1]
    Predictions: [8.6, 9.4, 10.0, 10.333333333333334, 13.666666666666666, 12.933333333333334, 13.0, 12.333333333333334, 12.733333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_53.png)
    


    Neighborhood: Riverside
    Actuals: [17, 12, 14, 6, 7, 15, 12, 7]
    Predictions: [9.866666666666667, 12.4, 12.533333333333333, 13.0, 10.866666666666667, 9.466666666666667, 10.733333333333333, 11.133333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_55.png)
    


    Neighborhood: Fruit Belt
    Actuals: [12, 8, 4, 7, 13, 7, 8, 10]
    Predictions: [9.066666666666666, 10.066666666666666, 9.4, 7.733333333333333, 7.266666666666667, 8.866666666666667, 8.266666666666667, 8.333333333333334]
    


    
![png](\img\posts\Buffalo-Crime\output_96_57.png)
    


    Neighborhood: Central Park
    Actuals: [19, 16, 9, 17, 14, 18, 22, 17, 1]
    Predictions: [12.6, 14.133333333333333, 14.333333333333334, 12.333333333333334, 13.866666666666667, 14.4, 15.4, 17.8, 18.133333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_59.png)
    


    Neighborhood: MLK Park
    Actuals: [10, 9, 7, 14, 14, 7, 4, 7]
    Predictions: [7.6, 7.866666666666666, 8.066666666666666, 7.933333333333334, 10.133333333333333, 11.666666666666666, 10.4, 8.333333333333334]
    


    
![png](\img\posts\Buffalo-Crime\output_96_61.png)
    


    Neighborhood: Parkside
    Actuals: [41, 17, 30, 32, 28, 24, 22, 27, 2]
    Predictions: [17.333333333333332, 24.133333333333333, 22.066666666666666, 24.8, 27.666666666666668, 28.866666666666667, 27.0, 25.6, 25.533333333333335]
    


    
![png](\img\posts\Buffalo-Crime\output_96_63.png)
    


    Neighborhood: Kaisertown
    Actuals: [5, 2, 4, 2, 5, 5, 2, 4]
    Predictions: [3.6666666666666665, 4.133333333333334, 3.4, 3.533333333333333, 3.066666666666667, 3.6, 4.066666666666666, 3.533333333333333]
    


    
![png](\img\posts\Buffalo-Crime\output_96_65.png)
    


    Neighborhood: Seneca-Cazenovia
    Actuals: [25, 13, 20, 21, 22, 15, 22, 19, 1]
    Predictions: [13.266666666666667, 16.333333333333332, 15.2, 16.533333333333335, 18.333333333333332, 20.333333333333332, 18.6, 19.866666666666667, 19.533333333333335]
    


    
![png](\img\posts\Buffalo-Crime\output_96_67.png)
    


    Neighborhood: Grant-Amherst
    Actuals: [11, 9, 7, 8, 7, 8, 12, 6]
    Predictions: [8.266666666666667, 9.2, 9.266666666666667, 8.533333333333333, 8.2, 7.8, 7.666666666666667, 9.066666666666666]
    


    
![png](\img\posts\Buffalo-Crime\output_96_69.png)
    


    Test RMSE: 11.405
    

# Crime Projection On The Last Eight Months Using Exponential Moving Average



```python
# Exponential Moving Average
predTot = list()
testTot = list()
alpha = 0.6

# Get unique neighborhood names
unique_neighborhoods = data_test['neighborhood_1'].unique()

# Walk forward over time steps in test
for neighNum, neighborhood in enumerate(unique_neighborhoods):

    history = train_d[neighNum]
    test = test_d[neighNum]

    # Check if there is test data for this neighborhood
    if len(test) == 0:
        continue  # Skip neighborhoods with no test data

    preds = []
    lastPred = 0
    for t in range(len(test)):
        yhat = ((1-alpha)*lastPred + (alpha*history[-1]))
        lastPred = yhat
        obs = test[t]
        preds.append(yhat)
        history.append(obs)

    # Plot
    plt.figure(figsize=(8, 4))  # Adjust figure size
    plt.plot(test, color='yellowgreen')
    plt.plot(preds, color='steelblue')

    # Add neighborhood name as annotation
    plt.annotate(neighborhood, (0.02, 0.9), xycoords='axes fraction', fontsize=12, color='black')

    plt.title(f'Exponential Moving Average - {neighborhood}')
    plt.xlabel('Months Staring in Jan')
    plt.ylabel('Number Of Crimes')
    plt.legend(['Test Data', 'Predictions'])
    plt.show()

        #print('Neighborhood: {}'.format(neighNum+1))
    print('Neighborhood: {}'.format(neighborhood))
    print('Actuals: {}'.format(test))
    print('Predictions: {}'.format(preds))

    testTot = testTot + test
    predTot = predTot + preds

error = mean_squared_error(predTot, testTot) ** .5
print('Test RMSE: %.3f' % error)

```


    
![png](\img\posts\Buffalo-Crime\output_98_0.png)
    


    Neighborhood: South Park
    Actuals: [67, 50, 72, 65, 63, 58, 45, 55, 1]
    Predictions: [0.6, 40.44, 46.176, 61.6704, 63.66816, 63.267264, 60.1069056, 51.04276224, 53.417104896]
    


    
![png](\img\posts\Buffalo-Crime\output_98_2.png)
    


    Neighborhood: Hopkins-Tifft
    Actuals: [30, 16, 17, 35, 16, 27, 18, 33, 2]
    Predictions: [1.2, 18.48, 16.992, 16.9968, 27.79872, 20.719488, 24.4877952, 20.59511808, 28.038047232]
    


    
![png](\img\posts\Buffalo-Crime\output_98_4.png)
    


    Neighborhood: Lower West Side
    Actuals: [28, 16, 24, 25, 31, 30, 35, 27, 1]
    Predictions: [0.6, 17.04, 16.416, 20.9664, 23.386560000000003, 27.954624, 29.1818496, 32.67273984, 29.269095936]
    


    
![png](\img\posts\Buffalo-Crime\output_98_6.png)
    


    Neighborhood: Central
    Actuals: [16, 11, 7, 15, 16, 20, 16, 13, 2]
    Predictions: [1.2, 10.08, 10.632, 8.4528, 12.38112, 14.552448, 17.8209792, 16.72839168, 14.491356672000002]
    


    
![png](\img\posts\Buffalo-Crime\output_98_8.png)
    


    Neighborhood: Lovejoy
    Actuals: [23, 13, 23, 21, 28, 17, 15, 23, 3]
    Predictions: [1.7999999999999998, 14.52, 13.608, 19.2432, 20.29728, 24.918912, 20.1675648, 17.06702592, 20.626810368]
    


    
![png](\img\posts\Buffalo-Crime\output_98_10.png)
    


    Neighborhood: North Park
    Actuals: [18, 19, 21, 24, 19, 28, 34, 33, 1]
    Predictions: [0.6, 11.04, 15.815999999999999, 18.9264, 21.97056, 20.188223999999998, 24.875289600000002, 30.35011584, 31.940046336]
    


    
![png](\img\posts\Buffalo-Crime\output_98_12.png)
    


    Neighborhood: Kensington-Bailey
    Actuals: [32, 27, 30, 36, 40, 34, 34, 41]
    Predictions: [24.599999999999998, 29.04, 27.816, 29.1264, 33.25056, 37.300224, 35.3200896, 34.52803584]
    


    
![png](\img\posts\Buffalo-Crime\output_98_14.png)
    


    Neighborhood: Elmwood Bryant
    Actuals: [41, 39, 52, 44, 45, 45, 42, 53]
    Predictions: [31.799999999999997, 37.31999999999999, 38.327999999999996, 46.5312, 45.01248, 45.004992, 45.0019968, 43.20079872]
    


    
![png](\img\posts\Buffalo-Crime\output_98_16.png)
    


    Neighborhood: Pratt-Willert
    Actuals: [8, 10, 14, 13, 17, 20, 18, 29, 1]
    Predictions: [0.6, 5.04, 8.016, 11.6064, 12.44256, 15.177024, 18.0708096, 18.02832384, 24.611329536]
    


    
![png](\img\posts\Buffalo-Crime\output_98_18.png)
    


    Neighborhood: Masten Park
    Actuals: [33, 21, 20, 10, 19, 15, 14, 24]
    Predictions: [14.399999999999999, 25.560000000000002, 22.824, 21.129600000000003, 14.451840000000002, 17.180736000000003, 15.872294400000001, 14.748917760000001]
    


    
![png](\img\posts\Buffalo-Crime\output_98_20.png)
    


    Neighborhood: West Hertel
    Actuals: [39, 28, 26, 34, 39, 47, 50, 62, 1]
    Predictions: [0.6, 23.639999999999997, 26.256, 26.102400000000003, 30.840960000000003, 35.736384, 42.4945536, 46.99782144, 55.999128576]
    


    
![png](\img\posts\Buffalo-Crime\output_98_22.png)
    


    Neighborhood: University Heights
    Actuals: [45, 46, 43, 41, 43, 60, 90, 58, 2]
    Predictions: [1.2, 27.48, 38.592, 41.2368, 41.094719999999995, 42.237888, 52.895155200000005, 75.15806208000001, 64.863224832]
    


    
![png](\img\posts\Buffalo-Crime\output_98_24.png)
    


    Neighborhood: Broadway Fillmore
    Actuals: [56, 21, 27, 39, 53, 60, 51, 48, 1]
    Predictions: [0.6, 33.84, 26.136000000000003, 26.654400000000003, 34.06176, 45.424704, 54.1698816, 52.26795264, 49.707181055999996]
    


    
![png](\img\posts\Buffalo-Crime\output_98_26.png)
    


    Neighborhood: Elmwood Bidwell
    Actuals: [34, 22, 19, 24, 24, 30, 43, 43, 3]
    Predictions: [1.7999999999999998, 21.119999999999997, 21.647999999999996, 20.059199999999997, 22.423679999999997, 23.369472, 27.3477888, 36.73911552, 40.495646208]
    


    
![png](\img\posts\Buffalo-Crime\output_98_28.png)
    


    Neighborhood: Genesee-Moselle
    Actuals: [32, 26, 29, 29, 34, 26, 33, 38]
    Predictions: [22.8, 28.32, 26.928, 28.1712, 28.66848, 31.867392, 28.3469568, 31.138782720000002]
    


    
![png](\img\posts\Buffalo-Crime\output_98_30.png)
    


    Neighborhood: Upper West Side
    Actuals: [18, 7, 18, 14, 10, 19, 13, 27]
    Predictions: [16.2, 17.28, 11.112000000000002, 15.2448, 14.49792, 11.799168000000002, 16.119667200000002, 14.24786688]
    


    
![png](\img\posts\Buffalo-Crime\output_98_32.png)
    


    Neighborhood: West Side
    Actuals: [36, 24, 40, 40, 52, 60, 51, 52, 5]
    Predictions: [3.0, 22.799999999999997, 23.519999999999996, 33.408, 37.3632, 46.14528, 54.458112, 52.3832448, 52.15329792]
    


    
![png](\img\posts\Buffalo-Crime\output_98_34.png)
    


    Neighborhood: Hamlin Park
    Actuals: [30, 15, 11, 13, 30, 19, 20, 15, 1]
    Predictions: [0.6, 18.24, 16.296, 13.1184, 13.047360000000001, 23.218944, 20.6875776, 20.275031040000002, 17.110012416000004]
    


    
![png](\img\posts\Buffalo-Crime\output_98_36.png)
    


    Neighborhood: Ellicott
    Actuals: [27, 15, 17, 16, 21, 20, 35, 21, 1]
    Predictions: [0.6, 16.439999999999998, 15.576, 16.4304, 16.172159999999998, 19.068863999999998, 19.627545599999998, 28.85101824, 24.140407296]
    


    
![png](\img\posts\Buffalo-Crime\output_98_38.png)
    


    Neighborhood: Seneca Babcock
    Actuals: [33, 19, 40, 34, 37, 40, 44, 30, 1]
    Predictions: [0.6, 20.04, 19.416, 31.7664, 33.10656, 35.442624, 38.177049600000004, 41.67081984, 34.668327936]
    


    
![png](\img\posts\Buffalo-Crime\output_98_40.png)
    


    Neighborhood: Kenfield
    Actuals: [13, 14, 16, 14, 20, 16, 17, 14]
    Predictions: [8.4, 11.16, 12.864, 14.7456, 14.29824, 17.719296, 16.6877184, 16.875087360000002]
    


    
![png](\img\posts\Buffalo-Crime\output_98_42.png)
    


    Neighborhood: nan
    Actuals: [14, 18, 22, 25, 30, 22, 16, 11]
    Predictions: [6.6, 11.040000000000001, 15.216, 19.2864, 22.71456, 27.085824000000002, 24.0343296, 19.21373184]
    


    
![png](\img\posts\Buffalo-Crime\output_98_44.png)
    


    Neighborhood: First Ward
    Actuals: [29, 26, 29, 39, 30, 36, 37, 26]
    Predictions: [15.6, 23.64, 25.056, 27.4224, 34.36896, 31.747584000000003, 34.2990336, 35.91961344]
    


    
![png](\img\posts\Buffalo-Crime\output_98_46.png)
    


    Neighborhood: Allentown
    Actuals: [33, 27, 26, 37, 39, 30, 32, 41]
    Predictions: [24.599999999999998, 29.64, 28.056, 26.822400000000002, 32.928960000000004, 36.571584, 32.6286336, 32.25145344]
    


    
![png](\img\posts\Buffalo-Crime\output_98_48.png)
    


    Neighborhood: Black Rock
    Actuals: [12, 15, 14, 24, 21, 25, 23, 15]
    Predictions: [9.0, 10.799999999999999, 13.32, 13.728000000000002, 19.891199999999998, 20.55648, 23.222592, 23.0890368]
    


    
![png](\img\posts\Buffalo-Crime\output_98_50.png)
    


    Neighborhood: Delavan Grider
    Actuals: [16, 5, 13, 16, 24, 18, 18, 20]
    Predictions: [12.0, 14.4, 8.760000000000002, 11.304, 14.1216, 20.04864, 18.819456, 18.327782399999997]
    


    
![png](\img\posts\Buffalo-Crime\output_98_52.png)
    


    Neighborhood: Schiller Park
    Actuals: [12, 12, 11, 20, 9, 13, 11, 14, 1]
    Predictions: [0.6, 7.4399999999999995, 10.175999999999998, 10.670399999999999, 16.26816, 11.907264000000001, 12.5629056, 11.62516224, 13.050064896]
    


    
![png](\img\posts\Buffalo-Crime\output_98_54.png)
    


    Neighborhood: Riverside
    Actuals: [17, 12, 14, 6, 7, 15, 12, 7]
    Predictions: [4.2, 11.879999999999999, 11.951999999999998, 13.1808, 8.87232, 7.748928, 12.0995712, 12.03982848]
    


    
![png](\img\posts\Buffalo-Crime\output_98_56.png)
    


    Neighborhood: Fruit Belt
    Actuals: [12, 8, 4, 7, 13, 7, 8, 10]
    Predictions: [6.0, 9.6, 8.64, 5.856, 6.542400000000001, 10.41696, 8.366783999999999, 8.1467136]
    


    
![png](\img\posts\Buffalo-Crime\output_98_58.png)
    


    Neighborhood: Central Park
    Actuals: [19, 16, 9, 17, 14, 18, 22, 17, 1]
    Predictions: [0.6, 11.64, 14.256, 11.1024, 14.64096, 14.256384, 16.5025536, 19.80102144, 18.120408576]
    


    
![png](\img\posts\Buffalo-Crime\output_98_60.png)
    


    Neighborhood: MLK Park
    Actuals: [10, 9, 7, 14, 14, 7, 4, 7]
    Predictions: [4.2, 7.68, 8.472, 7.5888, 11.43552, 12.974208, 9.3896832, 6.15587328]
    


    
![png](\img\posts\Buffalo-Crime\output_98_62.png)
    


    Neighborhood: Parkside
    Actuals: [41, 17, 30, 32, 28, 24, 22, 27, 2]
    Predictions: [1.2, 25.08, 20.232, 26.0928, 29.63712, 28.654848, 25.861939200000002, 23.54477568, 25.617910272]
    


    
![png](\img\posts\Buffalo-Crime\output_98_64.png)
    


    Neighborhood: Kaisertown
    Actuals: [5, 2, 4, 2, 5, 5, 2, 4]
    Predictions: [2.4, 3.96, 2.784, 3.5136, 2.6054399999999998, 4.0421759999999995, 4.6168704, 3.04674816]
    


    
![png](\img\posts\Buffalo-Crime\output_98_66.png)
    


    Neighborhood: Seneca-Cazenovia
    Actuals: [25, 13, 20, 21, 22, 15, 22, 19, 1]
    Predictions: [0.6, 15.24, 13.896, 17.5584, 19.623359999999998, 21.049343999999998, 17.419737599999998, 20.167895039999998, 19.467158016]
    


    
![png](\img\posts\Buffalo-Crime\output_98_68.png)
    


    Neighborhood: Grant-Amherst
    Actuals: [11, 9, 7, 8, 7, 8, 12, 6]
    Predictions: [3.5999999999999996, 8.04, 8.616, 7.6464, 7.85856, 7.343424000000001, 7.7373696, 10.294947839999999]
    Test RMSE: 13.853
    

# Conclusion
<a id='conclusion'></a>

In conclusion, the graphs and charts presented throughout the project have been instrumental in conveying critical insights:

- We observed a remarkable annual decline in the total number of crimes since 2009.

- The year 2022 accounted for a relatively modest 3.95% of the total crimes recorded in the dataset spanning from 2009 to the present day.

- While Fridays appeared to exhibit a slightly higher incidence of crimes when compared to other days, the difference was not markedly significant.

- February consistently registered the lowest number of crimes per month, as evident from the graphical representations.

- The annual crime rate displayed a declining trend, characterized by a distinctive zigzag pattern, with crime receding during colder seasons and resurging during hotter months.

- The specific week within a month appeared to have minimal impact on crime rates, with the observation that the fifth week recorded fewer incidents, attributed to its shorter duration.

- Our hypothesis regarding the decrease in crime during the blizzard was disproven; it was solely attributable to the typical February weather and its influence on crime due to the freezing conditions.

- The predominant type of crime across Buffalo neighborhoods was larceny or theft, with the noteworthy exception of the Delavan Grider neighborhood, where assault was the dominant category.

In terms of forecasting accuracy, we obtained Root Mean Square Errors (RMSE) for crime predictions per neighborhood:

- Simple Moving Average: RMSE of 11.41
- Weighted Moving Average: RMSE of 11.405
- Exponential Moving Average: RMSE of 13.85
