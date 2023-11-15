---
layout: post
title: "Erie County Redfin Housing Sales"
subtitle: "Erie County Housing Sales Analysis using Refin API, Pandas and Tableau"
date: 2023-11-02
background: '/img/posts/Redfin-Housing-Sales/Redfinlogo.png'

#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---


### Table of Contents

- [Summary](#summary)
- [Python Code](#imports)
- [Tableau Screenshots](#screenshots)


### Summary
<a id='summary'></a>

In this post, we explore real estate market trends in New York State using Python and Tableau. We leverage various Python libraries such as pandas, numpy, matplotlib, seaborn, and others to analyze and visualize the data. Our primary data source is a Redfin market tracker dataset for zip codes, which we load and process to gain insights into the real estate market.

Data Loading and Exploration:
We start by importing necessary packages and loading the dataset from the Redfin market tracker. The dataset is a comprehensive source of information, and we print the number of rows, columns, and display the first few rows to get a glimpse of the data.

Geospatial Data:
To enhance our analysis, we integrate geospatial data that includes information about states, counties, and zip codes. We use this data to filter and focus on a specific state, in this case, New York. Visualizing the geographical distribution of data adds context to our analysis, and we display an image of Erie County for reference.

Data Cleaning:
To ensure the accuracy of our analysis, we check the validity of zip codes in the geospatial data. Invalid zip codes are identified and reported. We create a cleaned dataset containing only valid zip codes for further analysis.

Merging Datasets:
We merge the Redfin market data with the cleaned geospatial data, combining relevant information such as zip codes, cities, and counties. This merged dataset sets the foundation for our subsequent analysis.

Exploratory Data Analysis (EDA):
We perform exploratory data analysis by selecting a specific zip code and property type. This allows us to analyze trends over time, specifically focusing on metrics such as median days on market. Additionally, we correct an error in the calculation of month-over-month changes in median days on market.

Feature Engineering:
We enhance our dataset by adding features like the most recent date flag and correcting the median days on market based on chronological order. These features contribute to a more comprehensive analysis of real estate market trends.

Conclusion and Download:
In conclusion, we provide a downloadable CSV file containing the processed dataset to further explore the data in Tableau.


### Python Code
<a id='imports'></a>
```python
# import packages

import pandas as pd
import numpy as np

import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import os


plt.style.use('seaborn-v0_8-darkgrid')
# warnings ignore
import warnings
# set warnings to ignore
#warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None  # default='warn'
```


```python
pd.set_option('display.max_columns', None)
```


```python

url = 'https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz'

# start time to read file
startTime = time.time()
df = pd.read_csv(url, compression='gzip', sep='\t', on_bad_lines='skip')
# end time
executionTime = (time.time() - startTime)
print('Execution time in minutes: ' + str(round(executionTime / 60, 2)))
print('Num of rows:', len(df))
print('Num of cols:', len(df.columns))
df.head()
```

    Execution time in minutes: 3.0
    Num of rows: 7476840
    Num of cols: 58
    





  <div id="df-6e2f8487-d3d0-49b7-9a20-2f70191885d2" class="colab-df-container">
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
      <th>period_begin</th>
      <th>period_end</th>
      <th>period_duration</th>
      <th>region_type</th>
      <th>region_type_id</th>
      <th>table_id</th>
      <th>is_seasonally_adjusted</th>
      <th>region</th>
      <th>city</th>
      <th>state</th>
      <th>state_code</th>
      <th>property_type</th>
      <th>property_type_id</th>
      <th>median_sale_price</th>
      <th>median_sale_price_mom</th>
      <th>median_sale_price_yoy</th>
      <th>median_list_price</th>
      <th>median_list_price_mom</th>
      <th>median_list_price_yoy</th>
      <th>median_ppsf</th>
      <th>median_ppsf_mom</th>
      <th>median_ppsf_yoy</th>
      <th>median_list_ppsf</th>
      <th>median_list_ppsf_mom</th>
      <th>median_list_ppsf_yoy</th>
      <th>homes_sold</th>
      <th>homes_sold_mom</th>
      <th>homes_sold_yoy</th>
      <th>pending_sales</th>
      <th>pending_sales_mom</th>
      <th>pending_sales_yoy</th>
      <th>new_listings</th>
      <th>new_listings_mom</th>
      <th>new_listings_yoy</th>
      <th>inventory</th>
      <th>inventory_mom</th>
      <th>inventory_yoy</th>
      <th>months_of_supply</th>
      <th>months_of_supply_mom</th>
      <th>months_of_supply_yoy</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>avg_sale_to_list</th>
      <th>avg_sale_to_list_mom</th>
      <th>avg_sale_to_list_yoy</th>
      <th>sold_above_list</th>
      <th>sold_above_list_mom</th>
      <th>sold_above_list_yoy</th>
      <th>price_drops</th>
      <th>price_drops_mom</th>
      <th>price_drops_yoy</th>
      <th>off_market_in_two_weeks</th>
      <th>off_market_in_two_weeks_mom</th>
      <th>off_market_in_two_weeks_yoy</th>
      <th>parent_metro_region</th>
      <th>parent_metro_region_metro_code</th>
      <th>last_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-02-01</td>
      <td>2017-04-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>2921</td>
      <td>f</td>
      <td>Zip Code: 08037</td>
      <td>NaN</td>
      <td>New Jersey</td>
      <td>NJ</td>
      <td>Multi-Family (2-4 Unit)</td>
      <td>4</td>
      <td>233000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>149900.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>104.719101</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>78.693331</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>135.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.913725</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Atlantic City, NJ</td>
      <td>12100</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-12-01</td>
      <td>2017-02-28</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>36342</td>
      <td>f</td>
      <td>Zip Code: 85303</td>
      <td>NaN</td>
      <td>Arizona</td>
      <td>AZ</td>
      <td>All Residential</td>
      <td>-1</td>
      <td>182750.0</td>
      <td>0.065598</td>
      <td>0.068713</td>
      <td>189900.0</td>
      <td>0.026486</td>
      <td>0.085143</td>
      <td>110.310638</td>
      <td>0.002052</td>
      <td>0.089112</td>
      <td>117.628681</td>
      <td>0.005186</td>
      <td>0.124287</td>
      <td>68.0</td>
      <td>-0.190476</td>
      <td>-0.252747</td>
      <td>35.0</td>
      <td>0.346154</td>
      <td>0.166667</td>
      <td>107.0</td>
      <td>0.000000</td>
      <td>0.019048</td>
      <td>82.0</td>
      <td>-0.136842</td>
      <td>0.138889</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>53.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>0.983675</td>
      <td>-0.004671</td>
      <td>0.004752</td>
      <td>0.205882</td>
      <td>-0.044118</td>
      <td>0.030058</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.257143</td>
      <td>0.026374</td>
      <td>-0.109524</td>
      <td>Phoenix, AZ</td>
      <td>38060</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-07-01</td>
      <td>2018-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5832</td>
      <td>f</td>
      <td>Zip Code: 15644</td>
      <td>NaN</td>
      <td>Pennsylvania</td>
      <td>PA</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>116500.0</td>
      <td>0.030973</td>
      <td>0.165583</td>
      <td>99450.0</td>
      <td>-0.074022</td>
      <td>0.050158</td>
      <td>106.165882</td>
      <td>0.099922</td>
      <td>0.059423</td>
      <td>86.666579</td>
      <td>-0.013773</td>
      <td>-0.115633</td>
      <td>58.0</td>
      <td>-0.016949</td>
      <td>-0.033333</td>
      <td>10.0</td>
      <td>-0.285714</td>
      <td>-0.285714</td>
      <td>61.0</td>
      <td>-0.217949</td>
      <td>-0.298851</td>
      <td>104.0</td>
      <td>-0.009524</td>
      <td>-0.111111</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.0</td>
      <td>-16.0</td>
      <td>-13.5</td>
      <td>0.959806</td>
      <td>-0.005870</td>
      <td>0.001646</td>
      <td>0.103448</td>
      <td>-0.082992</td>
      <td>-0.096552</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-0.285714</td>
      <td>-0.071429</td>
      <td>Pittsburgh, PA</td>
      <td>38300</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-09-01</td>
      <td>2014-11-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>17223</td>
      <td>f</td>
      <td>Zip Code: 40507</td>
      <td>NaN</td>
      <td>Kentucky</td>
      <td>KY</td>
      <td>Condo/Co-op</td>
      <td>3</td>
      <td>238000.0</td>
      <td>0.012766</td>
      <td>-0.004184</td>
      <td>196950.0</td>
      <td>-0.083527</td>
      <td>-0.218452</td>
      <td>188.000000</td>
      <td>0.026643</td>
      <td>0.031127</td>
      <td>194.026540</td>
      <td>0.040589</td>
      <td>-0.002149</td>
      <td>9.0</td>
      <td>-0.181818</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>-0.454545</td>
      <td>1.000000</td>
      <td>19.0</td>
      <td>-0.173913</td>
      <td>-0.441176</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104.5</td>
      <td>-21.5</td>
      <td>-68.5</td>
      <td>0.979014</td>
      <td>0.017979</td>
      <td>0.034035</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>0.111111</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Lexington, KY</td>
      <td>30460</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-01</td>
      <td>2021-04-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>36264</td>
      <td>f</td>
      <td>Zip Code: 85212</td>
      <td>NaN</td>
      <td>Arizona</td>
      <td>AZ</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>454500.0</td>
      <td>0.041822</td>
      <td>0.218662</td>
      <td>451275.0</td>
      <td>0.005067</td>
      <td>0.236370</td>
      <td>209.806157</td>
      <td>0.039224</td>
      <td>0.242289</td>
      <td>212.842720</td>
      <td>0.013018</td>
      <td>0.225578</td>
      <td>265.0</td>
      <td>0.019231</td>
      <td>0.031128</td>
      <td>92.0</td>
      <td>-0.041667</td>
      <td>0.150000</td>
      <td>294.0</td>
      <td>0.069091</td>
      <td>-0.023256</td>
      <td>68.0</td>
      <td>0.096774</td>
      <td>-0.595238</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>-6.0</td>
      <td>-27.0</td>
      <td>1.025972</td>
      <td>0.007849</td>
      <td>0.031341</td>
      <td>0.630189</td>
      <td>0.084035</td>
      <td>0.392835</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.630435</td>
      <td>0.026268</td>
      <td>0.242935</td>
      <td>Phoenix, AZ</td>
      <td>38060</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6e2f8487-d3d0-49b7-9a20-2f70191885d2')"
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
        document.querySelector('#df-6e2f8487-d3d0-49b7-9a20-2f70191885d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6e2f8487-d3d0-49b7-9a20-2f70191885d2');
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


<div id="df-265462c0-f9da-4578-923a-4f800e4f01ad">
  <button class="colab-df-quickchart" onclick="quickchart('df-265462c0-f9da-4578-923a-4f800e4f01ad')"
            title="Suggest charts"
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
        document.querySelector('#df-265462c0-f9da-4578-923a-4f800e4f01ad button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# filter on state code
state_code = 'NY'
df_filter =df.loc[df['state_code'] == state_code]
print('Num of rows:', len(df_filter))
df_filter.head()
```

    Num of rows: 483632
    





  <div id="df-ec43ab6c-5456-4da4-a606-1d7f684fb999" class="colab-df-container">
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
      <th>period_begin</th>
      <th>period_end</th>
      <th>period_duration</th>
      <th>region_type</th>
      <th>region_type_id</th>
      <th>table_id</th>
      <th>is_seasonally_adjusted</th>
      <th>region</th>
      <th>city</th>
      <th>state</th>
      <th>state_code</th>
      <th>property_type</th>
      <th>property_type_id</th>
      <th>median_sale_price</th>
      <th>median_sale_price_mom</th>
      <th>median_sale_price_yoy</th>
      <th>median_list_price</th>
      <th>median_list_price_mom</th>
      <th>median_list_price_yoy</th>
      <th>median_ppsf</th>
      <th>median_ppsf_mom</th>
      <th>median_ppsf_yoy</th>
      <th>median_list_ppsf</th>
      <th>median_list_ppsf_mom</th>
      <th>median_list_ppsf_yoy</th>
      <th>homes_sold</th>
      <th>homes_sold_mom</th>
      <th>homes_sold_yoy</th>
      <th>pending_sales</th>
      <th>pending_sales_mom</th>
      <th>pending_sales_yoy</th>
      <th>new_listings</th>
      <th>new_listings_mom</th>
      <th>new_listings_yoy</th>
      <th>inventory</th>
      <th>inventory_mom</th>
      <th>inventory_yoy</th>
      <th>months_of_supply</th>
      <th>months_of_supply_mom</th>
      <th>months_of_supply_yoy</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>avg_sale_to_list</th>
      <th>avg_sale_to_list_mom</th>
      <th>avg_sale_to_list_yoy</th>
      <th>sold_above_list</th>
      <th>sold_above_list_mom</th>
      <th>sold_above_list_yoy</th>
      <th>price_drops</th>
      <th>price_drops_mom</th>
      <th>price_drops_yoy</th>
      <th>off_market_in_two_weeks</th>
      <th>off_market_in_two_weeks_mom</th>
      <th>off_market_in_two_weeks_yoy</th>
      <th>parent_metro_region</th>
      <th>parent_metro_region_metro_code</th>
      <th>last_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>2018-08-01</td>
      <td>2018-10-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>3452</td>
      <td>f</td>
      <td>Zip Code: 10509</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Townhouse</td>
      <td>13</td>
      <td>250000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>314000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>156.250000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>179.985868</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>139.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.980392</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New York, NY</td>
      <td>35614</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2016-07-01</td>
      <td>2016-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>4280</td>
      <td>f</td>
      <td>Zip Code: 12547</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Multi-Family (2-4 Unit)</td>
      <td>4</td>
      <td>155000.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>619949.5</td>
      <td>1.584200</td>
      <td>NaN</td>
      <td>184.523810</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>134.106454</td>
      <td>0.233176</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.837838</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kingston, NY</td>
      <td>28740</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2012-08-01</td>
      <td>2012-10-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>3790</td>
      <td>f</td>
      <td>Zip Code: 11572</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>396250.0</td>
      <td>0.031901</td>
      <td>-0.016749</td>
      <td>399499.5</td>
      <td>-0.110246</td>
      <td>-0.076300</td>
      <td>231.925656</td>
      <td>-0.021452</td>
      <td>-0.022505</td>
      <td>238.468989</td>
      <td>-0.029403</td>
      <td>-0.069139</td>
      <td>50.0</td>
      <td>-0.019608</td>
      <td>0.111111</td>
      <td>5.0</td>
      <td>-0.545455</td>
      <td>-0.375000</td>
      <td>70.0</td>
      <td>-0.135802</td>
      <td>0.129032</td>
      <td>142.0</td>
      <td>0.028986</td>
      <td>-0.027397</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104.5</td>
      <td>-0.5</td>
      <td>20.5</td>
      <td>0.945461</td>
      <td>0.000605</td>
      <td>0.001794</td>
      <td>0.040000</td>
      <td>0.000784</td>
      <td>-0.071111</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>-0.090909</td>
      <td>0.000000</td>
      <td>Nassau County, NY</td>
      <td>35004</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2019-05-01</td>
      <td>2019-07-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>4440</td>
      <td>f</td>
      <td>Zip Code: 12878</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>158000.0</td>
      <td>-0.363417</td>
      <td>0.166052</td>
      <td>164900.0</td>
      <td>0.157193</td>
      <td>-0.586717</td>
      <td>107.513499</td>
      <td>0.020027</td>
      <td>-0.997177</td>
      <td>93.163842</td>
      <td>-0.035048</td>
      <td>-0.512194</td>
      <td>4.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.500000</td>
      <td>0.250000</td>
      <td>5.0</td>
      <td>0.000000</td>
      <td>-0.166667</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>230.0</td>
      <td>-74.0</td>
      <td>-200.0</td>
      <td>0.917599</td>
      <td>-0.049489</td>
      <td>-0.100128</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Glens Falls, NY</td>
      <td>24020</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2021-09-01</td>
      <td>2021-11-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>3818</td>
      <td>f</td>
      <td>Zip Code: 11706</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>465000.0</td>
      <td>0.015284</td>
      <td>0.134146</td>
      <td>450000.0</td>
      <td>-0.020664</td>
      <td>0.058824</td>
      <td>302.572385</td>
      <td>-0.032961</td>
      <td>0.248304</td>
      <td>319.148936</td>
      <td>0.022469</td>
      <td>0.153220</td>
      <td>122.0</td>
      <td>0.079646</td>
      <td>-0.122302</td>
      <td>25.0</td>
      <td>-0.264706</td>
      <td>-0.074074</td>
      <td>105.0</td>
      <td>-0.062500</td>
      <td>-0.355828</td>
      <td>63.0</td>
      <td>-0.030769</td>
      <td>-0.343750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>-3.0</td>
      <td>1.040465</td>
      <td>0.002822</td>
      <td>0.036925</td>
      <td>0.737705</td>
      <td>0.029740</td>
      <td>0.226914</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.44</td>
      <td>0.087059</td>
      <td>0.291852</td>
      <td>Nassau County, NY</td>
      <td>35004</td>
      <td>2023-10-29 14:25:50</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ec43ab6c-5456-4da4-a606-1d7f684fb999')"
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
        document.querySelector('#df-ec43ab6c-5456-4da4-a606-1d7f684fb999 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ec43ab6c-5456-4da4-a606-1d7f684fb999');
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


<div id="df-d2f1e9df-df4d-4409-8510-8aac8c6bcf48">
  <button class="colab-df-quickchart" onclick="quickchart('df-d2f1e9df-df4d-4409-8510-8aac8c6bcf48')"
            title="Suggest charts"
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
        document.querySelector('#df-d2f1e9df-df4d-4409-8510-8aac8c6bcf48 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
df_filter.describe()
```





  <div id="df-5b9a7bfc-ed1a-4c34-b4bf-58edeaf6ac2a" class="colab-df-container">
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
      <th>period_duration</th>
      <th>region_type_id</th>
      <th>table_id</th>
      <th>city</th>
      <th>property_type_id</th>
      <th>median_sale_price</th>
      <th>median_sale_price_mom</th>
      <th>median_sale_price_yoy</th>
      <th>median_list_price</th>
      <th>median_list_price_mom</th>
      <th>median_list_price_yoy</th>
      <th>median_ppsf</th>
      <th>median_ppsf_mom</th>
      <th>median_ppsf_yoy</th>
      <th>median_list_ppsf</th>
      <th>median_list_ppsf_mom</th>
      <th>median_list_ppsf_yoy</th>
      <th>homes_sold</th>
      <th>homes_sold_mom</th>
      <th>homes_sold_yoy</th>
      <th>pending_sales</th>
      <th>pending_sales_mom</th>
      <th>pending_sales_yoy</th>
      <th>new_listings</th>
      <th>new_listings_mom</th>
      <th>new_listings_yoy</th>
      <th>inventory</th>
      <th>inventory_mom</th>
      <th>inventory_yoy</th>
      <th>months_of_supply</th>
      <th>months_of_supply_mom</th>
      <th>months_of_supply_yoy</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>avg_sale_to_list</th>
      <th>avg_sale_to_list_mom</th>
      <th>avg_sale_to_list_yoy</th>
      <th>sold_above_list</th>
      <th>sold_above_list_mom</th>
      <th>sold_above_list_yoy</th>
      <th>price_drops</th>
      <th>price_drops_mom</th>
      <th>price_drops_yoy</th>
      <th>off_market_in_two_weeks</th>
      <th>off_market_in_two_weeks_mom</th>
      <th>off_market_in_two_weeks_yoy</th>
      <th>parent_metro_region_metro_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>483632.0</td>
      <td>483632.0</td>
      <td>483632.000000</td>
      <td>0.0</td>
      <td>483632.000000</td>
      <td>4.836310e+05</td>
      <td>458476.000000</td>
      <td>411219.000000</td>
      <td>4.411990e+05</td>
      <td>416001.000000</td>
      <td>381840.000000</td>
      <td>4.725180e+05</td>
      <td>447056.000000</td>
      <td>400187.000000</td>
      <td>4.349210e+05</td>
      <td>409627.000000</td>
      <td>375851.000000</td>
      <td>483632.000000</td>
      <td>458478.000000</td>
      <td>411221.000000</td>
      <td>307990.000000</td>
      <td>249870.000000</td>
      <td>236135.000000</td>
      <td>441428.000000</td>
      <td>416233.000000</td>
      <td>382079.000000</td>
      <td>447182.000000</td>
      <td>423817.000000</td>
      <td>388225.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>481521.00000</td>
      <td>456316.000000</td>
      <td>409097.000000</td>
      <td>481160.000000</td>
      <td>455978.000000</td>
      <td>408895.000000</td>
      <td>483632.000000</td>
      <td>458478.000000</td>
      <td>411221.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>307990.000000</td>
      <td>249870.000000</td>
      <td>236135.000000</td>
      <td>483632.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>4344.397916</td>
      <td>NaN</td>
      <td>3.440525</td>
      <td>4.171076e+05</td>
      <td>0.069758</td>
      <td>0.218782</td>
      <td>4.482294e+05</td>
      <td>0.045579</td>
      <td>0.146212</td>
      <td>2.442654e+02</td>
      <td>0.135518</td>
      <td>0.457926</td>
      <td>2.669602e+02</td>
      <td>0.053183</td>
      <td>1.116265</td>
      <td>18.299653</td>
      <td>0.074915</td>
      <td>0.243026</td>
      <td>6.713721</td>
      <td>0.257825</td>
      <td>0.368467</td>
      <td>25.987049</td>
      <td>0.068378</td>
      <td>0.176109</td>
      <td>32.909323</td>
      <td>0.031351</td>
      <td>0.062185</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>124.88643</td>
      <td>-0.807652</td>
      <td>-10.342834</td>
      <td>0.964606</td>
      <td>0.000505</td>
      <td>0.005243</td>
      <td>0.213572</td>
      <td>0.002272</td>
      <td>0.025362</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.235077</td>
      <td>0.000291</td>
      <td>0.026445</td>
      <td>31491.533658</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1548.475951</td>
      <td>NaN</td>
      <td>3.950440</td>
      <td>9.143662e+05</td>
      <td>7.179519</td>
      <td>14.369080</td>
      <td>9.020977e+05</td>
      <td>2.910467</td>
      <td>5.015077</td>
      <td>4.023996e+03</td>
      <td>24.710078</td>
      <td>62.409171</td>
      <td>4.333815e+03</td>
      <td>5.245511</td>
      <td>231.303425</td>
      <td>27.660098</td>
      <td>0.457406</td>
      <td>1.051684</td>
      <td>8.744142</td>
      <td>1.000105</td>
      <td>1.334226</td>
      <td>36.932721</td>
      <td>0.467056</td>
      <td>0.962321</td>
      <td>52.466888</td>
      <td>0.350431</td>
      <td>0.930395</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>215.24094</td>
      <td>137.166799</td>
      <td>223.626941</td>
      <td>0.060563</td>
      <td>0.034060</td>
      <td>0.062259</td>
      <td>0.256130</td>
      <td>0.144508</td>
      <td>0.260508</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.307035</td>
      <td>0.325926</td>
      <td>0.327081</td>
      <td>10856.871769</td>
    </tr>
    <tr>
      <th>min</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>1.000000e+00</td>
      <td>-0.999454</td>
      <td>-0.999974</td>
      <td>1.230000e+02</td>
      <td>-0.999381</td>
      <td>-0.996364</td>
      <td>2.666667e-04</td>
      <td>-0.999640</td>
      <td>-0.999986</td>
      <td>1.500000e-03</td>
      <td>-0.999947</td>
      <td>-0.999612</td>
      <td>1.000000</td>
      <td>-0.900000</td>
      <td>-0.967742</td>
      <td>1.000000</td>
      <td>-0.968750</td>
      <td>-0.969697</td>
      <td>1.000000</td>
      <td>-0.937500</td>
      <td>-0.969697</td>
      <td>1.000000</td>
      <td>-0.937500</td>
      <td>-0.982759</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00000</td>
      <td>-9536.000000</td>
      <td>-9961.000000</td>
      <td>0.500000</td>
      <td>-0.902985</td>
      <td>-1.158557</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-2.000000</td>
      <td>-1.666667</td>
      <td>10580.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>3759.000000</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>1.310000e+05</td>
      <td>-0.044776</td>
      <td>-0.095863</td>
      <td>1.424000e+05</td>
      <td>-0.049774</td>
      <td>-0.081622</td>
      <td>8.241040e+01</td>
      <td>-0.036262</td>
      <td>-0.074294</td>
      <td>9.030120e+01</td>
      <td>-0.038677</td>
      <td>-0.060868</td>
      <td>2.000000</td>
      <td>-0.142857</td>
      <td>-0.250000</td>
      <td>1.000000</td>
      <td>-0.333333</td>
      <td>-0.320000</td>
      <td>4.000000</td>
      <td>-0.168224</td>
      <td>-0.250000</td>
      <td>5.000000</td>
      <td>-0.111111</td>
      <td>-0.310345</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.00000</td>
      <td>-9.500000</td>
      <td>-41.000000</td>
      <td>0.941320</td>
      <td>-0.007863</td>
      <td>-0.017047</td>
      <td>0.000000</td>
      <td>-0.020202</td>
      <td>-0.052632</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-0.111888</td>
      <td>-0.079365</td>
      <td>24100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>4213.000000</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>2.275000e+05</td>
      <td>0.000000</td>
      <td>0.053459</td>
      <td>2.490000e+05</td>
      <td>0.000000</td>
      <td>0.044413</td>
      <td>1.354059e+02</td>
      <td>0.000000</td>
      <td>0.051813</td>
      <td>1.460769e+02</td>
      <td>0.000000</td>
      <td>0.044410</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>-0.076923</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.00000</td>
      <td>0.000000</td>
      <td>-6.000000</td>
      <td>0.968767</td>
      <td>0.000000</td>
      <td>0.004262</td>
      <td>0.138889</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35614.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>4853.000000</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>4.600000e+05</td>
      <td>0.055650</td>
      <td>0.226415</td>
      <td>4.999000e+05</td>
      <td>0.058881</td>
      <td>0.188413</td>
      <td>2.661241e+02</td>
      <td>0.047213</td>
      <td>0.199768</td>
      <td>2.911184e+02</td>
      <td>0.047801</td>
      <td>0.165407</td>
      <td>22.000000</td>
      <td>0.166667</td>
      <td>0.400000</td>
      <td>8.000000</td>
      <td>0.500000</td>
      <td>0.625000</td>
      <td>32.000000</td>
      <td>0.200000</td>
      <td>0.322581</td>
      <td>38.000000</td>
      <td>0.100000</td>
      <td>0.187500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>146.50000</td>
      <td>9.000000</td>
      <td>22.000000</td>
      <td>0.994012</td>
      <td>0.008618</td>
      <td>0.027534</td>
      <td>0.333333</td>
      <td>0.025000</td>
      <td>0.119048</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.400000</td>
      <td>0.111111</td>
      <td>0.160714</td>
      <td>39100.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.0</td>
      <td>2.0</td>
      <td>41778.000000</td>
      <td>NaN</td>
      <td>13.000000</td>
      <td>6.500000e+07</td>
      <td>3179.500000</td>
      <td>6379.000000</td>
      <td>5.850000e+07</td>
      <td>1733.347609</td>
      <td>1674.806452</td>
      <td>1.450000e+06</td>
      <td>11316.924528</td>
      <td>18459.106814</td>
      <td>1.600000e+06</td>
      <td>1723.068340</td>
      <td>65209.916515</td>
      <td>322.000000</td>
      <td>28.000000</td>
      <td>57.000000</td>
      <td>117.000000</td>
      <td>30.000000</td>
      <td>87.000000</td>
      <td>527.000000</td>
      <td>26.000000</td>
      <td>111.000000</td>
      <td>783.000000</td>
      <td>39.000000</td>
      <td>100.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10647.00000</td>
      <td>9953.000000</td>
      <td>10062.000000</td>
      <td>1.979950</td>
      <td>0.833108</td>
      <td>1.009360</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>1.666667</td>
      <td>1.250000</td>
      <td>48060.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5b9a7bfc-ed1a-4c34-b4bf-58edeaf6ac2a')"
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
        document.querySelector('#df-5b9a7bfc-ed1a-4c34-b4bf-58edeaf6ac2a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5b9a7bfc-ed1a-4c34-b4bf-58edeaf6ac2a');
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


<div id="df-9457086f-e398-4b3b-ae27-9ba17f21f7d6">
  <button class="colab-df-quickchart" onclick="quickchart('df-9457086f-e398-4b3b-ae27-9ba17f21f7d6')"
            title="Suggest charts"
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
        document.querySelector('#df-9457086f-e398-4b3b-ae27-9ba17f21f7d6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
geo_data_url = 'https://raw.githubusercontent.com/scpike/us-state-county-zip/master/geo-data.csv'
df_geo = pd.read_csv(geo_data_url)
print('Num of rows:' , len(df_geo))
print('Num of columns:' , len(df_geo.columns))
df_geo.head()
```

    Num of rows: 33103
    Num of columns: 6
    





  <div id="df-5d00a867-c739-4558-a62b-f7c6b70ff338" class="colab-df-container">
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
      <th>state_fips</th>
      <th>state</th>
      <th>state_abbr</th>
      <th>zipcode</th>
      <th>county</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alabama</td>
      <td>AL</td>
      <td>35004</td>
      <td>St. Clair</td>
      <td>Acmar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Alabama</td>
      <td>AL</td>
      <td>35005</td>
      <td>Jefferson</td>
      <td>Adamsville</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Alabama</td>
      <td>AL</td>
      <td>35006</td>
      <td>Jefferson</td>
      <td>Adger</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Alabama</td>
      <td>AL</td>
      <td>35007</td>
      <td>Shelby</td>
      <td>Keystone</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Alabama</td>
      <td>AL</td>
      <td>35010</td>
      <td>Tallapoosa</td>
      <td>New site</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5d00a867-c739-4558-a62b-f7c6b70ff338')"
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
        document.querySelector('#df-5d00a867-c739-4558-a62b-f7c6b70ff338 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5d00a867-c739-4558-a62b-f7c6b70ff338');
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


<div id="df-ee9f51a6-794d-41da-9d6b-705bbac207c5">
  <button class="colab-df-quickchart" onclick="quickchart('df-ee9f51a6-794d-41da-9d6b-705bbac207c5')"
            title="Suggest charts"
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
        document.querySelector('#df-ee9f51a6-794d-41da-9d6b-705bbac207c5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
from IPython.display import Image, display
display(Image('https://www3.erie.gov/dhses/sites/www3.erie.gov.dhses/files/2022-06/psap_map.jpg'))
```


    
![jpeg](/img/posts/Redfin-Housing-Sales/output_9_0.jpg)
    



```python
# filter on erie county
df_geo_county = df_geo.loc[(df_geo['state_abbr'] == 'NY') &
                           (df_geo['county'].isin(['Erie']))]

# create function to check if zip code is valid
df_geo_county['valid_zip_code'] = df_geo_county.apply(lambda x: x['zipcode'].isnumeric(), axis =1)
print('Dataset:')
print(df_geo_county.groupby(['valid_zip_code', 'county'])['zipcode'].count())
print(' ')
print('Invalid zip codes:', df_geo_county.loc[df_geo_county['valid_zip_code'] == False]['zipcode'].unique())
```

    Dataset:
    valid_zip_code  county
    False           Erie       1
    True            Erie      61
    Name: zipcode, dtype: int64
     
    Invalid zip codes: ['142HH']
    


```python
# filter on only valid zip codes
df_geo_county_valid = df_geo_county.loc[df_geo_county['valid_zip_code'] == True]
print('Num of valid zip codes:', len(df_geo_county_valid))
df_geo_county_valid.head(1)
```

    Num of valid zip codes: 61
    





  <div id="df-a120db45-3cf2-4763-8b3e-561eb9a491d2" class="colab-df-container">
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
      <th>state_fips</th>
      <th>state</th>
      <th>state_abbr</th>
      <th>zipcode</th>
      <th>county</th>
      <th>city</th>
      <th>valid_zip_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20220</th>
      <td>36</td>
      <td>New york</td>
      <td>NY</td>
      <td>14001</td>
      <td>Erie</td>
      <td>Akron</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a120db45-3cf2-4763-8b3e-561eb9a491d2')"
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
        document.querySelector('#df-a120db45-3cf2-4763-8b3e-561eb9a491d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a120db45-3cf2-4763-8b3e-561eb9a491d2');
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
  </div>





```python
# get zip code from region field
df_filter['zipcode'] = df_filter.apply(lambda x: x['region'].split(':')[1].strip(), axis=1)
# merge market data & geo zip codes
df_merge = pd.merge(
    df_filter,
    df_geo_county_valid[['zipcode', 'city', 'county']],
    how = 'inner', # only return zipcodes in both tables
    on = ['zipcode'] # column to join on
)

print('Num of rows:', len(df_merge))
df_merge.head()
```

    Num of rows: 25811
    





  <div id="df-741dfa10-54ae-4aa2-9350-c3bc3e6de0c7" class="colab-df-container">
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
      <th>period_begin</th>
      <th>period_end</th>
      <th>period_duration</th>
      <th>region_type</th>
      <th>region_type_id</th>
      <th>table_id</th>
      <th>is_seasonally_adjusted</th>
      <th>region</th>
      <th>city_x</th>
      <th>state</th>
      <th>state_code</th>
      <th>property_type</th>
      <th>property_type_id</th>
      <th>median_sale_price</th>
      <th>median_sale_price_mom</th>
      <th>median_sale_price_yoy</th>
      <th>median_list_price</th>
      <th>median_list_price_mom</th>
      <th>median_list_price_yoy</th>
      <th>median_ppsf</th>
      <th>median_ppsf_mom</th>
      <th>median_ppsf_yoy</th>
      <th>median_list_ppsf</th>
      <th>median_list_ppsf_mom</th>
      <th>median_list_ppsf_yoy</th>
      <th>homes_sold</th>
      <th>homes_sold_mom</th>
      <th>homes_sold_yoy</th>
      <th>pending_sales</th>
      <th>pending_sales_mom</th>
      <th>pending_sales_yoy</th>
      <th>new_listings</th>
      <th>new_listings_mom</th>
      <th>new_listings_yoy</th>
      <th>inventory</th>
      <th>inventory_mom</th>
      <th>inventory_yoy</th>
      <th>months_of_supply</th>
      <th>months_of_supply_mom</th>
      <th>months_of_supply_yoy</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>avg_sale_to_list</th>
      <th>avg_sale_to_list_mom</th>
      <th>avg_sale_to_list_yoy</th>
      <th>sold_above_list</th>
      <th>sold_above_list_mom</th>
      <th>sold_above_list_yoy</th>
      <th>price_drops</th>
      <th>price_drops_mom</th>
      <th>price_drops_yoy</th>
      <th>off_market_in_two_weeks</th>
      <th>off_market_in_two_weeks_mom</th>
      <th>off_market_in_two_weeks_yoy</th>
      <th>parent_metro_region</th>
      <th>parent_metro_region_metro_code</th>
      <th>last_updated</th>
      <th>zipcode</th>
      <th>city_y</th>
      <th>county</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-06-01</td>
      <td>2021-08-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>All Residential</td>
      <td>-1</td>
      <td>220000.0</td>
      <td>0.000000</td>
      <td>0.157895</td>
      <td>182450.0</td>
      <td>-0.039484</td>
      <td>0.035471</td>
      <td>145.823035</td>
      <td>-0.038151</td>
      <td>0.200821</td>
      <td>136.376569</td>
      <td>0.007440</td>
      <td>0.101305</td>
      <td>73.0</td>
      <td>-0.026667</td>
      <td>0.280702</td>
      <td>38.0</td>
      <td>0.407407</td>
      <td>0.151515</td>
      <td>115.0</td>
      <td>0.116505</td>
      <td>0.064815</td>
      <td>31.0</td>
      <td>-0.060606</td>
      <td>0.068966</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>-2.0</td>
      <td>1.116532</td>
      <td>-0.009422</td>
      <td>0.107955</td>
      <td>0.794521</td>
      <td>-0.032146</td>
      <td>0.320836</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.684211</td>
      <td>-0.130604</td>
      <td>0.078150</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>2023-03-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>223900.0</td>
      <td>-0.013656</td>
      <td>0.066190</td>
      <td>199900.0</td>
      <td>0.025391</td>
      <td>0.057672</td>
      <td>153.019024</td>
      <td>-0.036131</td>
      <td>0.012689</td>
      <td>152.173913</td>
      <td>0.021311</td>
      <td>0.021061</td>
      <td>37.0</td>
      <td>-0.119048</td>
      <td>-0.195652</td>
      <td>25.0</td>
      <td>1.777778</td>
      <td>1.500000</td>
      <td>45.0</td>
      <td>0.730769</td>
      <td>0.363636</td>
      <td>9.0</td>
      <td>-0.250000</td>
      <td>1.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>-4.5</td>
      <td>0.0</td>
      <td>1.018800</td>
      <td>-0.024009</td>
      <td>-0.070206</td>
      <td>0.378378</td>
      <td>-0.097812</td>
      <td>-0.317274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.840000</td>
      <td>-0.160000</td>
      <td>-0.160000</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-11-01</td>
      <td>2018-01-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>All Residential</td>
      <td>-1</td>
      <td>145000.0</td>
      <td>-0.003436</td>
      <td>0.132812</td>
      <td>139900.0</td>
      <td>-0.060759</td>
      <td>0.076572</td>
      <td>102.077364</td>
      <td>-0.030680</td>
      <td>0.092081</td>
      <td>99.483806</td>
      <td>0.038098</td>
      <td>0.078621</td>
      <td>77.0</td>
      <td>-0.114943</td>
      <td>0.184615</td>
      <td>15.0</td>
      <td>0.071429</td>
      <td>0.875000</td>
      <td>46.0</td>
      <td>-0.303030</td>
      <td>-0.080000</td>
      <td>31.0</td>
      <td>-0.031250</td>
      <td>-0.380000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>-25.0</td>
      <td>0.991330</td>
      <td>-0.000183</td>
      <td>0.022958</td>
      <td>0.376623</td>
      <td>-0.002687</td>
      <td>0.068931</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.333333</td>
      <td>0.047619</td>
      <td>0.208333</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-07-01</td>
      <td>2021-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Townhouse</td>
      <td>13</td>
      <td>220000.0</td>
      <td>-0.015660</td>
      <td>0.156677</td>
      <td>219900.0</td>
      <td>-0.011241</td>
      <td>0.024220</td>
      <td>148.648649</td>
      <td>0.264239</td>
      <td>0.250711</td>
      <td>151.043956</td>
      <td>-0.021620</td>
      <td>0.126584</td>
      <td>5.0</td>
      <td>1.500000</td>
      <td>-0.285714</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>-0.500000</td>
      <td>10.0</td>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>-22.5</td>
      <td>-1.0</td>
      <td>1.076753</td>
      <td>0.060382</td>
      <td>0.068323</td>
      <td>0.800000</td>
      <td>0.300000</td>
      <td>0.228571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500000</td>
      <td>-0.500000</td>
      <td>0.500000</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-07-01</td>
      <td>2019-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>NaN</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>151700.0</td>
      <td>0.000000</td>
      <td>0.039041</td>
      <td>144900.0</td>
      <td>-0.000690</td>
      <td>-0.033356</td>
      <td>119.293078</td>
      <td>-0.036400</td>
      <td>0.156225</td>
      <td>105.507893</td>
      <td>-0.068096</td>
      <td>-0.066909</td>
      <td>71.0</td>
      <td>0.290909</td>
      <td>0.126984</td>
      <td>14.0</td>
      <td>-0.363636</td>
      <td>-0.263158</td>
      <td>64.0</td>
      <td>-0.189873</td>
      <td>-0.228916</td>
      <td>25.0</td>
      <td>0.041667</td>
      <td>-0.305556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>-6.0</td>
      <td>1.024315</td>
      <td>0.000577</td>
      <td>0.010307</td>
      <td>0.619718</td>
      <td>-0.016645</td>
      <td>-0.015202</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.357143</td>
      <td>-0.097403</td>
      <td>-0.063910</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-741dfa10-54ae-4aa2-9350-c3bc3e6de0c7')"
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
        document.querySelector('#df-741dfa10-54ae-4aa2-9350-c3bc3e6de0c7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-741dfa10-54ae-4aa2-9350-c3bc3e6de0c7');
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


<div id="df-ed62be55-6280-4c20-9c2d-c23de010370b">
  <button class="colab-df-quickchart" onclick="quickchart('df-ed62be55-6280-4c20-9c2d-c23de010370b')"
            title="Suggest charts"
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
        document.querySelector('#df-ed62be55-6280-4c20-9c2d-c23de010370b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# choose a zip code & property type
df_merge.loc[(df_merge['zipcode'] == '14213') &
             (df_merge['property_type'] == 'Single Family Residential')]\
             .sort_values(by=['period_begin']).tail(2)\
             [['period_begin', 'median_dom', 'median_dom_mom', 'median_dom_yoy']]
```





  <div id="df-956e13fe-189a-42ab-b8c7-8af955a8678e" class="colab-df-container">
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
      <th>period_begin</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19944</th>
      <td>2023-06-01</td>
      <td>14.5</td>
      <td>-0.5</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>20047</th>
      <td>2023-07-01</td>
      <td>16.0</td>
      <td>1.5</td>
      <td>6.5</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-956e13fe-189a-42ab-b8c7-8af955a8678e')"
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
        document.querySelector('#df-956e13fe-189a-42ab-b8c7-8af955a8678e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-956e13fe-189a-42ab-b8c7-8af955a8678e');
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


<div id="df-08f465d2-60db-49f9-a868-b21276ba41e0">
  <button class="colab-df-quickchart" onclick="quickchart('df-08f465d2-60db-49f9-a868-b21276ba41e0')"
            title="Suggest charts"
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
        document.querySelector('#df-08f465d2-60db-49f9-a868-b21276ba41e0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# test to correct the error
df_test = df_merge.copy()
df_test['median_dom_mom_correction'] = df_test.sort_values(by=['period_begin'])['median_dom'].pct_change()
df_test.loc[(df_test['zipcode'] == '14213') &
            (df_test['property_type'] == 'Single Family Residential')] \
            .sort_values(by=['period_begin']).tail(2)\
            [['period_begin', 'median_dom', 'median_dom_mom', 'median_dom_yoy', 'median_dom_mom_correction']]
```





  <div id="df-f453c65f-e93c-4085-aeba-393d5ff427d7" class="colab-df-container">
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
      <th>period_begin</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>median_dom_mom_correction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19944</th>
      <td>2023-06-01</td>
      <td>14.5</td>
      <td>-0.5</td>
      <td>4.5</td>
      <td>-0.500000</td>
    </tr>
    <tr>
      <th>20047</th>
      <td>2023-07-01</td>
      <td>16.0</td>
      <td>1.5</td>
      <td>6.5</td>
      <td>0.333333</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f453c65f-e93c-4085-aeba-393d5ff427d7')"
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
        document.querySelector('#df-f453c65f-e93c-4085-aeba-393d5ff427d7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f453c65f-e93c-4085-aeba-393d5ff427d7');
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


<div id="df-00fe17df-4bb3-4a40-a5cf-617745744d07">
  <button class="colab-df-quickchart" onclick="quickchart('df-00fe17df-4bb3-4a40-a5cf-617745744d07')"
            title="Suggest charts"
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
        document.querySelector('#df-00fe17df-4bb3-4a40-a5cf-617745744d07 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# add features
df_features = df_merge.copy()
print('Max begin date:', df_features['period_begin'].max())
print('Max end date:', df_features['period_end'].max())
max_end_date = df_features['period_end'].max()

# flag the most recent date
df_features['latest_period'] = df_features.apply(
    lambda x: True if x['period_end'] == max_end_date else False, axis=1)

# remove Redfin city and keep Geo city
df_features = df_features.drop(columns=['city_x'])
df_features = df_features.rename(columns={'city_y': 'city'})

# fix median days on market
df_features['median_dom_mom'] = df_features.sort_values(by=['period_begin'])['median_dom'].pct_change()

df_features.head()
```

    Max begin date: 2023-07-01
    Max end date: 2023-09-30
    





  <div id="df-fb8af3f1-fe5e-45bb-8f3d-b5c973adef35" class="colab-df-container">
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
      <th>period_begin</th>
      <th>period_end</th>
      <th>period_duration</th>
      <th>region_type</th>
      <th>region_type_id</th>
      <th>table_id</th>
      <th>is_seasonally_adjusted</th>
      <th>region</th>
      <th>state</th>
      <th>state_code</th>
      <th>property_type</th>
      <th>property_type_id</th>
      <th>median_sale_price</th>
      <th>median_sale_price_mom</th>
      <th>median_sale_price_yoy</th>
      <th>median_list_price</th>
      <th>median_list_price_mom</th>
      <th>median_list_price_yoy</th>
      <th>median_ppsf</th>
      <th>median_ppsf_mom</th>
      <th>median_ppsf_yoy</th>
      <th>median_list_ppsf</th>
      <th>median_list_ppsf_mom</th>
      <th>median_list_ppsf_yoy</th>
      <th>homes_sold</th>
      <th>homes_sold_mom</th>
      <th>homes_sold_yoy</th>
      <th>pending_sales</th>
      <th>pending_sales_mom</th>
      <th>pending_sales_yoy</th>
      <th>new_listings</th>
      <th>new_listings_mom</th>
      <th>new_listings_yoy</th>
      <th>inventory</th>
      <th>inventory_mom</th>
      <th>inventory_yoy</th>
      <th>months_of_supply</th>
      <th>months_of_supply_mom</th>
      <th>months_of_supply_yoy</th>
      <th>median_dom</th>
      <th>median_dom_mom</th>
      <th>median_dom_yoy</th>
      <th>avg_sale_to_list</th>
      <th>avg_sale_to_list_mom</th>
      <th>avg_sale_to_list_yoy</th>
      <th>sold_above_list</th>
      <th>sold_above_list_mom</th>
      <th>sold_above_list_yoy</th>
      <th>price_drops</th>
      <th>price_drops_mom</th>
      <th>price_drops_yoy</th>
      <th>off_market_in_two_weeks</th>
      <th>off_market_in_two_weeks_mom</th>
      <th>off_market_in_two_weeks_yoy</th>
      <th>parent_metro_region</th>
      <th>parent_metro_region_metro_code</th>
      <th>last_updated</th>
      <th>zipcode</th>
      <th>city</th>
      <th>county</th>
      <th>latest_period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-06-01</td>
      <td>2021-08-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>New York</td>
      <td>NY</td>
      <td>All Residential</td>
      <td>-1</td>
      <td>220000.0</td>
      <td>0.000000</td>
      <td>0.157895</td>
      <td>182450.0</td>
      <td>-0.039484</td>
      <td>0.035471</td>
      <td>145.823035</td>
      <td>-0.038151</td>
      <td>0.200821</td>
      <td>136.376569</td>
      <td>0.007440</td>
      <td>0.101305</td>
      <td>73.0</td>
      <td>-0.026667</td>
      <td>0.280702</td>
      <td>38.0</td>
      <td>0.407407</td>
      <td>0.151515</td>
      <td>115.0</td>
      <td>0.116505</td>
      <td>0.064815</td>
      <td>31.0</td>
      <td>-0.060606</td>
      <td>0.068966</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>-0.280000</td>
      <td>-2.0</td>
      <td>1.116532</td>
      <td>-0.009422</td>
      <td>0.107955</td>
      <td>0.794521</td>
      <td>-0.032146</td>
      <td>0.320836</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.684211</td>
      <td>-0.130604</td>
      <td>0.078150</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>2023-03-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>223900.0</td>
      <td>-0.013656</td>
      <td>0.066190</td>
      <td>199900.0</td>
      <td>0.025391</td>
      <td>0.057672</td>
      <td>153.019024</td>
      <td>-0.036131</td>
      <td>0.012689</td>
      <td>152.173913</td>
      <td>0.021311</td>
      <td>0.021061</td>
      <td>37.0</td>
      <td>-0.119048</td>
      <td>-0.195652</td>
      <td>25.0</td>
      <td>1.777778</td>
      <td>1.500000</td>
      <td>45.0</td>
      <td>0.730769</td>
      <td>0.363636</td>
      <td>9.0</td>
      <td>-0.250000</td>
      <td>1.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>-0.541667</td>
      <td>0.0</td>
      <td>1.018800</td>
      <td>-0.024009</td>
      <td>-0.070206</td>
      <td>0.378378</td>
      <td>-0.097812</td>
      <td>-0.317274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.840000</td>
      <td>-0.160000</td>
      <td>-0.160000</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-11-01</td>
      <td>2018-01-31</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>New York</td>
      <td>NY</td>
      <td>All Residential</td>
      <td>-1</td>
      <td>145000.0</td>
      <td>-0.003436</td>
      <td>0.132812</td>
      <td>139900.0</td>
      <td>-0.060759</td>
      <td>0.076572</td>
      <td>102.077364</td>
      <td>-0.030680</td>
      <td>0.092081</td>
      <td>99.483806</td>
      <td>0.038098</td>
      <td>0.078621</td>
      <td>77.0</td>
      <td>-0.114943</td>
      <td>0.184615</td>
      <td>15.0</td>
      <td>0.071429</td>
      <td>0.875000</td>
      <td>46.0</td>
      <td>-0.303030</td>
      <td>-0.080000</td>
      <td>31.0</td>
      <td>-0.031250</td>
      <td>-0.380000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>-0.688889</td>
      <td>-25.0</td>
      <td>0.991330</td>
      <td>-0.000183</td>
      <td>0.022958</td>
      <td>0.376623</td>
      <td>-0.002687</td>
      <td>0.068931</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.333333</td>
      <td>0.047619</td>
      <td>0.208333</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-07-01</td>
      <td>2021-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>New York</td>
      <td>NY</td>
      <td>Townhouse</td>
      <td>13</td>
      <td>220000.0</td>
      <td>-0.015660</td>
      <td>0.156677</td>
      <td>219900.0</td>
      <td>-0.011241</td>
      <td>0.024220</td>
      <td>148.648649</td>
      <td>0.264239</td>
      <td>0.250711</td>
      <td>151.043956</td>
      <td>-0.021620</td>
      <td>0.126584</td>
      <td>5.0</td>
      <td>1.500000</td>
      <td>-0.285714</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>-0.500000</td>
      <td>10.0</td>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>-0.619048</td>
      <td>-1.0</td>
      <td>1.076753</td>
      <td>0.060382</td>
      <td>0.068323</td>
      <td>0.800000</td>
      <td>0.300000</td>
      <td>0.228571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500000</td>
      <td>-0.500000</td>
      <td>0.500000</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-07-01</td>
      <td>2019-09-30</td>
      <td>90</td>
      <td>zip code</td>
      <td>2</td>
      <td>5005</td>
      <td>f</td>
      <td>Zip Code: 14043</td>
      <td>New York</td>
      <td>NY</td>
      <td>Single Family Residential</td>
      <td>6</td>
      <td>151700.0</td>
      <td>0.000000</td>
      <td>0.039041</td>
      <td>144900.0</td>
      <td>-0.000690</td>
      <td>-0.033356</td>
      <td>119.293078</td>
      <td>-0.036400</td>
      <td>0.156225</td>
      <td>105.507893</td>
      <td>-0.068096</td>
      <td>-0.066909</td>
      <td>71.0</td>
      <td>0.290909</td>
      <td>0.126984</td>
      <td>14.0</td>
      <td>-0.363636</td>
      <td>-0.263158</td>
      <td>64.0</td>
      <td>-0.189873</td>
      <td>-0.228916</td>
      <td>25.0</td>
      <td>0.041667</td>
      <td>-0.305556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>-0.333333</td>
      <td>-6.0</td>
      <td>1.024315</td>
      <td>0.000577</td>
      <td>0.010307</td>
      <td>0.619718</td>
      <td>-0.016645</td>
      <td>-0.015202</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.357143</td>
      <td>-0.097403</td>
      <td>-0.063910</td>
      <td>Buffalo, NY</td>
      <td>15380</td>
      <td>2023-10-29 14:25:50</td>
      <td>14043</td>
      <td>Depew</td>
      <td>Erie</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fb8af3f1-fe5e-45bb-8f3d-b5c973adef35')"
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
        document.querySelector('#df-fb8af3f1-fe5e-45bb-8f3d-b5c973adef35 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fb8af3f1-fe5e-45bb-8f3d-b5c973adef35');
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


<div id="df-cf765127-ce9f-466a-8814-a4b66b019355">
  <button class="colab-df-quickchart" onclick="quickchart('df-cf765127-ce9f-466a-8814-a4b66b019355')"
            title="Suggest charts"
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
        document.querySelector('#df-cf765127-ce9f-466a-8814-a4b66b019355 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Export new data set
```


```python
date_str = datetime.today().strftime('%Y-%m')
print('Current year/month:', date_str)
```

    Current year/month: 2023-11
    


```python
from google.colab import files
```


```python
# download file
df_features.to_csv('zip_realtor_{0}_{1}.csv'.format(state_code, date_str), index=False)
files.download('zip_realtor_{0}_{1}.csv'.format(state_code, date_str))
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


### Tableau Screenshots
<a id='screenshots'></a>

##### No filters all of Erie County
![png](\img\posts\Redfin-Housing-Sales\RedfinTableau.png)

##### Filtered by the City of Buffalo
![png](\img\posts\Redfin-Housing-Sales\RedfinTableau2.png)


##### Filtered by Multi-Family in the City of Buffalo
![png](\img\posts\Redfin-Housing-Sales\RedfinTableau3.png)
