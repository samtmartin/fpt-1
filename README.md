# FPT-1 (Financial Projections Transformer 1)

![Michael Burry](/assets/burry_the_big_short.png?raw=true "Michael Burry in The Big Short")

## Overview

Financial Projections Transformer 1 is a batch learning, multivariate regression model that predicts future financial performance given historical performance. The model is 'sector agnostic' and therefore is able to accurately predict performance for companies in many different industries. Similar to the GPT class of models, FPT uses the decoder-only transformer architecture (i.e. causal or autoregressive attention).

The project is designed to be a proof of concept. The concept will be considered 'proven' when the model performs better than the baseline predictions. See the Performance section below for more detail.

## Data

The dataset used to train the model contains quarterly financial data (only ~5.1k rows) from ~60 companies across 10 business industries. The financial data consists of 33 coarse-grained, general business metrics (i.e. not data such as number of sales reps or revenue backlog). The dataset can also be modified to include up to four high-level macro environment metrics.

### Company-specific

The company-specific financial data is quarterly and contains a mix of Income Statement, Balance Sheet, Cash Flow Statement, and trading multiples. The company-specific metrics are:

* Revenue (IS)
* Gross profit (IS)
* Operating income (IS)
* Net income (IS)	
* Selling, general, and administrative expense (IS)
* Research and development expense (IS)
* Depreciation and amortization (IS)
* Other operating expense (IS)
* Interest expense (IS) 
* Interest income	(IS)
* Foreign currency exchange gain (IS)
* Other non-operating income (IS)
* Unusual expenses (IS)
* Tax expense (IS)
* Income from discontinued operations (IS)
* Extraordinary items (IS)
* Minority interest (IS)
* Total assets (BS)
* Total liabilities (BS)
* Total equity (BS)
* Cash and short term investments (BS)
* Receivables (BS)
* Current assets (BS)
* Net PP&E (BS)
* Current liabilities (BS)
* Total debt (BS)
* Cash from operations (CFS)
* Cash from investment (CFS)
* Cash from financing (CFS)
* Capital expenditures (CFS)
* Debt issued (CFS)
* Debt repaid (CFS)
* Total enterpise value over last twelve month revenue (trading multiple)

### Macro

<b>Current environment</b><br />
The metrics used to represent the current health of the macro environment are the monthly Personal Consumption Expenditures[1] and General Freight Trucking Price Index[2]. The Personal Consumption Expenditures (PCE) measures consumer spending on goods and services among households in the US. The PCE is used as a mechanism to gauge how much earned income of households is being spent on current consumption for various goods and services. The trucking price index measures the price of shipping general freight by truck over long distances. 

[1] https://fred.stlouisfed.org/series/PCE<br/>
[2] https://fred.stlouisfed.org/series/PCU4841214841212<br/>

<b>Forward-looking environment</b><br />
The current period metrics used to represent investor sentiment towards the near-term economic outlook are the 3-month[1] and 10-year[2] US treasury yields. Treasury yields are an imperfect metric for this use case, as rates are influenced by a variety of factors (e.g. geopolitical risk, monetary policy, etc). 

[1] https://fred.stlouisfed.org/series/DGS3MO<br/>
[2] https://fred.stlouisfed.org/series/DGS10<br/>

<b>Macro data availability</b><br />
Public companies seem to report quarterly results ~30 days after the period ends (e.g. the financial data for the period ended December 31st is reported in early February). Monthly FRED (Federal Reserve Bank of St. Louis) economic data seems to be published 1-30 days after the period. As such, for a given fiscal quarter, every month in the quarter has the FRED data for each macro metric in feature set (i.e. for a quarter ended December 31st, there will be macro data for December (and earlier if lag param is set) for each metric in feature set). Note that the csv keys in macro files have 01 in the day position (e.g. 2023-01-01), but it appears that the value may be representative of the end of the month (e.g. 2023-01-31).

## Data structure

The structure of the data varies significantly from having smooth trends and clear periodicity (left) to having few/no discernible patterns or recurring cycles (right) as well as hybrids that have a mix of both. 

![Data Structure](/assets/data_structure.png?raw=true "Data Structure")

## Data preparation

The metrics, particularly the company-specific financial metrics, tend to have long tails (see below). `DataLoader` supports multiple scaling strategies, but replacing features with their log10 value tends to perform best.

![Data Distributions](/assets/data_distributions.png?raw=true "Data Distributions")

## Training

The configuration at head in `train.py` performed roughly the best.

Adam optimizer was selected because it's consistent with the influenza time series paper[1]. Huber loss was used because it's consistent with paper and also because it performed compareably to other common regresson task performance measures (e.g. RMSE and MAE).

![Train Val Loss](/assets/train_val_loss.png "Train Val Loss")

[1] https://arxiv.org/pdf/2001.08317.pdf<br />

## Performance

In order to assess the quality of the model's quarterly financial predictions, a baseline prediction is used. The baseline prediction applies the previous quarter's year-over-year growth rate to the current quarter's prior year value. This is a common method for forecasting business performance when there aren't fine-grained P&L drivers available. Median percent error (i.e. `abs(estimate - actual) / abs(actual) * 100.0`) is used to compare model predictions to the baseline predictions.

The 'proof of concept' mentioned above (i.e. that a transformer model can accurately predict business performance given only high-level financial data) is not currently proven. More work will be done to bring down percent error.

![Median Percent Error](/assets/median_percent_error.png "Median Percent Error")

## Where does the loss come from?

* Model architecture and related 'inefficiencies'
* Very limited dataset size
* No forward-looking, company-specific data (e.g. company guidance, consensus research estimates, etc)
* Only course-grained company-specific data. More detailed company information (e.g. sku-level metrics) would very likely improve the model's forecasting quality
* No company-specific 'structural' change signals (e.g. acquisitions, divestitures, etc)

## How to run

Reach out to me to get the `public_company_financials.csv` file. With that, you can instantiate `DataLoader` and run `train.py` and/or `forecast.py`.