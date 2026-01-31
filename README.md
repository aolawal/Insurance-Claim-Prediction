# Insurance Claim Prediction Using Building Characteristics

## Project Overview
Insurance companies face significant financial risk due to unexpected claims. This project aims to build a predictive machine learning model that estimates the probability of a building having at least one insurance claim during its insured period based on building characteristics.

The project follows a complete data science lifecycle, including data preprocessing, exploratory data analysis, modeling, and evaluation.

### Objective: 
The project aims to build a predictive model to estimate the probability that a building will have at least one insurance claim during the insured period based on characteristics such as Building Dimension,
Building Age, Date of Occupancy, Number of Windows, Year of Observation and other factors.

Target Variable:
Claim = 1 → At least one claim 
Claim = 0 → No claim

### Dataset overview
The dataset contains information about insured buildings, including:
Building characteristics
Location details
Insurance duration
Claim history

Files used:
In this project we will be working on 2 CSV files:
Train_data.csv – contains the datasets that we will use to build, train and test our model
Variable Description.csv – metadata explaining each variable (Gives more details about content each of the features / columns)

## Methodology
## Data Preprocessing

#### Data Cleaning

I dropped non-informative identifier: Customer Id
I Checked missing values using 'insurance_df.isnull().sum()'
I imputed the median value for the numerical features where we have null values
For Categorical features → I imputed most frequent value
I Corrected the data types where necessary

#### b) Feature Handling
I dropped some variables (features) that had little or no correlation with our target variable (Claim). While, I encoded the relevant Categorical variables such as Building Type and Residential using One-Hot Encoding

##### Numerical variables:
I scaled the model using StandardScaler to prevent features with large ranges dominating the model

#### Train-Test Split
I used Stratified split to preserve claim distribution 80% training / 20% testing

## Exploratory Data Analysis (EDA)

### Claim distribution (class imbalance)

Claim rate vs:

Residential vs non-residential

Building type

Settlement (Urban vs Rural)

Building size

Correlation heatmap for numerical features

### Tools & Technologies
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook
GitHub
Snipping Tool



