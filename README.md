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

### Claim distribution 

#### Claim by Settlement (Urban vs Rural:
As per the below analysis, a higher proportion of the customers insured in rural area (R) filed a claim when compared to urban area(U)

![Settlement](https://github.com/user-attachments/assets/31ec2186-aca4-4db0-8fe7-61edec42f08b)

#### Residential vs non-residential
The analysis below shows that building that are not residential had higher proportion of claim compared to residential building

![Residential](https://github.com/user-attachments/assets/2e5b0a7a-42be-4850-b66c-069ac5f99239)

#### Fence
A not-fenced plot (V) has higher proportion of claim than fenced plot (V) which implies a not fenced plot owner is likely to make a claim than a fenced plot owner

![Fence](https://github.com/user-attachments/assets/0bfeb78f-c104-426b-91d4-716c26fb019d)

#### Building type
Building type 3 & 4 had the highest proportion of claim out of total insured, while building type 2 had the lowest proportion. This implies building type 3 & 4 are more likely to make claim than other type of building

![Building_Type](https://github.com/user-attachments/assets/4311e4d3-5de4-41bb-a6ee-0fa441d24a94)

### Class Imbalance
![Settlement_imb](https://github.com/user-attachments/assets/34f1e174-aed2-4daa-af18-f6bcf87663a1)

![Residential_imb](https://github.com/user-attachments/assets/e42ba820-8ea9-4c3d-9ec0-f0cfa49e1fe5)

![Fence_imb](https://github.com/user-attachments/assets/25cfca4b-565d-46a1-accb-004aace1aa1d)

### Correlation heatmap for numerical features

The correlation in the heatmap shows positive correlation between The Insurance claim and the following features:
Insured Period
Residential
Building Dimension
Building Type and
Date of Occupancy
While Building Age and Year of Observation have a negative correlation with the Insurance claim.

![Heatmap](https://github.com/user-attachments/assets/fa2ce11a-5aa1-4e8f-bf6d-338d988def9c)

## Feature Selection
From the Feature Selection analysis, the following are the 5 most important features that determine whether there would be an insurance claim on a building or not.

Building Dimension
Building Age
Date of Occupancy
Number of Windows
Year of Observation

Contrary to expectation, the Insured period and Building Type have little impact on potential claim by the insured

![Feature Importance](https://github.com/user-attachments/assets/7f8f7686-8a73-4db8-9867-a27f0cd92818)

## Modeling
o	Logistic Regression
o	Random Forest Classifier

## Model Evaluation
o	ROC-AUC score
o	Model comparison and interpretation

## Conclusion

This project successfully built a predictive model to estimate insurance claim probability using building characteristics. After extensive preprocessing and experimentation on the classification, the model achieved the best Accuracy score of 0.76. Future improvements could include advanced feature engineering and gradient boosting models.

The project demonstrates how structured data preprocessing and careful feature handling can significantly impact predictive performance. Random Forest Classifier proved to be a strong baseline model for insurance claim prediction. Future work could explore advanced ensemble models and additional feature engineering.


### Tools & Technologies
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook
GitHub
Snipping Tool



