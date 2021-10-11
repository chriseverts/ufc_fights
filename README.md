# Drivers that effect the outcome of UFC fights

## Project Overview

In this project, I will be working with a ufc data set to discover which attributes effect the outcome of a UFC fight.


## Project Description

I will be using classification machine learning algorithms to discover the best model at predicting statistical signficance in the outcome of UFC fights. The data set I will be working with is downloadable from the kaggle servers. Specifically the years 1993 - 2019

## Goals

Deliver a Jupyter notebook going through the steps of the data science pipeline
Create a classification model
Discover features that contribute to wins/losses in the UFC
Present a notebook about my findings

## Deliverables
Finalized Jupyter notebook complete with comments

A README.md with executive summary, contents, data dictionary, conclusion and next steps, and how to recreate this project.

Here is a provided link to my [trello board]

## Project Summary
I incorporated clustering to discover keys drivers in logerror of zestimates using a Zillow data frame.

## Data Dictionary 

| Column Name               | Description                              |
|---------------------------|------------------------------------------|
| R_ and B_                 | prefix signifies red and blue corner fighter stats respectively
KD                          |is number of knockdowns
SIG_STR is                  |no. of significant strikes 'landed of attempted'
SIG_STR_pct is              |significant strikes percentage
TOTAL_STR is                |total strikes 'landed of attempted'
TD is                       |no. of takedowns
TD_pct is                   |takedown percentages
SUB_ATT is                  |no. of submission attempts
PASS is                     |no. times the guard was passed?
REV?
HEAD is no. of significant strinks to the head 'landed of attempted'
BODY is no. of significant strikes to the body 'landed of attempted'
CLINCH is no. of significant strikes in the clinch 'landed of attempted'
GROUND is no. of significant strikes on the ground 'landed of attempted'
win_by is method of win
last_round is last round of the fight (ex. if it was a KO in 1st, then this will be 1)
last_round_time is when the fight ended in the last round
Format is the format of the fight (3 rounds, 5 rounds etc.)
Referee is the name of the Ref
date is the date of the fight
location is the location in which the event took place
Fight_type is which weight class and whether it's a title bout or not
Winner is the winner of the fight

<br>

##  Hypothesis 

- Logerror is affected by squared feet over 1700 sq ft. 

- Logerror is affected by the number of bedrooms

- Logerror is affected by the number of acres

- Logerror is affected by location

- Logerror is affected by tax value per square feet

- Logerror is effected by a combintaion of house features and also location + land

## Findings and Next Steps 
 - There appears to be distinct groups between longitude/latitude, and dollar per sqft/number of bedrooms, and acreage 
 - Using recursive feature elimination, it selected dollar per sqft, number of bathrooms, county, census block, and calculatedbathnbr .


Next steps would be:
 - gather more information on location
 - Try out new combinations for clustering with these as well as other columns with other property features.


# The Pipeline

## Planning 
Goal: Plan out the project I will be seeing how square footage, bedroom count, longitude, latitude, acreage, age, and county relate to log error of Zestimates. I will try to cluster by location and by land features to see if it'll be helpful to a supervised regression model.

First, I will begin by bringing in my data and exploring features to assure that I want to continue with clustering these (and/or others), I can then turn it into a cluster column and use feature selection to see if the clustering helps.



Hypotheses: Square footage, beds, acreage and location will have an effect on the logerror


## Acquire 
Goal: Have Zillow dataframe ready to prepare in first part of wrangle.py In this stage, I used a connection URL to access the CodeUp database. Using a SQL query, I brought in the 2017 Zillow dataset with only properties set for single use, and joined them with other tables via parcelid to get all of their features. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline.

## Prep 
Goal: Have Zillow dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in our wrangle.py file as well. In this stage, I handled missing values by dropping any rows and columns with more than 50% missing data.

Duplicates were dropped (in parcelid)

Nulls in square footage, lotsize, tax value, and tax amount were imputed with median. (after splitting)

Nulls in calculatedbathnbr, full bath count, region id city, regionidzip, and censustractandblock were imputed with most frequent. (after splitting)

Any remaining nulls after these were dropped. I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test. Last, I scaled it on a StandardScaler scaler (I made sure to drop outliers first!) and also returned X_train, X_validate, and X_test scaled.

## Explore 
Goal: Visualize the data. Explore relationships, and make clusters. Use the visuals and statistics tests to help answer my questions. I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset.

I ran a few t-tests with the features in respect to log error to test for difference in means. Also did a few correlation tests for continuous variables.

I found that square footage, bedroom count, and acres over 2 were all statistically significant. They are not independent to logerror. Square footage less then 1500 did not have an effect on logerror

## Modeling and Evaluation
Goal: develop a regression model that performs better than the baseline.

The models worked best with $/sqft, acres, cluster, and locations. Polynomial Regression performed the best, so I did a test on it.

| Model                            | RMSE Training | RMSE Validate | R^2   |
|----------------------------------|---------------|---------------|-------|
| Baseline                         | 0.1718        | 0.1605        | 0.000 |
| OLS LinearRegression             | 0.1716        | 0.1602        | 0.003 |
| LassoLars                        | 0.1718        | 0.1604        | 0.000 |
| TweedieRegressor                 | 0.1717        | 0.1603        | 0.002 |
| PolynomialRegression (2 degrees) | 0.1715        | 0.1602        | 0.003 |
<br>

Test for OLS Linear Regression:
 - RMSE of 0.174
 - R^2 of 0.003



## Delivery 
A final notebook walkthrough of the my findings will be given 
 - All acquire and prepare .py files are uploaded for easy replication.
 - This README 
 - Final notebook that documents a commented walkthrough of my process

# Conclusion 


My clustering didn't help with my supervised model, however, I could not find the right combinations to make my model beat the baseline for predicting log error either.

 - Log error was different for properties depending on county, number of bedrooms, dollar per square foot, and acres.
 - I made clusters with tax value and square footage, longitude and latitude, and based on property features like age, dollar per sqft, and acreage. I also made one based on location (neighborhoods) which consisted of longitude, latitude, and acreage bins.
  - My best model was my quadratic model (2 degrees), but even though it surpassed the baseline on train and validate, it did not perform better on the test. The RMSE to beat was 0.160, but mine was 0.174. It did better on r^2 at only 0.003 though. 


# How to Recreate Project

 - You'll need your own username/pass/host credentials in order to use the get_connection function in my acquire.py to access the Zillow database
 - Have a copy of my acquire, prep, explore .py files. 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.
