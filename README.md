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

| Column Name               | Description                                                            |
|---------------------------|------------------------------------------------------------------------|
| R_ and B_                 | prefix signifies red and blue corner fighter stats respectively
KD                          |is number of knockdowns
SIG_STR                     |no. of significant strikes 'landed of attempted'
SIG_STR_pct                 |significant strikes percentage
TOTAL_STR                   |total strikes 'landed of attempted'
TD                          |no. of takedowns
TD_pct                      |takedown percentages
SUB_ATT                     |no. of submission attempts
PASS                        |no. times the guard was passed?
HEAD                        |no. of significant strinks to the head 'landed of attempted'
BODY                        |no. of significant strikes to the body 'landed of attempted'
CLINCH                      |no. of significant strikes in the clinch 'landed of attempted'
GROUND                      |no. of significant strikes on the ground 'landed of attempted'
win_by                      | method of win
last_round                  | last round of the fight (ex. if it was a KO in 1st, then this will be 1)
last_round_time             | when the fight ended in the last round
Format                      | the format of the fight (3 rounds, 5 rounds etc.)
Referee                     | the name of the Ref
date                        | the date of the fight
location                    | the location in which the event took place
Fight_type                  | weight class and whether it's a title bout or not
Winner                      |the winner of the fight

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
Goal: Plan out the project I will be seeing what features drive the outcome of a UFC fight.

First, I will begin by bringing in my data and exploring features to assure that I want to continue with in the classification modeling


## Acquire 
Goal: Have UFC dataframe ready to prepare in first part of wrangle.py In this stage, I used a csv file that was downloaded from the kaggle database. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline.

## Prep 
Goal: Have UFC dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in our wrangle.py file as well. In this stage, I handled missing values by dropping any rows and columns with more than 50% missing data.

Duplicates were dropped 

---Nulls in square footage, lotsize, tax value, and tax amount were imputed with median. (after splitting)

---Nulls in calculatedbathnbr, full bath count, region id city, regionidzip, and censustractandblock were imputed with most frequent. (after splitting)

Any remaining nulls after these were dropped. I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test. Last, I scaled it on a StandardScaler scaler (I made sure to drop outliers first!) and also returned X_train, X_validate, and X_test scaled.

## Explore 
Goal: Visualize the data. Explore relationships. Use the visuals and statistics tests to help answer my questions. I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset.

I ran a few t-tests with the features in respect to winner to test for difference in means. Also did a few correlation tests for continuous variables.

----I found that square footage, bedroom count, and acres over 2 were all statistically significant. They are not independent to logerror. Square footage less then 1500 did not have an effect on logerror

## Modeling and Evaluation
----Goal: develop a regression model that performs better than the baseline.

----The models worked best with $/sqft, acres, cluster, and locations. Polynomial Regression performed the best, so I did a test on it.

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


----My clustering didn't help with my supervised model, however, I could not find the right combinations to make my model beat the baseline for predicting log error either.

 - Log error was different for properties depending on county, number of bedrooms, dollar per square foot, and acres.
 - I made clusters with tax value and square footage, longitude and latitude, and based on property features like age, dollar per sqft, and acreage. I also made one based on location (neighborhoods) which consisted of longitude, latitude, and acreage bins.
  - My best model was my quadratic model (2 degrees), but even though it surpassed the baseline on train and validate, it did not perform better on the test. The RMSE to beat was 0.160, but mine was 0.174. It did better on r^2 at only 0.003 though. 


# How to Recreate Project

 - You'll need to download the csv file from my repo
 - Have a copy of my prep, explore .py files. 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.
