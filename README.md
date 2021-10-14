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


## Project Summary
- Based on the features average submission attempts ,reach ,average significant strikes, 'average ground attempts ,average control time(seconds) there is a 64.5%   accuracy of predicting a winner. 

- My test accuracy of 64.5% did not beat my baseline of 65% 

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

- More submission attempts and ctrl time will effect the outcome
- The more rounds fought the more likely to win
- Reach and hieght will effect the outcome of a win
- Landing more sig shots will effect the outcome


## Findings and Next Steps 

- There is a significance in the number of submission attempts in determining the outcome of a fight

- There is no significance in the number of rounds fought with those 16 or  more rds fought

- There is no significance between reach and the winner

- There is significance between those with 50% or more significant strike percentage



Next steps would be:
 - gather more information on UFC fighers
 
 - Try out more combinations and also add in blue fighter stats

# The Pipeline

## Planning 
Goal: Plan out the project I will be seeing what features drive the outcome of a UFC fight.

First, I will begin by bringing in my data and exploring features to assure that I want to continue with in the classification modeling


## Acquire 
Goal: Have UFC dataframe ready to prepare in first part of wrangle.py In this stage, I used a csv file that was downloaded from the kaggle database. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline.

## Prep 
Goal: Have UFC dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in our prep.py file as well. In this stage, I handled missing values by dropping any rows and columns with missing data and also used imputation

Any remaining nulls after these were dropped. I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test. 

## Explore 
Goal: Visualize the data. Explore relationships. Use the visuals and statistics tests to help answer my questions. I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset.

I ran a few t-tests with the features in respect to winner to test for difference in means. 

---- I found that There is evidence to suggest there is a difference in a winner with more then one submission attempt vs no submission attempts

---- There is evidence to suggest there is a difference in a winner with more then 16 rounds fought vs less then 16 rounds fought

---- There is evidence to suggest there is a difference in a winner with a signifcant strike percentage over 50% vs less then 50%


## Modeling and Evaluation
----Goal: develop a classification model that performs better than the baseline.

----The models worked best with average submission attempts, reach, average significant strikes, average ground attempts, and average control time(seconds).

                      

 Baseline  .65                                    


| Model                            | Training      | Validate      | Accuracy  |
|----------------------------------|---------------|---------------|-----------|
| Decision Tree                    | 64.83%        | 64.56%        | 65%   
| Logistic Regression              | 64.59%        | 64.99%        | 65%
| RFM                              | 64.55%        | 64.56%        | 65%
| KNN                              | 65.39%        | 64.34%        | 65%

<br>

Test Data Result

| Model                            | Accuracy    
|----------------------------------|---------------|
| rfc	                             |    64.50
|	logistic_regression	             |    64.50
| svm	                             |    64.50
|	knn	                             |    64.14
|	naive_bayes	                     |    59.95
                          

## Delivery 
A final notebook walkthrough of the my findings will be given 
 - All acquire and prepare .py files are uploaded for easy replication.
 - This README 
 - Final notebook that documents a commented walkthrough of my process

# Conclusion 

- Based on the features average submission attempts ,reach ,average significant strikes, 'average ground attempts ,average control time(seconds) there is a 63%     accuracy of predicting a winner. 

- My test accuracy of 64.5% did not beat my baseline of 65% 

- The chart above, I ran my test data on multiple machine learning algo's to get a visual

- With more time I would like to try different combinations or maybe use all the features



# How to Recreate Project

 - You'll need to download the csv file from my repo
 - Have a copy of my prep, explore .py files. 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.
