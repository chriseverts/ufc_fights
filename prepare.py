import pandas as pd 
import numpy as np 

import acquire as a 

############# Functions for prepping the sales data ###############

# ~~~~~~~ Use the function prepare_sales_data() ~~~~~~~~~~~~~~~


def day_of_week(df):
    '''
    This function takes in a dataframe and returns the dataframe with a column called 
    'day_of_week' attached
    df must have time series index
    '''
    df['day_of_week'] = df.index.day_name()
    
    return df

#################

def add_month(df):
    '''
    This function takes in a dataframe with a time series index
    Returns the dataframe with a new column called 'month' attached
    '''
    df['month'] = df.index.month
    
    return df

#################

def create_sales_total(df):
    '''
    This function takes in the sales df and calculates a new column
    called 'sales_total' from item_price and sale_amount
    '''
    df['sales_total'] = df.item_price * df.sale_amount
    
    return df

#################

def make_datetime_index(df, col_name):
    '''
    This function takes in a dataframe 
    A column name of the column that is your date (as string)
    Performs basic to_datetime conversion and sets tha column as the index
    '''
    
    df[col_name] = pd.to_datetime(df[col_name])

    df = df.set_index(col_name)
    
    return df

 #################

def add_year(df):
    '''
    This function takes in a dataframe with a time series index
    Returns the dataframe with a new column called 'year' attached
    '''
    df['year'] = df.index.year
    
    return df

################# ~~~~~~~~~~~~~ Use this one ~~~~~~~~~~~~~ #################

def prepare_sales_data():
    '''
    This function acquires and prepares the sales dataframe 
    It uses the functions from the acquire module to acquire the data.
    It resets the index to the sale date (giving it a datetime index)
    Adds columns for day_of_week, month, and sales_total
    '''
    
    df = a.the_whole_shebang().reset_index()
    
    df = make_datetime_index(df, 'sale_date')
    
    df = day_of_week(df)
    
    df = add_month(df)
    
    df = create_sales_total(df)
    
    return df

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import pandas as pd

def prep_combine(df):
    df.sale_date = pd.to_datetime(df.sale_date)
    df = df.set_index('sale_date').sort_index()
    df['month'] = df.index.month
    df['day_of_week'] = df.index.day_name()
    df['sales_total'] = df.sale_amount * df.item_price
    return df

def prep_opsd(df):
    df = df.drop(columns=['Unnamed: 0'])
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date')
    df['month']=df.index.month
    df['year']=df.index.year
    df = df.fillna(0)
    return df

def prep_fitbit(df):
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date').sort_index()
    df['month'] = df.index.month
    df['day_of_week'] = df.index.day_name()
    return df

################# ~~~~~~~~~~~~~ Germany Data ~~~~~~~~~~~~~ #################

def prep_germany_data():
    '''
    This function preps the opsd germany data
    It uses the helper functions in prepare.py 
    Renames a column, makes the Date column the Datetime index
    Adds a column with the month and one with the year
    and fills the NaNs with 0
    '''
    df = a.get_germany_data()
    
    #rename wind+solar for ease of use
    df = df.rename(columns = {'Wind+Solar': 'wind_and_solar'})
    
    df = make_datetime_index(df, 'Date')
    
    df = add_month(df)
    
    df = add_year(df)
    
    # fill NaNs with 0s
    df = df.fillna(value = 0)
    
    return df