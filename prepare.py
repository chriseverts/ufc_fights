import numpy as np
import pandas as pd
from acquire import new_ufc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def prepare_ufc(df):
    ''' Prepare ufc data'''
    
    df = new_ufc()

    # impute median to these features
    imp_features = ['R_Weight_lbs', 'R_Height_cms', 'B_Height_cms', 'R_age', 'B_age', 'R_Reach_cms', 'B_Reach_cms']
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

    for feature in imp_features:
        imp_feature = imp_median.fit_transform(df[feature].values.reshape(-1,1))
        df[feature] = imp_feature
    # impute most frequent for these features
    imp_stance = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_R_stance = imp_stance.fit_transform(df['R_Stance'].values.reshape(-1,1))
    imp_B_stance = imp_stance.fit_transform(df['B_Stance'].values.reshape(-1,1))
    df['R_Stance'] = imp_R_stance
    df['B_Stance'] = imp_B_stance
    
    #Drop NAs
    df.dropna(how='any', inplace=True)
    
    #Drop Stances which are missing (as mean cannot be obtained from string value)
    df_clean = df.copy()
    df_clean = df_clean[df_clean['B_Stance'].notna()]
    df_clean = df_clean[df_clean['R_Stance'].notna()]
    count_row = df_clean.shape[0]  
     
    # Encode time

  
    """
Encoding stances
"Open Stance":0
"Orthodox":1
"Southpaw":2
"Switch":3
"Sideways":4
"""
    stances_dict ={"Open Stance":0,"Orthodox":1,"Southpaw":2,"Switch":3, "Sideways":4}
    b_stance = df_clean.loc[:,'B_Stance']
    r_stance= df_clean.loc[:,'R_Stance']
    b_stance_list = b_stance.tolist()
    r_stance_list = r_stance.tolist()
    b_stance_int_list = []
    r_stance_int_list = []
    counter = range(len(df_clean.index))
    for rows in counter:
        b_stance_int_list.append(stances_dict[b_stance_list[rows]])
        r_stance_int_list.append(stances_dict[r_stance_list[rows]])

    b_stanceValues = np.array(b_stance_int_list)
    r_stanceValues = np.array(r_stance_int_list)
    #Dropping previous 'Stance' columns
    df_clean.drop(['R_Stance','B_Stance'], axis=1, inplace = True)
    #Adding int stances into DF
    df_clean.insert(3, 'B_Stance', b_stanceValues)
    df_clean.insert(4, 'R_Stance', r_stanceValues)
      

    """
Encoding match results
Red win = 1, Red lose = 0 & draw = 2
Did not use label encoder, because it labels in Alphabetical order
"""
    match_results = df_clean.loc[:,'Winner']
    match_results_list = match_results.tolist()
    matchList = []
#print(match_results_list)
    for results in match_results_list:
        if "Blue" in results:
            matchList.append("0")
        elif "Red" in results: 
            matchList.append("1")
        else:
            matchList.append("2")
        
    resultValues = np.array(matchList)
#Dropping previous 'Winner' column
    df_clean.drop(['Winner'], axis=1, inplace = True)
#Adding results into DF
    df_clean.insert(6, 'Winner', resultValues)

#Dropping Draw winners
    df_clean.drop(df_clean[df_clean['Winner'] == '2' ].index , inplace=True)
    df_clean ['Winner'] = df_clean.Winner.astype(float)

    """
Encoding Title Bout
True = 1, False = 0
Using label encoder
"""
    title = df_clean.loc[:,'title_bout']
    title_list = title.tolist()
    titleValues = np.array(title_list)
    label_encoder_title = LabelEncoder()
    title_encoded = label_encoder_title.fit_transform(titleValues)
    #Dropping previous 'Winner' column
    df_clean.drop(['title_bout'], axis=1, inplace = True)
    #Adding results into df_clean
    df_clean.insert(7, 'title_bout', title_encoded)
    # dropping irrevelant columns
    df_clean.drop(['B_fighter','date','location','Referee'], axis=1, inplace = True)
        
       
    # dropped all blue oppenents features

    df_clean = df_clean.drop(df_clean.columns[df_clean.columns.str.contains('^B')], axis=1)
    
    
    df = df_clean
          
    return df

def ufc_split(df):
    #splitting our data
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.Winner)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.Winner)
    return train, validate, test


def prep_ufc(df):
    #cleaning and splitting our data
    df = prepare_ufc(df)
    train, validate, test = ufc_split(df)
    return train, validate, test


