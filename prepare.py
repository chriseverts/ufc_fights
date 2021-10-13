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


def prepare_ufc():
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

    weight_class = df_clean.loc[:,'weight_class']
    weight_class_list = weight_class.tolist()
    genderList = []
    genderCount = [0,0]
    weight_class_numbers = []
    weight_class_dict = {"WomenStrawweight":0,"WomenFlyweight":1,"WomenBantamweight":2,"WomenFeatherweight":3,"Flyweight":4,"Bantamweight":5,"Featherweight":6,"Lightweight":7,"Welterweight":8,"Middleweight":9,"LightHeavyweight":10,"Heavyweight":11,"CatchWeight":12,"OpenWeight":13}
    
    #print(weight_class_gender)
    for weights in weight_class_list:
        if "Women" in weights:
                genderList.append("f")
                genderCount[0]+=1
        else: 
            genderList.append("m")
            genderCount[1]+=1
            weight_class_numbers.append(weight_class_dict[weights])

    genderValues = np.array(genderList)
    # integer encode
    label_encoder = LabelEncoder()
    gender_encoded = label_encoder.fit_transform(genderValues)
    
    
    #Adding gender and weight class into DF as numbers
    df_clean.insert(0, 'gender', gender_encoded)
   
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
    
          
    return df_clean

def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



