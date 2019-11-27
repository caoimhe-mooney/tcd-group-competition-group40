import numpy as np
import pandas as pd
import math as maths
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
import os
from datetime import datetime


# ## Visual Exploration

for variable in ['Age','Size_City','Height','Crime','Work_Exp','Additional_Income']:
    print(variable)
    fig, ax = plt.subplots(figsize=(10,6))
    sample = df_train.sample(n=50000)
    if variable=='Additional_Income': 
        sample[variable] = sample[variable].apply(lambda x: maths.log(x) if x>0 else x)
    plt.scatter(x=sample[variable],y=sample['Income'])
    plt.show()

inc = df.Income.sort_values().reset_index(drop=True)
inc = inc.apply(lambda x: maths.log(x))
fig = plt.subplots(figsize=(10,6))
plt.axhline(7, linestyle='--', color='red')
plt.axhline(13, linestyle='--', color='red')
plt.scatter(x=inc.index, y=inc, color='black')


# ## Data Pre-Processing

def load(which, sample=1):
    if which=='train':
        path = '/Users/Christopher/Desktop/HPC/ML/kaggle group/tcd-ml-1920-group-income-train.csv'
    elif which=='test':
        path = '/Users/Christopher/Desktop/HPC/ML/kaggle group/tcd-ml-1920-group-income-test.csv'
    else:
        print('Data source not recognised.')
        return
    df = pd.read_csv(path, index_col='Instance')
    df.columns = ['Year_of_Record', 'Housing','Crime','Work_Exp', 'Satisfaction','Gender', 'Age','Country',
                    'Size_City', 'Profession','University_Degree', 'Wears_Glasses', 'Hair', 'Height',
                    'Additional_Income','Income']
    df = df.sample(frac=sample)
    return df


def clean_cols_drop(df):
    df.Work_Exp = df.Work_Exp.replace({'#NUM!':np.nan})
    df.Work_Exp = df.Work_Exp.astype(float)
    for col in df.columns:
        if type(df[col].iloc[0]) == str or col == 'Housing':
            df[col] = df[col].replace({'f':'female', 0:'other', '0':'other', '#NUM!':'other', 'unknown':'other',
                                       'nA':'other'})
    df.Additional_Income = df.Additional_Income.apply(lambda x: float(x.split()[0]))
    df.Year_of_Record = preprocessing.scale(df.Year_of_Record)
    df = df.dropna(axis=0)
    df = df.drop_duplicates(subset=['Income','Size_City','Crime'])
    return df


def clean_cols_impute(df):
    df.Work_Exp = df.Work_Exp.replace({'#NUM!':df['Work_Exp'].replace({'#NUM!':10}).median()})
    df.Work_Exp = df.Work_Exp.astype(float)
    for col in df.columns:
        if col in ['Housing','Gender','Satisfaction','Profession','Country','University_Degree','Hair']:
            df[col] = df[col].replace({'f':'female', 0:np.nan, '0':np.nan, '#NUM!':np.nan, 'unknown':np.nan,
                                       'nA':np.nan})
    df.Additional_Income = df.Additional_Income.apply(lambda x: float(x.split()[0]))
    
    nan_dict = {'Year_of_Record':1996,'University_Degree':'No','Satisfaction':'Happy','Gender':'male',
            'Profession':'other','Hair':'Blue','Country':'other','Housing':'Other housing'}
    for col in nan_dict.keys():
        df[col] = df[col].fillna(nan_dict[col])
    
    return df
    

def create_features(df, squares, cubes):
    for col in squares:
        df['{}_sqr'.format(col)] = df[col].apply(lambda x: x**2)
    for col in cubes:
        df['{}_cube'.format(col)] = df[col].apply(lambda x: x**3)
    
    df['No_add_inc'] = 0
    df.loc[(df.Additional_Income==0), 'No_add_inc'] = 1
    df['Small_city'] = 1
    df.loc[(df.Size_City>3000), 'Small_city'] = 0
    
    return df


def encoding(df, target, one_hot, train, smoothing_weight=0):

    df['Profession'] = df['Profession'].str.lower()
    df['Profession'] = df['Profession'].apply(lambda x: x.replace(' ','') if type(x)==str else x)
    df['Profession'] = df['Profession'].apply(lambda x: x.replace('.','') if type(x)==str else x)

    temp = df.copy()
    temp['count'] = 1
    group = temp.groupby('Profession')['count'].sum().sort_values(ascending=True)
    other_profs = group[group<5].index.tolist()
    df['Profession'] = df['Profession'].replace(other_profs, 'other')
    group = temp.groupby('Country')['count'].sum().sort_values(ascending=True)
    other_countries = group[group<5].index.tolist()
    df['Country'] = df['Country'].replace(other_countries, 'other')

    hot_cols = []
    if train==1:
        encodings = {}
        for col in target:
            target_encodings = (df.groupby(col)['Income'].mean()*(1-smoothing_weight))+             (df.Income.mean()*smoothing_weight)
            df[col] = df[col].map(target_encodings)
            encodings[col] = target_encodings
            
        inflation = df.groupby('Year_of_Record')['Income'].mean()
        inflation = inflation / (inflation.iloc[0])
        df['Inflation'] = df['Year_of_Record'].map(inflation)
        encodings['Inflation'] = inflation
    for col in one_hot:
        hot_encodings = pd.get_dummies(df[col])
        hot_cols.extend(hot_encodings.columns.tolist())
        df = pd.concat([df, hot_encodings],axis=1)     
    if train==1:
        return df, encodings, hot_cols
    else:
        return df, hot_cols


def create_cat_con(df, cats, cons, normalize=True):
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
            new_col = cat +'_'+ con
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col]/df[cat+'_FE_FULL']
    return df


def preprocess(df, handle_missing, train):
    if handle_missing == 'drop': df = clean_cols_drop(df)
    elif handle_missing == 'impute': df = clean_cols_impute(df)
    
    squares = ['Age','Height','Work_Exp','Inflation','Additional_Income']
    cubes = ['Inflation']
    target = []
    one_hot = []
    remove = ['Hair','Wears_Glasses']
    cats = ['Year_of_Record','Gender', 'Country','Profession', 'University_Degree','Housing','Satisfaction']
    cons = ['Size_City']
    remove.extend(one_hot)
    remove.extend(cats)
        
    if train==1:
        df = create_cat_con(df, cats, cons)
        df, encodings, hot_cols = encoding(df, target, one_hot, train)
        df = create_features(df, squares, cubes)
        cols = list(df.columns)
        cols = [i for i in cols if i not in remove]
        df.to_csv(os.path.abspath('Desktop/HPC/ML/kaggle group/training_clean.csv'))
        return df[cols], encodings
    else:
        df = create_cat_con(df, cats, cons)
        df, hot_cols = encoding(df, target, one_hot, train)
        df = create_features(df, squares, cubes)
        df = create_cat_con(df, cats, cons)
        cols = list(df.columns)
        cols = [i for i in cols if i not in remove]
        return df[cols]


# ## Model Estimation & Evaluation

def train_test_model(model, df):
    X = df.drop('Income', axis=1)
    Y = df.Income
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random.randrange(10))
    
    model.fit(X, Y)
    Y_pred = model.predict(X_test)
    accuracy = metrics.mean_absolute_error(Y_test, Y_pred)
    print('MAE: ', round(accuracy))
    
    imp = pd.DataFrame(X.columns.tolist(),columns=['feature'])
    imp['importance'] = model.feature_importances_
    imp = imp.sort_values('importance',ascending=False)

    num = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(10,10))
    ax.barh(num, imp.importance, align='center')
    ax.set_yticks(num)
    ax.set_yticklabels(imp.feature)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.show()


df_train = load(which='train', sample=1)
df_train, encodings = preprocess(df_train, handle_missing='impute', train=1)
df_train = df_train[~df_train.duplicated()]

models = [
#    ('LinearRegression', LinearRegression()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(
        loss='ls', n_estimators=500, criterion='friedman_mse', max_depth=4, warm_start=True))
]

start = datetime.now()
for name, model in models:
    train_test_model(model, df_train)
end = datetime.now()
print('Time taken: ', end - start)


def preprocess_test(sample, encodings):
    df = load(which='test', sample=sample)
    df = df.drop('Income', axis=1)
    df = clean_cols_impute(df)
    
    squares = ['Age','Height','Work_Exp','Inflation','Additional_Income']
    cubes = ['Inflation']
    target = []
    one_hot = []
    remove = ['Hair','Wears_Glasses']
    cats = ['Year_of_Record','Gender', 'Country','Profession', 'University_Degree','Housing','Satisfaction']
    cons = ['Size_City']
    remove.extend(one_hot)
    #remove.extend(cats)

    # map target encodings from training set to the test set
    df['Inflation'] = df['Year_of_Record'].map(encodings['Inflation'])
    
    #df = create_cat_con(df, cats, cons)
    df, hot_cols = encoding(df, target, one_hot, train=0)
    df = create_features(df, squares, cubes)
    cols = list(df.columns)
    cols = [i for i in cols if i not in remove]
    '''
    for col in ['Profession' ,'Country']:
        df[col] = df[col].map(encodings[col])
    df = df.fillna(df_train.Income.mean())
    '''
    return df[cols]

X = preprocess_test(1, encodings)

# indicate which of our models is currently generating predictions
print(name)
print('Desired number of observations: ', 369438)
print('Shape of testing set: ', X.shape)

Y_pred = model.predict(X)
attempt = X.reset_index()[['Instance']].join(pd.DataFrame(Y_pred,columns=['Income']))
attempt.columns = ['Instance','Total Yearly Income [EUR]']
attempt = attempt.sort_values('Instance',ascending=True)
attempt.to_csv('/Users/Christopher/Desktop/HPC/ML/kaggle group/submission.csv',index=False)
