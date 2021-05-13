import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


def feature_list(s):
    columns = ['Location', 'RainTomorrow'] #'Rainfall'
    for feature in FEATURES:
        #if feature != 'date':
        for N in range(1, 4):
            col_name = "{}_{}".format(feature, N)
            columns = columns+[col_name]
    return columns
    
    
def prepare_features(df):
    df = df.sort_values(by=['Location', 'Date'])
    
    df.set_index(['Location', 'Date'])
    
    #for feature in FEATURES:
    #    #if feature != 'date':
    #    for N in range(1, 4):
    #        col_name = "{}_{}".format(feature, N)
    #        df.loc[:,col_name] = df[feature].shift(N) 
    return df
    
def encode_values_dyn(df, df_train, typ):
    le = LabelEncoder()
    if typ == 'TRAIN':
        for idx in range(df.shape[1]):
                #if idx <> 0:
                df[:,idx] = le.fit_transform(df[:,idx])
    elif typ == 'TEST':
        for idx in range(df.shape[1]):
            le.fit(list(set(df_train.iloc[:,idx]))) # fit auf train     df_train[:,idx])
            df.iloc[:,idx] = le.transform(df.iloc[:,idx])  # transform auf test anwenden
            #le.fit(list(set(df_train[:,idx]))) # fit auf train     df_train[:,idx])
            #df[:,idx] = le.transform(df[:,idx])  # transform auf test anwenden
    return df    
            
        #print(idx)
    
def replace_empty(df):
    df.loc[pd.isnull(df['Sunshine']),'Sunshine'] = 0
    df.loc[pd.isnull(df['Cloud9am']) ,'Cloud9am'] = 0 #df[['Cloud9am']].fillna(value=0)
    df.loc[pd.isnull(df['Cloud3pm']),'Cloud3pm'] = 0
    df.loc[pd.isnull(df['WindGustDir']),'WindGustDir'] = 'N/A'
    df.loc[pd.isnull(df['WindDir9am']),'WindDir9am'] = 'N/A'
    df.loc[pd.isnull(df['WindDir3pm']),'WindDir3pm'] = 'N/A'
    df.loc[pd.isnull(df['Rainfall']),'Rainfall'] = 0
    df.loc[pd.isnull(df['RainToday']),'RainToday'] = "No"
    
    df=df.fillna(0)
    #df.loc[pd.isnull (df['RainToday']) ,'RainToday'] = df[['RainToday']].fillna(value='No')
    #df.loc[pd.isnull (df['Pressure9am']),'Pressure9am'] = df['Pressure9am'].mean().fillna(value=0)
    #df.loc[pd.isnull (df['Pressure3pm']),'Pressure3pm'] = df['Pressure3pm'].mean().fillna(value=0)
    
    
    #df.loc[df['MaxTemp'].isna(),'MaxTemp'] = df[['MaxTemp']].fillna(value=df[["Temp9am", "Temp3pm"]].max(axis=1))
    
    columns = df.dtypes[~ (df.dtypes == 'object')].reset_index()['index'].tolist()
    # iterate over the precip columns
    for precip_col in columns: 
        # create a boolean array of values representing nans
        missing_vals = pd.isnull(df[precip_col])
        df.loc[missing_vals, precip_col] = df[precip_col].fillna(value=0) 
    return df

def prepare_dataset_normalize(df, df_train):
    metrics = df_train.copy(deep=True)
    for precip_col in ['MinTemp', 'MaxTemp']: 
        #metrics = df_train.groupby('Location')[precip_col].describe()
        metrics2 = metrics.groupby('Location')[precip_col].describe()[['mean', 'std']].reset_index()
        df = df.merge(metrics2, left_on='Location', right_on='Location', suffixes=('', '_y'))#, how='left'
        #= df.loc[:precip_col]
        df[precip_col]  = df.apply(lambda x: ( x[precip_col] * x['mean_y'] ) / x['std_y'] , axis=1)
    #df = df.drop(columns=['std_y', 'mean_y'])
    return df

def prepare_dataset(df):
    #df.loc[:,'Date'] = pd.to_datetime(df['Date'],errors='coerce')#"%Y-%m-%d"
    #df = df.loc[df['Date'].map(lambda x: x.strftime("%Y")) > '2008'] #.count() dynamisch?
    print("prepare_dataset was called")
    df = replace_empty(df)
    df = private_prepare_dataset(df)
    return df

def postprepare_dataset(df, typ):
    if typ == 'TRAIN':
        #df = df.dropna(axis=0) #TODO
        df1 = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
        df = df.append(df1, ignore_index=True)
    
    columns = df.dtypes[~ (df.dtypes == 'object')].reset_index()['index'].tolist()
    # iterate over the precip columns
    for precip_col in columns: 
        # fill numeric
        missing_vals = pd.isnull(df[precip_col])
        df.loc[missing_vals, precip_col] = 0 
    
    columns = df.dtypes[(df.dtypes == 'object')].reset_index()['index'].tolist()
    for precip_col in columns:
        df.loc[pd.isnull(df[precip_col]),precip_col] = 'N/A'
    #df = df.fillna(value='0')    
    return df

def private_prepare_dataset(df):
    #df = df.loc[df['Location'] == 'Witchcliffe'] #filter
    
    #datatypes and cleaning                                       

    df.loc[:,'Temperature'] = np.nanmean(df[['MaxTemp', 'MinTemp']], axis=1)
    df.loc[:,'Temperature_gap'] = np.subtract(df['MaxTemp'], df['MinTemp'], axis=1).fillna(value= 0)
    df.loc[:,'Humidity9am'] = df['Humidity9am'].fillna(value=0) #todo
    df.loc[:,'Humidity'] = np.nanmean(df[['Humidity9am', 'Humidity3pm']], axis=1)
    df.loc[:,'Humidity'] = df['Humidity'].fillna(value=0)
    
    df.loc[:,'Humidity_gap'] = np.subtract(df['Humidity9am'], df['Humidity3pm'], axis=1).fillna(value= 0)
    
    df.loc[:,'Pressure'] = np.nanmean(df[['Pressure9am', 'Pressure3pm']], axis=1)
    df.loc[:,'Pressure_gap'] = np.subtract(df['Pressure9am'], df['Pressure3pm'], axis=1).fillna(value= 0)

    df.loc[df['WindGustDir'] == 'NWN','WindGustDir'] = 'NW'
    df.loc[df['WindGustDir'] == 'WNW','WindGustDir'] = 'NW'
    df.loc[df['WindGustDir'] == 'NNW','WindGustDir'] = 'NW'
    df.loc[df['WindGustDir'] == 'NNE','WindGustDir'] = 'NE'
    df.loc[df['WindGustDir'] == 'WSW','WindGustDir'] = 'SW'
    df.loc[df['WindGustDir'] == 'ESE','WindGustDir'] = 'SE'
    df.loc[df['WindGustDir'] == 'SSE','WindGustDir'] = 'SE'
    df.loc[df['WindGustDir'] == 'ENE','WindGustDir'] = 'NE'
    df.loc[df['WindGustDir'] == 'SSW','WindGustDir'] = 'SW'
    
    df.loc[df['WindDir9am'] == 'NWN','WindDir9am'] = 'NW'
    df.loc[df['WindDir9am'] == 'WNW','WindDir9am'] = 'NW'
    df.loc[df['WindDir9am'] == 'WSW','WindDir9am'] = 'SW'
    df.loc[df['WindDir9am'] == 'ESE','WindDir9am'] = 'SE'
    df.loc[df['WindDir9am'] == 'ENE','WindDir9am'] = 'NE'
    df.loc[df['WindDir9am'] == 'NNE','WindDir9am'] = 'NE'
    df.loc[df['WindDir9am'] == 'NNW','WindDir9am'] = 'NW'
    df.loc[df['WindDir9am'] == 'SSE','WindDir9am'] = 'SE'
    df.loc[df['WindDir9am'] == 'SSW','WindDir9am'] = 'SW'
    
    df.loc[df['WindDir3pm'] == 'NNE','WindDir3pm'] = 'NE'
    df.loc[df['WindDir3pm'] == 'NNW','WindDir3pm'] = 'NW'
    df.loc[df['WindDir3pm'] == 'NWN','WindDir3pm'] = 'NW'
    df.loc[df['WindDir3pm'] == 'WNW','WindDir3pm'] = 'NW'
    df.loc[df['WindDir3pm'] == 'WSW','WindDir3pm'] = 'SW'
    df.loc[df['WindDir3pm'] == 'ESE','WindDir3pm'] = 'SE'
    df.loc[df['WindDir3pm'] == 'SSE','WindDir3pm'] = 'SE'
    df.loc[df['WindDir3pm'] == 'ENE','WindDir3pm'] = 'NE'
    df.loc[df['WindDir3pm'] == 'SSW','WindDir3pm'] = 'SW'
    
    return df


def binning(df):
    columns = df.dtypes[~ (df.dtypes == 'object')].reset_index()['index'].tolist()
    #Humidity3pm_1: (53, 60) 
    #Temperature_gap_1:  (8, 10.5)
    #Pressure3pm_1: (1011, 1017)
    #WindGuestSpeed_1: (35, 44)
    #Rainfall_1: (0.5, 4)
    cut_labels_3 = ["0","1","2"]
    cut_labels_2 = [0,1]
    #labels=cut_labels_3
    df.loc[:,'Humidity3pm'] = pd.cut(df['Humidity3pm'], bins=[-1, 53, 60,101])
    df.loc[:,'Temperature_gap'] = pd.cut(df['Temperature_gap'], bins=[-100,8, 11,100])#7.4
    df.loc[:,'Pressure3pm'] = pd.cut(df['Pressure3pm'], bins=[-1,1011, 1017,2000])
    df.loc[:,'WindGustSpeed'] = pd.cut(df['WindGustSpeed'], bins=[-1,35, 44,1000])
    df.loc[:,'Rainfall'] = pd.cut(df['Rainfall'], bins=[-1,0.5, 4, 1000])
    df.loc[:,'Humidity_gap'] = pd.cut(df['Humidity_gap'], bins=[-100, 8, 20,101])
    
    
    # iterate over the precip columns
    #for precip_col in columns: 
        # create a boolean array of values representing nans
        #missing_vals = pd.isnull(df[precip_col])
        #distance = 
     #   df.loc[:, precip_col] = pd.qcut(df[precip_col], q=1, precision=0)
    return df

def prd_model(df, array):
    #array = ['RainTomorrow', 'Location', 'Humidity3pm', 'Temperature_gap', 'Pressure3pm','WindGustDir']
    array = ['RainTomorrow',"Location", 'WindGustDir', "RainToday",'Humidity3pm']
    df = prepare_dataset(df)
    df = prepare_features(df)
    df_shape = df.copy(deep=True).head(0)
    df = binning(df)
    X_train = df[array]
    idx = [*range(1, X_train.shape[1], 1)]  #[*range(1, 7, 1)] #[0]+
    y_train = X_train.iloc[:, [0]].values 
    X_train = X_train.iloc[:, idx ].values
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42, stratify=X_train[:,0])
    X_train_org = X_train.copy()
    X_train = encode_values_dyn(X_train, X_train_org, 'TRAIN')
    
    y_train_label= np.array(y_train[:,0])
    X_train_features = np.array(X_train, dtype=int)
    #y_test_label = np.array(y_test[:,0])
    #X_test_features = np.array(X_test, dtype=int)

    model = GaussianNB()
    model.fit(X_train_features, y_train_label.ravel())

    return X_train_org, X_train, y_train, df_shape,model

def prd_eingabe(array, X_train_org, array_features, df_structure):
    #array = [['AliceSprings', "S", 'No',1018]]
    array_features = ["Location", "WindGustDir", "RainToday",'Humidity3pm']
    numpy_data = np.array(array)

    df = pd.DataFrame(data=numpy_data, columns=array_features)
    df['Humidity3pm'] = df['Humidity3pm'].astype(int)
    
    df = pd.merge(df, df_structure.head(0), how="outer")
    df = prepare_dataset(df)
    df = binning(df)
    df = df[array_features]
    df.info()
    #print(X_train_org)
    #df = encode_values_dyn(df, X_train_org,'TEST')
    le = LabelEncoder()
    for idx in range(df.shape[1]):
        #print(X_train_org[:,idx])
        #print(df.iloc[:,idx])
        le.fit(list(set(X_train_org[:,idx]))) # fit auf train     df_train[:,idx])
        df.iloc[:,idx] = le.transform(df.iloc[:,idx])  # transform auf test anwenden
       # print(df.iloc[:,idx])
    
    return df

    

def abgabe_test(df, X_train_org): 
    df = prepare_dataset(df_test)
    df = df.binning(df)
    
    df = encode_values_dyn(X_test, X_train_org,'TEST')
    