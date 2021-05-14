# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import request
from flask import jsonify
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv
import math
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# due to memory restrictions concerning hobby accounts on Heroku we can use here a subset of rows for the hosted version
MAX_ROWS=45000
#MAX_ROWS=None

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

CORS(app)

print("Starting flask app")

#train = pd.read_csv('static/weather_train_data.csv', encoding = "ISO-8859-1", delimiter=',')





def prepare_dataset(df):
    print("prepare_dataset was called")
    result=df
    result['Humidity3pm']=result['Humidity3pm'].fillna(df['Humidity3pm'].mean())
    result['Pressure3pm']=result['Pressure3pm'].fillna(df['Pressure3pm'].mean())
    result['Cloud3pm']=result['Cloud3pm'].fillna(df['Cloud3pm'].mean())
    result['Sunshine']=result['Sunshine'].fillna(df['Sunshine'].mean())
    result['Rainfall']=result['Rainfall'].fillna(df['Rainfall'].mean())
    return result


def bin_v2(df, isTraining):
    print("bin_v2 was called")
    #result=df.copy(deep=True)
    result=df
    result["bin_humidity"]=result.apply(lambda x: "High" if x.Humidity3pm>60 else 'Low', axis=1)
    result["bin_pressure"]=result.apply(lambda x: "High" if x.Pressure3pm>1017 else ('Medium' if x.Pressure3pm>1011 else 'Low'), axis=1)
    result["bin_cloud"]=result.apply(lambda x: "High" if x.Cloud3pm>7 else ('Medium' if x.Cloud3pm>6 else 'Low'), axis=1)
    result["bin_sunshine"]=result.apply(lambda x: "High" if x.Sunshine>6 else 'Low', axis=1)
    result["bin_rainfall"]=result.apply(lambda x: "High" if x.Rainfall>1 else  ('Medium' if x.Rainfall>0.1 else 'Low'), axis=1)
    if (isTraining):
        return result[['RainTomorrow','Location','bin_humidity', 'bin_pressure', 'bin_cloud','bin_sunshine','bin_rainfall']]
    else:
        return result[['Location','bin_humidity', 'bin_pressure', 'bin_cloud','bin_sunshine','bin_rainfall']]


DATA_TRAIN = 'static/weather_train_data.csv'
DATA_TRAIN_Y = 'static/weather_train_label.csv'

print("reading training data")
df_train = pd.read_csv(DATA_TRAIN, encoding = "ISO-8859-1", nrows=MAX_ROWS, delimiter=',') #,delimiter=';'

print("reading training labels")
df_train_y = pd.read_csv(DATA_TRAIN_Y, encoding = "ISO-8859-1", nrows=MAX_ROWS,  header=None,delimiter=',')
#df_train_y = df_train_y.rename(columns = { 0: 'RainTomorrow'}, inplace = False)

df_train['RainTomorrow']=df_train_y
#df_train=pd.concat([df_train_y, df_train], axis=1)


# result is the prepared training data set
cleaned=prepare_dataset(df_train)
prepared=bin_v2(cleaned, True)


train_data=prepared[['Location', 'bin_humidity', 'bin_pressure', 'bin_cloud','bin_sunshine','bin_rainfall']]

train_labels=prepared['RainTomorrow']


def getModel():
    print ("getModel was called")
    le = LabelEncoder()

    # training data encoding
    encodedLocation=le.fit_transform(train_data['Location'])
    encodedHumidity=le.fit_transform(train_data['bin_humidity'])
    encodedPressure=le.fit_transform(train_data['bin_pressure'])
    encodedCloud=le.fit_transform(train_data['bin_cloud'])
    encodedSunshine=le.fit_transform(train_data['bin_sunshine'])
    encodedRainfall=le.fit_transform(train_data['bin_rainfall'])                                                 

    features=zip(encodedLocation, encodedHumidity,encodedPressure,encodedCloud, encodedSunshine, encodedRainfall)
    features = list(features)

    label=train_labels

    # DecisionTree
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(features, label)
   
    return model

model=getModel()


def predict(test_data):

    le = LabelEncoder()

    # test data encoding
    le.fit(train_data['Location'])
    testEncodedLocation=le.transform(test_data['Location'])
    
    le.fit(train_data['bin_humidity'])
    testEncodedHumidity=le.transform(test_data['bin_humidity'])

    le.fit(train_data['bin_pressure'])
    testEncodedPressure=le.transform(test_data['bin_pressure'])

    le.fit(train_data['bin_cloud'])
    testEncodedCloud=le.transform(test_data['bin_cloud'])

    le.fit(train_data['bin_sunshine'])
    testEncodedSunshine=le.transform(test_data['bin_sunshine'])

    le.fit(train_data['bin_rainfall'])   
    testEncodedRainfall=le.transform(test_data['bin_rainfall'])

    features_test=zip(testEncodedLocation, testEncodedHumidity,testEncodedPressure, testEncodedCloud, testEncodedSunshine, testEncodedRainfall)
    features_test = list(features_test)

    y_pred = model.predict(features_test)
    return y_pred



@app.route('/locations')
def getLocations():
    print("getLocations was called")
    result= train_data['Location'].unique().tolist()
    return jsonify(result)



@app.route('/weather')
def getWeather():
    paramLocation=request.args.get("location")
    paramRainfall=request.args.get('rainfall', type=int)
    paramHumidity=request.args.get('humidity', type=int)
    paramPressure=request.args.get("pressure", type=int)
    paramCloud=request.args.get("cloud", type=int)
    paramSunshine=request.args.get("sunshine", type=int)


    print("getWeather was called with params location: ", paramLocation, "rainfall: ",paramRainfall," humidity: ",paramHumidity, " pressure: ",paramPressure,"cloud: ",paramCloud,"sunshine: ",paramSunshine)


    params = {'Location': [paramLocation], 'Humidity3pm': [paramHumidity], 'Pressure3pm':[paramPressure],'Cloud3pm':[paramCloud],'Sunshine':[paramSunshine],'Rainfall':[paramRainfall]}
    paramsDf = pd.DataFrame(data=params)

    test_data=bin_v2(paramsDf, False)

    predicted=predict(test_data)
    #predicted=['Yes']
    return jsonify(
        rainNextDay=predicted[0],
        subset=MAX_ROWS
    )


