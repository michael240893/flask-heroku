# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import request
from flask import jsonify
import json
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn import metrics
import csv
import math
from flask_cors import CORS
from .preprocessing_utils import prd_model
from .preprocessing_utils import prd_eingabe


# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

CORS(app)

#train = pd.read_csv('static/weather_train_data.csv', encoding = "ISO-8859-1", delimiter=',')

DATA_TRAIN = 'static/weather_train_data.csv'
DATA_TRAIN_Y = 'static/weather_train_label.csv'

df_train = pd.read_csv(DATA_TRAIN, encoding = "ISO-8859-1", delimiter=',') #,delimiter=';'
df_train_y = pd.read_csv(DATA_TRAIN_Y, encoding = "ISO-8859-1",  header=None,delimiter=',')
df_train_y = df_train_y.rename(columns = { 0: 'RainTomorrow'}, inplace = False)

df_train=pd.concat([df_train_y, df_train], axis=1)


@app.route('/locations')
def getLocations():
    trainCopy=df_train.copy(deep=True)
    result= trainCopy['Location'].unique().tolist()
    return jsonify(result)

@app.route('/wind_directions')
def getWindDirections():
    trainCopy=df_train.copy(deep=True)
    result=trainCopy['WindGustDir'].fillna("No data")
    result= result.unique().tolist()
    return jsonify(result)


@app.route('/weather')
def getWeather():
    paramLocation=request.args.get("location")
    paramWindDirection = request.args.get('windDirection')
    paramRainToday=request.args.get('rainToday')
    paramPreassure=request.args.get("preassure", type=int)
    paramPreassure=80

    print("getWeather was called with params location: ", paramLocation, "windDirection: ",paramWindDirection," rainToday: ",paramRainToday)

    input_param_arr=[[paramLocation, paramWindDirection, paramRainToday, paramPreassure]]

    df = df_train.copy(deep=True)
    X_train_org, X_train, y_train, df_structure, model  = prd_model(df, [''])
    X_test = prd_eingabe(input_param_arr, X_train_org, ['array_features'], df_structure)
    predicted=model.predict(X_test)
    #predicted=['Yes']
    return jsonify(
        rainNextDay=predicted[0]
    )



    

   

  

@app.route('/dataframe')
# ‘/’ URL is bound with hello_world() function.
def dataframe():
    print(" the dataframe entry point was called")
    df = pd.read_csv('/home/michael/Wirtschaftsinformatik/SS2021/Data Mining/Assignment_2/data/company_data.csv')

    #jsonfiles = json.loads(df.to_json(orient='records'))
    return df.to_json(orient="records")


# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()