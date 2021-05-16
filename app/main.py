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
import pickle

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

CORS(app)

print("Starting flask app")


@app.route('/weather')
def getWeather():
    #paramLocation=request.args.get("location")
    paramHumidity=request.args.get('humidity', type=int)
    paramRainfall=request.args.get('rainfall', type=int)
    paramSunshine=request.args.get("sunshine", type=int)
    paramPressure=request.args.get("pressure", type=int)
    paramMinTemp=request.args.get("min_temp", type=int)
    paramWindGustSpeed=request.args.get("wind_gust_speed", type=int)

    #print("getWeather was called with params location: ", paramLocation, "rainfall: ",paramRainfall," humidity: ",paramHumidity, " pressure: ",paramPressure,"cloud: ",paramCloud,"sunshine: ",paramSunshine)
    print ('getWeather called with params Humidity3pm: ', paramHumidity,'Rainfall: ',paramRainfall, 'Sunshine: ',paramSunshine, 'Pressure3pm: ',paramPressure,'WindGustSpeed: ',paramWindGustSpeed,'MinTemp: ',paramMinTemp)


    with open('static/pickle_model.pkl', 'rb') as file:
        model = pickle.load(file)

    #['Humidity3pm', 'Rainfall', 'Sunshine', 'Pressure3pm', 'WindGustSpeed', 'MinTemp']]
    params = {'Humidity3pm': [paramHumidity],'Rainfall':[paramRainfall], 'Sunshine':[paramSunshine], 'Pressure3pm':[paramPressure],'WindGustSpeed':[paramWindGustSpeed],'MinTemp':[paramMinTemp]}
    paramsDf = pd.DataFrame(data=params)


    predicted=model.predict(paramsDf)
    #predicted=['Yes']
    return jsonify(
        rainNextDay=predicted[0]
    )


