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
from .outcome import OutcomeRainToday

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

CORS(app)

train = pd.read_csv('static/weather_train_data.csv', encoding = "ISO-8859-1", delimiter=',')


# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    print(" the hello world entry point was called")
    return 'Hello World'


@app.route('/locations')
# ‘/’ URL is bound with hello_world() function.
def getLocations():
    result= train['Location'].unique().tolist()
    return jsonify(result)
    #b = a.tolist() # nested lists with same data, indices

@app.route('/wind_directions')
# ‘/’ URL is bound with hello_world() function.
def getWindDirections():
    trainCopy=train.copy(deep=True)
    result=trainCopy['WindGustDir'].fillna("No data")
    result= result.unique().tolist()
    return jsonify(result)


@app.route('/weather')
# ‘/’ URL is bound with hello_world() function.
def getWeather():
    print(" the weather entry point was called")
    #train = pd.read_csv('/home/michael/Wirtschaftsinformatik/SS2021/Data Mining/Assignment_5/data/weather_train_data.csv', encoding = "ISO-8859-1", delimiter=',')

    paramLocation=request.args.get("location")
    paramWindDir = request.args.get('windDirection')
    paramRain=request.args.get('rainToday', type=int)
    #print("temperature: "+temperature)


    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.


    #print wheather_encoded

    prep_train=train[["Location","WindGustDir", "Rainfall" , "RainToday"]]
    prep_train_sub = prep_train
    #prep_train_sub["Rainfall"].fillna(0)
    prep_train_sub["ActualRain"]=prep_train_sub.apply(lambda x: 0 if math.isnan(x.Rainfall)|(x.Rainfall<1) else 1, axis=1)


    unWindGustDir=prep_train_sub['WindGustDir'].fillna("empty")

    encodedLoc=le.fit_transform(prep_train_sub['Location'])
    encodedWindGustDir=le.fit_transform(unWindGustDir)
    encodedActualRain=le.fit_transform(prep_train_sub['ActualRain'])

    features=zip(encodedLoc,encodedWindGustDir,encodedActualRain)

    #unRainToday=prep_train["RainToday"].fillna("empty")
    label=prep_train_sub['RainToday'].fillna("No")
    features = list(features)


    ##################



    le.fit(prep_train_sub['Location'])
    encodedLocTest=le.transform([paramLocation])

    le.fit(unWindGustDir)
    encodedWindGustDirTest=le.transform([paramWindDir])

    le.fit(prep_train_sub['ActualRain'])
    encodedActualRainTest=le.transform([paramRain])

    features_test=zip(encodedLocTest,encodedWindGustDirTest,encodedActualRainTest)
    features_test = list(features_test)


    #################

    #Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(features,label)

    #y_pred = model.predict(features_test)

    #print(features_test)

    predicted= model.predict(features_test) # 0:Overcast, 2:Mild
    print(predicted)

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