from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime #allows you to measure time and get time in python
from model.titanic import *


# Create a Blueprint for the Titanic API to structure your endpoints
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')

api = Api(titanic_api)

class TitanicAPI:
        def post(self):
            data = request.get_json() # Get passenger data
            name = data.get('name')
            pclass = data.get('pclass')
            sex = data.get('sex')
            age = data.get('age')
            sibsp = data.get('sibsp')
            parch  = data.get('parch')
            fare = data.fare('fare')
            embarked = data.get('embarked')
            alone = data.get('alone')
            passenger = pd.DataFrame({
                'name': [name],
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'sibsp': [sibsp], 
                'parch': [parch], 
                'fare': [fare], 
                'embarked': [embarked], 
                'alone': [alone]
            })
            return jsonify(passenger) #return the result
# api.add_resource(TitanicAPI, '/') #add resource