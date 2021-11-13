# import library
from flask import Flask, request
from flask_restful import Resource, Api

# Inisiasi object flask
app = Flask(__name__)

# inisiasi object flask_restful
api = Api(app)

# inisiasi variabel kosong bertipe dictionary
fitur = {} # variable global , dictionary = json

import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
 
# Membaca file iris.csv
iris = pd.read_csv('Iris.csv')

# menghilangkan kolom yang tidak penting
iris.drop('Id',axis=1,inplace=True)

# memisahkan atribut dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']
 
# membuat model Decision Tree
tree_model = DecisionTreeClassifier() 
 
# melakukan pelatihan model terhadap data
tree_model.fit(X, y)

# membuat class Resource
class MachineLearningResource(Resource):
    # metode post
    def post(self):
        fitur["sepal_length"] = request.form["sepal_length"]
        fitur["sepal_width"] = request.form["sepal_length"]
        fitur["petal_length"] = request.form["petal_length"]
        fitur["petal_width"] = request.form["petal_width"]

        # prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
        prediction = tree_model.predict([[fitur["sepal_length"], fitur["sepal_width"], fitur["petal_length"], fitur["petal_width"]]])
        response = {"prediction" : prediction[0]}
        return response

# setup resourcenya
api.add_resource(MachineLearningResource, "/api/predict", methods=["POST"])