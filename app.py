# import library
from flask import Flask, request
from flask_restful import Resource, Api
from jcopml.utils import load_model

# Inisiasi object flask
app = Flask(__name__)

# inisiasi object flask_restful
api = Api(app)

# load label encoder
lbl_encoder = load_model("model/lbl_encoder_nannomate.pkl")

# load machine learning model 
random_forest_model = load_model("model/random_forest_nannomate.pkl")

# inisiasi variabel kosong bertipe dictionary
fitur = {} # variable global , dictionary = json

# membuat class Resource
class MachineLearningResource(Resource):
    # metode post
    def post(self):
        fitur["jumlah_lengan"] = request.form["jumlah_lengan"]
        fitur["bercabang"] = request.form["bercabang"]
        fitur["knob"] = request.form["knob"]
        fitur["bentuk_lengan"] = request.form["bentuk_lengan"]
        fitur["ujung_lengan"] = request.form["ujung_lengan"]

        # prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
        prediction = random_forest_model.predict([[6,1,1,1,1]])[0]
        response = {"prediction" : prediction}
        return response

# setup resourcenya
api.add_resource(MachineLearningResource, "/api/predict", methods=["POST"])