# import library
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from jcopml.utils import load_model
from cerberus import Validator
import numpy as np

# Inisiasi object flask
app = Flask(__name__)

# inisiasi object flask_restful
api = Api(app)

# inisiasi object flask_cors
CORS(app)

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
        # mengambil data request json
        request_data = request.get_json()

        # schema validator
        schema = {
                    'jumlah_lengan': {'type': 'integer', 'min': 0},
                    'bercabang': {'type': 'integer', 'min': 0, 'max': 1},
                    'simetris': {'type': 'integer', 'min': 0, 'max': 1},
                    'knob': {'type': 'integer', 'min': 0, 'max': 1},
                    'ukuran_lengan': {'type': 'integer', 'min': 0, 'max': 1},
                    'bentuk_lengan': {'type': 'integer', 'min': 0, 'max': 1},
                    'ujung_lengan': {'type': 'integer', 'min': 0, 'max': 1},
                    'ujung_lengan_melengkung': {'type': 'integer', 'min': 0, 'max': 1}
                    }

        # inisialisasi validator            
        validator = Validator(schema)
        validator.require_all = True

        # validasi request
        if not(validator.validate(request_data)):
            error = { 
                "error": validator.errors
            }
            return error, 400
        else:
            fitur["jumlah_lengan"] = request_data["jumlah_lengan"]
            fitur["bercabang"] = request_data["bercabang"]
            fitur["simetris"] = request_data["simetris"]
            fitur["knob"] = request_data["knob"]
            fitur["ukuran_lengan"] = request_data["ukuran_lengan"]
            fitur["bentuk_lengan"] = request_data["bentuk_lengan"]
            fitur["ujung_lengan"] = request_data["ujung_lengan"]
            fitur["ujung_lengan_melengkung"] = request_data["ujung_lengan_melengkung"]

            # prediksi probabilitas masing-masing kelas
            probs = random_forest_model.predict_proba([[fitur["jumlah_lengan"], fitur["bercabang"], fitur["simetris"], fitur["knob"], fitur["ukuran_lengan"], fitur["bentuk_lengan"], fitur["ujung_lengan"], fitur["ujung_lengan_melengkung"]]])

            # mengurutkan probabilitas dari yang terbesar ke yang terkecil
            sorted_probs = sorted(probs[0], reverse=True)

            # mencari 5 kelas spesies dengan probabilitas terbesar
            best_five_species = np.argsort(probs)[:,-5:][0]
            
            # melakukan inverse transform pada 5 kelas spesies dengan probabilitas terbesar
            decoded_best_five_species = lbl_encoder.inverse_transform(best_five_species)

            # membuat response
            response = {
                            "prediction": {
                                "first_prediction": decoded_best_five_species[4],
                                "second_prediction": decoded_best_five_species[3],
                                "third_prediction": decoded_best_five_species[2],
                                "fourth_prediction": decoded_best_five_species[1],
                                "fifth_prediction": decoded_best_five_species[0],
                            },
                            "probabilities": {
                                "first_probability": round(sorted_probs[0]*100, 2),
                                "second_probability": round(sorted_probs[1]*100, 2), 
                                "third_probability": round(sorted_probs[2]*100, 2),
                                "fourth_probability": round(sorted_probs[3]*100, 2),
                                "fifth_probability": round(sorted_probs[4]*100, 2)
                            }
                        }
            return response, 200

# setup resource
api.add_resource(MachineLearningResource, "/api/predict", methods=["POST"])