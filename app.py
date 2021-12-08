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
                    'knob': {'type': 'integer', 'min': 0, 'max': 1},
                    'bentuk_lengan': {'type': 'integer', 'min': 0, 'max': 1},
                    'ujung_lengan': {'type': 'integer', 'min': 0, 'max': 1}
                    }

        # inisialisasi validator            
        validator = Validator(schema)
        validator.require_all = True

        # validasi request
        if not(validator.validate(request_data)):
            return validator.errors, 400
        else:
            fitur["jumlah_lengan"] = request_data["jumlah_lengan"]
            fitur["bercabang"] = request_data["bercabang"]
            fitur["knob"] = request_data["knob"]
            fitur["bentuk_lengan"] = request_data["bentuk_lengan"]
            fitur["ujung_lengan"] = request_data["ujung_lengan"]

            # prediksi probabilitas masing-masing kelas
            probs = random_forest_model.predict_proba([[fitur["jumlah_lengan"], fitur["bercabang"], fitur["knob"], fitur["bentuk_lengan"], fitur["ujung_lengan"]]])

            # mengurutkan probabilitas dari yang terbesar ke yang terkecil
            sorted_probs = sorted(probs[0], reverse=True)

            # mencari 3 kelas spesies dengan probabilitas terbesar
            best_three_species = np.argsort(probs)[:,-3:][0]
            
            # melakukan inverse transform pada 3 kelas spesies dengan probabilitas terbesar
            decoded_best_three_species = lbl_encoder.inverse_transform(best_three_species)

            # membuat response
            response = {
                            "prediction": {
                                "first_prediction": decoded_best_three_species[2],
                                "second_prediction": decoded_best_three_species[1],
                                "third_prediction": decoded_best_three_species[0]
                            },
                            "probabilities": {
                                "first_probability": round(sorted_probs[0]*100, 2),
                                "second_prediction": round(sorted_probs[1]*100, 2), 
                                "third_prediction": round(sorted_probs[2]*100, 2)
                            }
                        }
            return response, 200

# setup resource
api.add_resource(MachineLearningResource, "/api/predict", methods=["POST"])

if __name__ == "__main__":
    app.run(debug=True, port=5005)