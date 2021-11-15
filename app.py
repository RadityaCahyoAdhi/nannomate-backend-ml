# import library
from flask import Flask, request
from flask_restful import Resource, Api
from jcopml.utils import load_model
import numpy as np

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

        # prediksi probabilitas masing-masing kelas
        probs = random_forest_model.predict_proba([[fitur["jumlah_lengan"],fitur["bercabang"],fitur["knob"],fitur["bentuk_lengan"],fitur["ujung_lengan"]]])

        # mengurutkan probabilitas dari yang terbesar ke yang terkecil
        sorted_probs = sorted(probs[0], reverse=True)

        # mencari 3 kelas spesies dengan probabilitas terbesar
        best_three_species = np.argsort(probs)[:,-3:][0]
        
        # melakukan inverse transform pada 3 kelas spesies dengan probabilitas terbesar
        decoded_best_three_species = lbl_encoder.inverse_transform(best_three_species)

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