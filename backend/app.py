from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Definir la ruta base y cargar los modelos
base_path = os.path.dirname(os.path.abspath(__file__))
usernames = ["fabimontans", "DiegoPlusUltra", "Saborcito"]
models = {}

for username in usernames:
    model_path = os.path.join(
        base_path, "..", f"best_joined_today_predictor_{username}.pkl"
    )
    models[username] = joblib.load(model_path)


@app.route("/hello", methods=["GET"])
def hello_world():
    return "Hello, World!"


# Esperamos un JSON con la siguiente estructura:
# {
#     'daysAway': int, (racha de dias fuera de discord)
#     'consecutiveJoinedDays': int, (racha de dias seguidos entrando a discord)
#     'freeDay': int, (1 o 0)
#     'actualDay': int,
#     'actualMonth': int,
#     'actualYear': int
# }
@app.route("/predict/<username>", methods=["POST"])
def predict(username):
    if username not in models:
        return jsonify({"error": "Usuario no encontrado"}), 404

    model = models[username]
    data = request.json
    new_data = pd.DataFrame([data])
    prediction = model.predict(new_data)
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":
    app.run(debug=True)
