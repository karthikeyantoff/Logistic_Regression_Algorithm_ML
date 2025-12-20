from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("house_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    if request.method == "POST":
        income = float(request.form["income"])
        age = float(request.form["age"])
        rooms = float(request.form["rooms"])
        user_data = pd.DataFrame([[
            -118.0, 34.0, age, rooms, 100.0, 800.0, 300.0, income, 0
        ]], columns=[
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms',
            'population', 'households',
            'median_income', 'ocean_proximity'
        ])
        price = model.predict(user_data)[0]
    return render_template("index.html", price=price)
if __name__ == "__main__":
    app.run(debug=True)
