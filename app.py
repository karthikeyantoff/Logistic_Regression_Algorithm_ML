from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model
# Make sure you ran train.py first to create this file!
model = joblib.load("placement_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    
    if request.method == "POST":
        try:
            # 1. Get Input
            cgpa = float(request.form["cgpa"])
            
            # 2. Predict (0 = Not Placed, 1 = Placed)
            prediction = model.predict([[cgpa]])[0]
            
            # 3. Show Message
            if prediction == 1:
                result = "You are PLACED!"
            else:
                result = "You are NOT Placed."
                
        except:
            result = "Error: Please check your input."

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)