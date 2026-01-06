# 🧠 Logistic Regression ML Web Application

A Machine Learning web application that predicts **binary outcomes** (such as Purchase / Not Purchase, Yes / No, True / False) based on user input features.
The model is built using the **Logistic Regression** algorithm and deployed using **Flask**.

🔗 **Live Demo:**
👉 [https://logistic-regression-algorithm-ml-pi.vercel.app/](https://logistic-regression-algorithm-ml-pi.vercel.app/)

🔗 **Repository:**
👉 [https://github.com/karthikeyantoff/Logistic_Regression_Algorithm_ML](https://github.com/karthikeyantoff/Logistic_Regression_Algorithm_ML)

---

## 📌 About the Project

This project uses **Logistic Regression**, a supervised machine learning algorithm, to perform **classification tasks** based on structured input data.

Logistic Regression predicts the **probability of a class** and assigns a final class label (0 or 1).
It is widely used for problems such as:

* Purchase Prediction
* Pass / Fail Prediction
* Risk Analysis
* Binary Decision Systems

The application provides a simple **HTML frontend** where users enter input values, and a **Flask backend** that loads the trained model and returns predictions in real time.

The project is optimized for **deployment on Vercel**.

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn, Joblib
* **Algorithm:** Logistic Regression
* **Data Processing:** NumPy
* **Deployment:** Vercel

---

## ▶️ How to Run Locally

Follow these steps to run the project on your local system:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/karthikeyantoff/Logistic_Regression_Algorithm_ML.git
cd Logistic_Regression_Algorithm_ML
```

### 2️⃣ Install Dependencies

Make sure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App

```bash
python app.py
```

### 4️⃣ Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🤖 Model Details

* **Algorithm:** Logistic Regression
* **Library Used:** Scikit-Learn
* **Learning Type:** Supervised Learning
* **Model Saving:** Joblib (`.pkl` file)

---

## 🔢 Input Features (Example)

The model takes numerical input features such as:

* Feature 1 (e.g., Age)
* Feature 2 (e.g., Salary / Income)
* Feature 3 (e.g., Experience)
* Feature 4 (e.g., Usage / Score)
* Feature 5 (dataset-specific attributes)

*(Exact features depend on the dataset used in training.)*

---

## 📤 Output

* **Class Prediction:**

  * `0` → Negative Class
  * `1` → Positive Class

* **Prediction Result Displayed on Web UI**

---

## 📂 Project Structure

```
Logistic_Regression_Algorithm_ML/
│
├── .github/workflows/     # GitHub Actions
├── DATA_SETS/             # Dataset files
├── templates/             # HTML frontend
│   └── index.html
├── app.py                 # Flask backend
├── data_prp.py            # Data preprocessing logic
├── train.py               # Model training script
├── model.pkl              # Trained Logistic Regression model
├── requirements.txt       # Project dependencies
├── vercel.json            # Vercel deployment config
└── README.md
```

---

## 🌐 Frontend & Backend Flow

1. User enters feature values in the web form
2. Data is sent to Flask backend
3. Backend loads the trained Logistic Regression model
4. Model predicts class output
5. Result is displayed on the frontend

---

## 🤝 Contributing

Contributions are welcome 🚀
You can:

* Improve UI design
* Add accuracy, confusion matrix, ROC curve
* Optimize preprocessing
* Improve model performance

Fork the repository and submit a pull request.

---

## 👨‍💻 Author

**Karthikeyan T**
Machine Learning | Deep Learning | AI Engineering Enthusiast 🔥
