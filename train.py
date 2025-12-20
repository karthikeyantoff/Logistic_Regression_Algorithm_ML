# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# # from sklearn.preprocessing import StandardScaler
# # 1. LOAD DATA
# df = pd.read_csv('D:\DATA_SCIENCE_PROJECT\deployment_prep\DATA_SETS\housing_cleaned.csv')
# X = df.drop('median_house_value', axis=1) 
# y = df['median_house_value'] 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # scaler=StandardScaler()
# model=LinearRegression
# # model = LinearRegression()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)  
# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)
# print(f"MSE (Mean Squared Error):{mse:,.2f}")
# print(f"RMSE (Root Mean Sq Error):{rmse:,.2f}")
# print(f"MAE (Mean Absolute Error):{mae:,.2f}")
# print(f"R² Score (Accuracy %):{r2:.4f}")
# # 5. SAVE MODEL
# joblib.dump(model, 'house_model.pkl')
# print("Model Saved as 'house_model.pkl'")
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# IMPORT ALL 4 METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. LOAD DATA
df = pd.read_csv('D:\DATA_SCIENCE_PROJECT\deployment_prep\DATA_SETS\housing_cleaned.csv')
X = df.drop('median_house_value', axis=1) # Questions
y = df['median_house_value']              # Answers

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAIN MODEL
print("🤖 Training the Model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 4. PREDICT & EVALUATE (The 4 Metrics)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)  # Square Root of MSE
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📊 --- MODEL PERFORMANCE REPORT ---")
print(f"1️⃣  MSE (Mean Squared Error):      {mse:,.2f}")
print(f"2️⃣  RMSE (Root Mean Sq Error):     {rmse:,.2f}")
print(f"3️⃣  MAE (Mean Absolute Error):     {mae:,.2f}")
print(f"4️⃣  R² Score (Accuracy %):         {r2:.4f}")

# 5. SAVE MODEL
joblib.dump(model, 'house_model.pkl')
print("\n✅ Model Saved as 'house_model.pkl'")