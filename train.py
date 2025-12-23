import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url = r'D:\DATA_SCIENCE_PROJECT\deployment_prep\DATA_SETS\placement.csv'
df = pd.read_csv(url)

X = df[['cgpa']]
y = df['placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'placement_model.pkl')
print("Model Saved as 'placement_model.pkl'")