import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

NUM = ["study_hours", "sleep_hours", "attendance", "prev_gpa", "distraction"]
METHODS = ["practice_tests", "flashcards", "group_study", "reading_notes", "video_lectures"]

df = pd.read_csv("students.csv")
TARGET = "marks"

model = keras.models.load_model("study_predictor.keras")
scaler = joblib.load("scaler.pkl")

def preprocess(d):
    numeric = scaler.transform(d[NUM])
    onehot = (pd.get_dummies(d["method"])
                .reindex(columns=METHODS, fill_value=0)
                .values.astype(float))
    return np.hstack([numeric, onehot]).astype("float32")

_, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_test = preprocess(test_df)
y_true = test_df[TARGET].values
y_pred = np.clip(model.predict(X_test, verbose=0).flatten(), 0, 100)

print("MAE :", round(mean_absolute_error(y_true, y_pred), 2), "marks")
print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2), "marks")
print("R²  :", round(r2_score(y_true, y_pred), 3))
