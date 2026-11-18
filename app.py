import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from tensorflow import keras

NUM = ["study_hours", "sleep_hours", "attendance", "prev_gpa", "distraction"]
METHODS = ["practice_tests", "flashcards", "group_study", "reading_notes", "video_lectures"]

@st.cache_resource
def load_files():
    model = keras.models.load_model("study_predictor.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

def preprocess(d):
    numeric = scaler.transform(d[NUM])
    onehot = (pd.get_dummies(d["method"])
                .reindex(columns=METHODS, fill_value=0)
                .values.astype(float))
    return np.hstack([numeric, onehot]).astype("float32")

st.set_page_config(page_title="Study Method Predictor", page_icon="📚")
st.title("🧑‍🎓 Predict How You Learn Best")
st.write("Enter your study habits. The model predicts your marks with each study method and recommends the best one.")

col1, col2 = st.columns(2)
with col1:
    study_hours = st.slider("Study hours per week", 2, 30, 10)
    sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.0, 0.5)
    attendance = st.slider("Attendance (%)", 50, 100, 80)
with col2:
    prev_gpa = st.slider("Previous GPA", 1.5, 4.0, 3.0, 0.1)
    distraction = st.slider("Distraction level (0 = none, 10 = very high)", 0, 10, 5)

if st.button("Recommend a study method", type="primary"):
    student = {
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "prev_gpa": prev_gpa,
        "distraction": distraction,
    }

    # Animated progress bar
    progress = st.progress(0, text="Analyzing your study habits...")
    for i in range(0, 101, 5):
        time.sleep(0.03)
        progress.progress(i, text=f"Analyzing your study habits... {i}%")

    rows = pd.DataFrame([{**student, "method": m} for m in METHODS])
    preds = model.predict(preprocess(rows), verbose=0).flatten()
    progress.empty()

    results = pd.DataFrame({
        "key": METHODS,
        "Study method": [m.replace("_", " ").title() for m in METHODS],
        "Predicted marks": preds.round(1),
    }).sort_values("Predicted marks", ascending=False).reset_index(drop=True)

    best = results.iloc[0]

    st.balloons()
    st.success(
        f"Recommended method: **{best['Study method']}** "
        f"(predicted marks: {best['Predicted marks']})"
    )
    st.info(f"**Tip:** {ADVICE[best['key']]}")

    st.bar_chart(results.set_index("Study method")["Predicted marks"])
    st.dataframe(results.drop(columns="key"), hide_index=True)

st.caption("This is a prototype trained on synthetic data. It shows patterns, not guaranteed results.")
