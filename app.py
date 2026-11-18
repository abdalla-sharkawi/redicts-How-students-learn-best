import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
from tensorflow import keras

NUM = ["study_hours", "sleep_hours", "attendance", "prev_gpa", "distraction"]
METHODS = ["practice_tests", "flashcards", "group_study", "reading_notes", "video_lectures"]

METHOD_ADVICE = {
    "practice_tests": "Test yourself often. Answer questions from memory, then check your mistakes. Short quizzes every few days work well.",
    "flashcards": "Review short sets of key terms and formulas every day. Fifteen to twenty minutes daily beats one long session.",
    "group_study": "Study with classmates, but agree on a clear agenda and keep phones away. Explaining topics to each other helps most.",
    "reading_notes": "Rewrite your notes as summaries in your own words, then check them against the source. Re-reading alone is weak.",
    "video_lectures": "Pause videos often and try the problems yourself before watching the solution. Watching without practicing rarely raises marks.",
}

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

def performance_level(mark):
    if mark >= 80:
        return "Excellent", "🌟", "Your habits point to a strong result. Keep what's working and stay consistent."
    elif mark >= 65:
        return "Good", "👍", "There's clear room to improve. Switching to the recommended method can close part of the gap."
    else:
        return "Needs work", "💪", "The predicted marks are low. Combine the recommended method with the habit changes below, and ask a teacher or classmate for help with difficult topics."

def habit_tips(s):
    tips = []
    if s["sleep_hours"] < 7:
        tips.append("Sleep: aim for 7 to 8 hours. In this model, sleep far from 8 hours is linked to lower marks.")
    if s["distraction"] >= 7:
        tips.append("Focus: put your phone in another room during study sessions. High distraction lowers predicted marks the most.")
    if s["attendance"] < 75:
        tips.append("Attendance: regular class attendance is strongly linked to higher marks.")
    if s["study_hours"] < 10:
        tips.append("Study time: adding a few more weekly hours is linked to higher marks.")
    if s["prev_gpa"] < 2.5:
        tips.append("Foundations: start with short review sessions to rebuild confidence before tackling harder material.")
    if not tips:
        tips.append("Your habits look solid. Keep them consistent, since regular routines matter most.")
    return tips

st.set_page_config(page_title="Study Method Predictor", page_icon="📚")
st.title("📚 Predict How You Learn Best")
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

    # Staged animation
    steps = [
        "Reading your study habits",
        "Testing all five study methods",
        "Comparing predicted marks",
        "Choosing the best method",
    ]
    with st.status("Analyzing your study habits...", expanded=True) as status:
        bar = st.progress(0)
        for idx, step in enumerate(steps):
            st.write(f"⏳ {step}...")
            for pct in range(idx * 25, (idx + 1) * 25 + 1, 5):
                bar.progress(min(pct, 100))
                time.sleep(0.02)
            st.write(f"✅ {step}")

        rows = pd.DataFrame([{**student, "method": m} for m in METHODS])
        preds = np.clip(model.predict(preprocess(rows), verbose=0).flatten(), 0, 100)
        status.update(label="Analysis complete!", state="complete", expanded=False)

    results = pd.DataFrame({
        "key": METHODS,
        "Study method": [m.replace("_", " ").title() for m in METHODS],
        "Predicted marks": preds.round(1),
    }).sort_values("Predicted marks", ascending=False).reset_index(drop=True)

    best = results.iloc[0]
    worst = results.iloc[-1]
    best_mark = float(best["Predicted marks"])
    gap = round(best_mark - float(worst["Predicted marks"]), 1)

    # Summary metrics
    m1, m2 = st.columns(2)
    m1.metric("Best predicted marks", f"{best_mark:.1f}")
    m2.metric("Gap between best and worst method", f"{gap:.1f} marks")

    # Result and tips that depend on the predicted marks
    label, icon, level_text = performance_level(best_mark)
    st.success(f"{icon} **{label}.** Recommended method: **{best['Study method']}**")
    st.write(level_text)
    st.info(f"**How to use {best['Study method']}:** {METHOD_ADVICE[best['key']]}")

    st.markdown("**Habit tips based on your inputs:**")
    for tip in habit_tips(student):
        st.markdown(f"- {tip}")

    # Effects
    if best_mark >= 80:
        st.balloons()
    else:
        st.snow()

    st.bar_chart(results.set_index("Study method")["Predicted marks"])
    st.dataframe(results.drop(columns="key"), hide_index=True)

st.caption("This is a prototype trained on synthetic data. It shows patterns, not guaranteed results.")
