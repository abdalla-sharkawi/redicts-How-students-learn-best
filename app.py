import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import base64
from pathlib import Path
from tensorflow import keras

TITLE = "🎓 Study Smarter"
SUBTITLE = "Enter your study habits. The model predicts your marks for each study method and recommends the one most likely to work for you."

NUM = ["study_hours", "sleep_hours", "attendance", "prev_gpa", "distraction"]
METHODS = ["practice_tests", "flashcards", "group_study", "reading_notes", "video_lectures"]

METHOD_ADVICE = {
    "practice_tests": "Test yourself often. Answer questions from memory, then check your mistakes. Short quizzes every few days work well.",
    "flashcards": "Review short sets of key terms and formulas every day. Fifteen to twenty minutes daily beats one long session.",
    "group_study": "Study with classmates, but agree on a clear agenda and keep phones away. Explaining topics to each other helps most.",
    "reading_notes": "Rewrite your notes as summaries in your own words, then check them against the source. Re-reading alone is weak.",
    "video_lectures": "Pause videos often and try the problems yourself before watching the solution. Watching without practicing rarely raises marks.",
}

st.set_page_config(page_title="Study Smarter", page_icon="🎓", layout="centered")

BACKGROUND_CSS = """
<style>
.stApp {
    background-image: url("data:image/jpeg;base64,__BG_DATA__");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    background: rgba(8, 20, 45, 0.88);
    border-radius: 18px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}
h1, h2, h3, h4, p, label, li, .stMarkdown, .stCaption,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}
</style>
"""

LOADER_HTML = """
<div style="background:#0f2344;border:1px solid #ff6b1a;border-radius:14px;padding:20px;margin:10px 0;">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:26px;height:26px;border:4px solid #ffffff33;border-top:4px solid #ff6b1a;border-radius:50%;animation:spin 0.9s linear infinite;"></div>
        <div style="color:#ffffff;font-size:16px;">__STEP_TEXT__</div>
    </div>
    <div style="margin-top:14px;background:#ffffff22;border-radius:8px;height:10px;overflow:hidden;">
        <div style="width:__PCT__%;height:100%;background:linear-gradient(90deg,#ff6b1a,#ffa54d);transition:width 0.4s ease;"></div>
    </div>
    <div style="color:#cfd8e6;font-size:13px;margin-top:6px;">__PCT__% complete</div>
</div>
<style>
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>
"""

def set_background(image_file):
    if not Path(image_file).exists():
        return
    data = base64.b64encode(Path(image_file).read_bytes()).decode()
    st.markdown(BACKGROUND_CSS.replace("__BG_DATA__", data), unsafe_allow_html=True)

def loader_html(step_text, fraction):
    pct = str(int(fraction * 100))
    return LOADER_HTML.replace("__STEP_TEXT__", step_text).replace("__PCT__", pct)

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

set_background("background.jpg")

st.markdown("<h1>" + TITLE + "</h1>", unsafe_allow_html=True)
st.write(SUBTITLE)

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

    steps = [
        "Reading your study habits",
        "Testing all five study methods",
        "Comparing predicted marks",
        "Choosing the best method",
    ]
    placeholder = st.empty()
    for idx, step in enumerate(steps):
        placeholder.markdown(loader_html(step + "...", (idx + 1) / len(steps)), unsafe_allow_html=True)
        time.sleep(0.6)

    rows = pd.DataFrame([{**student, "method": m} for m in METHODS])
    preds = np.clip(model.predict(preprocess(rows), verbose=0).flatten(), 0, 100)
    placeholder.empty()

    results = pd.DataFrame({
        "key": METHODS,
        "Study method": [m.replace("_", " ").title() for m in METHODS],
        "Predicted marks": preds.round(1),
    }).sort_values("Predicted marks", ascending=False).reset_index(drop=True)

    best = results.iloc[0]
    worst = results.iloc[-1]
    best_mark = float(best["Predicted marks"])
    gap = round(best_mark - float(worst["Predicted marks"]), 1)

    m1, m2 = st.columns(2)
    m1.metric("Best predicted marks", f"{best_mark:.1f}")
    m2.metric("Gap between best and worst method", f"{gap:.1f} marks")

    label, icon, level_text = performance_level(best_mark)
    st.success(icon + " **" + label + ".** Recommended method: **" + best["Study method"] + "**")
    st.write(level_text)
    st.info("**How to use " + best["Study method"] + ":** " + METHOD_ADVICE[best["key"]])

    st.markdown("**Habit tips based on your inputs:**")
    for tip in habit_tips(student):
        st.markdown("- " + tip)

    if best_mark >= 80:
        st.balloons()
    else:
        st.snow()

    st.bar_chart(results.set_index("Study method")["Predicted marks"])
    st.dataframe(results.drop(columns="key"), hide_index=True)

st.caption("This is a prototype trained on synthetic data. It shows patterns, not guaranteed results.")
