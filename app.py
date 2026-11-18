import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

st.set_page_config(page_title="Study Smarter", page_icon="🎓", layout="centered")

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
.stButton > button[kind="primary"] {
    background: #ff6b1a;
    border: none;
    color: #ffffff;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: #ff8a4c;
}
</style>
"""

CARD_CSS = (
    "<style>"
    "@keyframes pulse { 0% { opacity: 0.45; } 50% { opacity: 1; } 100% { opacity: 0.45; } }"
    "@keyframes glow { 0% { box-shadow: 0 0 6px #ff6b1a; } 50% { box-shadow: 0 0 24px #ff6b1a; } 100% { box-shadow: 0 0 6px #ff6b1a; } }"
    ".mcards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }"
    ".mcard { background: #0f2344; border: 1px solid #ffffff33; border-radius: 12px; padding: 12px 8px; text-align: center; }"
    ".mcard.scan { animation: pulse 0.8s ease-in-out infinite; border-color: #ff6b1a; }"
    ".mcard.done { border-color: #ff6b1a; }"
    ".mcard.best { animation: glow 1.4s ease-in-out infinite; border-color: #ffa54d; }"
    ".mname { font-size: 13px; color: #cfd8e6; min-height: 32px; }"
    ".mscore { font-size: 22px; font-weight: 700; color: #ffffff; margin-top: 6px; }"
    ".mbar { height: 8px; background: #ffffff22; border-radius: 6px; overflow: hidden; margin-top: 8px; }"
    ".mfill { height: 100%; background: linear-gradient(90deg, #ff6b1a, #ffa54d); transition: width 0.5s ease; }"
    "@media (max-width: 700px) { .mcards { grid-template-columns: repeat(2, 1fr); } }"
    "</style>"
)

EVAL_CSS = (
    "<style>"
    ".eval-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px; }"
    ".eval-card { background: #0f2344; border: 1px solid #ff6b1a; border-radius: 14px; padding: 16px 12px; text-align: center; }"
    ".eval-label { color: #cfd8e6; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }"
    ".eval-value { color: #ffffff; font-size: 30px; font-weight: 700; margin-top: 6px; }"
    ".eval-unit { color: #ffa54d; font-size: 13px; margin-top: 2px; }"
    ".eval-note { color: #cfd8e6; font-size: 13px; margin-top: 10px; }"
    "@media (max-width: 700px) { .eval-grid { grid-template-columns: 1fr; } }"
    "</style>"
)

def set_background(image_file):
    if not Path(image_file).exists():
        return
    data = base64.b64encode(Path(image_file).read_bytes()).decode()
    st.markdown(BACKGROUND_CSS.replace("__BG_DATA__", data), unsafe_allow_html=True)

def render_cards(names, scores, states, stage_text, pct):
    # states: 0 waiting, 1 scanning, 2 revealed, 3 best
    parts = [
        CARD_CSS,
        '<div style="color:#ffffff;font-size:16px;margin-bottom:10px;">' + stage_text + "</div>",
        '<div style="background:#ffffff22;border-radius:8px;height:10px;overflow:hidden;margin-bottom:14px;">'
        '<div style="width:' + str(pct) + '%;height:100%;background:linear-gradient(90deg,#ff6b1a,#ffa54d);transition:width 0.3s ease;"></div></div>',
        '<div class="mcards">',
    ]
    for i, name in enumerate(names):
        state = states[i]
        if state == 0:
            cls, score_text, width = "mcard", "–", 0
        elif state == 1:
            cls, score_text, width = "mcard scan", "testing...", 0
        elif state == 2:
            cls, score_text, width = "mcard done", "%.1f" % scores[i], scores[i]
        else:
            cls, score_text, width = "mcard best", "%.1f" % scores[i], scores[i]
        width = max(0, min(100, float(width)))
        parts.append(
            '<div class="' + cls + '">'
            '<div class="mname">' + name + "</div>"
            '<div class="mscore">' + score_text + "</div>"
            '<div class="mbar"><div class="mfill" style="width:' + str(round(width, 1)) + '%;"></div></div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)

def performance_level(mark):
    if mark >= 80:
        return "Excellent", "🌟", "Your habits point to a strong result. Keep what's working and stay consistent."
    elif mark >= 65:
        return "Good", "👍", "There's clear room to improve. Switching to the recommended method can close part of the gap."
    else:
        return "Needs work", "💪", "The predicted marks are low. Combine the recommended method with the habit changes below, and ask a teacher or classmate for help with difficult topics."

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

def predict_methods(student):
    rows = pd.DataFrame([{**student, "method": m} for m in METHODS])
    return np.clip(model.predict(preprocess(rows), verbose=0).flatten(), 0, 100)

def build_scenarios(student):
    changes = [
        ("Sleeping closer to 8 hours", {"sleep_hours": 8.0}),
        ("Lowering distraction by 3 points", {"distraction": max(0, student["distraction"] - 3)}),
        ("Adding 5 study hours per week", {"study_hours": min(30, student["study_hours"] + 5)}),
        ("Raising attendance by 10 points", {"attendance": min(100, student["attendance"] + 10)}),
        ("Raising previous GPA by 0.5", {"prev_gpa": min(4.0, student["prev_gpa"] + 0.5)}),
    ]
    scenarios = [("base", {})]
    for label, change in changes:
        if any(student[k] != v for k, v in change.items()):
            scenarios.append((label, change))
    return scenarios

def predictive_tips(student):
    scenarios = build_scenarios(student)
    rows = []
    for _, change in scenarios:
        for m in METHODS:
            rows.append({**student, **change, "method": m})
    preds = np.clip(model.predict(preprocess(pd.DataFrame(rows)), verbose=0).flatten(), 0, 100)
    n = len(METHODS)
    blocks = [preds[i * n:(i + 1) * n] for i in range(len(scenarios))]

    base_best = float(blocks[0].max())
    gains = []
    for (label, _), block in zip(scenarios[1:], blocks[1:]):
        new_best = float(block.max())
        gains.append((label, base_best, new_best, new_best - base_best))
    gains.sort(key=lambda g: g[3], reverse=True)
    return [g for g in gains if g[3] >= 0.5]

@st.cache_data
def get_model_metrics():
    df_all = pd.read_csv("students.csv")
    _, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    y_true = test_df["marks"].values
    y_pred = np.clip(model.predict(preprocess(test_df), verbose=0).flatten(), 0, 100)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "n": int(len(test_df)),
        "y_true": y_true,
        "y_pred": y_pred,
    }

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

    names = [m.replace("_", " ").title() for m in METHODS]
    scores = predict_methods(student)
    best_idx = int(np.argmax(scores))
    states = [0] * len(METHODS)

    slot = st.empty()
    slot.markdown(render_cards(names, scores, states, "Loading your study profile...", 0), unsafe_allow_html=True)
    time.sleep(0.6)

    for i in range(len(METHODS)):
        states[i] = 1
        slot.markdown(render_cards(names, scores, states, "Testing " + names[i] + "...", i * 20), unsafe_allow_html=True)
        time.sleep(0.5)
        states[i] = 2
        slot.markdown(render_cards(names, scores, states, "Predicted marks for " + names[i], (i + 1) * 20), unsafe_allow_html=True)
        time.sleep(0.3)

    states[best_idx] = 3
    slot.markdown(render_cards(names, scores, states, "Best method found", 100), unsafe_allow_html=True)
    time.sleep(0.8)

    results = pd.DataFrame({
        "key": METHODS,
        "Study method": names,
        "Predicted marks": scores.round(1),
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

    st.markdown("**What would change your predicted marks:**")
    tips = predictive_tips(student)
    if tips:
        for name, base, new, gain in tips[:3]:
            st.markdown(
                f"- {name}: predicted best marks rise from {base:.1f} to {new:.1f} (+{gain:.1f})"
            )
        st.caption("These are model estimates from changing one habit at a time, not guaranteed results.")
    else:
        st.write("Your inputs already sit near the model's best settings, so the biggest gain now comes from using the recommended method consistently.")

    st.info("**How to use " + best["Study method"] + ":** " + METHOD_ADVICE[best["key"]])
    st.warning(
        "**Avoid relying on " + worst["Study method"] + " alone.** It has the lowest predicted marks, "
        + f"{gap:.1f} marks below your best method."
    )

    if best_mark >= 80:
        st.balloons()
    else:
        st.snow()

    st.bar_chart(results.set_index("Study method")["Predicted marks"])
    st.dataframe(results.drop(columns="key"), hide_index=True)

st.markdown("## Model performance")

if Path("students.csv").exists():
    m = get_model_metrics()
    st.markdown(
        EVAL_CSS
        + '<div class="eval-grid">'
        + '<div class="eval-card"><div class="eval-label">MAE</div>'
        + '<div class="eval-value">' + f"{m['MAE']:.2f}" + "</div>"
        + '<div class="eval-unit">average error in marks</div></div>'
        + '<div class="eval-card"><div class="eval-label">RMSE</div>'
        + '<div class="eval-value">' + f"{m['RMSE']:.2f}" + "</div>"
        + '<div class="eval-unit">typical error in marks</div></div>'
        + '<div class="eval-card"><div class="eval-label">R²</div>'
        + '<div class="eval-value">' + f"{m['R2']:.3f}" + "</div>"
        + '<div class="eval-unit">share of variation explained</div></div>'
        + "</div>"
        + '<div class="eval-note">Measured on ' + str(m["n"])
        + " students the model did not train on (20% of the data, held out before training).</div>",
        unsafe_allow_html=True,
    )

    chart_df = pd.DataFrame({"Actual marks": m["y_true"], "Predicted marks": m["y_pred"]})
    st.scatter_chart(chart_df, x="Actual marks", y="Predicted marks", color="#ff6b1a")
else:
    st.write("Upload students.csv to the repository to show model performance.")

st.caption("This is a prototype trained on synthetic data. It shows patterns, not guaranteed results.")
