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

def set_background(image_file):
    if not Path(image_file).exists():
        return
    data = base64.b64encode(Path(image_file).read_bytes()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(8, 20, 45, 0.88);
        border-radius: 18px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }}
    h1, h2, h3, h4, p, label, li, .stMarkdown, .stCaption, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
        color: #ffffff !important;
    }}
    .stSlider [data-testid="stWidgetLabel"] p {{
        color: #ffffff !important;
    }}
    .title-accent {{
        color: #ff6b1a;
    }}
    </style>
    """, unsafe_allow_html=True)

def loader_html(step_text, fraction):
    pct = int(fraction * 100)
    return f"""
    <div style="background:#0f2344;border:1px solid #ff6b1a;border-radius:14px;padding:20px;margin:10px 0;">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:26px;height:26px;border:4px solid #ffffff33;border-top:4px solid #ff6b1a;
                        border-radius:50%;animation:spin 0.9s linear infinite;"></div>
            <div style="color:#ffffff;font-size:16px;">{step_text}
