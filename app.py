import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp{background:#f6f5ef;color:#243b53}
.block-container{max-width:1150px;padding-top:2rem}
.hero{background:linear-gradient(120deg,#e8f0e8,#e9eff4);padding:25px 30px;border-radius:16px;border:1px solid #d7e2d8;margin-bottom:20px}
.hero h1{color:#243b53;margin:0}
.result{background:#e8f1e9;border-left:6px solid #6d927d;border-radius:12px;padding:18px 22px}
.alert{background:#f8e9e6;border-left-color:#bd6b5e}

/* Dropdown control and opened menu */
div[data-baseweb="select"] > div{
    background:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
    border-radius:8px !important;
}
div[data-baseweb="select"] *{color:#111827 !important}
ul[role="listbox"]{background:#ffffff !important}
li[role="option"]{background:#ffffff !important;color:#111827 !important}
li[role="option"]:hover,
li[role="option"][aria-selected="true"]{
    background:#e9eafc !important;
    color:#111827 !important;
}

/* Tweet/message text area */
div[data-baseweb="textarea"]{
    background:#ffffff !important;
    border:1px solid #cbd5e1 !important;
    border-radius:8px !important;
}
div[data-baseweb="textarea"] textarea{
    background:#ffffff !important;
    color:#111827 !important;
    caret-color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}
div[data-baseweb="textarea"] textarea::placeholder{
    color:#64748b !important;
    -webkit-text-fill-color:#64748b !important;
    opacity:1 !important;
}
.stSelectbox label,.stTextArea label{color:#243b53 !important}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    return joblib.load("cyberbullying_artifacts.joblib")

try:
    data = load_artifacts()
    model, tfidf = data["model"], data["tfidf"]
    STOPWORDS = set(data["stopwords"])
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    class_names = list(model.classes_)
except Exception as e:
    st.error("Could not load cyberbullying_artifacts.joblib. Add it beside app.py in your GitHub repository.")
    st.exception(e)
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(stemmer.stem(t) for t in tokens)

def predict_proba_pipeline(texts):
    cleaned = [clean_text(t) for t in texts]
    cleaned = [t if t.strip() else "empty" for t in cleaned]
    return model.predict_proba(tfidf.transform(cleaned))

@st.cache_resource
def create_explainer(names):
    return LimeTextExplainer(class_names=names, random_state=42)

explainer = create_explainer(class_names)

st.markdown("""
<div class="hero">
<h1>🛡️ Cyberbullying Detection Dashboard</h1>
<p>Explainable text classification using TF-IDF, Stacking Ensemble Learning and LIME.</p>
