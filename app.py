
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from huggingface_hub import hf_hub_download
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide"
)

# -------------------- CUSTOM STYLING --------------------

st.markdown("""
<style>
.stApp {
    background: #f6f5ef;
    color: #243b53;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(120deg,#e8f0e8,#e9eff4);
    padding: 25px 30px;
    border-radius: 16px;
    border: 1px solid #d7e2d8;
    margin-bottom: 20px;
}

.hero h1 {
    color: #243b53;
    margin: 0;
}

.result {
    background: #e8f1e9;
    border-left: 6px solid #6d927d;
    border-radius: 12px;
    padding: 18px 22px;
}

.alert {
    background: #f8e9e6;
    border-left-color: #bd6b5e;
}

/* FORCE LIGHT DROPDOWN */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] [role="combobox"] * {
    color: #111827 !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #ffffff !important;
}

[role="option"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background-color: #e9eafc !important;
    color: #111827 !important;
}

/* FORCE LIGHT TEXT AREA */
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] [data-baseweb="textarea"] > div,
[data-testid="stTextArea"] [data-baseweb="base-input"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
