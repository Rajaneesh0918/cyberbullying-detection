
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
    border-radius: 8px !important;
    box-shadow: none !important;
}

[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

/* LABELS */
.stSelectbox label,
.stTextArea label {
    color: #243b53 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(
        repo_id="rajaneesh0918/cyberbullying-artifacts",
        filename="cyberbullying_artifacts.joblib"
    )
    return joblib.load(model_path)


try:
    data = load_artifacts()

    model = data["model"]
    tfidf = data["tfidf"]
    STOPWORDS = set(data["stopwords"])

    stemmer = PorterStemmer()
    class_names = list(model.classes_)

except Exception as e:
    st.error("Could not load the cyberbullying model.")
    st.exception(e)
    st.stop()


# -------------------- TEXT PREPROCESSING --------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = [
        t for t in text.split()
        if t not in STOPWORDS and len(t) > 1
    ]

    return " ".join(stemmer.stem(t) for t in tokens)


# -------------------- PREDICTION PIPELINE --------------------

def predict_proba_pipeline(texts):
    cleaned = [clean_text(t) for t in texts]
    cleaned = [t if t.strip() else "empty" for t in cleaned]

    return model.predict_proba(tfidf.transform(cleaned))


# -------------------- LIME EXPLAINER --------------------

@st.cache_resource
def create_explainer(names):
    return LimeTextExplainer(
        class_names=names,
        random_state=42
    )


explainer = create_explainer(class_names)


# -------------------- DASHBOARD HEADER --------------------

st.markdown("""
<div class="hero">
    <h1>🛡️ Cyberbullying Detection Dashboard</h1>
    <p>Explainable text classification using TF-IDF, Stacking Ensemble Learning and LIME.</p>
</div>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("About this project")

    st.write(
        "Classifies text into the five categories "
        "learned by the trained model."
    )

    st.markdown("""
    **Classes**
    - Age
    - Ethnicity
    - Gender
    - Not Cyberbullying
    - Religion
    """)

    st.caption(
        "Academic demonstration. Automated predictions may "
        "be incorrect; human review is recommended."
    )


# -------------------- USER INPUT SECTION --------------------

st.subheader("Analyse a tweet")

samples = {
    "Choose a sample": "",

    "Age-related example":
        "You are too old to understand anything about modern technology.",

    "Gender-related example":
        "Women deserve equal opportunities and respect in every profession.",

    "Religion-related example":
        "People should not be judged because of their religion.",

    "Ethnicity-related example":
        "People from every ethnic background deserve equal respect.",

    "Not cyberbullying example":
        "Good morning everyone. Have a wonderful day!",

    "General insult example":
        "You are such an idiot and a bully."
}

choice = st.selectbox(
    "Quick sample",
    list(samples.keys())
)

tweet = st.text_area(
    "Tweet / message",
    value=samples[choice],
    height=130,
    placeholder="Paste a tweet or message here..."
)

analyse = st.button(
    "🔍 Analyse Tweet",
    type="primary",
    use_container_width=True
)


# -------------------- ANALYSIS AND PREDICTION --------------------

if analyse:

    if not tweet.strip():
        st.warning("Please enter a tweet first.")

    else:

        with st.spinner("Analysing tweet..."):

            probs = predict_proba_pipeline([tweet])[0]

            idx = int(np.argmax(probs))
            predicted = class_names[idx]
            confidence = float(probs[idx]) * 100

        st.divider()

        # Classification result
        st.subheader("Classification result")

        title = predicted.replace("_", " ").title()

        if predicted == "not_cyberbullying":

            st.markdown(
                f"""
                <div class="result">
                    <b>NOT CYBERBULLYING</b>
                    <h2>{title}</h2>
                    <b>Model probability: {confidence:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result alert">
                    <b>CYBERBULLYING CATEGORY</b>
                    <h2>{title}</h2>
                    <b>Model probability: {confidence:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Probability analysis
        st.subheader("Class probability analysis")

        prob_df = pd.DataFrame({
            "Category": [
                c.replace("_", " ").title()
                for c in class_names
            ],
            "Probability (%)": probs * 100
        }).sort_values(
            "Probability (%)",
            ascending=False
        )

        st.bar_chart(
            prob_df.set_index("Category"),
            horizontal=True
        )

        st.dataframe(
            prob_df.style.format({
                "Probability (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

        # LIME explanation
        st.subheader("LIME explanation")

        st.caption(
            "Green bars support the predicted class; "
            "red bars oppose it."
        )

        try:

            exp = explainer.explain_instance(
                tweet,
                predict_proba_pipeline,
                num_features=10,
                labels=[idx],
                num_samples=1000
            )

            items = exp.as_list(label=idx)

            words, weights = zip(*items)

            colors = [
                "#6d927d" if w > 0 else "#c96e61"
                for w in weights
            ]

            fig, ax = plt.subplots(
                figsize=(9, max(3, len(words) * 0.4))
            )

            ax.barh(
                list(words)[::-1],
                list(weights)[::-1],
                color=list(colors)[::-1]
            )

            ax.axvline(
                0,
                color="#40566b",
                linestyle="--",
                linewidth=0.8
            )

            ax.set_xlabel("LIME feature contribution")
            ax.set_title(f"Local explanation: {title}")

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:

            st.warning(
                "Prediction succeeded, but LIME explanation "
                "could not be generated."
            )

            st.caption(str(e))

        # Technical details
        with st.expander("Technical details"):

            st.write("Model: Stacking Ensemble")

            st.write(
                "Cleaned input:",
                clean_text(tweet)
            )

            st.write(
                "Class order:",
                class_names
            )

            st.write(
                "Probability values:",
                probs.tolist()
            )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "Research prototype: the model may misclassify sarcasm, "
    "context-dependent language and general insults. "
    "Predictions are not final judgements; human moderation is recommended."
)
