
import joblib
import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="HeartCare | Heart Disease Prediction",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f4f8fd;
    color: #183153;
}

/* Main content width and spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #123e78, #1976d2);
    padding: 30px 34px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(25, 80, 145, 0.15);
}
.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #e5f1ff;
    font-size: 16px;
    margin-bottom: 0;
}

/* Section cards */
.section-card {
    background: white;
    padding: 20px 22px;
    border: 1px solid #e1eaf5;
    border-radius: 15px;
    margin-bottom: 14px;
}
.section-title {
    color: #164b88;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 5px;
}
.section-caption {
    color: #6a7d93;
    font-size: 13px;
    margin-bottom: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eaf3ff;
    border-right: 1px solid #d9e7f8;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #164b88;
}

/* Input labels */
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    color: #24476c !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #1769c2;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 22px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s ease;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0e4f98;
    color: white;
    border: none;
}

/* Result cards */
.result-card {
    background: white;
    padding: 24px;
    border-radius: 15px;
    border: 1px solid #dce8f5;
    margin-top: 20px;
}
.footer {
    color: #71839a;
    font-size: 12px;
    text-align: center;
    padding: 25px 0 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED FILES ----------------
@st.cache_resource
def load_files():
    model = joblib.load("LR_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_files()
except Exception as e:
    st.error("Unable to load the saved model files. Check that LR_heart.pkl, scaler.pkl, and columns.pkl are in the project folder.")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 💙 HeartCare")
    st.caption("Heart health prediction dashboard")
    st.divider()

    st.markdown("### About this app")
    st.write(
        "Enter the requested health measurements to get a prediction "
        "from the trained machine-learning model."
    )

    st.markdown("### How to use")
    st.write("1. Enter the information.")
    st.write("2. Click **Predict**.")
    st.write("3. Review the model result.")

    st.divider()
    st.caption("Educational demonstration only. Not a medical diagnosis.")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>💙 Heart Disease Prediction</h1>
    <p>A machine-learning demonstration with a clean, easy-to-use medical dashboard.</p>
</div>
""", unsafe_allow_html=True)

# Small information cards
info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🩺 Health inputs</div>
        <div class="section-caption">Enter the available measurements below.</div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🤖 ML prediction</div>
        <div class="section-caption">Uses your saved trained model.</div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🔒 Your session</div>
        <div class="section-caption">Use the result for learning, not diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- INPUT FORM ----------------
st.markdown("### Patient information")
st.caption("Complete the fields below, then submit to run the model.")

with st.form("heart_prediction_form"):
    # Personal and general measurements
    st.markdown("""
    <div class="section-title">👤 Basic information</div>
    <div class="section-caption">Age and sex</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 18, 100, 40)

    with col2:
        sex = st.selectbox("Sex", ["M", "F"],
                           format_func=lambda x: "Male" if x == "M" else "Female")

    st.divider()

    # Chest pain
    st.markdown("""
    <div class="section-title">❤️ Symptoms and blood pressure</div>
    <div class="section-caption">Select the available clinical measurements.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        chest_pain = st.selectbox(
            "Chest pain type",
            ["ATA", "NAP", "TA", "ASY"],
            format_func=lambda x: {
                "ATA": "ATA — Atypical angina",
                "NAP": "NAP — Non-anginal pain",
                "TA": "TA — Typical angina",
                "ASY": "ASY — Asymptomatic"
            }[x]
        )

    with col2:
        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=80, max_value=200, value=120
        )

    col1, col2 = st.columns(2)

    with col1:
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=200
        )

    with col2:
        fasting_bs = st.selectbox(
            "Fasting blood sugar above 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.divider()

    # Heart measurements
    st.markdown("""
    <div class="section-title">📊 Heart measurements</div>
    <div class="section-caption">Enter the remaining measurements.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Maximum heart rate (bpm)",
            min_value=60, max_value=220, value=150
        )

    with col2:
        exercise_angina = st.selectbox(
            "Exercise-induced angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )

        oldpeak = st.slider(
            "Oldpeak (ST depression)",
            min_value=0.0, max_value=6.0, value=1.0, step=0.1
        )

    st_slope = st.selectbox(
        "ST slope",
        ["Flat", "Up", "Down"]
    )

    st.write("")
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ---------------- PREDICTION ----------------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    try:
        input_df = pd.DataFrame([raw_input])

        # Add missing one-hot encoded columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep exactly the feature order used during training
        input_df = input_df[expected_columns]

        # Scale input and predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Prediction result")

        if prediction == 1:
            st.error(
                "⚠️ The model predicts a higher likelihood of heart disease."
            )
        else:
            st.success(
                "✅ The model predicts a lower likelihood of heart disease."
            )

        st.caption(
            "This result is generated by a machine-learning model and "
            "does not confirm or rule out heart disease."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed. Please check the input features and saved model compatibility.")
        st.exception(e)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    HeartCare · Educational machine-learning project · Not for medical diagnosis
</div>
""", unsafe_allow_html=True)
