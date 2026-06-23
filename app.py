import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Illinois House Predictor", page_icon="🏠", layout="centered")

# 2. Custom styling -- dusty rose & plum design system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Quicksand:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stMarkdownContainer"] p,
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'Quicksand', sans-serif !important;
    color: #E7E4F2;
}

.stApp {
    background: linear-gradient(160deg, #0B3142 0%, #0F5257 100%);
}

/* Turn the main content area into the "card" itself */
.main .block-container {
    background: #0E2A38;
    border-radius: 22px;
    padding: 2.4rem 2.6rem 2.8rem !important;
    margin-top: 2rem;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(156, 146, 163, 0.35);
    max-width: 700px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 2.5rem;
    color: #F5F3FA;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: 0.3px;
}

.hero-subtitle {
    text-align: center;
    color: #B4A7E6;
    font-size: 0.98rem;
    margin-bottom: 1.8rem;
    font-weight: 500;
}

.eyebrow {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #B4A7E6;
    font-weight: 700;
    margin-top: 1.4rem;
    margin-bottom: 0.2rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #B4A7E6, transparent);
    border: none;
    margin: 1.8rem 0 1.4rem 0;
    opacity: 0.6;
}

.result-card {
    background: linear-gradient(135deg, #9C92A3 0%, #D6D3F0 100%);
    border-left: 5px solid #0F5257;
    border-radius: 16px;
    padding: 1.5rem 1.7rem 1.3rem;
    margin-top: 1rem;
}

.result-label {
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #0F5257;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #0B3142;
    margin: 0;
}

.result-caption {
    font-size: 0.85rem;
    color: #0B3142;
    margin-top: 0.5rem;
    line-height: 1.5;
    opacity: 0.85;
}

div.stButton > button {
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 0.6rem 1.6rem;
    border: none;
    width: 100%;
    background-color: #B4A7E6;
    color: #0B3142;
}

div.stButton > button:hover {
    background-color: #F5F3FA;
    color: #0B3142;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border-radius: 12px;
    border-color: #B4A7E6;
    background-color: #0F5257;
}
</style>
""", unsafe_allow_html=True)

# 3. Load the Illinois Model and Town Map Dictionary
@st.cache_resource
def load_illinois_artifacts():
    model = joblib.load('illinois_housing_model.pkl')
    town_map = joblib.load('illinois_town_map.pkl')
    meta = joblib.load('town_map_meta.pkl')
    return model, town_map, meta

model, illinois_town_map, town_map_meta = load_illinois_artifacts()

# 4. Header
st.markdown('<div class="hero-title">🏠 Illinois House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A data-driven estimate built on 300,000+ verified Cook County home sales</div>',
    unsafe_allow_html=True,
)

# 5. Location
st.markdown('<div class="eyebrow">Location</div>', unsafe_allow_html=True)

town_options = {
    "Schaumburg (Township 35)": 35,
    "Elk Grove Village (Township 16)": 16,
    "Palatine (Township 29)": 29,
    "Northfield / Glenview area (Township 25)": 25,
    "Chicago - Hyde Park / South Side (Township 70)": 70,
    "Other / Countywide Average": "AVERAGE",
}

selected_town_label = st.selectbox("Submarket", options=list(town_options.keys()), label_visibility="collapsed")
chosen_code = town_options[selected_town_label]

if chosen_code == "AVERAGE":
    town_price_per_sf = town_map_meta['global_mean']
else:
    town_price_per_sf = illinois_town_map.get(chosen_code, town_map_meta['global_mean'])

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# 6. Structure
st.markdown('<div class="eyebrow">Size & Layout</div>', unsafe_allow_html=True)

total_sf = st.slider("Total Building Square Footage", min_value=500, max_value=6000, value=1800, step=50)

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3, step=1)
with col2:
    total_baths = st.slider("Total Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# 7. Condition
st.markdown('<div class="eyebrow">Condition</div>', unsafe_allow_html=True)
house_age = st.slider("Age of the Property (Years)", min_value=0, max_value=150, value=25, step=1)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# 8. Run Prediction Pipeline
if st.button("Calculate Valuation"):

    input_data = {
        'BuildingSF': [total_sf],
        'TotalBaths': [total_baths],
        'Bedrooms': [bedrooms],
        'HouseAge': [house_age],
        'Town_PricePerSF': [town_price_per_sf],
    }
    input_df = pd.DataFrame(input_data)

    predicted_log_price = model.predict(input_df)
    final_dollar_price = np.expm1(predicted_log_price)[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Market Value</div>
        <div class="result-amount">${final_dollar_price:,.0f}</div>
        <div class="result-caption">
            Anchored to a local submarket weight of ${town_price_per_sf:.2f}/sq. ft.<br>
            Typical model error on held-out test data: ~26% (median). Treat this as a ballpark estimate, not an appraisal.
        </div>
    </div>
    """, unsafe_allow_html=True)
