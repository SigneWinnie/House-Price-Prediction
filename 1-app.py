import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Using joblib as we did in our notebook
import json
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# PAGE CONFIGURATION

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="https://cdn-icons-png.flaticon.com/512/25/25694.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS STYLING 

custom_css = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .main { background-color: #f5f7fa; }
    h1, h2, h3 { color: #1f3a93; font-weight: 700; }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 20px 0;
    }
    .prediction-price { font-size: 48px; font-weight: bold; margin: 20px 0; }
    .info-box { background: #e8f4f8; border-left: 5px solid #0288d1; padding: 20px; border-radius: 8px; margin: 15px 0; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); text-align: center; border-top: 4px solid #667eea; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1f3a93 0%, #2c5aa0 100%); color: white; }
    [data-testid="stSidebar"] label { color: white !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# LOAD MODEL AND DATA

@st.cache_resource
def load_assets():
    try:
        # These names MUST match what you saved in your Jupyter Notebook
        model = joblib.load('knn_house_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model, scaler, features
    except FileNotFoundError:
        st.error(" Model files not found. Please run your Jupyter Notebook first to generate .pkl files.")
        st.stop()

@st.cache_data
def load_data():
    return pd.read_csv('House-Data.csv')

knn_model, scaler, feature_names = load_assets()
original_data = load_data()


# SIDEBAR - INPUT FEATURES

# Render the sidebar title with explicit white color to ensure contrast on blue background
st.sidebar.markdown("<h2 style='color: #FFFFFF; margin-bottom: 6px;'><i class='fas fa-home'></i> House Features</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Grouping inputs for a better user experience
# Use white subheader text to contrast with the blue sidebar
st.sidebar.markdown("<h4 style='color:#FFFFFF; margin-bottom:4px;'><i class='fas fa-info-circle'></i> Basic Info</h4>", unsafe_allow_html=True)
bedrooms = st.sidebar.slider("Bedrooms", 1, 13, 3)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 8.0, 2.5, 0.25)
floors = st.sidebar.selectbox("Floors", [1, 1.5, 2, 2.5, 3, 3.5])

st.sidebar.markdown("<h4 style='color:#FFFFFF; margin-bottom:4px;'><i class='fas fa-ruler'></i> Space &amp; Quality</h4>", unsafe_allow_html=True)
sqft_living = st.sidebar.number_input("Living Space (sqft)", value=2000)
sqft_lot = st.sidebar.number_input("Lot Size (sqft)", value=5000)
grade = st.sidebar.slider("Building Grade", 1, 13, 7)
condition = st.sidebar.slider("Condition", 1, 5, 3)

st.sidebar.markdown("<h4 style='color:#FFFFFF; margin-bottom:4px;'><i class='fas fa-map-marker-alt'></i> Location &amp; History</h4>", unsafe_allow_html=True)
yr_built = st.sidebar.slider("Year Built", 1900, 2015, 1990)
zipcode = st.sidebar.number_input("Zipcode", value=98101)
latitude = st.sidebar.number_input("Latitude", value=47.51, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=-122.25, format="%.4f")


# LOGIC & PREDICTION


# 1. Create the dictionary (Ordering must match 'features.pkl')
input_dict = {
    'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
    'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': 0, 'view': 0,
    'condition': condition, 'grade': grade, 'sqft_above': sqft_living,
    'sqft_basement': 0, 'yr_built': yr_built, 'yr_renovated': 0,
    'zipcode': zipcode, 'lat': latitude, 'long': longitude,
    'sqft_living15': sqft_living, 'sqft_lot15': sqft_lot
}

# 2. Convert to DataFrame and Scale
input_df = pd.DataFrame([input_dict])
scaled_input = scaler.transform(input_df)

# 3. Predict Price
predicted_price = knn_model.predict(scaled_input)[0]

# 4. Find Similar Houses (The "Optimization" proof)
# This finds real houses from your CSV that are mathematically closest to the user input
X_orig = original_data.drop(['id', 'date', 'price'], axis=1)
X_orig_scaled = scaler.transform(X_orig)
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_orig_scaled)
distances, indices = nn.kneighbors(scaled_input)
similar_houses = original_data.iloc[indices[0]]


# MAIN DISPLAY

st.markdown("<h1 style='text-align: center;'>Smart House Price Predictor</h1>", unsafe_allow_html=True)
st.write(f"<p style='text-align: center;'>Developed by: <b>MBA SIGNE</b></p>", unsafe_allow_html=True)

# Prediction Result Card
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"""
    <div class='prediction-card'>
        <h2>Estimated Market Value</h2>
        <div class='prediction-price'>${predicted_price:,.2f}</div>
        <p>Calculated using Optimized KNN Analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3><i class='fas fa-chart-bar'></i> Model Context</h3>
        <p>Your input is being compared to <b>{len(original_data)}</b> historical sales.</p>
        <p>Optimal K-Value used: <b>11</b></p>
    </div>
    """, unsafe_allow_html=True)

# Tabs Section
tab1, tab2, tab3 = st.tabs(["Similar Properties", "Statistics", "How it Works"])

with tab1:
    st.subheader("Houses similar to your selection")
    st.write("These are the 5 'Nearest Neighbors' found in our database:")
    st.dataframe(similar_houses[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'grade', 'yr_built']])

with tab2:
    st.subheader("Price Distribution")
    fig = px.histogram(original_data, x="price", title="Market Price Overview")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    ### <i class='fas fa-code'></i> How KNN Works
    KNN finds the **K** houses most similar to yours based on the features you selected. 
    It then averages their prices (weighted by distance) to give you an estimate.
    
    ### <i class='fas fa-cogs'></i> Why this is Optimized
    - **Standardization:** All inputs are scaled so that 'sqft' doesn't outweigh 'bedrooms'.
    - **Distance Weighting:** Houses that are extremely similar have more influence than those slightly less similar.
    """, unsafe_allow_html=True)
    