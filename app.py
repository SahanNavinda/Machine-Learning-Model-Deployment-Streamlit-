import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Boston Housing Price Predictor", layout="wide")

@st.cache_data
def load_data():
    # If using built-in dataset, save CSV to data/boston.csv; here load from CSV
    return pd.read_csv('data/boston.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Visualizations", "Model Prediction", "Model Performance", "About"])

df = load_data()
model = None
if page in ["Model Prediction", "Model Performance"]:
    try:
        model = load_model()
    except:
        st.sidebar.error("Failed to load model. Check model.pkl.")
        st.stop()

if page == "Home":
    st.title("Boston Housing Price Predictor")
    st.write("Predict median house prices in Boston neighborhoods using an interactive ML-powered app.")

if page == "Data Explorer":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.subheader("Feature filters")
    numerical = df.select_dtypes(include=np.number).columns.tolist()
    feature = st.selectbox("Select Feature to filter:", numerical, index=numerical.index('RM'))
    vmin, vmax = float(df[feature].min()), float(df[feature].max())
    vrange = st.slider(f"{feature} range", vmin, vmax, (vmin, vmax))
    st.write(f"Rows matching: {((df[feature] >= vrange[0]) & (df[feature] <= vrange[1])).sum()}")
    st.dataframe(df[df[feature].between(*vrange)].head())

if page == "Visualizations":
    st.header("Visualizations")
    fig1 = px.histogram(df, x='MEDV', nbins=30, title='Distribution of Median Value (MEDV)')
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x='LSTAT', y='MEDV', title='MEDV vs % Lower Status (LSTAT)')
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = px.scatter(df, x='RM', y='MEDV', color='PTRATIO', title='MEDV vs RM colored by PTRATIO')
    st.plotly_chart(fig3, use_container_width=True)
    feat = st.selectbox("Select feature to compare with MEDV:", ['RM','LSTAT','PTRATIO','NOX'])
    figx = px.scatter(df, x=feat, y='MEDV', trendline='ols', title=f"MEDV vs {feat}")
    st.plotly_chart(figx, use_container_width=True)

if page == "Model Prediction":
    st.header("Predict House Price")
    st.write("Enter feature values to estimate the median housing price (in $1000s).")
    with st.form("predict_form"):
        inputs = {}
        for col in df.columns.drop('MEDV'):
            inputs[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].median()))
        submitted = st.form_submit_button("Predict")
    if submitted:
        sample = pd.DataFrame([inputs])
        with st.spinner("Predicting..."):
            try:
                price = model.predict(sample)[0]
                st.success(f"Predicted median house price: ${price*1000:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if page == "Model Performance":
    st.header("Model Performance")
    st.write("Metrics from model training")
    # Provide numeric scores manually or recompute
    st.write("""
    - Cross-validated RMSE, R² for each model (see notebook)
    - Final test RMSE: *…*  
    - Final test R²: *…*
    """)
    # Optionally, plot scatter of y_test vs y_pred if you load test set
    try:
        test = pd.read_csv('data/boston_test.csv')
        y_true = test['MEDV']
        X_test = test.drop(columns=['MEDV'])
        y_pred = model.predict(X_test)
        fig = px.scatter(x=y_true, y=y_pred, labels={'x':'True MEDV','y':'Predicted MEDV'}, title='True vs Predicted')
        st.plotly_chart(fig, use_container_width=True)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        st.write(f"Test RMSE: {rmse:.3f}")
        st.write(f"Test R²: {r2:.3f}")
    except Exception:
        st.info("To show test performance visually, include `data/boston_test.csv` with MEDV column.")

if page == "About":
    st.header("About this project")
    st.markdown("""
    - **Dataset**: Boston Housing (from scikit-learn)
    - **Models**: Linear Regression, Random Forest Regressor
    - **Goal**: Predict median house prices using neighborhood features.
    - App created with Streamlit, trained with scikit-learn.
    """)
