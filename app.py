import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Sydney Property Price Predictor", page_icon="🏠")

@st.cache_resource
def train_model():
    df = pd.read_csv("SIT720_8.1D_Recent_Sydney_Housing_120_Properties.csv")
    df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])
    df["Sale_Month"] = df["Sale_Date"].dt.month
    features = ["Suburb","Property_Type","Bedrooms","Bathrooms","Parking","Land_Area_sqm","Sale_Month"]
    X, y = df[features], df["Sale_Price_AUD"]
    numeric = ["Bedrooms","Bathrooms","Parking","Land_Area_sqm","Sale_Month"]
    categorical = ["Suburb","Property_Type"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical)
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ])
    model.fit(X, y)
    return model, sorted(df["Property_Type"].dropna().unique())

model, types = train_model()
st.title("Sydney Property Price Predictor")
st.write("Enter property details to generate an estimated sale price using the Gradient Boosting model.")

suburb = st.selectbox("Suburb", ["Blacktown","Balmain","Bellevue Hill"])
ptype = st.selectbox("Property type", types)
bedrooms = st.number_input("Bedrooms", 0, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
parking = st.number_input("Parking spaces", 0, 10, 1)
land = st.number_input("Land area (m²)", 0.0, 5000.0, 500.0, 10.0)
month = st.selectbox("Sale month", list(range(1,13)))

if st.button("Predict Sale Price", type="primary"):
    x = pd.DataFrame([{
        "Suburb":suburb, "Property_Type":ptype, "Bedrooms":bedrooms,
        "Bathrooms":bathrooms, "Parking":parking,
        "Land_Area_sqm": np.nan if land == 0 else land, "Sale_Month":month
    }])
    pred = max(float(model.predict(x)[0]), 0)
    st.success(f"Estimated Sale Price: ${pred:,.0f}")
    st.caption("Project estimate only; not a professional property valuation.")
