import streamlit as st
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
floors = st.number_input("Floors")
year = st.number_input("Year Built")
location = st.number_input("Location (Encoded)")
condition = st.number_input("Condition (Encoded)")
garage = st.number_input("Garage (0 = No, 1 = Yes)")

if st.button("Predict Price"):
    prediction = model.predict([[1001, area, bedrooms, bathrooms, floors, year, location, condition, garage]])
    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")