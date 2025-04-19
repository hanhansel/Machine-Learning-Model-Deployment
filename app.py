import streamlit as st
import numpy as np
import pandas as pd
from hotel_booking_model import HotelBookingModel

st.set_page_config(page_title="UTS Model Deployment", layout="centered")
st.title("Hotel Booking Cancellation Prediction")

with st.form("booking_form"):
    no_of_adults = st.slider("Input Number of Adults", 0, 4, 2)
    no_of_children = st.slider("Input Number of Children", 0, 10, 0)
    no_of_weekend_nights = st.slider("Input Weekend Nights", 0, 7, 1)
    no_of_week_nights = st.slider("Input Week Nights", 0, 17, 2)
    type_of_meal_plan = st.selectbox("Select Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    required_car_parking_space = st.selectbox("Select Requires Car Parking?", [0, 1])
    room_type_reserved = st.selectbox("Select Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    lead_time = st.slider("Input Lead Time (days)", 0, 443, 30)
    arrival_year = st.selectbox("Input Arrival Year", [2017, 2018])
    arrival_month = st.slider("Input Arrival Month", 1, 12, 6)
    arrival_date = st.slider("Input Arrival Date", 1, 31, 15)
    market_segment_type = st.selectbox("Select Market Segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
    repeated_guest = st.selectbox("Select Repeated Guest", [0, 1])
    no_of_previous_cancellations = st.slider("Input Previous Cancellations", 0, 13, 0)
    no_of_previous_bookings_not_canceled = st.slider("Input Previous Non-Canceled Bookings", 0, 58, 0)
    avg_price_per_room = st.slider("Input Avg Price per Room", 0.0, 540.0, 75.0)
    no_of_special_requests = st.slider("Input Special Requests", 0, 5, 0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }])

    model = HotelBookingModel()
    model.load_model("best_model.pkl")
    
    input_df['type_of_meal_plan'] = input_df['type_of_meal_plan'].map({
    'Not Selected': 0, 
    'Meal Plan 1': 1, 
    'Meal Plan 2': 2, 
    'Meal Plan 3': 3
    })

    input_df['room_type_reserved'] = input_df['room_type_reserved'].map({
    'Room_Type 1': 1, 
    'Room_Type 2': 2, 
    'Room_Type 3': 3,
    'Room_Type 4': 4, 
    'Room_Type 5': 5, 
    'Room_Type 6': 6, 
    'Room_Type 7': 7
    })

    input_df['market_segment_type'] = input_df['market_segment_type'].map({
    'Corporate': 0, 
    'Offline': 1, 
    'Complementary': 2, 
    'Online': 3, 
    'Aviation': 4
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: **{prediction}**")
