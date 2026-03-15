import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import datetime

# Load mô hình XGBoost
with open("Model/xgboost_optuna_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Dự đoán doanh thu hàng tuần của siêu thị Walmart")

# Giao diện nhập liệu
store = st.number_input("Store (ID cửa hàng)", min_value=1, step=1, max_value= 45)
holiday_flag_str = st.selectbox("Có phải tuần nghỉ lễ?", options=["Có", "Không"])
temperature = st.number_input("Nhiệt độ trung bình (°F)", min_value= -129, max_value= 134)
fuel_price = st.number_input("Giá xăng (USD)", min_value= 0.0)
cpi = st.number_input("Chỉ số giá tiêu dùng (CPI)", min_value= 0.0)
unemployment = st.number_input("Tỷ lệ thất nghiệp (%)", min_value=0.0, max_value= 100.0)
# Cho người dùng chọn ngày
ngay = st.date_input("Chọn ngày", value=datetime.date(2012, 1, 1), min_value=datetime.date(2010, 1, 1), max_value=datetime.date(2012, 12, 31))

#Chuyển holiday_flag sang dạng số
if holiday_flag_str == "Có":
    holiday_flag = 1
else:
    holiday_flag = 0

# Trích xuất week, month, year
week = ngay.isocalendar()[1]  # Tuần ISO
month = ngay.month
year = ngay.year

# Khi nhấn nút "Dự đoán"
if st.button("Dự đoán doanh số"):
    # Tạo dataframe đầu vào
    input_data = pd.DataFrame([[
        store, holiday_flag,     temperature, fuel_price, cpi,
        unemployment, week, month, year
    ]], columns=[
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
        'Unemployment', 'Week', 'Month', 'Year'
    ])

    # Dự đoán với mô hình
    prediction = model.predict(input_data)
    st.success(f"Dự đoán doanh số: ${prediction[0]:,.2f}")
