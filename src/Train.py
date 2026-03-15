import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Đọc dữ liệu
df = pd.read_csv("Data/Walmart.csv")

# 2. Xử lý ngày tháng
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# 3. Tạo tập đặc trưng và biến mục tiêu
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'Week', 'Month', 'Year']
X = df[features]
y = df['Weekly_Sales']

# 4. Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Hàm đánh giá mô hình
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    return {'Model': name, 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'MAPE': mape}

# 6. Huấn luyện và đánh giá
results = []

# Linear Regression (chuẩn hóa dữ liệu)
scaler = StandardScaler()
X_train_lr = X_train.copy()
X_test_lr = X_test.copy()
continuous_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
X_train_lr[continuous_features] = scaler.fit_transform(X_train_lr[continuous_features])
X_test_lr[continuous_features] = scaler.transform(X_test_lr[continuous_features])

lr = LinearRegression().fit(X_train_lr, y_train)
y_pred_lr = lr.predict(X_test_lr)
results.append(evaluate_model("Linear Regression", y_test, y_pred_lr))

# GBoost 
xgb = XGBRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results.append(evaluate_model("XGBoost", y_test, y_pred_xgb))

# Random Forest 
rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, y_pred_rf))
