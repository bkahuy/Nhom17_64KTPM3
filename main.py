from datetime import date
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64



#uvicorn main:app --reload

# Khởi tạo ứng dụng API
app = FastAPI()

# Đọc dữ liệu từ tệp CSV và xử lý
df = pd.read_csv("giavang.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# Chuyển cột Date thành định dạng datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Chuyển đổi cột Date thành số ngày kể từ ngày đầu tiên trong dataset
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days

# Biểu đồ Pair Plot cho tất cả các biến
sns.pairplot(df[["Price", "Open", "Vol"]])
plt.show()

# Chuyển đổi cột Date thành số ngày kể từ ngày đầu tiên trong dataset
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days

# Sử dụng các cột Open, Vol, và số ngày từ đầu dataset
x = df[['Open', 'Vol', 'Days_Since_Start']].values
y = df['Price'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# 1. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 2. Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(x_train, y_train)

y_pred_gb = gb_model.predict(x_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# 3. SVR (Support Vector Regression)
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_model.fit(x_train, y_train)
y_pred_svr = svr_model.predict(x_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# 4. Hồi quy tuyến tính
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 5. Hồi quy Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
y_pred_lasso = lasso_model.predict(x_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# 6. Mạng nơ-ron
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(x_train, y_train)
y_pred_nn = nn_model.predict(x_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# 7. Bagging Regressor
bagging_model = BaggingRegressor(n_estimators=100, random_state=42)
bagging_model.fit(x_train, y_train)
y_pred_bagging = bagging_model.predict(x_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    open: float
    vol: float
    days_since_start: int

# Hàm dự đoán giá vàng cho các mô hình
def predict_price_rf(open_value: float, vol_value: float, days_since_start: int) -> float:
    input_data = scaler.transform([[open_value, vol_value, days_since_start]])
    return rf_model.predict(input_data)[0]

def predict_price_gb(open_value: float, vol_value: float, days_since_start: int) -> float:
    input_data = scaler.transform([[open_value, vol_value, days_since_start]])
    return gb_model.predict(input_data)[0]

def predict_price_svr(open_value: float, vol_value: float, days_since_start: int) -> float:
    input_data = scaler.transform([[open_value, vol_value, days_since_start]])
    return svr_model.predict(input_data)[0]

def predict_price_bagging(open_value: float, vol_value: float, days_since_start: int) -> float:
    input_data = scaler.transform([[open_value, vol_value, days_since_start]])
    return bagging_model.predict(input_data)[0]

# Endpoint dự đoán
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open
    vol_value = input_data.vol
    days_since_start = input_data.days_since_start

    try:
        # Dự đoán giá vàng
        predicted_rf = predict_price_rf(open_value, vol_value, days_since_start)
        predicted_gb = predict_price_gb(open_value, vol_value, days_since_start)
        predicted_svr = predict_price_svr(open_value, vol_value, days_since_start)
        predicted_bagging = predict_price_bagging(open_value, vol_value, days_since_start)

        return {
            "predicted_rf": predicted_rf,
            "predicted_gb": predicted_gb,
            "predicted_svr": predicted_svr,
            "predicted_bagging": predicted_bagging,
            "mse_rf": mse_rf,
            "r2_rf": r2_rf,
            "mse_gb": mse_gb,
            "r2_gb": r2_gb,
            "mse_svr": mse_svr,
            "r2_svr": r2_svr,
            "mse_bagging": mse_bagging,
            "r2_bagging": r2_bagging,
            "mse_lr": mse_lr,
            "r2_lr": r2_lr,
            "mse_lasso": mse_lasso,
            "r2_lasso": r2_lasso,
            "mse_nn": mse_nn,
            "r2_nn": r2_nn
        }
    except Exception as e:
        return {"error": str(e)}

# Trang chính với form nhập liệu
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dự đoán giá vàng</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-color: #f8f8f8;
            }
            .container {
                width: 600px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="text-center">Dự đoán giá vàng nhóm 17</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="open">Giá mở cửa:</label>
                    <input type="number" id="open" name="open" class="form-control" step="any" required>
                </div>
                <div class="form-group">
                    <label for="vol">Khối lượng vàng giao dịch:</label>
                    <input type="number" id="vol" name="vol" class="form-control" step="any" required>
                </div>
                <div class="form-group">
                    <label for="days_since_start">Số ngày từ ngày đầu tiên:</label>
                    <input type="number" id="days_since_start" name="days_since_start" class="form-control" step="1" required>
                </div>
                <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
            </form>
            <div id="result" class="mt-3"></div>
        </div>
        <script>
        async function predict() {
            const open = document.getElementById('open').value;
            const vol = document.getElementById('vol').value;
            const days_since_start = document.getElementById('days_since_start').value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    open: parseFloat(open),
                    vol: parseFloat(vol),
                    days_since_start: parseInt(days_since_start),
                }),
            });

        const data = await response.json();
        document.getElementById('result').innerHTML = `
            <h4>Kết quả dự đoán:</h4>
            <p><strong>Giá vàng dự đoán theo Hồi quy tuyến tính:</strong> ${data.predicted_rf.toFixed(2)} USD</p>
            <p><strong>Giá vàng dự đoán theo Hồi quy Lasso:</strong> ${data.predicted_gb.toFixed(2)} USD</p>
            <p><strong>Giá vàng dự đoán theo Neural Network:</strong> ${data.predicted_svr.toFixed(2)} USD</p>
            <hr>
            <h5>Hồi quy tuyến tính:</h5>
            <p><strong>MSE:</strong> ${data.mse_lr.toFixed(4)}</p>
            <p><strong>R²:</strong> ${data.r2_lr.toFixed(4)}</p>
            <h5>Hồi quy Lasso:</h5>
            <p><strong>MSE:</strong> ${data.mse_lasso.toFixed(4)}</p>
            <p><strong>R²:</strong> ${data.r2_lasso.toFixed(4)}</p>
            <h5>Mạng nơ-ron:</h5>
            <p><strong>MSE:</strong> ${data.mse_nn.toFixed(4)}</p>
            <p><strong>R²:</strong> ${data.r2_nn.toFixed(4)}</p>
            <h5>Bagging:</h5>
            <p><strong>MSE:</strong> ${data.mse_bagging.toFixed(4)}</p>
            <p><strong>R²:</strong> ${data.r2_bagging.toFixed(4)}</p>
        `;
    }
    </script>

    </body>
    </html>
    """

