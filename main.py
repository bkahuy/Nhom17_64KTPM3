from datetime import date
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from numpy import double
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
import io, os
import base64

#uvicorn main:app --reload

# Khởi tạo ứng dụng API
app = FastAPI()

# Rest of your code...
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Đọc dữ liệu từ tệp CSV và xử lý dữ liệu
df = pd.read_csv('giavangnew.csv')
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol", "Change%"]

# 1. Biểu đồ phân bổ dữ liệu:
# Tạo dữ liệu mẫu để tạo bảng phân bổ dữ liệu
data = {
    "Date": df["Date"].values,
    "Price": df["Price"].values,
    "Open": df["Open"].values,
    "High": df["High"].values,
    "Low": df["Low"].values,
    "Vol": df["Vol"].values,
    "Change%": df["Change%"].values
}
# Chuyển đổi thành DataFrame
df = pd.DataFrame(data)

# Chuyển cột Date thành định dạng datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sử dụng các cột Open và Vol
df[['Price', 'Open', 'High', 'Low']] = df[['Price', 'Open', 'High', 'Low']].replace({',': ''}, regex=True)
df['Vol'] = df['Vol'].replace({'K': ''}, regex=True).astype(float) * 1000
df.drop(['Change%'], axis=1, inplace=True)
# Biểu đồ Pair Plot cho tất cả các biến
sns.pairplot(df[["Price", "Open", "High", "Low", "Vol"]])
plt.show()

df.dropna(inplace=True)
df.head()
x = df[['Open', 'High', 'Low', 'Vol']].values
y = df['Price'].values



# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Chia dữ liệu thành tập huấn luyện và kiểm tra (70% - 30%)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình MLPRegressor
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42, verbose=True)

# Huấn luyện và lưu lại lịch sử hội tụ
nn_model.fit(x_train, y_train)

# Truy cập vào lịch sử hội tụ (hàm mất mát tại mỗi lần lặp)
loss_values = nn_model.loss_curve_

# Vẽ đồ thị hội tụ của hàm mất mát
plt.plot(loss_values)
plt.title("Độ hội tụ của hàm mất mát theo số vòng lặp")
plt.xlabel("Số vòng lặp")
plt.ylabel("Hàm mất mát (Loss)")
plt.grid(True)
plt.show()

# Vẽ biểu đồ MSE giữa giá trị thực và giá trị dự đoán
y_pred_nn = nn_model.predict(x_test)
plt.scatter(y_test, y_pred_nn)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")  # Đường lý tưởng (y_test = y_pred)
plt.title("Biểu đồ MSE giữa giá trị thực và dự đoán")
plt.xlabel("Giá trị thực (Actual Prices)")
plt.ylabel("Giá trị dự đoán (Predicted Prices)")
plt.show() 


# 1. Hồi quy tuyến tính
lr_model = LinearRegression()

lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 2. Hồi quy Lasso
lasso_model = Lasso(alpha=0.1)

lasso_model.fit(x_train, y_train)
y_pred_lasso = lasso_model.predict(x_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# 3. Mạng nơ-ron
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42, verbose=True)

nn_model.fit(x_train, y_train)
y_pred_nn = nn_model.predict(x_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# 4. Bagging Regressor
bagging_model = BaggingRegressor(n_estimators=100, random_state=42)

bagging_model.fit(x_train, y_train)
y_pred_bagging = bagging_model.predict(x_test)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    open: float
    high: float
    low: float
    vol: float

# Hàm dự đoán giá vàng cho các mô hình

def predict_price_lr(open_value: float, high_value: float, low_value: float, vol_value: float) -> float:
    input_data = scaler.transform([[open_value, high_value, low_value, vol_value]])
    return lr_model.predict(input_data)[0]

def predict_price_lasso(open_value: float, high_value: float, low_value: float, vol_value: float) -> float:
    input_data = scaler.transform([[open_value, high_value, low_value, vol_value]])
    return lasso_model.predict(input_data)[0]

def predict_price_nn(open_value: float, high_value: float, low_value: float, vol_value: float) -> float:
    input_data = scaler.transform([[open_value, high_value, low_value, vol_value]])
    return nn_model.predict(input_data)[0]

def predict_price_bagging(open_value: float, high_value: float, low_value: float, vol_value: float) -> float:
    input_data = scaler.transform([[open_value, high_value, low_value, vol_value]])
    return bagging_model.predict(input_data)[0]

# Endpoint dự đoán
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open
    high_value = input_data.high
    low_value = input_data.low
    vol_value = input_data.vol

    try:
        # high_value, low_value = get_high_low(open_value, vol_value)

        # Dự đoán giá vàng
        predicted_lr = predict_price_lr(open_value, high_value, low_value, vol_value)
        predicted_lasso = predict_price_lasso(open_value, high_value, low_value, vol_value)
        predicted_nn = predict_price_nn(open_value, high_value, low_value, vol_value)
        predicted_bagging = predict_price_bagging(open_value, high_value, low_value, vol_value)
        return {
            "predicted_lr": predicted_lr,
            "predicted_lasso": predicted_lasso,
            "predicted_nn": predicted_nn,
            "predicted_bagging": predicted_bagging,
            "mse_lr": mse_lr,
            "r2_lr": r2_lr,
            "mse_lasso": mse_lasso,
            "r2_lasso": r2_lasso,
            "mse_nn": mse_nn,
            "r2_nn": r2_nn,
            "mse_bagging": mse_bagging,
            "r2_bagging": r2_bagging
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
                    <label for="high">Giá cao nhất:</label>
                    <input type="number" id="high" name="high" class="form-control" step="any" required>
                </div>
                <div class="form-group">
                    <label for="low">Giá thấp nhất:</label>
                    <input type="number" id="low" name="low" class="form-control" step="any" required>
                </div>
                <div class="form-group">
                    <label for="vol">Khối lượng vàng giao dịch:</label>
                    <input type="number" id="vol" name="vol" class="form-control" step="any" required>
                </div>
                <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
            </form>
            <div id="result" class="mt-3"></div>
        </div>
        <script>
        async function predict() {
            const open = document.getElementById('open').value;
            const high = document.getElementById('high').value;
            const low = document.getElementById('low').value;
            const vol = document.getElementById('vol').value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    open: parseFloat(open),
                    high: parseFloat(high),
                    low: parseFloat(low),
                    vol: parseFloat(vol),
                }),
            });

            const data = await response.json();
            if (data.error) {
                document.getElementById('result').innerHTML = `<p class="text-danger">${data.error}</p>`;
            } else {
                document.getElementById('result').innerHTML = `
                    <h4>Kết quả dự đoán:</h4>
                    <p><strong>Giá vàng dự đoán theo Hồi quy tuyến tính:</strong> ${data.predicted_lr.toFixed(2)} USD</p>
                    <p><strong>Giá vàng dự đoán theo Hồi quy Lasso:</strong> ${data.predicted_lasso.toFixed(2)} USD</p>
                    <p><strong>Giá vàng dự đoán theo Neural Network:</strong> ${data.predicted_nn.toFixed(2)} USD</p>
                    <p><strong>Giá vàng dự đoán theo Bagging:</strong> ${data.predicted_bagging.toFixed(2)} USD</p>
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
        }
        </script>
    </body>
    </html>
    """
