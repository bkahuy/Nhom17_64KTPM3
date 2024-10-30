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