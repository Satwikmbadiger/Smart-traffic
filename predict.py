from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic(counts_over_time, timestamps, predict_interval):
    future_counts = {}
    time_array = np.array(timestamps).reshape(-1, 1)

    for route, counts in counts_over_time.items():
        counts_array = np.array(counts)
        model = LinearRegression()
        model.fit(time_array, counts_array)
        future_time = np.array([[timestamps[-1] + predict_interval]])
        prediction = model.predict(future_time)
        future_counts[route] = max(0, int(prediction[0]))

    return future_counts