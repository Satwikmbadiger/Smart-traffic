from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval):
    
    counts_over_time = {
        "A_D": A_D,
        "A_F": A_F,
        "E_D": E_D,
        "E_B": E_B,
        "C_B": C_B
    }

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