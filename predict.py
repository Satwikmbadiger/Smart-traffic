import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

def predict_traffic_flow(traffic_counts, forecast_duration_sec=90):
    # Forecast duration in terms of frames (assuming 1-second intervals for simplicity)
    arima_model = ARIMA(traffic_counts, order=(1, 1, 1))
    arima_result = arima_model.fit()
    future_counts = arima_result.forecast(steps=forecast_duration_sec)

    # Plot original and predicted traffic counts
    plt.figure(figsize=(10, 5))
    plt.plot(traffic_counts, label="Original Traffic Count")
    plt.plot(range(len(traffic_counts), len(traffic_counts) + forecast_duration_sec), future_counts, label="Predicted Traffic Count")
    plt.xlabel("Frames")
    plt.ylabel("Traffic Count")
    plt.legend()
    plt.title("Traffic Count Prediction")
    plt.show()

    return future_counts