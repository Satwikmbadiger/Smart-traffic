from detect_track import detect_and_track
from predict import predict_traffic_flow

if __name__ == "__main__":
    # Detect and track objects in video
    traffic_counts = detect_and_track('blr.mp4')

    # Predict traffic flow
    predicted_count = predict_traffic_flow(traffic_counts, forecast_duration_sec=90)