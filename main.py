from detection import detect_track
from predict import predict_traffic_flow

if __name__ == "__main__":
    # Detect and track objects in the video
    try:
        counts_over_time, timestamps, predict_interval = detect_track('blr.mp4')
    except Exception as e:
        print(f"Error during detection and tracking: {e}")
        counts_over_time, timestamps, predict_interval = {}, [], 0

    # Proceed only if detection was successful
    if counts_over_time and timestamps:
        # Predict traffic flow
        predicted_count = predict_traffic_flow(counts_over_time, timestamps, predict_interval)

        if predicted_count:
            print("Predicted traffic count:", predicted_count)
        else:
            print("Prediction could not be made due to insufficient or inconsistent data.")
    else:
        print("No data available from detection for prediction.")