from detection import detect_track
from predict import predict_traffic_flow

if __name__ == "__main__":
    # Detect and track objects in the video
    try:
        A_D, A_F, E_D, E_B, C_B = detect_track('blr.mp4')
    except Exception as e:
        print(f"An error occurred: {e}")

    # Predict traffic flow
    try:
        timestamps = list(range(1, 63))  #Assuming 1-second intervals for 62 seconds
        predict_interval = 90  #Predict 1.5 minutes ahead (90 seconds)
        predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)
        print(predicted_traffic)
    except Exception as e:
        print(f"An error occurred: {e}")