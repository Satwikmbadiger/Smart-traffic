import streamlit as st
from detect_ import detect_track
from predict import predict_traffic_flow
import json
import os

# Function to handle file upload and processing
def process_video(video_file):
    # Save the uploaded video to a temporary directory
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    # Call the detection function (you should modify detect_track to return counts)
    try:
        A_D, A_F, E_D, E_B, C_B = detect_track('uploaded_video.mp4')
        return A_D, A_F, E_D, E_B, C_B
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
        return None, None, None, None, None

# Streamlit UI elements
st.title("Traffic Detection and Prediction")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Display the uploaded video
    st.video(video_file)

    # Process the video to get vehicle counts
    A_D, A_F, E_D, E_B, C_B = process_video(video_file)
    
    if A_D and A_F and E_D and E_B and C_B:
        # Display traffic counts
        st.subheader("Traffic Counts for Each Lane:")
        st.write(f"Lane A_D: {A_D}")
        st.write(f"Lane A_F: {A_F}")
        st.write(f"Lane E_D: {E_D}")
        st.write(f"Lane E_B: {E_B}")
        st.write(f"Lane C_B: {C_B}")

        # Generate timestamps (assume 1 second per count for simplicity)
        timestamps = list(range(1, len(A_D) + 1))

        # Set prediction interval (1.5 minutes)
        predict_interval = 90

        # Predict future traffic flow
        try:
            predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)
            st.subheader("Predicted Traffic Flow for Next Interval:")
            st.json(predicted_traffic)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")