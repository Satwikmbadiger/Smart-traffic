import streamlit as st
from detection import detect_track
from predict import predict_traffic_flow
import matplotlib.pyplot as plt
import pandas as pd

def process_video(video_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())
    try:
        return detect_track('uploaded_video.mp4')
    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
        return None, None, None, None, None

html_temp = """
<div style="background-color:green;padding:10px">
<h1 style="color:white;text-align:center;">Smart Traffic Solution</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

video_file = st.file_uploader("Upload the clip", type=["mp4", "avi", "mov"])

if st.button("Detect"):
    if video_file is not None:
        st.video(video_file)
        A_D, A_F, E_D, E_B, C_B = process_video(video_file)
        if A_D and A_F and E_D and E_B and C_B:
            st.subheader("Traffic Counts for Each Lane:")
            st.write(f"Lane A_D: {A_D}")
            st.write(f"Lane A_F: {A_F}")
            st.write(f"Lane E_D: {E_D}")
            st.write(f"Lane E_B: {E_B}")
            st.write(f"Lane C_B: {C_B}")

            st.write("### Traffic Counts per Lane")
            try:
                # Ensure all counts are numeric and handle invalid data
                categories = ["A_D", "A_F", "E_D", "E_B", "C_B"]
                counts = int([A_D, A_F, E_D, E_B, C_B])

                # Validate that all values are numeric
                counts = [float(count) if isinstance(count, (int, float)) else 0 for count in counts]

                fig, ax = plt.subplots()
                ax.bar(categories, counts, color='skyblue')
                ax.set_xlabel("Lanes")
                ax.set_ylabel("Counts")
                ax.set_title("Traffic Counts per Lane")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error generating bar chart: {e}")

            try:
                timestamps = [62]
                predict_interval = 90
                predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)
                
                # Debugging step: print the predicted_traffic to check the structure
                #st.write("Predicted Traffic Data:")
                #st.write(predicted_traffic)

                st.subheader("Predicted Traffic Flow for Next Interval:")
                st.write(f"Predicted traffic flow for Route A_D: {predicted_traffic.get('A_D')}")
                st.write(f"Predicted traffic flow for Route A_F: {predicted_traffic.get('A_F')}")
                st.write(f"Predicted traffic flow for Route E_D: {predicted_traffic.get('E_D')}")
                st.write(f"Predicted traffic flow for Route E_B: {predicted_traffic.get('E_B')}")
                st.write(f"Predicted traffic flow for Route C_B: {predicted_traffic.get('C_B')}")

                
                predicted_counts = [
                    predicted_traffic.get('A_D'),
                    predicted_traffic.get('A_F'),
                    predicted_traffic.get('E_D'),
                    predicted_traffic.get('E_B'),
                    predicted_traffic.get('C_B')
                ] 
                
                if isinstance(predicted_counts, list) and all(isinstance(x, (int, float)) for x in predicted_counts):
                    predicted_data = pd.DataFrame({
                        "Lane": ["A_D", "A_F", "E_D", "E_B", "C_B"], 
                        "Predicted Count": predicted_counts
                    })

                    if 'Lane' in predicted_data.columns:
                        st.write("Predicted Traffic Flow per Lane")
                        st.line_chart(predicted_data.set_index("Lane"), use_container_width=True)
                    else:
                        st.error("Predicted data does not contain 'Lane' column.")
                else:
                    st.error("Predicted counts must be a list of numbers.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.write("Data from CSV file:")
        st.dataframe(df)
        
        required_columns = ['Timestamp', 'A_D', 'A_F', 'E_D', 'E_B', 'C_B']
        if all(col in df.columns for col in required_columns):
            A_D = df['A_D'].tolist()
            A_F = df['A_F'].tolist()
            E_D = df['E_D'].tolist()
            E_B = df['E_B'].tolist()
            C_B = df['C_B'].tolist()
            timestamps = df['Timestamp'].tolist()

            predict_interval = st.number_input("Prediction interval (in seconds)", min_value=1, value=60)
            predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)

            # Debugging step: print the predicted_traffic to check the structure
            #st.write("Predicted Traffic Data:")
            #st.write(predicted_traffic)

            st.subheader("Predicted Traffic Flow for Next Interval:")
            st.write(f"Predicted traffic flow for Route A_D: {predicted_traffic.get('A_D')}")
            st.write(f"Predicted traffic flow for Route A_F: {predicted_traffic.get('A_F')}")
            st.write(f"Predicted traffic flow for Route E_D: {predicted_traffic.get('E_D')}")
            st.write(f"Predicted traffic flow for Route E_B: {predicted_traffic.get('E_B')}")
            st.write(f"Predicted traffic flow for Route C_B: {predicted_traffic.get('C_B')}")

            predicted_counts = [
                    predicted_traffic.get('A_D'),
                    predicted_traffic.get('A_F'),
                    predicted_traffic.get('E_D'),
                    predicted_traffic.get('E_B'),
                    predicted_traffic.get('C_B')
                ]   
            if isinstance(predicted_counts, list) and all(isinstance(x, (int, float)) for x in predicted_counts):
                predicted_data = pd.DataFrame({
                    "Lane": ["A_D", "A_F", "E_D", "E_B", "C_B"], 
                    "Predicted Count": predicted_counts
                })
                
                if 'Lane' in predicted_data.columns:
                    st.write("Predicted Traffic Flow per Lane")
                    st.line_chart(predicted_data.set_index("Lane"), use_container_width=True)
                else:
                    st.error("Predicted data does not contain 'Lane' column.")
                

            else:
                st.error("Predicted counts must be a list of numbers.")
        else:
            st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")