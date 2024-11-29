from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval):
    # Ensure that counts and timestamps are aligned and not empty
    if not all([len(timestamps) == len(A_D), len(timestamps) == len(A_F),
                len(timestamps) == len(E_D), len(timestamps) == len(E_B),
                len(timestamps) == len(C_B)]):
        raise ValueError("Timestamps and counts must have the same length.")
    
    counts_over_time = {
        "A_D": A_D,
        "A_F": A_F,
        "E_D": E_D,
        "E_B": E_B,
        "C_B": C_B
    }

    future_counts = {}
    time_array = np.array(timestamps).reshape(-1, 1)  # Ensure time_array is 2D

    for route, counts in counts_over_time.items():
        counts_array = np.array(counts)
        
        # Ensure counts_array is 1D (flatten it if necessary)
        if counts_array.ndim > 1:
            counts_array = counts_array.flatten()

        # Initialize the linear regression model
        model = LinearRegression()
        
        # Fit the model: X is time_array, y is counts_array
        model.fit(time_array, counts_array)
        
        # Predict future count based on the last timestamp + predict_interval
        future_time = np.array([[timestamps[-1] + predict_interval]])
        prediction = model.predict(future_time)
        
        # Store the prediction (ensure no negative counts)
        future_counts[route] = max(0, int(prediction[0]))

    return future_counts