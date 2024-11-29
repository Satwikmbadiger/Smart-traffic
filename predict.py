from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval):
    """
    Predicts future traffic flow for given routes using linear regression.

    Args:
        A_D, A_F, E_D, E_B, C_B: Lists of traffic counts for the respective routes.
        timestamps: List of timestamps corresponding to the traffic counts.
        predict_interval: Time interval to predict into the future.

    Returns:
        A dictionary containing predicted traffic counts for each route.
    """

    # Ensure counts are in list format, even if they are integers
    def ensure_list(var, size):
        if isinstance(var, int):
            return [var] * size
        return var

    # Ensure each count is in a list format (matching the length of timestamps)
    size = len(timestamps)
    A_D = ensure_list(A_D, size)
    A_F = ensure_list(A_F, size)
    E_D = ensure_list(E_D, size)
    E_B = ensure_list(E_B, size)
    C_B = ensure_list(C_B, size)

    # Check if the lengths of all inputs are consistent
    lengths = [len(A_D), len(A_F), len(E_D), len(E_B), len(C_B), len(timestamps)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Input arrays must have the same length. Lengths are: {lengths}")

    # Create a dictionary with routes and their counts over time
    counts_over_time = {
        "A_D": A_D,
        "A_F": A_F,
        "E_D": E_D,
        "E_B": E_B,
        "C_B": C_B
    }

    # Initialize a dictionary to store the predicted traffic flow
    future_counts = {}
    time_array = np.array(timestamps).reshape(-1, 1)  # Reshape timestamps for use in linear regression

    # For each route, fit a linear regression model and predict the next value
    for route, counts in counts_over_time.items():
        counts_array = np.array(counts)

        # Initialize the linear regression model
        model = LinearRegression()

        # Fit the model to the current route's counts and timestamps
        model.fit(time_array, counts_array)

        # Predict the traffic flow at the next time step (current last timestamp + prediction interval)
        future_time = np.array([[timestamps[-1] + predict_interval]])
        prediction = model.predict(future_time)

        # Ensure no negative values and store the prediction
        future_counts[route] = max(0, int(prediction[0]))

    return future_counts