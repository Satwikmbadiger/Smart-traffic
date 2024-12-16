from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval):
    # Ensure inputs are lists and have the same length as timestamps
    def ensure_list(var):
        return [var] * len(timestamps) if isinstance(var, int) else var
    
    # Ensure all input arrays are the same length
    A_D, A_F, E_D, E_B, C_B = map(ensure_list, [A_D, A_F, E_D, E_B, C_B])
    
    # Prepare data
    time_array = np.array(timestamps).reshape(-1, 1)
    routes = {'A_D': A_D, 'A_F': A_F, 'E_D': E_D, 'E_B': E_B, 'C_B': C_B}
    future_counts = {}

    # Train model and predict for each route
    for route, counts in routes.items():
        model = LinearRegression()
        model.fit(time_array, np.array(counts))
        future_time = np.array([[timestamps[-1] + predict_interval]])
        future_counts[route] = max(0, round(model.predict(future_time)[0]))

    return future_counts