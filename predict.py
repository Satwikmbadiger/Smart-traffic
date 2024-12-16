from sklearn.linear_model import LinearRegression
import numpy as np

def predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval):
    #check input is list and have same time stamp length
    def ensure_list(var):
        return [var] * len(timestamps) if isinstance(var, int) else var
    
    #check if input arrays ahve same length
    A_D, A_F, E_D, E_B, C_B = map(ensure_list, [A_D, A_F, E_D, E_B, C_B])
    
    #Prepare data
    time_array = np.array(timestamps).reshape(-1, 1)
    routes = {'A_D': A_D, 'A_F': A_F, 'E_D': E_D, 'E_B': E_B, 'C_B': C_B}
    future_counts = {}

    #train and predict model
    for route, counts in routes.items():
        model = LinearRegression()
        model.fit(time_array, np.array(counts))
        future_time = np.array([[timestamps[-1] + predict_interval]])
        future_counts[route] = max(0, round(model.predict(future_time)[0]))

    return future_counts

'''
#sample data
A_D = [250, 180, 300, 120, 200]  # Traffic flow for Route A_D
A_F = [220, 80, 280, 150, 200]   # Traffic flow for Route A_F
E_D = [210, 170, 130, 270, 90]   # Traffic flow for Route E_D
E_B = [140, 230, 190, 290, 110]  # Traffic flow for Route E_B
C_B = [160, 70, 260, 200, 110]   # Traffic flow for Route C_B

timestamps = [30, 60, 90, 120, 150]  # Timestamps in seconds

predict_interval = 30  # Predict 30 seconds ahead

predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)

print(predicted_traffic)
'''

'''
A_D = [100]
A_F = [80]
E_D = [90]
E_B = [110]
C_B = [70]

#timestamps = [30] #list(range(1, len(A_D) + 1))

predict_interval = 60

predicted_traffic = predict_traffic_flow(A_D, A_F, E_D, E_B, C_B, timestamps, predict_interval)

print(predicted_traffic)
'''