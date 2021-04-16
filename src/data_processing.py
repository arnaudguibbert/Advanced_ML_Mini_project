import numpy as np 
import pandas as pd 

def extract(data_path):
    data = pd.read_csv(data_path,skiprows=[1])
    data["Class"] = "Class " + data["Class"].astype(str)
    return data

def data_params(data):
    nb_data_points = data.shape[0]
    features = data.shape[1]
    missing_values = data[(data == "?").any(axis=1)].shape[0]
    Miss_sensor = []
    count_class = data.groupby("Class").count()
    new_cols = {"User": "Number of data points"}
    class_dist = count_class[["User"]]
    class_dist = class_dist.rename(columns=new_cols)
    class_dist = class_dist.reset_index()
    for i in range(2,38,3):
        df = data.iloc[:,[i,i+1,i+2]]
        missing_sensor = df[(df == "?").any(axis=1)].shape[0]
        Miss_sensor.append(missing_sensor)
    Sensor_name = ["Sensor " + str(i) for i in range(0,12)]
    Sensor_df = pd.DataFrame(list(zip(Sensor_name,Miss_sensor)),columns=["Sensors","Missing datapoints"])
    return nb_data_points, features, missing_values, Sensor_df, class_dist

def normalize(data,mean=True,std=False):
    if mean:
        mean_data = np.mean(data,axis=0,keepdims=True)
    else:
        mean_data = 0
    if std: 
        std_data = np.std(data,axis=0,keepdims=True)
    else:
        std_data = 1
    norm_data = (data - mean_data)/std_data
    return norm_data

def complete(data):
    return 0