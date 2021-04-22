import numpy as np 
import pandas as pd 

def extract(data_path):
    data = pd.read_csv(data_path,skiprows=[1])
    data["Class"] = "Class " + data["Class"].astype(str)
    data = data.drop(columns=["User"])
    return data

def data_params(data):
    nb_data_points = data.shape[0]
    features = data.shape[1]
    missing_values = data[(data == "?").any(axis=1)].shape[0]
    Miss_sensor = []
    count_class = data.groupby("Class").count()
    new_cols = {"X0": "Number of data points"}
    class_dist = count_class[["X0"]]
    class_dist = class_dist.rename(columns=new_cols)
    class_dist = class_dist.reset_index()
    for i in range(1,37,3):
        df = data.iloc[:,[i,i+1,i+2]]
        missing_sensor = df[(df == "?").any(axis=1)].shape[0]
        Miss_sensor.append(missing_sensor)
    Sensor_name = ["Sensor " + str(i) for i in range(0,12)]
    Sensor_df = pd.DataFrame(list(zip(Sensor_name,Miss_sensor)),columns=["Sensors","Missing datapoints"])
    return nb_data_points, features, missing_values, Sensor_df, class_dist

def normalize(data,index,mean=None,std=None):
    if mean is None:
        mean = np.mean(data[:,index],axis=0,keepdims=True)
    if std is None: 
        std = np.std(data[:,index],axis=0,keepdims=True)
    norm_data = data
    norm_data[:,index] = (data[:,index] - mean)/std
    return norm_data, mean, std

def one_hot_encoding(data):
    one_hot = pd.get_dummies(data["Class"])
    new_data = pd.concat([one_hot,data],axis=1)
    new_data = new_data.drop(columns=["Class"])
    return new_data

def split_data(data,k,one_hot=True,nb_class=5):
    if one_hot:
        new_data = one_hot_encoding(data)
    subdataset = [[] for i in range(k)]
    for i in range(0,nb_class):
        if isinstance(data,np.ndarray):
            data_class = data[data[:,i] == 1]
        else:
            data_class = new_data[new_data.iloc[:,i] == 1].to_numpy()
        index = np.random.permutation(data_class.shape[0])
        data_class = data_class[index].astype(float)
        subdata = np.array_split(data_class,k,axis=0)
        for j,sub in enumerate(subdataset):
            sub.append(subdata[j])
    subdataset = [np.concatenate(sub,axis=0) for sub in subdataset]
    return subdataset

        
