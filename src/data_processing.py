import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import KNeighborsClassifier as KNN

def normalize(data,mean=None,std=None):
    if mean is None:
        mean = np.mean(data[:,1:],axis=0,keepdims=True)
    if std is None: 
        std = np.std(data[:,1:],axis=0,keepdims=True)
    norm_data = data
    norm_data[:,1:] = (data[:,1:] - mean)/std
    return norm_data, mean, std

def split_data(data,ratio):
    Classes = np.unique(data[:,0]).tolist()
    train_set = []
    validation_set = []
    test_set = []
    for myclass in Classes:
        data_class = data[data[:,0] == myclass]
        nb_data = data_class.shape[0]
        permu = np.random.permutation(nb_data)
        index_train = round(nb_data*ratio[0])
        index_validation = index_train + round(nb_data*ratio[1])
        train_set.append(data_class[permu[:index_train]])
        if ratio[1] != 0: 
            validation_set.append(data_class[permu[index_train:index_validation]])
        test_set.append(data_class[permu[index_validation:]])
    train_set = np.concatenate(train_set,axis=0)
    if ratio[1] != 0: 
        validation_set = np.concatenate(validation_set,axis=0)
    test_set = np.concatenate(test_set,axis=0)
    return train_set, validation_set, test_set

def generate_pairplot(data_np,class_name,components_to_show):
    nb_features = data_np.shape[1] - 1
    columns = ["Class"] + ["Component " + str(i) for i in range(1,nb_features+1)]
    data_pd = pd.DataFrame(data_np,columns=columns).iloc[:,:components_to_show+1]
    sns.set_style("darkgrid")
    sns.color_palette("tab10")
    ax = sns.pairplot(data_pd,hue="Class",palette="tab10")

def KNN_classifier(X,y,n_neighbors=10):
    classifier = KNN(n_neighbors=n_neighbors)
    classifier.fit(X,y)
    return classifier

def find_hyper(data,range_components,range_neighbors,method='standard'):
    x, y = np.meshgrid(range_components, range_neighbors)
    metric = np.empty_like(x).astype(float)
    KNN_metric = np.empty_like(x).astype(float)
    data, _ ,_ = normalize(data)
    row_format = '{:<25}{:<25}{:<25}{:<25}' # Define the display format
    header = ["Number of Components","Number of neighbors","Reconstruction error","KNN accuracy"]
    subheader = ["-"*len(head) for head in header]
    print(row_format.format(*header))
    print(row_format.format(*subheader))
    for i,components in enumerate(range_components):
        for j,neighbors in enumerate(range_neighbors):
            eigen_solver = "auto"
            if method == "modified":
                eigen_solver = "dense"
            embedding = LLE(n_components=components,n_neighbors=neighbors,method=method,eigen_solver=eigen_solver)
            temp_data = embedding.fit_transform(data[:,1:])
            temp_data = np.concatenate((data[:,[0]],temp_data),axis=1)
            train_set, _ , test_set = split_data(temp_data,[0.7,0])
            classifier = KNN_classifier(train_set[:,1:],train_set[:,0])
            accuracy = classifier.score(test_set[:,1:],test_set[:,0])
            KNN_metric[j,i] = accuracy
            metric[j,i] = embedding.reconstruction_error_/components
            row = [components,neighbors,metric[j,i],accuracy]
            print(row_format.format(*row))
    return x,y,metric,KNN_metric