from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd


def Mynet(X,y,hidden,ratio=0.3,max_iter=1000,seed=0,return_MLP=False):
    hidden = tuple(hidden)
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        random_state=seed,
                                                        test_size=ratio)
    MLP = MLPRegressor(hidden_layer_sizes=hidden,
                       random_state=seed, 
                       max_iter=max_iter)
    std_x = np.std(X_train[:,5:],axis=0,keepdims=True)
    #std_y = np.std(y_train[:,:5],axis=1,keepdims=True)
    X_train[:,5:] = (X_train[:,5:] - np.mean(X_train[:,5:],axis=0,keepdims=True))/std_x
    y_train = y_train - np.mean(y_train,axis=0,keepdims=True)
    X_test[:,5:] = (X_test[:,5:] - np.mean(X_train[:,5:],axis=0,keepdims=True))/std_x
    y_test = y_test - np.mean(y_train,axis=0,keepdims=True)
    MLP.fit(X_train,y_train)
    if return_MLP:
        return MLP
    else:
        predicted_train = MLP.predict(X_train)
        loss_train = np.mean(np.sum((y_train - predicted_train)**2,axis=1))
        predicted = MLP.predict(X_test)
        loss_test = np.mean(np.sum((y_test - predicted)**2,axis=1))
        return loss_test, loss_train

class cross_validation_net():

    def __init__(self,model,start,stop,step,runs=10,max_iter=1000,ratio=0.3,layers=3):
        self.model = model
        self.runs = runs
        self.scores = []
        hidden_size = np.arange(stop,start,step).reshape(-1,1).astype(int)
        hidden = np.tile(hidden_size,(1,layers))
        self.hidden = list(hidden)

    def assess(self,X,y):
        args = [[X,y,tuple(hidden)] for hidden in self.hidden]
        for _ in range(0,self.runs):
            results = [self.model(myargs) for myargs in args]
            self.scores.append(results)
        self.scores = np.array(self.scores).reshape(-1,1)

    def plot_graph(self):
        labels = np.array([str(tuple(hidden)) for hidden in self.hidden]).reshape(-1,1)
        labels = np.tile(labels,(self.runs,10))
        data_score = np.concatenate((labels,self.scores),axis=1)
        columns = ["Network parameters","Loss"]
        df_scores = pd.DataFrame(data_score,columns=columns)
        sns.boxplot(data=df_scores,y="Loss",x="Network parameters")
        plt.show()
        return 0
