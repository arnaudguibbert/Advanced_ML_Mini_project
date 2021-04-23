import numpy as np 
from sklearn.svm import SVR
from data_processing import normalize

class SVRmodel():

    def __init__(self,subdatasets,indexY,range_C,range_gamma,kernel="linear"):
        self.kernel = kernel
        self.range_C = range_C
        self.range_gamma = range_gamma
        self.grid_C, self.grid_gamma = np.meshgrid(range_C,range_gamma)
        self.subdatasets = subdatasets
        self.indexY = indexY
        self.cross_train, self.cross_test = self.cross_validation()
        self.test_grid = np.empty_like(self.grid_C)
        self.train_grid = np.empty_like(self.grid_gamma)
    
    def cross_validation(self):
        k = len(self.subdatasets)
        cross_train = []
        cross_test = []
        for i in range(0,k):
            sub = self.subdatasets.copy()
            cross_test.append(sub.pop(i))
            train = np.concatenate(sub,axis=0)
            cross_train.append(train)
        return cross_train, cross_test

    def SVR_fit(self,data,C,gamma):
        svr_model = SVR(kernel=self.kernel, 
                        C=C, 
                        gamma=gamma, 
                        epsilon=.1)
        y = data[:,self.indexY].reshape(-1,1)
        X = np.delete(data,self.indexY,axis=1)
        index = np.arange(5,X.shape[1])
        X_clean,meanx,stdx = normalize(X,index)
        y_clean,meany,stdy = normalize(y,[0])
        y_clean = y_clean.reshape(-1)
        svr_model.fit(X_clean,y_clean)
        return svr_model, meanx, stdx, meany, stdy

    def compute_loss(self,model,test,meanx,stdx,meany,stdy):
        y = test[:,self.indexY].reshape(-1,1)
        X = np.delete(test,self.indexY,axis=1)
        index = np.arange(5,X.shape[1])
        X_clean,meanx,stdx = normalize(X,index,meanx,stdx)
        y_pred = model.predict(X_clean)
        y_pred = y_pred*stdy + meany
        loss = np.mean((y - y_pred)**2)
        return loss

    def train_models(self):
        for j,C in enumerate(self.range_C):
            for k,gamma in enumerate(self.range_gamma):
                print("C =",C," gamma =",gamma)
                loss_train = np.empty(len(self.cross_train))
                loss_test = np.empty(len(self.cross_test))
                for i,sub in enumerate(self.cross_train):
                    svr_model, meanx, stdx, meany, stdy = self.SVR_fit(sub,C,gamma)
                    test = self.cross_test[i]
                    model_loss_test = self.compute_loss(svr_model,
                                                        test,
                                                        meanx,stdx,meany,stdy)
                    model_loss_train = self.compute_loss(svr_model,
                                                         sub,
                                                         meanx,stdx,meany,stdy)
                    loss_test[i] = model_loss_test
                    loss_train[i] = model_loss_train
                self.train_grid[k,j] = np.mean(loss_train)
                self.test_grid[k,j] = np.mean(loss_test)
            

            


    