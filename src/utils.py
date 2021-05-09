import numpy as np 
import pandas as pd 
import seaborn as sns
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVC
from sklearn.metrics import f1_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
import plotly.graph_objects as go
import plotly.express as px


def split_two(data):
    classes = np.unique(data[:,0])
    final_data = []
    for myclass in classes:
        data_class = data[data[:,0] == myclass]
        nb_data = data_class.shape[0]
        permu = np.random.permutation(nb_data)
        data_class = data_class[permu]
        subsets = np.array_split(data_class,2,axis=0)
        final_data.append(subsets[0])
    final_data = np.concatenate(final_data,axis=0)
    return final_data

def normalize(data,mean=None,std=None):
    if mean is None:
        mean = np.mean(data[:,1:],axis=0,keepdims=True)
    if std is None: 
        std = np.std(data[:,1:],axis=0,keepdims=True)
    norm_data = data.copy()
    norm_data[:,1:] = (data[:,1:] - mean)/std
    return norm_data, mean, std

class Assessment():

    def __init__(self,data,range_compo,range_neighbors,
                 method='standard',k=3,run=3,norm=True,KNN_neigh=10,check=True,seed=0):
        warnings.filterwarnings('ignore')
        self.method = method
        self.range_compo = range_compo
        self.range_neighbors = range_neighbors
        self.data = data.copy()
        self.norm = norm
        self.row_format = '{:<12}{:<12}{:<5}{:<25}{:<15}{:<10}{:<15}{:<10}{:<18}' # Define the display format
        self.k = k
        self.run = run 
        self.KNN_neigh = KNN_neigh
        self.seed = seed
        if check:
            self.sanity_check()
        self.x , self.y = np.meshgrid(self.range_compo, self.range_neighbors)
        self.recon_error = []
        self.KNN_accu = np.empty_like(self.x).astype(float)
        self.KNN_F1 = np.empty_like(self.x).astype(float)
        self.KNN_accu_std = np.empty_like(self.x).astype(float)
        self.KNN_F1_std = np.empty_like(self.x).astype(float)
        self.time_perf = []

    def reset(self):
        self.time_perf = []
        self.recon_error = []
        self.KNN_accu = np.empty_like(self.x).astype(float)
        self.KNN_F1 = np.empty_like(self.x).astype(float)
        self.KNN_accu_std = np.empty_like(self.x).astype(float)
        self.KNN_F1_std = np.empty_like(self.x).astype(float)

    def crossksets(self,data):
        classes = np.unique(data[:,0])
        sets = [np.empty((0,data.shape[1])) for i in range(self.k)]
        train_sets = []
        test_sets = []
        for myclass in classes:
            data_class = data[data[:,0] == myclass]
            nb_data = data_class.shape[0]
            permu = np.random.permutation(nb_data)
            data_class = data_class[permu]
            subsets = np.array_split(data_class,self.k,axis=0)
            sets = [np.concatenate((sets[i],subsets[i]),axis=0) for i in range(self.k)]
        for i in range(self.k):
            cp_set = sets.copy()
            test_sets.append(cp_set.pop(i))
            train_sets.append(np.concatenate(cp_set,axis=0))
        return train_sets, test_sets 

    def SVM_metric(self,train_set,test_set):
        X_train, y_train = train_set[:,1:], train_set[:,0]
        X_test, y_test = test_set[:,1:], test_set[:,0]
        classifier = SVC().fit(X_train,y_train)
        accuracy = classifier.score(X_test,y_test)
        y_pred = classifier.predict(X_test)
        F1_measure = f1_score(y_test,y_pred)
        return 100*accuracy, F1_measure

    def KNN_metric(self,train_set,test_set):
        X_train, y_train = train_set[:,1:], train_set[:,0]
        X_test, y_test = test_set[:,1:], test_set[:,0]
        classifier = KNN(n_neighbors=self.KNN_neigh).fit(X_train,y_train)
        accuracy = classifier.score(X_test,y_test)
        y_pred = classifier.predict(X_test)
        F1_measure = f1_score(y_test,y_pred)
        return 100*accuracy, F1_measure

    def sanity_check(self):
        print("Sanity check launched")
        if self.norm:
            mydata,_,_ = normalize(self.data)
        else:
            mydata = self.data.copy()
        index_remove = []
        for j,neighbors in enumerate(self.range_neighbors):
            print("Sanity check for neighbors = ",neighbors)
            try:
                embedding = LLE(n_components=4,
                            n_neighbors=neighbors,
                            method=self.method,
                            max_iter=200,
                            random_state=self.seed,
                            eigen_solver="auto")
                embedding.fit(mydata[:,1:])
            except ValueError:
                try:
                    embedding = LLE(n_components=4,
                                n_neighbors=neighbors,
                                method=self.method,
                                eigen_solver="dense")
                    embedding.fit(mydata[:,1:])
                except ValueError:
                    index_remove.append(j)
                    print("Problem matrix totally singular with neighbors = ",neighbors)
        self.range_neighbors = np.delete(self.range_neighbors,index_remove)

    def find_hyper(self):
        if self.norm:
            mydata,_,_ = normalize(self.data)
        else:
            mydata = self.data.copy()
        header = ["Components",
                  "Neighbors",
                  "Run",
                  "Reconstruction error",
                  "KNN accuracy",
                  "KNN F1",
                  "STD accuracy",
                  "STD F1",
                  "Wall clock time"]
        subheader = ["-"*len(head) for head in header]
        print(self.row_format.format(*header))
        print(self.row_format.format(*subheader))
        for j,neighbors in enumerate(self.range_neighbors):
            for i,components in enumerate(self.range_compo):
                if neighbors < components and self.method == 'modified':
                    self.recon_error.append([1e20,components,neighbors])
                    self.KNN_F1_std[j,i] = 0
                    self.KNN_accu_std[j,i] = 0
                    self.KNN_F1[j,i] = 0
                    self.KNN_accu[j,i] = 0
                    continue
                #reco_error_run = np.empty(self.run)
                KNN_F1_run = np.empty(self.run)
                KNN_accu_run = np.empty(self.run)
                KNN_F1_std_run = np.empty(self.run)
                KNN_accu_std_run = np.empty(self.run)
                for run_index in range(self.run):
                    try:
                        start = perf_counter()
                        embedding = LLE(n_components=components,
                                    n_neighbors=neighbors,
                                    method=self.method,
                                    max_iter=200,
                                    random_state=self.seed,
                                    eigen_solver="auto")
                        reduc_data = embedding.fit_transform(mydata[:,1:])
                        end = perf_counter()
                    except ValueError:
                        start = perf_counter()
                        embedding = LLE(n_components=components,
                                    n_neighbors=neighbors,
                                    method=self.method,
                                    eigen_solver="dense")
                        reduc_data = embedding.fit_transform(mydata[:,1:])
                        end = perf_counter()
                    elapsed = end - start
                    #reco_error_run[run_index] = embedding.reconstruction_error_
                    self.time_perf.append([elapsed,components,neighbors])
                    self.recon_error.append([embedding.reconstruction_error_,components,neighbors])
                    reduc_data = np.concatenate((mydata[:,[0]],reduc_data),axis=1)
                    train_sets, test_sets = self.crossksets(reduc_data)
                    accu_temp_KNN = np.empty(self.k)
                    F1_temp_KNN = np.empty(self.k)
                    for s,train in enumerate(train_sets):
                        accu_temp_KNN[s], F1_temp_KNN[s] = self.KNN_metric(train,test_sets[s])
                    KNN_accu_run[run_index] = np.mean(accu_temp_KNN)
                    KNN_F1_run[run_index] = np.mean(F1_temp_KNN)
                    KNN_accu_std_run[run_index] = np.std(accu_temp_KNN)
                    KNN_F1_std_run[run_index] = np.std(F1_temp_KNN)
                    row = [components,
                           neighbors,
                           run_index,
                           self.recon_error[-1][0],
                           round(KNN_accu_run[run_index],2),
                           round(KNN_F1_run[run_index],2),
                           round(KNN_accu_std_run[run_index],2),
                           round(KNN_F1_std_run[run_index],2),
                           round(elapsed,1)]
                    print(self.row_format.format(*row))
                #self.recon_error.append([float(np.mean(reco_error_run)),components,neighbors])
                self.KNN_F1_std[j,i] = np.mean(KNN_F1_std_run)
                self.KNN_accu_std[j,i] = np.mean(KNN_accu_std_run)
                self.KNN_F1[j,i] = np.mean(KNN_F1_run)
                self.KNN_accu[j,i] = np.mean(KNN_accu_run)

    def plot_cumulative_error(self,title,figsize=[12,7],save_file=None):
        fig = plt.figure(figsize=figsize)
        recon_err_np = np.array(self.recon_error)
        recon_err_np[recon_err_np[:,0] < 0,0] *= -1
        line_keep = min(self.range_neighbors.shape[0],8)
        small_subset_neighbors = np.linspace(0,self.range_neighbors.shape[0]-1,
                                             line_keep,dtype=int)
        small_range_neighbors = self.range_neighbors[small_subset_neighbors]
        print(small_range_neighbors)
        recon_err_np = recon_err_np[np.isin(recon_err_np[:,2],small_range_neighbors)]
        for neighbors in small_range_neighbors:
            err = recon_err_np[(recon_err_np[:,2] == neighbors) & (recon_err_np[:,1] == self.range_compo[-1]),0]
            mean_err = np.mean(err)
            recon_err_np[recon_err_np[:,2] == neighbors,0] /= mean_err
        columns =  ["Cumulative sum of eigenvalues (normalized)",
                    "Eigenvalue index",
                    "Number of neighbors"]
        recon_err_pd = pd.DataFrame(recon_err_np,columns=columns)
        ax = fig.add_subplot(1,1,1)
        sns.set_style("darkgrid")
        sns.lineplot(data=recon_err_pd,
                     x="Eigenvalue index",
                     y="Cumulative sum of eigenvalues (normalized)",
                     hue="Number of neighbors",
                     style="Number of neighbors",
                     palette=sns.color_palette("hls", line_keep),
                     ci="sd",
                     ax=ax)
        ax.set_yscale('log')
        ax.set_xlim(self.range_compo[0],self.range_compo[-1])
        ax.set_title(title,fontsize=16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.legend(fontsize=12,title="Number of neighbors",title_fontsize=12,loc="lower right")
        ax.set_xticks(self.range_compo)
        if save_file is not None:
            fig.savefig("figures/" + save_file + ".svg",dpi=200)
        plt.close("all")

    def generate_pairplot(self,neighbors,components,title,components_to_show=4,save_file=None):
        if self.norm:
            mydata,_,_ = normalize(self.data)
        else:
            mydata = self.data.copy()
        try:
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        max_iter=200,
                        random_state=self.seed,
                        eigen_solver="auto")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        except ValueError:
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        eigen_solver="dense")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        nb_features = reduc_data.shape[1]
        reduc_data = np.concatenate((mydata[:,[0]],reduc_data),axis=1)
        columns = ["Class"] + ["Component " + str(i) for i in range(1,nb_features+1)]
        data_pd = pd.DataFrame(reduc_data,columns=columns).iloc[:,:components_to_show+1] 
        sns.set_style("darkgrid")
        sns.color_palette("tab10")
        pairplot = sns.pairplot(data_pd,hue="Class",palette="tab10")
        handles = pairplot._legend_data.values()
        labels = ['Not Spam', 'Spam']
        pairplot._legend.remove()
        pairplot.fig.legend(handles=handles, 
                            labels=labels, 
                            loc='upper left', 
                            fontsize=13,
                            ncol=1)
        pairplot._legend.set_bbox_to_anchor((0.5, 0.5))
        pairplot.fig.suptitle(title, fontsize=16)
        pairplot.fig.subplots_adjust(top=0.91)
        if save_file:
            pairplot.fig.savefig("figures/" + save_file + ".svg",dpi=200)
        plt.close('all')

    def generate_3Dplot(self,neighbors,title,components=3):
        if self.norm:
            mydata,_,_ = normalize(self.data)
        else:
            mydata = self.data.copy()
        try:
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        max_iter=200,
                        random_state=self.seed,
                        eigen_solver="auto")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        except ValueError:
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        eigen_solver="dense")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        reduc_data = reduc_data[:,:3]
        reduc_data = np.concatenate((mydata[:,[0]],reduc_data),axis=1)
        columns = ["Class"] + ["Component " + str(i) for i in range(1,components+1)]
        data_pd = pd.DataFrame(reduc_data,columns=columns)
        fig = px.scatter_3d(data_pd, 
                            x='Component 1', 
                            y='Component 2', 
                            z='Component 3',
                            color='Class')
        plt.title(title)
        fig.show()

    def plot_time_perf(self,title,save_file=None):
        fig = plt.figure(figsize=[12,7])
        ax = fig.add_subplot(1,1,1)
        time_perf_np = np.array(self.time_perf)
        columns = ["Wall clock time [s]","Components","Neighbors"]
        time_perf_pd = pd.DataFrame(data=time_perf_np,columns=columns)
        sns.set_style("darkgrid")
        sns.lineplot(data=time_perf_pd,
                     x="Neighbors",
                     y="Wall clock time [s]",
                     hue="Components",
                     style="Components",
                     palette=sns.color_palette("hls", self.range_compo[-1]))
        ax.set_xlim(self.range_neighbors[0],self.range_neighbors[-1])
        ax.set_title(title,fontsize=16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.legend(fontsize=12,title="Number of components",title_fontsize=12,loc="upper left")
        if save_file is not None:
            fig.savefig("figures/" + save_file + ".svg",dpi=200)
        plt.close("all")

    def generate_contour(self,z,fig,subplot,title,cmap="viridis"):
        ax = fig.add_subplot(*subplot)
        cs = ax.contourf(self.x,self.y,z,cmap=cmap)
        ax.set_xlabel("Number of components",fontsize=12)
        ax.set_ylabel("Number of neighbors",fontsize=12)
        ax.set_title(title,fontsize=13)
        ax.set_xticks(self.range_compo)
        plt.colorbar(cs,ax=ax)

    def generate_all(self,size=[12,10],save_file=None):
        fig = plt.figure(figsize=size)
        self.generate_contour(self.KNN_accu,
                              fig,[2,2,1],
                              "KNN accuracy")
        self.generate_contour(self.KNN_F1,
                              fig,[2,2,2],
                              "KNN F1 measure")
        self.generate_contour(self.KNN_accu_std,
                              fig,[2,2,3],
                              "KNN accuracy standard deviation",
                              cmap="magma")
        self.generate_contour(self.KNN_F1_std,
                              fig,[2,2,4],
                              "KNN F1 standard deviation",
                              cmap="magma")
        plt.subplots_adjust(hspace=0.4)
        tit = "LLE"
        if self.method == "modified":
            tit = "MLLE"
        fig.suptitle(tit + " additional metrics",fontsize=16)
        if save_file is not None:
            fig.savefig("figures/" + save_file + ".svg",dpi=200)
        plt.close('all')

