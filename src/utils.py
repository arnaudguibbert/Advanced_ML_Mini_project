import numpy as np 
import pandas as pd 
import seaborn as sns
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVC
from sklearn.metrics import f1_score
import warnings
import plotly.express as px

def plot_time_comparison(path_LLE_full,path_MLLE_full,path_LLE_semi,path_MLLE_semi,save_file=None):
    fig = plt.figure(figsize=[12,7])
    ax = fig.add_subplot(111)
    data_LLE_full = pd.read_csv(path_LLE_full)
    data_MLLE_full = pd.read_csv(path_MLLE_full)
    data_LLE_semi = pd.read_csv(path_LLE_semi)
    data_MLLE_semi = pd.read_csv(path_MLLE_semi)
    data_LLE_full["type"] = [0]*data_LLE_full.shape[0]
    data_MLLE_full["type"] = [1]*data_MLLE_full.shape[0]
    data_LLE_semi["type"] = [2]*data_LLE_semi.shape[0]
    data_MLLE_semi["type"] = [3]*data_MLLE_semi.shape[0]
    full_data = pd.concat((data_LLE_full,
                           data_LLE_semi,
                           data_MLLE_full,
                           data_MLLE_semi),ignore_index=True)
    sns.set_style("darkgrid")
    sns.lineplot(data=full_data,y="Wall clock time [s]",
                 x="Neighbors",hue="type",ax=ax,
                 palette=sns.color_palette("hls",4))
    handles, labels = ax.get_legend_handles_labels()
    true_labels = ["LLE (whole data set)",
                   "MLLE (whole data set)",
                   "LLE (semi data set)",
                   "MLLE (semi data set)"]
    labels = [true_labels[int(float(i))] for i in labels]
    ax.set_title("Time performances comparison",fontsize=16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.legend(handles,labels,
              fontsize=12,
              title=None,
              title_fontsize=12,loc="upper left")
    ax.set_xlim(20,450)
    if save_file is not None:
        fig.savefig("figures/" + save_file + ".svg")
    

def split_two(data):
    """
    Goal:
    Split randomly the data into two small subsets and keep only the fisrt one
    preserve the class distribution
    Inputs:
    data = np.ndarray - size Nx(D+1) (N number of data points, D dimension of the data points)
           the first column of the data shall be the classes
    Outputs:
    final_data = np.ndarray - size N//2xD (N number of data points, D dimension of the data points)
                 first subset
    """
    classes = np.unique(data[:,0]) # Extract the classes
    final_data = []
    for myclass in classes:
        data_class = data[data[:,0] == myclass] # Get the data of the class
        nb_data = data_class.shape[0] # Number of points
        permu = np.random.permutation(nb_data) # Permutation of the indexes
        data_class = data_class[permu] # Shuffle the data
        subsets = np.array_split(data_class,2,axis=0)
        final_data.append(subsets[0]) 
    final_data = np.concatenate(final_data,axis=0) # Concatenate the data set of each class
    return final_data

def normalize_0_100(data,upper_bound=40):
    data_norm = data.copy()
    min = np.min(data_norm[:,-3:],axis=0,keepdims=True)
    max = np.max(data_norm[:,-3:],axis=0,keepdims=True)
    data_norm[:,-3:] = upper_bound*(data_norm[:,-3:] - min)/(max - min)
    data_norm[:,1:] -= np.mean(data_norm[:,1:],axis=0,keepdims=True)
    return data_norm

def normalize(data,mean=None,std=None):
    """
    Goal:
    Normalize the data - idest substracting the mean, divide by the standard deviation
    Inputs:
    data = np.ndarray - size Nx(D+1) (N number of data points, D dimension of the data points)
           the first column of the data shall be the classes
    mean = np.ndarray - size 1xD (D dimension of the data points)
           if the mean vector is passed, then it will directly use this mean vector instead of computing it
    std = np.ndarray - size 1xD (D dimension of the data points)
          if the std vector is passed, then it will directly use this std vector instead of computing it
    Outputs:
    data = np.ndarray - size Nx(D+1) (N number of data points, D dimension of the data points)
           Normalized data
    mean = np.ndarray - size 1xD (D dimension of the data points)
           Mean of the data, or mean passed as argument
    std = np.ndarray - size 1xD (D dimension of the data points)
          standard deviation of the data, or std passed as argument
    """
    if mean is None:
        mean = np.mean(data[:,1:],axis=0,keepdims=True) # Compute the mean
    if std is None: 
        std = np.std(data[:,1:],axis=0,keepdims=True) # Compute the std
    norm_data = data.copy() 
    norm_data[:,1:] = (data[:,1:] - mean)/std # Normalize
    return norm_data, mean, std

class Assessment():

    def __init__(self,data,range_compo,range_neighbors,
                 method='standard',k=3,run=1,KNN_neigh=10,
                 check=True,seed=0,norm=True,norm_0100=False):
        """
        Goal:
        Inputs:
        data = np.ndarray - size Nx(D+1) (N number of data points, D dimension of the data points)
               the first column of the data shall be the classes 
        range_compo = np.ndarray int - array of the component for which you want to assess the algorithm
        range_neighbors = np.ndarray int - array of the neighbors for which you want to assess the algorithm (hyperparameter)
        method = string - method to be used "standard" -> LLE | "modified" -> MLLE
        k = int - parameter for the cross k validation for KNN
        run = int - number of time you rerun the dimensionality reduction
        norm = Boolean - specify if you want to normalize the data
        KNN_neigh = int - number of neighbors to use for the KNN metric
        check = Boolean - Perform a sanity check if set to True
        seed = int - seed for the random state of the eigen_solver to get reproducibility
        Outputs:
        """
        self.method = method
        self.range_compo = range_compo
        self.range_neighbors = range_neighbors
        self.data = data.copy()
        self.norm = norm
        self.norm_0100 = norm_0100
        self.row_format = '{:<12}{:<12}{:<5}{:<25}{:<15}{:<10}{:<15}{:<10}{:<18}' # Define the display format
        self.k = k
        self.run = run 
        self.KNN_neigh = KNN_neigh
        self.seed = seed
        if check:
            self.sanity_check()
        self.x , self.y = np.meshgrid(self.range_compo, self.range_neighbors)
        self.recon_error = [] # The reconstruction error is stored here
        # The KNN metric will be stored there
        self.KNN_accu = np.empty_like(self.x).astype(float)
        self.KNN_F1 = np.empty_like(self.x).astype(float)
        self.KNN_accu_std = np.empty_like(self.x).astype(float)
        self.KNN_F1_std = np.empty_like(self.x).astype(float)
        # The time performances will be stored there
        self.time_perf = []
        warnings.filterwarnings('ignore')

    def reset(self):
        """
        Goal:
        Reset the attributes of the instance
        Inputs:
        Outputs:
        """
        self.time_perf = []
        self.recon_error = []
        self.KNN_accu = np.empty_like(self.x).astype(float)
        self.KNN_F1 = np.empty_like(self.x).astype(float)
        self.KNN_accu_std = np.empty_like(self.x).astype(float)
        self.KNN_F1_std = np.empty_like(self.x).astype(float)

    def crossksets(self,data):
        """
        Goal:
        Generate k different training/testing sets using cross-k-validation method
        Preserve the classes distribution for each set
        Inputs:
        data = np.ndarray - size Nx(D+1) (N number of data points, D dimension of the data points)
               the first column of the data shall be the classes 
        Outputs:
        """
        classes = np.unique(data[:,0]) # Get the classes
        sets = [np.empty((0,data.shape[1])) for i in range(self.k)]
        train_sets = [] # Training sets will be stored here
        test_sets = [] # Testing sets will be stored here
        for myclass in classes:
            data_class = data[data[:,0] == myclass]
            nb_data = data_class.shape[0]
            permu = np.random.permutation(nb_data) # Shuffle the indexes
            data_class = data_class[permu] # Shuffle the data
            subsets = np.array_split(data_class,self.k,axis=0)
            sets = [np.concatenate((sets[i],subsets[i]),axis=0) for i in range(self.k)]
        for i in range(self.k):
            cp_set = sets.copy()
            # Select one set to be the testing set
            test_sets.append(cp_set.pop(i))
            # Concatenate the other ones to get the training set
            train_sets.append(np.concatenate(cp_set,axis=0))
        return train_sets, test_sets 

    def SVM_metric(self,train_set,test_set):
        """
        Goal:
        Inputs:
        Outputs:
        """
        X_train, y_train = train_set[:,1:], train_set[:,0]
        X_test, y_test = test_set[:,1:], test_set[:,0]
        classifier = SVC().fit(X_train,y_train)
        accuracy = classifier.score(X_test,y_test)
        y_pred = classifier.predict(X_test)
        F1_measure = f1_score(y_test,y_pred)
        return 100*accuracy, F1_measure

    def KNN_metric(self,train_set,test_set):
        """
        Goal:
        Given a training set and a testing set, returns the accuracy and the F1-measure
        on the testing set
        Inputs:
        train_set = np.ndarray - size NtrainxD (Ntrain number of data points for training, D dimension of the data points)
        test_set = np.ndarray - size NtestxD (Ntest number of data points for testing, D dimension of the data points)
        Outputs:
        accuracy = float - Accuracy on the testing set
        F1_measure = float - F1 measure on the testing set
        """
        X_train, y_train = train_set[:,1:], train_set[:,0] # Get the data points and labels for training
        X_test, y_test = test_set[:,1:], test_set[:,0] # Get the data points and labels for testing
        # Initialize the classifier 
        classifier = KNN(n_neighbors=self.KNN_neigh).fit(X_train,y_train)
        accuracy = classifier.score(X_test,y_test) # Compute accuracy on training set
        y_pred = classifier.predict(X_test) 
        F1_measure = f1_score(y_test,y_pred) # Compute F1 measure
        return 100*accuracy, F1_measure

    def sanity_check(self):
        """
        Goal:
        Check that for each neighbors in range_neighbors, the dimensionality reduction is well performed
        actually because of the eigensolvers, sometimes the build in function is not able to compute
        the quantity of interest.
        Inputs:
        Outputs:
        """
        print("Sanity check launched")
        if self.norm: # Normalize the data
            mydata,_,_ = normalize(self.data)
        elif self.norm_0100:
            mydata = normalize_0_100(self.data)
        else:
            mydata = self.data.copy()
        index_remove = [] # Neighbors to remove
        for j,neighbors in enumerate(self.range_neighbors):
            print("Sanity check for neighbors = ",neighbors)
            try:
                # Try to perform dimensionality reduction with arpack solver
                embedding = LLE(n_components=4,
                            n_neighbors=neighbors,
                            method=self.method,
                            max_iter=200,
                            random_state=self.seed,
                            eigen_solver="auto")
                embedding.fit(mydata[:,1:])
            except ValueError:
                try:
                    # If fail try to perform dimensionality reduction with dense solver
                    embedding = LLE(n_components=4,
                                n_neighbors=neighbors,
                                method=self.method,
                                eigen_solver="dense")
                    embedding.fit(mydata[:,1:])
                except ValueError:
                    # If it fails remove the neighors from the list
                    index_remove.append(j)
                    print("Problem matrix totally singular with neighbors = ",neighbors)
        self.range_neighbors = np.delete(self.range_neighbors,index_remove)

    def find_hyper(self):
        """
        Goal:
        Perform a grid search on range_neighbors x range_compo. 
        On each position compute the reconstruction error and an estimation of the other metrics
        Inputs:
        Outputs:
        """
        if self.norm: # Normalize the data
            mydata,_,_ = normalize(self.data)
        elif self.norm_0100:
            mydata = normalize_0_100(self.data)
        else:
            mydata = self.data.copy()
        # Header fo the logs
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
        # Display the header
        print(self.row_format.format(*header))
        print(self.row_format.format(*subheader))
        for j,neighbors in enumerate(self.range_neighbors):
            for i,components in enumerate(self.range_compo):
                # For MLLE neighbors > components
                if neighbors <= components and self.method == 'modified':
                    self.recon_error.append([1e20,components,neighbors])
                    self.KNN_F1_std[j,i] = 0
                    self.KNN_accu_std[j,i] = 0
                    self.KNN_F1[j,i] = 0
                    self.KNN_accu[j,i] = 0
                    continue
                KNN_F1_run = np.empty(self.run)
                KNN_accu_run = np.empty(self.run)
                KNN_F1_std_run = np.empty(self.run)
                KNN_accu_std_run = np.empty(self.run)
                for run_index in range(self.run):
                    try:
                        # Try with "arpack solver"
                        start = perf_counter() # start the chrono
                        embedding = LLE(n_components=components,
                                    n_neighbors=neighbors,
                                    method=self.method,
                                    max_iter=200,
                                    random_state=self.seed,
                                    eigen_solver="auto")
                        reduc_data = embedding.fit_transform(mydata[:,1:]) # Dimensionality reduction
                        end = perf_counter() # Stop the chrono
                    except ValueError:
                        start = perf_counter() # Start the chrono
                        # else Try with "dense" solver 
                        embedding = LLE(n_components=components,
                                    n_neighbors=neighbors,
                                    method=self.method,
                                    eigen_solver="dense")
                        reduc_data = embedding.fit_transform(mydata[:,1:]) # Dimensionality reduction
                        end = perf_counter() # Stop the chrono
                    elapsed = end - start # Compute the elapsed time
                    self.time_perf.append([elapsed,components,neighbors]) # Store the time performance
                    # Store the reconstruction error
                    self.recon_error.append([abs(embedding.reconstruction_error_),components,neighbors])
                    # Readd the classes to the reduced data
                    reduc_data = np.concatenate((mydata[:,[0]],reduc_data),axis=1)
                    # Get the k pairs of train/test sets
                    train_sets, test_sets = self.crossksets(reduc_data)
                    accu_temp_KNN = np.empty(self.k) # Initialize the accuracy vector
                    F1_temp_KNN = np.empty(self.k) # Initialize the F1 vector
                    for s,train in enumerate(train_sets): # Compute k times the metrics on different data set
                        accu_temp_KNN[s], F1_temp_KNN[s] = self.KNN_metric(train,test_sets[s])
                    # Compute the means
                    KNN_accu_run[run_index] = np.mean(accu_temp_KNN) 
                    KNN_F1_run[run_index] = np.mean(F1_temp_KNN)
                    # Compute the standard deviations
                    KNN_accu_std_run[run_index] = np.std(accu_temp_KNN)
                    KNN_F1_std_run[run_index] = np.std(F1_temp_KNN)
                    # Row to be displayed
                    row = [components,
                           neighbors,
                           run_index,
                           self.recon_error[-1][0],
                           round(KNN_accu_run[run_index],2),
                           round(KNN_F1_run[run_index],2),
                           round(KNN_accu_std_run[run_index],2),
                           round(KNN_F1_std_run[run_index],2),
                           round(elapsed,1)]
                    # Display the row
                    print(self.row_format.format(*row))
                # Compute the mean of the metrics over different run for the space if run = 1 does not change anything
                self.KNN_F1_std[j,i] = np.mean(KNN_F1_std_run)
                self.KNN_accu_std[j,i] = np.mean(KNN_accu_std_run)
                self.KNN_F1[j,i] = np.mean(KNN_F1_run)
                self.KNN_accu[j,i] = np.mean(KNN_accu_run)

    def plot_cumulative_error(self,title,figsize=[12,7],save_file=None):
        """
        Goal:
        Plot the reconstruction error curve for different value of the hyperparameter (number of neighbors)
        Inputs:
        title = string - title of the graph
        figsize = list of size 2 - size of the graph
        save_file = string - save the file at "figures/save_file.svg"
        Outputs:
        """
        if self.norm: # Normalize the data
            mydata,_,_ = normalize(self.data)
        elif self.norm_0100:
            mydata = normalize_0_100(self.data)
        else:
            mydata = self.data.copy()
        fig = plt.figure(figsize=figsize) # Initialize the figure
        recon_err_np = np.array(self.recon_error)
        # Only 8 hyperparameter values are kept for the figure
        if self.method == "modified":
            valid_neighbors_value = self.range_neighbors[self.range_neighbors > self.data.shape[1] - 1]
        else:
            valid_neighbors_value = self.range_neighbors
        line_keep = min(valid_neighbors_value.shape[0],8) 
        # Perform a linspace to know which values to keep
        small_subset_neighbors = np.linspace(0,valid_neighbors_value.shape[0]-1,
                                             line_keep,dtype=int)
        # Extract the values to keep
        small_range_neighbors = valid_neighbors_value[small_subset_neighbors]
        # Retain the samples that have the values retained for the hyperparameter
        recon_err_np = recon_err_np[np.isin(recon_err_np[:,2],small_range_neighbors)]
        for j,neighbors in enumerate(small_range_neighbors):
            try:
                # Try with "arpack solver"
                embedding = LLE(n_components=self.data.shape[1] - 1,
                            n_neighbors=neighbors,
                            method=self.method,
                            max_iter=200,
                            random_state=self.seed,
                            eigen_solver="auto")
                embedding.fit(mydata[:,1:]) # Dimensionality reduction
            except ValueError:
                # else Try with "dense" solver 
                embedding = LLE(n_components=self.data.shape[1] - 1,
                            n_neighbors=neighbors,
                            method=self.method,
                            eigen_solver="dense")
                embedding.fit(mydata[:,1:]) # Dimensionality reduction
            err = embedding.reconstruction_error_
            recon_err_np[recon_err_np[:,2] == neighbors,0] /= err
        columns =  ["Cumulative sum of eigenvalues (normalized)",
                    "Eigenvalue index",
                    "Number of neighbors"]
        # Convert to pandas 
        recon_err_pd = pd.DataFrame(recon_err_np,columns=columns)
        ax = fig.add_subplot(1,1,1)
        # Set darkgrid style
        sns.set_style("darkgrid")
        # Plot the graph
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
        """
        Goal:
        Generate a pairplot to visualize the distribution of the points
        after having performed the dimensionelity reduction
        Inputs:
        neighbors = int - number of neighbors to use for the dimensionality reduction
        components = int - number of components to use for the dimensionality reduction
        title = string - title of the graph 
        components_to_show = int - number of components to show on the pairplot
        save_file = string - save the file at "figures/save_file.svg"
        Outputs:
        """
        if self.norm: # Normalize the data
            mydata,_,_ = normalize(self.data)
        elif self.norm_0100:
            mydata = normalize_0_100(self.data)
        else:
            mydata = self.data.copy()
        try:
            # Try with arparck solver
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        max_iter=200,
                        random_state=self.seed,
                        eigen_solver="auto")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        except ValueError:
            # Try with eigen solver 
            embedding = LLE(n_components=components,
                        n_neighbors=neighbors,
                        method=self.method,
                        eigen_solver="dense")
            reduc_data = embedding.fit_transform(mydata[:,1:])
        # Get the number of dimension
        nb_features = reduc_data.shape[1]
        # Add the classes
        reduc_data = np.concatenate((mydata[:,[0]],reduc_data),axis=1)
        columns = ["Class"] + ["Component " + str(i) for i in range(1,nb_features+1)]
        # Convert to pandas data frame and keep only some components
        data_pd = pd.DataFrame(reduc_data,columns=columns).iloc[:,:components_to_show+1] 
        sns.set_style("darkgrid")
        sns.color_palette("tab10")
        # Generate the pairplot
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
        """
        Goal:
        Inputs:
        Outputs:
        """
        if self.norm: # Normalize the data
            mydata,_,_ = normalize(self.data)
        elif self.norm_0100:
            mydata = normalize_0_100(self.data)
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

    def plot_time_perf(self,title,save_file=None,save_data=None):
        """
        Goal:
        Plot the time performances with respect to the hyperparameter value (number of neighbors)
        Inputs:
        title = string - title of the graph 
        save_file = string - save the file at "figures/save_file.svg"
        Outputs:
        """
        fig = plt.figure(figsize=[12,7]) # Create the figure
        ax = fig.add_subplot(1,1,1)
        time_perf_np = np.array(self.time_perf) 
        columns = ["Wall clock time [s]","Components","Neighbors"]
        # Convert to pandas data frame
        time_perf_pd = pd.DataFrame(data=time_perf_np,columns=columns)
        sns.set_style("darkgrid")
        # Plot the graph 
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
        if save_data is not None:
            time_perf_pd.to_csv("data_time/" + save_data + ".csv",index=False)

    def generate_contour(self,z,fig,subplot,title,cmap="viridis"):
        """
        Goal:
        Plot a countour to visualize the metrics with KNN 
        Inputs:
        z = np.ndarray - size range_neighors x range_compo - metric value
        fig = figure object matplotlib
        subplot = list of size 3 - where the graph sould be plotted
        title = string - title of the graph 
        cmap = string - color map
        Outputs:
        """
        ax = fig.add_subplot(*subplot)
        cs = ax.contourf(self.x,self.y,z,cmap=cmap)
        ax.set_xlabel("Number of components",fontsize=12)
        ax.set_ylabel("Number of neighbors",fontsize=12)
        ax.set_title(title,fontsize=13)
        ax.set_xticks(self.range_compo)
        plt.colorbar(cs,ax=ax)

    def generate_all(self,size=[12,10],save_file=None):
        """
        Goal:
        Generate the four contours for the two metrics associated to KNN
        (two contours by metric)
        Inputs:
        size = list of size é - size of the graph
        save_file = string - save the file at "figures/save_file.svg"
        Outputs:
        """
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

