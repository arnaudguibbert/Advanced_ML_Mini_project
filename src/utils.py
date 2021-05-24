import numpy as np 
import pandas as pd 
import seaborn as sns
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import LinearSVC as SVC
from sklearn.metrics import f1_score
from scipy.spatial.distance import pdist
import warnings
import plotly.express as px

def plot_residual_variance(data_path,title,figsize=[12,7],save_file=None,compo_to_retain=[2,4]):
    """
    Goal:
    Plot the residual variance with respect to the number of neighbors
    Inputs:
    data_path = string - path to the data where the residual variance values are stored
    title = string - title of the graph
    figsize = list of size 2 - dimensions of the figure
    save_file = string - name of the file for the graph
    compo_to_retain = list - specify for which components you want to plot the graph
    Outputs:
    """
    fig = plt.figure(figsize=figsize) # Initialize the figure
    # Extract the data
    residual_pd = pd.read_csv(data_path)
    residual_pd = residual_pd.rename(columns={"Output dimension":"Output_dimension"})
    # Retain the components specified 
    condition = ["Output_dimension == " + str(i) for i in compo_to_retain]
    condition = " | ".join(condition)
    residual_pd = residual_pd.query(condition)
    residual_pd = residual_pd.rename(columns={"Output_dimension":"Output dimension"})
    ax = fig.add_subplot(1,1,1)
    # Set darkgrid style
    sns.set_style("darkgrid")
    # Plot the graph
    sns.lineplot(data=residual_pd,
                    x="Number of neighbors",
                    y="Residual variance",
                    hue="Output dimension",
                    palette=sns.color_palette("hls", len(compo_to_retain)),
                    ci=90,
                    ax=ax)
    max_neigh = residual_pd["Number of neighbors"].max()
    min_neigh = residual_pd["Number of neighbors"].min()
    ax.set_xlim(min_neigh,max_neigh)
    ax.set_title(title,fontsize=16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.legend(fontsize=12,title="Output dimension",title_fontsize=12,loc="best")
    # Save the graphs in different formats
    if save_file is not None:
        fig.savefig("figures/svg/" + save_file + ".svg",dpi=200)
        fig.savefig("figures/pdf/" + save_file + ".pdf",dpi=200)
    plt.close("all")

def plot_cumulative_error(data_path,title,figsize=[12,7],save_file=None,compo_to_retain=[2,4]):
    """
    Goal:
    Plot the reconstruction error with respect to the number of neighbors
    Inputs:
    data_path = string - path to the data where the residual variance values are stored
    title = string - title of the graph
    figsize = list of size 2 - dimensions of the figure
    save_file = string - name of the file for the graph
    compo_to_retain = list - specify for which components you want to plot the graph
    Outputs:
    """
    fig = plt.figure(figsize=figsize) # Initialize the figure
    # Extract the data
    residual_pd = pd.read_csv(data_path)
    # Retain the components specified 
    residual_pd = residual_pd.rename(columns={"Output dimension":"Output_dimension"})
    condition = ["Output_dimension == " + str(i) for i in compo_to_retain]
    condition = " | ".join(condition)
    residual_pd = residual_pd.query(condition)
    residual_pd = residual_pd.rename(columns={"Output_dimension":"Output dimension"})
    ax = fig.add_subplot(1,1,1)
    # Set darkgrid style
    sns.set_style("darkgrid")
    # Plot the graph
    sns.lineplot(data=residual_pd,
                    x="Number of neighbors",
                    y="Reconstruction error",
                    hue="Output dimension",
                    palette=sns.color_palette("hls", len(compo_to_retain)),
                    ci=90,
                    ax=ax)
    max_neigh = residual_pd["Number of neighbors"].max()
    min_neigh = residual_pd["Number of neighbors"].min()
    ax.set_yscale('log')
    ax.set_xlim(min_neigh,max_neigh)
    ax.set_title(title,fontsize=16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.legend(fontsize=12,title="Output dimension",title_fontsize=12,loc="best")
    # Save the graphs in different formats
    if save_file is not None:
        fig.savefig("figures/svg/" + save_file + ".svg",dpi=200)
        fig.savefig("figures/pdf/" + save_file + ".pdf",dpi=200)
    plt.close("all")

def plot_time_comparison(path_LLE_full,path_MLLE_full,path_LLE_semi,path_MLLE_semi,save_file=None):
    """
    Goal:
    Plot the wall clock time with respect to the number of neighbors
    Plot four different curves:
    * Wall clock time of LLE on the whole data set
    * Wall clock time of MLLE on the whole data set
    * Wall clock time of LLE on the semi data set
    * Wall clock time of MLLE on the semi data set
    Inputs:
    path_LLE_full = string - path to the data where the wall clock time values are stored for LLE on the whole data set
    path_MLLE_full = string - path to the data where the wall clock time values are stored for MLLE on the whole data set
    path_LLE_semi = string - path to the data where the wall clock time values are stored for LLE on the semi data set
    path_MLLE_semi = string - path to the data where the wall clock time values are stored for MLLE on the semi data set
    save_file = string - name of the file for the graph
    Outputs:
    """
    fig = plt.figure(figsize=[12,7])
    ax = fig.add_subplot(111)
    # Extract the data for each file
    data_LLE_full = pd.read_csv(path_LLE_full)
    data_MLLE_full = pd.read_csv(path_MLLE_full)
    data_LLE_semi = pd.read_csv(path_LLE_semi)
    data_MLLE_semi = pd.read_csv(path_MLLE_semi)
    data_LLE_full["type"] = [0]*data_LLE_full.shape[0]
    data_MLLE_full["type"] = [1]*data_MLLE_full.shape[0]
    data_LLE_semi["type"] = [2]*data_LLE_semi.shape[0]
    data_MLLE_semi["type"] = [3]*data_MLLE_semi.shape[0]
    # Merge them in one data set
    full_data = pd.concat((data_LLE_full,
                           data_LLE_semi,
                           data_MLLE_full,
                           data_MLLE_semi),ignore_index=True)
    sns.set_style("darkgrid")
    # Plot the graph
    sns.lineplot(data=full_data,y="Wall clock time [s]",
                 x="Neighbors",hue="type",ax=ax,
                 palette=sns.color_palette("hls",4))
    handles, labels = ax.get_legend_handles_labels()
    true_labels = ["LLE (whole data set)",
                   "MLLE (whole data set)",
                   "LLE (semi data set)",
                   "MLLE (semi data set)"]
    labels = [true_labels[int(float(i))] for i in labels]
    max_neigh = full_data["Neighbors"].max()
    min_neigh = full_data["Neighbors"].min()
    ax.set_xlim(min_neigh,max_neigh)
    ax.set_title("Time performances comparison",fontsize=16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.legend(handles,labels,
              fontsize=12,
              title=None,
              title_fontsize=12,loc="upper left")
    # Save figure in different formats
    if save_file is not None:
        fig.savefig("figures/svg/" + save_file + ".svg",dpi=200)
        fig.savefig("figures/pdf/" + save_file + ".pdf",dpi=200)

def residual_correlation(DX,DY,X_norm=False,Y_norm=True):
    """
    Goal:
    Compute the residual variance
    Inputs:
    DX = np.array - size M(M-1)/2 - pairwise distance matrix flattened (only the upper diagonal)
    DY = np.array - size M(M-1)/2 - pairwise distance matrix flattened (only the upper diagonal)
    X_norm = Boolean - specify if it is necessary to set the mean to 0 and variance to 1 for DX
    Y_norm = Boolean - specify if it is necessary to set the mean to 0 and variance to 1 for DY
    Outputs:
    residual = float - residual variance 
    """
    # Normalize if necessary
    if X_norm:
        DX_norm = (DX - np.mean(DX))/np.std(DX)
    else:
        DX_norm = DX
    if Y_norm:
        DY_norm = (DY - np.mean(DY))/np.std(DY)
    else:
        DY_norm = DY
    # Compute the correlation coefficient
    rho = np.mean(DX_norm*DY_norm)
    residual = 1 - rho**2
    return residual

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
    """
    Goal:
    Rescale and center the last three features in [0,upper_bound]
    Inputs:
    data = data to be normalized
    upper_bound = float - upper bound of the rescaling
    Outputs:
    """
    data_norm = data.copy() # copy to avoid any issue with the original data
    min = np.min(data_norm[:,-3:],axis=0,keepdims=True)
    max = np.max(data_norm[:,-3:],axis=0,keepdims=True)
    # Normalize into [0,upper_bound]
    data_norm[:,-3:] = upper_bound*(data_norm[:,-3:] - min)/(max - min)
    # Center the data
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
                 method='standard',k=3,KNN_neigh=10,
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
        self.row_format = '{:<12}{:<12}{:<25}{:<25}{:<20}' # Define the display format
        self.k = k
        self.KNN_neigh = KNN_neigh
        self.seed = seed
        if check:
            self.sanity_check()
        self.x , self.y = np.meshgrid(self.range_compo, self.range_neighbors)
        self.recon_error = [] # The reconstruction error is stored here
        self.residual = [] # Residual variance will be stored there
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
        self.residual = [] # Residual variance will be stored there
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
        Actually this function was no longer used 
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
        Dist_X = pdist(mydata[:,1:])
        Dist_X = (Dist_X - np.mean(Dist_X))/np.std(Dist_X)
        # Header fo the logs
        header = ["Components",
                  "Neighbors",
                  "Reconstruction error",
                  "Residual Variance",
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
                Dist_Y = pdist(reduc_data)
                residual_var = residual_correlation(Dist_X,Dist_Y)
                self.residual.append([residual_var,components,neighbors])
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
                # Compute the mean of the metrics over different run for the space if run = 1 does not change anything
                self.KNN_F1_std[j,i] = np.std(F1_temp_KNN)
                self.KNN_accu_std[j,i] = np.std(accu_temp_KNN)
                self.KNN_F1[j,i] = np.mean(F1_temp_KNN)
                self.KNN_accu[j,i] = np.mean(accu_temp_KNN)
                # Row to be displayed
                row = [components,
                        neighbors,
                        self.recon_error[-1][0],
                        round(self.residual[-1][0],3),
                        round(elapsed,1)]
                # Display the row
                print(self.row_format.format(*row))

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
            pairplot.fig.savefig("figures/svg/" + save_file + ".svg",dpi=200)
            pairplot.fig.savefig("figures/pdf/" + save_file + ".pdf",dpi=200)
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
            time_perf_pd.to_csv("data/" + save_data + ".csv",index=False)

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
            fig.savefig("figures/svg/" + save_file + ".svg",dpi=200)
            fig.savefig("figures/pdf/" + save_file + ".pdf",dpi=200)
        plt.close('all')

    def save_all_data(self,save_data):
        recon_error_np = np.array(self.recon_error)
        residual_np = np.array(self.residual)
        data = np.concatenate((recon_error_np[:,[0]],residual_np),axis=1)
        columns =  ["Reconstruction error",
                    "Residual variance",
                    "Output dimension",
                    "Number of neighbors"]
        data_pd = pd.DataFrame(data,columns=columns)
        KNN_acc_vec = self.KNN_accu.reshape(-1,1)
        KNN_F1_vec = self.KNN_F1.reshape(-1,1)
        KNN_acc_std_vec = self.KNN_accu_std.reshape(-1,1)
        KNN_F1_std_vec = self.KNN_F1_std.reshape(-1,1)
        compo_vec = self.x.reshape(-1,1)
        neighbors_vec = self.y.reshape(-1,1)
        data_KNN = np.concatenate((KNN_acc_vec,
                                   KNN_F1_vec,KNN_acc_std_vec,
                                   KNN_F1_std_vec,compo_vec,
                                   neighbors_vec),axis=1)
        columns_KNN = ["KNN Accuracy","KNN F1 measure",
                       "KNN accuracy std","KNN F1 measure std",
                       "Output dimension","Number of neighbors"]
        data_KNN_pd = pd.DataFrame(data_KNN,columns=columns_KNN)
        full_data = data_pd.merge(data_KNN_pd, on=["Output dimension","Number of neighbors"])
        full_data.to_csv("data/" + save_data + ".csv",index=False)
