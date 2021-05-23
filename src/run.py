import pandas as pd
import numpy as np
import seaborn as sns
import os
from utils import split_two, Assessment
from utils import plot_time_comparison, plot_cumulative_error, plot_residual_variance

Linux = True # Specify if you work on Linux or windows
# Choose what you want to run
hyperparams_search = True
best_neighbors_LLE_full = None
best_neighbors_MLLE_full = None
best_components_MLLE_full = None
best_neighbors_LLE_semi = None
best_neighbors_MLLE_semi = None
best_components_MLLE_semi = None
semi_data_set = True
full_data_set = True

# Define the parameters for the comparison
runs_semi_data_set = 5
norm = False
norm_0100 = True
san_check = True
k = 5
KNN_neigh = 10
min_components, max_components, step_components = 1,5,1
min_neigh_LLE, max_neigh_LLE, nb_neigh_LLE = 5,200,100
min_neigh_MLLE, max_neigh_MLLE, nb_neigh_MLLE = 5,200,100

# Let's the code do the rest

print("################# PARAMETERS FOR THE RUN #################")

print("hyperparameter search : ",hyperparams_search)
print("Full data set evaluation : ",full_data_set)
print("Semi data set evaluation : ",semi_data_set)
print("Parameters LLE : ",min_neigh_LLE, max_neigh_LLE, nb_neigh_LLE)
print("Parameters MLLE : ",min_neigh_MLLE, max_neigh_MLLE, nb_neigh_MLLE)
print("Sanity Check : ",san_check)
print("\n")

directories = ["figures","data","figures/svg","figures/pdf"]
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

sns.set_style("darkgrid")
range_components = np.arange(min_components,max_components,step_components)
range_neighbors_LLE = np.linspace(min_neigh_LLE,max_neigh_LLE,nb_neigh_LLE,dtype=int)
range_neighbors_MLLE = np.linspace(min_neigh_MLLE,max_neigh_MLLE,nb_neigh_MLLE,dtype=int)

print("Import the data set \n")
# Import the data set 
columns = ["Class"]
columns += ["Frequence word " + str(i) for i in range(1,49)]
columns += ["Special character " + str(i) for i in range(1,7)]
columns += ["Capital length mean"]
columns += ["Capital length longest"]
columns += ["Sum of captital length"]

data = pd.read_csv("../Data/spambasedata.csv",names=columns)
data_np = data.to_numpy().astype(float)

if full_data_set:

    dataset = data_np

    print("Evaluation on the full data set, Initialization of the objects \n")

    if san_check:
        print("\n################# SANITY CHECK ################# \n")

    # Initialize the objects
    LLE_algo = Assessment(dataset,range_components,range_neighbors_LLE,k=k,
                          KNN_neigh=KNN_neigh,
                          check=san_check,norm=norm,norm_0100=norm_0100)
    print("\n")
    MLLE_algo = Assessment(dataset,range_components,range_neighbors_MLLE,k=k,
                           KNN_neigh=KNN_neigh,method="modified",
                           check=san_check,norm=norm,norm_0100=norm_0100)

    if hyperparams_search:
        ############ LLE ############
        print("\n################# Launch the evaluation of LLE full data set ################# \n")

        LLE_algo.find_hyper()
        LLE_algo.generate_all(save_file="LLE_KNN_metric_full_set")
        LLE_algo.plot_time_perf(title="LLE time performances",
                                save_data="LLE_time_perf_full_set")
        LLE_algo.save_all_data("LLE_full_set")

        print("\nData saved in data folder \n")

        print("\n################# Launch the evaluation of MLLE full data set ################# \n")

        MLLE_algo.find_hyper()
        MLLE_algo.generate_all(save_file="MLLE_KNN_metric_full_set")
        MLLE_algo.plot_time_perf(title="MLLE time performances",
                                 save_data="MLLE_time_perf_full_set")
        MLLE_algo.save_all_data("MLLE_full_set")

        print("\n################# Data saved in data folder #################\n")

    if best_neighbors_LLE_full is not None:
        LLE_algo.generate_pairplot(best_neighbors_LLE_full,
                                   4,
                                   title="LLE neighbors = " + str(best_neighbors_LLE_full),
                                   save_file="LLE_pairplot_full_set")
        print("\n################# Pairplot LLE saved in the figures folder #################\n")


    if best_neighbors_MLLE_full is not None and best_components_MLLE_full is not None:
        MLLE_algo.generate_pairplot(best_neighbors_MLLE_full,
                                    best_components_MLLE_full,
                                    title="MLLE neighbors = " + str(best_neighbors_MLLE_full) + " components = " + str(best_components_MLLE_full),
                                    save_file="MLLE_pairplot_full_set")
        print("\n################# Pairplot MLLE saved in the figures folder #################\n")

################################### Assess sensitivity to a smaller data set ###################################

if semi_data_set:


    if hyperparams_search:

        full_data_frame_LLE = []
        full_data_frame_MLLE = []
        full_data_frame_LLE_time = []
        full_data_frame_MLLE_time = []

        for i in range(runs_semi_data_set):

            dataset = split_two(data_np)

            # Initialize the objects

            if san_check:
                print("\n################# SANITY CHECK ################# \n")

            LLE_algo = Assessment(dataset,range_components,range_neighbors_LLE,k=k,
                                KNN_neigh=KNN_neigh,
                                check=san_check,norm=norm,norm_0100=norm_0100)
                    
            print("\n")

            MLLE_algo = Assessment(dataset,range_components,range_neighbors_MLLE,k=k,
                                KNN_neigh=KNN_neigh,method="modified",
                                check=san_check,norm=norm,norm_0100=norm_0100)
            ############ LLE ############
            print("\n################# Launch the evaluation of LLE semi data set: semi data set n° " + str(i) +" #################\n")

            LLE_algo.find_hyper()
            LLE_algo.generate_all(save_file="LLE_KNN_metric_semi_set")
            LLE_algo.plot_time_perf(title="LLE time performances",
                                    save_data="LLE_time_perf_semi_set_" + str(i))
            LLE_algo.save_all_data("LLE_semi_set_" + str(i))

            path_file_LLE = "data/LLE_semi_set_" + str(i) + ".csv"
            if Linux:
                cmd_LLE = "rm data/LLE_semi_set_" + str(i) + ".csv"
            else:
                cmd_LLE = "del data\\LLE_semi_set_" + str(i) + ".csv"
            full_data_frame_LLE.append(pd.read_csv(path_file_LLE))
            os.system(cmd_LLE)

            path_file_LLE_time = "data/LLE_time_perf_semi_set_" + str(i) + ".csv"
            if Linux:
                cmd_LLE_time = "rm data/LLE_time_perf_semi_set_" + str(i) + ".csv"
            else:
                cmd_LLE_time = "del data\\LLE_time_perf_semi_set_" + str(i) + ".csv"
            full_data_frame_LLE_time.append(pd.read_csv(path_file_LLE_time))
            os.system(cmd_LLE_time)

            print("\n################# Launch the evaluation of MLLE semi data set: semi data set n° " + str(i) +" #################\n")

            MLLE_algo.find_hyper()
            MLLE_algo.generate_all(save_file="MLLE_KNN_metric_semi_set")
            MLLE_algo.plot_time_perf(title="MLLE time performances",
                                        save_data="MLLE_time_perf_semi_set_" + str(i))
            MLLE_algo.save_all_data("MLLE_semi_set_" + str(i))

            path_file_MLLE = "data/MLLE_semi_set_" + str(i) + ".csv"
            if Linux:
                cmd_MLLE = "rm data/MLLE_semi_set_" + str(i) + ".csv"
            else:
                cmd_MLLE = "del data\\MLLE_semi_set_" + str(i) + ".csv"
            full_data_frame_MLLE.append(pd.read_csv(path_file_MLLE))
            os.system(cmd_MLLE)

            path_file_MLLE_time = "data/MLLE_time_perf_semi_set_" + str(i) + ".csv"
            if Linux:
                cmd_MLLE_time = "rm data/MLLE_time_perf_semi_set_" + str(i) + ".csv"
            else:
                cmd_MLLE_time = "del data\\MLLE_time_perf_semi_set_" + str(i) + ".csv"
            full_data_frame_MLLE_time.append(pd.read_csv(path_file_MLLE_time))
            os.system(cmd_MLLE_time)

        full_data_frame_LLE = pd.concat(full_data_frame_LLE)
        full_data_frame_MLLE = pd.concat(full_data_frame_MLLE)
        full_data_frame_LLE_time = pd.concat(full_data_frame_LLE_time)
        full_data_frame_MLLE_time = pd.concat(full_data_frame_MLLE_time)

        full_data_frame_LLE.to_csv("data/LLE_semi_set.csv",index=False)
        full_data_frame_MLLE.to_csv("data/MLLE_semi_set.csv",index=False)
        full_data_frame_LLE_time.to_csv("data/LLE_time_perf_semi_set.csv",index=False)
        full_data_frame_MLLE_time.to_csv("data/MLLE_time_perf_semi_set.csv",index=False)

        print("\n################# Data saved in data folder #################\n")

    if best_neighbors_LLE_semi is not None:

        dataset = split_two(data_np)

        # Initialize the objects
        LLE_algo = Assessment(dataset,range_components,range_neighbors_LLE,k=k,
                            KNN_neigh=KNN_neigh,
                            check=san_check,norm=norm,norm_0100=norm_0100)

        LLE_algo.generate_pairplot(best_neighbors_LLE_full,
                                    4,
                                    title="LLE neighbors = " + str(best_neighbors_LLE_semi),
                                    save_file="LLE_pairplot_semi_set")
        print("\n################# Pairplot LLE saved in the figures folder #################\n")


    if best_neighbors_MLLE_semi is not None and best_components_MLLE_semi is not None:

        MLLE_algo = Assessment(dataset,range_components,range_neighbors_MLLE,k=k,
                            KNN_neigh=KNN_neigh,method="modified",
                            check=san_check,norm=norm,norm_0100=norm_0100)

        MLLE_algo.generate_pairplot(best_neighbors_MLLE_full,
                                    best_components_MLLE_full,
                                    title="MLLE neighbors = " + str(best_neighbors_MLLE_semi) + " components = " + str(best_components_MLLE_semi),
                                    save_file="MLLE_pairplot_semi_set")

        print("\n################# Pairplot MLLE saved in the figures folder #################\n")


print("\n################# Plotting graphs #################\n")

files_data = [["data/LLE_full_set.csv","data/LLE_semi_set.csv","LLE_residual_metric","LLE residual variance"],
              ["data/MLLE_full_set.csv","data/MLLE_semi_set.csv","MLLE_residual_metric","MLLE residual variance"],
              ["data/LLE_full_set.csv","data/LLE_semi_set.csv","LLE_reconstruction_metric","LLE reconstruction error"],
              ["data/MLLE_full_set.csv","data/MLLE_semi_set.csv","MLLE_reconstruction_metric","MLLE reconstruction error"]]

for file in files_data:
    if os.path.exists(file[0]):
        if "reconstruction" in file[-1]:
            plot_cumulative_error(file[0],file[-1],
                                  save_file=file[-2] + "_full_set")
        if "residual" in file[-1]:
            plot_residual_variance(file[0],file[-1],
                                   save_file=file[-2] + "_full_set")
    if os.path.exists(file[1]):
        if "reconstruction" in file[-1]:
            plot_cumulative_error(file[1],file[-1],
                                  save_file=file[-2] + "_semi_set")
        if "residual" in file[-1]:
            plot_residual_variance(file[1],file[-1],
                                   save_file=file[-2] + "_semi_set")

files = ["data/LLE_time_perf_full_set.csv",
         "data/MLLE_time_perf_full_set.csv",
         "data/LLE_time_perf_semi_set.csv",
         "data/MLLE_time_perf_semi_set.csv"]
plot_time = True

for file in files:
    if not os.path.exists(file):
        plot_time = False
        break

if plot_time:
    plot_time_comparison(*files,save_file="time_performances_comparison")

print("\n################# Graphs available in the figure folder #################\n")