import numpy as np
import torch
import pandas as pd
import matplotlib as plt
import cv2
import os

move = False
Linux = False
nb_samples = 5000
forbidden = [str(i) + ".jpg" for i in range(130822,130841)]
# Import the classes associated to each image
data_attr = pd.read_csv("../Data/list_attr_celeba.csv")
for forbid in forbidden:
    data_attr = data_attr[data_attr["Path"] != forbid]

# Class selected for the study
Class_selected = ["Eyeglasses","Wearing_Hat","Wearing_Necklace","Goatee"]
nb_classes = len(Class_selected)
data_class = [] # List where we will store a subset containing each class
# For each class extract all the data points
for i,type_class in enumerate(Class_selected):
    df = data_attr.query(type_class + " == 1")
    Class_bis = Class_selected.copy()
    Class_bis.pop(i)
    for other_class in Class_bis:
        df = df.query(other_class + " == -1")
    df = df.sample(frac=1,random_state=0)
    data_class.append(df)
    
Image_selected = [] # Here we will store the name of the file of each image
Class = [] # Here we will store the class associated to each image
# For each class extract these information
for i,data_f in enumerate(data_class):
    selection = data_f["Path"].iloc[:nb_samples//nb_classes].to_numpy()
    Image_selected.append(selection)
    class_selection = np.ones_like(selection)*i
    Class.append(class_selection)
# Concatenate this into a single array
Image_selected = np.concatenate(Image_selected,axis=0).reshape(-1,1)
Class = np.concatenate(Class,axis=0).reshape(-1,1)
# Put this into a pandas dataframe
Final_base = pd.DataFrame(np.concatenate((Image_selected,Class),axis=1),columns=["Path","Class"])
# Finally write this information into a csv file
Final_base.to_csv("image_selected.csv",index=False)
Image_list = Image_selected.reshape(-1).tolist()

if move:
    print("move the selected images to a specific folder")
    if Linux:
        initial_folder = "img_align_celeba/"
        final_folder = "selected_images/"
    else:
        initial_folder = "img_align_celeba\\"
        final_folder = "selected_images\\"
    for i,img_path in enumerate(Image_list):
        if Linux:
            cmd = "cp " + initial_folder + img_path + " " + final_folder + img_path
        else:
            cmd = "copy " + initial_folder + img_path + " " + final_folder + img_path
        os.system(cmd)
        if (i+1)%10 == 0:
            print(i+1," images has been moved into the right folder")
    print("Transfer sucessfully completed")

print("Go to image extration")

path_to_img = ["selected_images/" + path_img for path_img in Image_list]

img_tensor = torch.empty(nb_samples,1,218,178)

for i,path_img in enumerate(path_to_img):
    print(path_img)
    gray = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2GRAY).astype(float)/255 # gray scale image
    assert(gray.shape == (218,178))
    img_tensor[i,0,:,:] = torch.from_numpy(gray)
    if (i+1)%100 == 0:
        print(str(i+1) + " images have been converted")

print("Conversion done, Averaging now")
    
Avg_op = torch.nn.AvgPool2d(2, stride=2)
reduc_img_tensor = Avg_op(img_tensor).view(nb_samples,-1)

print("Averaging done")

reduc_img_tensor = np.concatenate((Image_selected,Class,reduc_img_tensor.numpy()),axis=1)
data_cleaned = pd.DataFrame(reduc_img_tensor)
data_cleaned.to_csv("data_frame.csv",index=False)