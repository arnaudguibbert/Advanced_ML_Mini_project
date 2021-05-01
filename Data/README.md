# Advanced_ML_mini_project
Mini project in the scope of advanced machine learning class.

## 16/04/2021

### Récapitulatif:
Présentation des données et attribution des tâches. 
### Job à faire
Everyone: Commenter le code  
Nessreddine: Commencer à tester les algorithmes (LLE & MLLE) sur un data set complet  
Arnaud: Tester des techniques pour prédire les data points manquants (Support Vector Regression)  

## Getting Started 

5000 face images have been selected among 200 000 face images. They are located into the Data\selected_images folder. Each image face belongs to one class among four classes. The classes are: 
##### Class 1 : Eyeglasses
##### Class 2 : Wearing Hat
##### Class 3 : Wavy Hair
##### Class 4 : Goatee
Since the original images have size 3x218x178 (116 412 features), a small preprocessing is needed to divide at least by 10 this initial dimension. To do so the images are first converted to gray scale and then an average pooling filter of size (2x2) is appplied on the images. The final images have size 109x89 (9 701 features). You can visualize such a transformation by running the main.ipynb in the src folder. 

### Generate the data set
The data set is not directly available on the repository (because too heavy 900 MB), only the images selected are available in the Data\selected_images folder. But no worries, to generate it the only thing you have to do is to run the preprocessing.py python file located in the Data folder. If you are working on Linux modify the first line of the code by setting "Linux = True". Then the dataset "data_frame.csv" will be created in the Data folder.  

### Locally Linear Embeddings (LLE)

### Modified Locally Linear Embeddings (MLLE)