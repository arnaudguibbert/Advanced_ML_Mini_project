import os 

os.system("pdf-crop-margins figures/LLE_residual_metric_full_set.pdf -o figures/crop/LLE_residual_metric_full_set.pdf -p 0")
os.system("pdf-crop-margins figures/MLLE_residual_metric_full_set.pdf -o figures/crop/MLLE_residual_metric_full_set.pdf -p 0")
os.system("pdf-crop-margins figures/LLE_residual_metric_semi_set.pdf -o figures/crop/LLE_residual_metric_semi_set.pdf -p 0")
os.system("pdf-crop-margins figures/MLLE_residual_metric_semi_set.pdf -o figures/crop/MLLE_residual_metric_semi_set.pdf -p 0")
os.system("pdf-crop-margins figures/LLE_reconstruction_metric_full_set.pdf -o figures/crop/LLE_reconstruction_metric_full_set.pdf -p 0")
os.system("pdf-crop-margins figures/MLLE_reconstruction_metric_full_set.pdf -o figures/crop/MLLE_reconstruction_metric_full_set.pdf -p 0")
os.system("pdf-crop-margins figures/LLE_reconstruction_metric_semi_set.pdf -o figures/crop/LLE_reconstruction_metric_semi_set.pdf -p 0")
os.system("pdf-crop-margins figures/MLLE_reconstruction_metric_semi_set.pdf -o figures/crop/MLLE_reconstruction_metric_semi_set.pdf -p 0")
os.system("pdf-crop-margins figures/time_performances_comparison.pdf -o figures/crop/time_performances_comparison.pdf -p 0")
