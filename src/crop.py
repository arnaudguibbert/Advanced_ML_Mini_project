import os 

mypath = "figures/pdf/"
out_path = "figures/crop/"
cmd_bas = "pdf-crop-margins "

for file in os.listdir(mypath):
    cmd = cmd_bas + mypath + file + " -o " + out_path + file + " -p 0"
    os.system(cmd)