# it makes gif of input, 28 layers and 1 output reconstruction

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

direc = "v2_1_oppDiceloss_prelu_relu_repeat_100"
#direc = "v2_1_modDiceloss_prelu_relu"
filenames = []
filenames.append(direc+"/3channels_QCD_1event.png")
for i in range(28):
    filename = direc+'/predictedQCD_m%d.png'%(i+1)
    filenames.append(filename)
    print(filenames[i])

filenames.append(direc+"/predictedQCD_ae.png")

# build gif
with imageio.get_writer('mygif.mp4', fps=2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
