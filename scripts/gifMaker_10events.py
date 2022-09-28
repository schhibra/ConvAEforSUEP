# it makes gif of inputs and outputs of 10 events

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

direc = "v2_1_oppDiceloss_prelu_relu_repeat_100"
filenames = []
for i in range(10):
    filename = direc+'/3channels_SUEP_1event_%d.png'%(i+1)
    filenames.append(filename)
    filename = direc+'/predictedSUEP_ae%d.png'%(i+1)
    filenames.append(filename)
    print(filenames[i])

# build gif
with imageio.get_writer('mygif.mp4', fps=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
