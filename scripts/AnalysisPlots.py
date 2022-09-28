##########################################
import setGPU

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import glob

import ROOT as R

import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt

from matplotlib.colors    import LogNorm
from sklearn.metrics      import roc_curve, auc

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
####config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras import backend as K
##########################################


########################################## 
isinvpt = True

mass = "125"
version = "v2_1_oppDiceloss_prelu_relu_repeat_100"
plotdir = "plots_suep"+mass+"_"+version+"/"
filelist = glob.glob(os.path.join(plotdir, "*"))
for f in filelist:
    os.remove(f)
if (not os.path.exists(plotdir)):
    os.mkdir(plotdir)

rawImages       = 0
predictedImages = 0

nevts = 4000

hf = h5py.File(plotdir+'data.h5', 'w')
########################################## 


##########################################
def images(inputpath):
    ImageTrk = np.array([])
    ImageECAL = np.array([])
    ImageHCAL = np.array([])

    for fileIN in glob.glob(inputpath):
        print("Appending %s" %fileIN)
        f = h5py.File(fileIN, 'r')

        if (ImageTrk.shape[0] > nevts): break
    
        myImageTrk  = np.array(f.get("ImageTrk_PUcorr")[:], dtype=np.float32)
        ImageTrk    = np.concatenate([ImageTrk, myImageTrk], axis=0) if ImageTrk.size else myImageTrk
        
        myImageECAL = np.array(f.get("ImageECAL")[:], dtype=np.float32)
        ImageECAL   = np.concatenate([ImageECAL, myImageECAL], axis=0) if ImageECAL.size else myImageECAL
        
        myImageHCAL = np.array(f.get("ImageHCAL")[:], dtype=np.float32)
        ImageHCAL   = np.concatenate([ImageHCAL, myImageHCAL], axis=0) if ImageHCAL.size else myImageHCAL

        del myImageTrk, myImageECAL, myImageHCAL

    ImageTrk  = ImageTrk.reshape(ImageTrk.shape[0], ImageTrk.shape[1], ImageTrk.shape[2], 1)
    ImageECAL = ImageECAL.reshape(ImageECAL.shape[0], ImageECAL.shape[1], ImageECAL.shape[2], 1)
    ImageHCAL = ImageHCAL.reshape(ImageHCAL.shape[0], ImageHCAL.shape[1], ImageHCAL.shape[2], 1)

    Image3D = np.concatenate([ImageTrk, ImageECAL, ImageHCAL], axis=-1)
    #Image3D = ImageECAL

    Image3D_zero = np.zeros((Image3D.shape[0], 288, 360, 3), dtype=np.float32)
    Image3D_zero[:, 1:287, :, :] += Image3D
    if (isinvpt == True): Image3D_zero = np.divide(Image3D_zero, 2000., dtype=np.float32)

    del ImageTrk, ImageECAL, ImageHCAL, Image3D

    print("Image3D_zero.shape ", Image3D_zero.shape)
    return Image3D_zero

Image3D_QCD  = images("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_qcd_v1/gensim/output/test/qcd_gensim_101_*.h5")
Image3D_SUEP = images("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_suep_v1/gensim/output/test/suep_gensim_h"+mass+"_phi2_dp0p7_dt2_*_*.h5")
##########################################

##########################################
def plotimages(Image3D, index):
    if (index == 0): label = ["Trk_QCD", "ECAL_QCD", "HCAL_QCD"]
    else:            label = ["Trk_SUEP", "ECAL_SEUP", "HCAL_SUEP"]
    for i in range(3):
        plt.figure()
        SUM_Image = np.sum(Image3D[:nevts,:,:,i], axis = 0)
        print("SUM_Image N events min max = ", np.min(SUM_Image), np.max(SUM_Image))
        
        if (np.max(SUM_Image) <= 1): 
            plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=np.max(SUM_Image)*0.01))
        elif (np.min(SUM_Image) == 0):
            plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.1))
        elif (np.min(SUM_Image) <  0 or np.max(SUM_Image) > 1):
            plt.imshow(SUM_Image.T, origin='lower', vmin=np.min(SUM_Image), vmax=np.max(SUM_Image))
        plt.colorbar()
        plt.title(label[i], fontsize=15)
        plt.xlabel("$\eta$ cell", fontsize=15)
        plt.ylabel("$\phi$ cell", fontsize=15)
        plt.savefig(plotdir+label[i]+'_%dEvents.pdf'%(nevts), dpi=1000)
        plt.close()
    for i in range(3):
        plt.figure()
        SUM_Image = np.sum(Image3D[:1,:,:,i], axis = 0)
        print("SUM_Image (1 event) min max = ", np.min(SUM_Image), np.max(SUM_Image))
        if (np.max(SUM_Image) <= 1):
            plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=np.max(SUM_Image)*0.01))
        elif (np.min(SUM_Image) == 0):
                plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.1))
        elif (np.min(SUM_Image) <  0 or np.max(SUM_Image) > 1):
            plt.imshow(SUM_Image.T, origin='lower', vmin=np.min(SUM_Image), vmax=np.max(SUM_Image))
        plt.colorbar()
        plt.title(label[i], fontsize=15)
        plt.xlabel("$\eta$ cell", fontsize=15)
        plt.ylabel("$\phi$ cell", fontsize=15)
        plt.savefig(plotdir+label[i]+'_1Event.pdf', dpi=1000)
        plt.close()
##########################################

##########################################
def intersection(y_true, y_pred):

    intersection = []
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1]
        y_true_tmp  = tf.gather(y_true_tmp[0,:], idx_keep_in)
        y_pred_tmp  = tf.gather(y_pred_tmp[0,:], idx_keep_in)

        y_true_tmp = tf.reshape(y_true_tmp, shape=(1, y_true_tmp.shape[0]))
        y_pred_tmp = tf.reshape(y_pred_tmp, shape=(y_pred_tmp.shape[0], 1))

        intersection.append(K.sum(K.dot(y_true_tmp, y_pred_tmp)))
    return intersection

def sum_y_true(y_true, y_pred):
    return K.sum(y_true, axis=[1,2,3])

def sum_y_pred(y_true, y_pred):
    return K.sum(y_pred, axis=[1,2,3])

new_model = tf.keras.models.load_model(version+'/model_'+version, compile=False, custom_objects = {"intersection":intersection, "sum_y_true":sum_y_true, "sum_y_pred":sum_y_pred})
new_model_encoder = tf.keras.models.load_model(version+'/encoder_'+version, compile=False)
new_model.summary()
new_model_encoder.summary()
##########################################

##########################################
def predict(Image3D, index):#0: model, 1: encoder, 2: encoder_c2, 3: decoder_c2
    predicted = np.array([])
    if (index == 0): predicted = new_model.predict(Image3D)
    if (index == 1): predicted = new_model_encoder.predict(Image3D)
    if (index == 2): predicted = new_model_encoder_c2.predict(Image3D)
    if (index == 3): predicted = new_model_decoder_c2.predict(Image3D)

    print("predicted.shape ", predicted.shape)
    return predicted
########################################## 

########################################## 
def plotpredimages(SUM_Image, name, index):
    plt.figure()
    print("SUM_Image N events min max = ", np.min(SUM_Image), np.max(SUM_Image))
    if (np.max(SUM_Image) <= 1 and np.min(SUM_Image) > 0):
        plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=np.max(SUM_Image)*0.01))
    elif (np.min(SUM_Image) == 0):
        plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.1))
    elif (np.min(SUM_Image) <  0 or np.max(SUM_Image) > 1):
        plt.imshow(SUM_Image.T, origin='lower', vmin=np.min(SUM_Image), vmax=np.max(SUM_Image))
    plt.colorbar()
    if (index == -1): plt.title("predicted (channels 1+2+3)", fontsize=15)
    if (index != -1): plt.title("predicted (channel %d)"%(j), fontsize=15)
    plt.xlabel("$\eta$ cell", fontsize=15)
    plt.ylabel("$\phi$ cell", fontsize=15)
    plt.savefig(plotdir+name+".pdf", dpi=1000)
    plt.close()
########################################## 

##########################################
def Chamferloss(y_true, y_pred): # check lossFunctions.py

    chamfer = np.empty([nevts, 1])
    for i in range(parts):
        inputs_R  = y_true[i,:,:,0]
        outputs_R = y_pred[i,:,:,0]
        inputs_G  = y_true[i,:,:,1]
        outputs_G = y_pred[i,:,:,1]
        inputs_B  = y_true[i,:,:,2]
        outputs_B = y_pred[i,:,:,2]

        y_true_R = np.reshape(inputs_R , shape=(1, (288*360))) #(1, 288*360)
        y_pred_R = np.reshape(outputs_R, shape=(1, (288*360)))
        y_true_G = np.reshape(inputs_G , shape=(1, (288*360)))
        y_pred_G = np.reshape(outputs_G, shape=(1, (288*360)))
        y_true_B = np.reshape(inputs_B , shape=(1, (288*360)))
        y_pred_B = np.reshape(outputs_B, shape=(1, (288*360)))

        ####################################                                                                                                                                                                
        t_true_keep_R = np.take(y_true_R[i], np.nonzero(y_true_R[i]))#(N, ) where N is # non-zero elements 
        t_pred_keep_R = np.take(y_true_R[i], np.nonzero(y_true_R[i]))
        t_true_keep_G = np.take(y_true_G[i], np.nonzero(y_true_R[i]))
        t_pred_keep_G = np.take(y_true_G[i], np.nonzero(y_true_R[i]))
        t_true_keep_B = np.take(y_true_B[i], np.nonzero(y_true_R[i]))
        t_pred_keep_B = np.take(y_true_B[i], np.nonzero(y_true_R[i]))

        t_true_keep_R = np.reshape(t_true_keep_R, shape=(t_true_keep_R.shape[0], 1)) #(N, 1) where N is # non-zero elements
        t_pred_keep_R = np.reshape(t_pred_keep_R, shape=(t_pred_keep_R.shape[0], 1))
        t_true_keep_G = np.reshape(t_true_keep_G, shape=(t_true_keep_G.shape[0], 1))
        t_pred_keep_G = np.reshape(t_pred_keep_G, shape=(t_pred_keep_G.shape[0], 1))
        t_true_keep_B = np.reshape(t_true_keep_B, shape=(t_true_keep_B.shape[0], 1))
        t_pred_keep_B = np.reshape(t_pred_keep_B, shape=(t_pred_keep_B.shape[0], 1))

        t_true_keep_R_conc = np.concatenate([t_true_keep_R, t_true_keep_G, t_true_keep_B], axis=-1) #(N, 3) where N is # non-zero elements
        t_pred_keep_R_conc = np.concatenate([t_pred_keep_R, t_pred_keep_G, t_pred_keep_B], axis=-1)

        expand_inputs  = np.expand_dims(t_true_keep_R_conc, 1)
        expand_outputs = np.expand_dims(t_pred_keep_R_conc, 0)

        distances = np.math.reduce_sum(np.math.squared_difference(expand_inputs, expand_outputs), -1)
        min_dist_to_inputs = np.math.reduce_min(distances,0)
        min_dist_to_outputs = np.math.reduce_min(distances,1)

        chamfer[i] = np.math.reduce_sum(min_dist_to_inputs, 0) + np.math.reduce_sum(min_dist_to_outputs, 0)
    return chamfer

def mod_sse(y_true, y_pred):  # sum of square of errors for non-zero elements only  # check lossFunctions.py

    y_true = y_true.reshape((nevts, (288*360*3)))
    y_pred = y_pred.reshape((nevts, (288*360*3)))

    t_true_keep = []
    t_pred_keep = []
    sq = []
    sse = np.empty([nevts, 1])
    for i in range(nevts):
        t_true_keep = np.take(y_true[i], np.nonzero(y_true[i]))
        t_pred_keep = np.take(y_pred[i], np.nonzero(y_true[i]))
        #print("t_true_keep.shape", t_true_keep[i].shape)

        sq = (t_true_keep - t_pred_keep) * (t_true_keep - t_pred_keep)
        #print("sq.shape", sq[i].shape)

        sse[i] = sq.sum(-1)
        #print("sse ", sse[i].shape)
    
    #print("sse ", sse.size)
    #print("sse ", np.sum(sse, axis=0))
    return sse

def sse(data_in, data_out): # sum of squares of errors  # check lossFunctions.py
    sse = np.array([], dtype=np.float32)
    sse = (data_out-data_in)*(data_out-data_in)
    
    index_1 = sse.shape[1]
    index_2 = sse.shape[2]
    index_3 = sse.shape[3]

    sse = sse.sum(-1)
    sse = sse.sum(-1)
    sse = sse.sum(-1)
    return sse

def Diceloss(targets, inputs, smooth=1e-6):  # dice coefficient  # check lossFunctions.py

    dice_final = np.empty([nevts, 1])
    for i in range(nevts):
        inputs_tmp  = inputs[i].reshape(((288*360*3), 1))
        targets_tmp = targets[i].reshape((1, (288*360*3)))

        intersection = np.sum(np.dot(targets_tmp, inputs_tmp))
        dice = (2*intersection + smooth) / (np.sum(targets_tmp * targets_tmp) + np.sum(inputs_tmp * inputs_tmp) + smooth)
        dice_final[i] = 1 - dice
    return dice_final

def oppDiceloss(targets, inputs, smooth=1e-6):  # inverse of dice coefficient  # check lossFunctions.py

    dice_final = np.empty([nevts, 1])
    for i in range(nevts):
        inputs_tmp  = inputs[i].reshape(((288*360*3), 1))
        targets_tmp = targets[i].reshape((1, (288*360*3)))

        intersection = np.sum(np.dot(targets_tmp, inputs_tmp))
        dice = (np.sum(targets_tmp * targets_tmp) + np.sum(inputs_tmp * inputs_tmp) + smooth) / (2*intersection + smooth)
        dice_final[i] = dice
    return dice_final
##########################################

########################################## 
def roc_compare(pQCD, pSUEP, name):

    ######################
    maxScore = np.float32(max(np.max(pQCD), np.max(pSUEP)))
    minScore = np.float32(min(np.max(pQCD), np.min(pSUEP)))
    print("maxScore ", maxScore, "minScore ", minScore)
    if (maxScore >  9999999.): maxScore =  9999999
    if (minScore < -9999999.): minScore = -9999999
    print("maxScore ", maxScore, "minScore ", minScore)

    plt.figure()
    if (name == "AE_loss"):
        minScore = 0.
        maxScore = 1000
    if (name == "True_E"):
        minScore = 0.
        maxScore = 6
    if (name == "inv_Reconstructed_E"):
        minScore = 0.
        maxScore = 550
    plt.hist(pQCD, bins=100, label='QCD', density=False, range=(minScore, maxScore), histtype='step', fill=False)
    plt.hist(pSUEP, bins=100, label='SUEP('+mass+' GeV)', density=False, range=(minScore, maxScore), histtype='step', fill=False)
    plt.semilogy()
    plt.xlabel(name)
    plt.ylabel("#events")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(plotdir+"compare_"+name+".pdf",dpi=1000)
    plt.close()

    ######################
    targetSUEP = np.ones(pSUEP.shape[0])
    targetQCD  = np.zeros(pQCD.shape[0])
    trueVal    = np.concatenate((targetSUEP, targetQCD))
    predVal    = np.concatenate((pSUEP, pQCD))
    
    fpr, tpr, threshold = roc_curve(trueVal, predVal)
    auc1 = auc(fpr, tpr)
    
    if (name == "AE_loss"):
        hf.create_dataset('fpr_AE_loss', data=fpr)
        hf.create_dataset('tpr_AE_loss', data=tpr)
    if (name == "inv_Reconstructed_E"):
        hf.create_dataset('fpr_invpt', data=fpr)
        hf.create_dataset('tpr_intpt', data=tpr)
    if (name == "Reconstructed_E"):
        hf.create_dataset('fpr_pt', data=fpr)
        hf.create_dataset('tpr_pt', data=tpr)
    if (name == "inv_True_E"):
        hf.create_dataset('fpr_truept', data=fpr)
        hf.create_dataset('tpr_truept', data=tpr)

    plt.figure()
    plt.plot(tpr,fpr,label='SUEP('+mass+' GeV) Anomaly Detection, auc = %0.1f%%'%(auc1*100.))
    plt.title(name, fontsize=15)
    plt.xlabel("sig. efficiency (TPR)")
    plt.ylabel("bkg. mistag rate (FPR)")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(plotdir+"roc_"+name+".pdf",dpi=1000)
    if (name == "AE_loss" or name == "True_E" or name == "Reconstructed_E" or name == "inv_Reconstructed_E" or name == "Mean_Reconstructed_E_LatentSpace"):
        plt.yscale('log')##########
        plt.savefig(plotdir+"roc_"+name+"_log.pdf",dpi=1000)
    plt.close()

def sumall(data_in):
    data_in = data_in.sum(-1)
    data_in = data_in.sum(-1)
    data_in = data_in.sum(-1)
    return data_in
########################################## 

########################################## 
#Plot 3 channels separately of input images (1 event and N events)
if(rawImages == True):
    plotimages(Image3D_QCD, 0)
    plotimages(Image3D_SUEP, 1)

#compare and roc for true energy
pQCD  = sumall(Image3D_QCD[:nevts])
pSUEP = sumall(Image3D_SUEP[:nevts])
roc_compare(pQCD, pSUEP, "True_E")

pQCD          = 1/pQCD
pSUEP         = 1/pSUEP
roc_compare(pQCD, pSUEP, "inv_True_E")
del pQCD, pSUEP

#compare and roc for AE predicted energy (also plot summed predicted energy of 3-channels (1 event and N events))
predictedQCD  = predict(Image3D_QCD[:nevts], 0)
predictedSUEP = predict(Image3D_SUEP[:nevts], 0)
pQCD          = sumall(predictedQCD)
pSUEP         = sumall(predictedSUEP)
roc_compare(pQCD, pSUEP, "Reconstructed_E")

#pQCD[pQCD == 0] = 1e-6 
pQCD          = 1/pQCD
#pSUEP[pSUEP == 0] = 1e-6
pSUEP         = 1/pSUEP
roc_compare(pQCD, pSUEP, "inv_Reconstructed_E")
del pQCD, pSUEP

if(predictedImages):
    plotpredimages(np.sum(np.sum(predictedQCD, axis=-1), axis=0), "AE_pred_QCD_%dEvents"%(nevts), -1)
    plotpredimages(np.sum(np.sum(predictedSUEP, axis=-1), axis=0), "AE_pred_SUEP_%dEvents"%(nevts), -1)
    plotpredimages(np.sum(np.sum(predictedQCD[:1, :, :, :], axis=-1), axis=0), "AE_pred_QCD_%dEvent"%(1), -1)
    plotpredimages(np.sum(np.sum(predictedSUEP[:1, :, :, :], axis=-1), axis=0), "AE_pred_SUEP_%dEvent"%(1), -1)
    plotpredimages(np.sum(np.sum(predictedQCD[:100, :, :, :], axis=-1), axis=0), "AE_pred_QCD_%dEvent"%(100), -1)
    plotpredimages(np.sum(np.sum(predictedSUEP[:100, :, :, :], axis=-1), axis=0), "AE_pred_SUEP_%dEvent"%(100), -1)

#compare and roc for AE loss
lossQCD  = oppDiceloss(Image3D_QCD[:nevts], predictedQCD)
lossSUEP = oppDiceloss(Image3D_SUEP[:nevts], predictedSUEP)
pQCD     = lossQCD
pSUEP    = lossSUEP
roc_compare(pQCD, pSUEP, "AE_loss")
del pQCD, pSUEP

#delete not needed stuff
del predictedQCD, predictedSUEP, lossQCD, lossSUEP

#import sys
#sys.exit(0)

#compare and roc for encoder-only predicted energy
predictedQCD_enc  = predict(Image3D_QCD[:nevts], 1)
predictedSUEP_enc = predict(Image3D_SUEP[:nevts], 1)
#lspaceQCD  = np.array([240[nevts]], dtype=np.float32)
#lspaceSUEP = np.array([240[nevts]], dtype=np.float32)

lspaceQCD  = np.zeros((240, nevts), dtype=np.float32)
lspaceSUEP = np.zeros((240, nevts), dtype=np.float32)

for j in range(predictedQCD_enc.shape[1]):
    if(predictedImages):
        #plot summed encoder-only predicted energy for 2 channels separately (1 event and N events)
        plotpredimages(np.sum(predictedQCD_enc[:, :, :, j], axis=0), "Enc_pred_QCD_%dEvents_layer%d"%(nevts, j), j)
        plotpredimages(np.sum(predictedSUEP_enc[:, :, :, j], axis=0), "Enc_pred_SUEP_%dEvents_layer%d"%(nevts, j), j)
        plotpredimages(np.sum(predictedQCD_enc[:1, :, :, j], axis=0), "Enc_pred_QCD_1Event_layer%d"%(j), j)
        plotpredimages(np.sum(predictedSUEP_enc[:1, :, :, j], axis=0), "Enc_pred_SUEP_1Event_layer%d"%(j), j)

    for k in range(predictedQCD_enc.shape[2]):
        for l in range(predictedQCD_enc.shape[3]):
            pQCD  = np.sum(np.sum(np.sum(predictedQCD_enc[:,j:j+1,k:k+1,l:l+1], axis=-1), axis=-1), axis=-1)#(2000, 2, 5, 2) #[:,1:2,*:*,0:1]             
            pSUEP = np.sum(np.sum(np.sum(predictedSUEP_enc[:,j:j+1,k:k+1,l:l+1], axis=-1), axis=-1), axis=-1)
            #roc_compare(pQCD, pSUEP, "Reconstructed_E_encoder_%d_%d_%d"%(j, k, l))

            for m in range(240):
                lspaceQCD[m]  = pQCD
                lspaceSUEP[m] = pSUEP
            del pQCD, pSUEP

print("lspaceQCD shape", lspaceQCD.shape, "sum ", np.sum(lspaceQCD, axis=0))
print("lspaceSUEP shape", lspaceSUEP.shape, "sum ", np.sum(lspaceSUEP, axis=0))
lspaceQCD  = np.mean(lspaceQCD, axis=0)
lspaceSUEP = np.mean(lspaceSUEP, axis=0)
print("lspaceQCD shape", lspaceQCD.shape, "mean ", lspaceQCD)
print("lspaceSUEP shape", lspaceQCD.shape, "mean ", lspaceSUEP)
roc_compare(lspaceQCD, lspaceSUEP, "Mean_Reconstructed_E_LatentSpace") 

#delete not needed stuff
del Image3D_QCD, Image3D_SUEP, predictedQCD_enc, predictedSUEP_enc

hf.close()
########################################## 
