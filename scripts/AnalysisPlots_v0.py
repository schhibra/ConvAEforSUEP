##########################################
#import setGPU
import os
###for CPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
#####config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##########################################


########################################## 
plotdir = "plots_suep1000/"
filelist = glob.glob(os.path.join(plotdir, "*"))
for f in filelist:
    os.remove(f)
if (not os.path.exists(plotdir)):
    os.mkdir(plotdir)

rawImages       = True
predictedImages = True
########################################## 


##########################################
def images(inputpath):
    ImageTrk = np.array([])
    ImageECAL = np.array([])
    ImageHCAL = np.array([])

    for fileIN in glob.glob(inputpath):
        print("Appending %s" %fileIN)
        f = h5py.File(fileIN, 'r')
    
        myImageTrk  = np.array(f.get("ImageTrk_PUcorr")[:], dtype=np.float16)
        ImageTrk    = np.concatenate([ImageTrk, myImageTrk], axis=0) if ImageTrk.size else myImageTrk
        
        myImageECAL = np.array(f.get("ImageECAL")[:], dtype=np.float16)
        ImageECAL    = np.concatenate([ImageECAL, myImageECAL], axis=0) if ImageECAL.size else myImageECAL
        
        myImageHCAL = np.array(f.get("ImageHCAL")[:], dtype=np.float16)
        ImageHCAL    = np.concatenate([ImageHCAL, myImageHCAL], axis=0) if ImageHCAL.size else myImageHCAL

        del myImageTrk, myImageECAL, myImageHCAL

    ImageTrk  = ImageTrk.reshape(ImageTrk.shape[0], ImageTrk.shape[1], ImageTrk.shape[2], 1)
    ImageECAL = ImageECAL.reshape(ImageECAL.shape[0], ImageECAL.shape[1], ImageECAL.shape[2], 1)
    ImageHCAL = ImageHCAL.reshape(ImageHCAL.shape[0], ImageHCAL.shape[1], ImageHCAL.shape[2], 1)

    Image3D = np.concatenate([ImageTrk, ImageECAL, ImageHCAL], axis=-1)

    Image3D_zero = np.zeros((Image3D.shape[0], 288, 360, 3))
    Image3D_zero[:, 1:287, :, :] += Image3D

    del ImageTrk, ImageECAL, ImageHCAL, Image3D

    print("Image3D_zero.shape ", Image3D_zero.shape)
    return Image3D_zero

Image3D_QCD  = images("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_qcd_v1/gensim/output/test/qcd_gensim_101_*.h5")
Image3D_SUEP = images("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_suep_v1/gensim/output/test/suep_gensim_h1000_phi2_dp0p7_dt2_10*_*.h5")
##########################################

##########################################
def plotimages(Image3D, index):
    if (index == 0): label = ["Trk_QCD", "ECAL_QCD", "HCAL_QCD"]
    else:            label = ["Trk_SUEP", "ECAL_SEUP", "HCAL_SUEP"]
    for i in range(3):
        plt.figure()
        SUM_Image = np.sum(Image3D[:,:,:,i], axis = 0)
        plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.01))
        plt.colorbar()
        plt.title(label[i], fontsize=15)
        plt.xlabel("$\eta$ cell", fontsize=15)
        plt.ylabel("$\phi$ cell", fontsize=15)
        plt.savefig(plotdir+label[i]+'_AllEvents.pdf', dpi=1000)
    for i in range(3):
        plt.figure()
        SUM_Image = np.sum(Image3D[:1,:,:,i], axis = 0)
        plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.01))
        plt.colorbar()
        plt.title(label[i], fontsize=15)
        plt.xlabel("$\eta$ cell", fontsize=15)
        plt.ylabel("$\phi$ cell", fontsize=15)
        plt.savefig(plotdir+label[i]+'_1Event.pdf', dpi=1000)

if(rawImages == True):
    plotimages(Image3D_QCD, 0)
    plotimages(Image3D_SUEP, 1)
##########################################

##########################################
max_Energy = 2000.

new_model = tf.keras.models.load_model('model_v1', compile=False)
new_model_encoder = tf.keras.models.load_model('model_encoder_v1', compile=False)
#new_model_encoder_c2 = tf.keras.models.load_model('model_encoder_c2_v1', compile=False)
#new_model_decoder_c2 = tf.keras.models.load_model('model_decoder_c2_v1', compile=False)
new_model.summary()
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

predictedQCD        = predict(Image3D_QCD, 0)
predictedQCD_enc    = predict(Image3D_QCD, 1)
#predictedQCD_enc_c2 = predict(Image3D_QCD, 2)
#predictedQCD_dec_c2 = predict(Image3D_QCD, 3)

predictedSUEP        = predict(Image3D_SUEP, 0)
predictedSUEP_enc    = predict(Image3D_SUEP, 1)
#predictedSUEP_enc_c2 = predict(Image3D_SUEP, 2)
#predictedSUEP_dec_c2 = predict(Image3D_SEUP, 3)
########################################## 

########################################## 
def plotpredimages(SUM_Image, name, index):
    plt.figure()
    print("np.min(SUM_Image)", np.min(SUM_Image))
    print("np.max(SUM_Image)", np.max(SUM_Image))
    if (np.min(SUM_Image) >= 0.0): plt.imshow(SUM_Image.T, origin='lower', norm=LogNorm(vmin=0.01))
    if (np.min(SUM_Image) < 0.0): plt.imshow(SUM_Image.T, origin='lower', vmin=np.min(SUM_Image), vmax=np.max(SUM_Image))
    plt.colorbar()
    if (index == -1): 
        plt.title("predicted (channels 1+2+3)", fontsize=15)
    else: 
        plt.title("predicted (channel %d)"%(j), fontsize=15)
    plt.xlabel("$\eta$ cell", fontsize=15)
    plt.ylabel("$\phi$ cell", fontsize=15)
    plt.savefig(plotdir+name, dpi=1000)

if(predictedImages):
    plotpredimages(np.sum(np.sum(predictedQCD, axis=-1), axis=0), "AE_pred_QCD_AllEvents.pdf", -1)
    plotpredimages(np.sum(np.sum(predictedSUEP, axis=-1), axis=0), "AE_pred_SUEP_AllEvents.pdf", -1)

    for j in range(predictedQCD_enc.shape[3]):
        plotpredimages(np.sum(predictedQCD_enc[:, :, :, j], axis=0), "Enc_pred_QCD_AllEvents_layer%d.pdf"%(j), j)
        plotpredimages(np.sum(predictedSUEP_enc[:, :, :, j], axis=0), "Enc_pred_SUEP_AllEvents_layer%d.pdf"%(j), j)
########################################## 

##########################################
def mse(data_in, data_out):
    mse = (data_out-data_in)*(data_out-data_in)
    
    index_1 = mse.shape[1]
    index_2 = mse.shape[2]
    index_3 = mse.shape[3]

    mse = mse.sum(-1)
    mse = mse.sum(-1)
    mse = mse.sum(-1)
    mse /= (index_1*index_2*index_3)
    return mse

lossQCD  = mse(Image3D_QCD, predictedQCD)
lossSUEP = mse(Image3D_SUEP, predictedSUEP)
##########################################

########################################## 
def roc_compare(pQCD, pSUEP, name):

    ######################
    maxScore = max(np.max(pQCD), np.max(pSUEP))
    minScore = min(np.min(pQCD), np.min(pSUEP))
    print("maxScore ", maxScore, "minScore ", minScore)

    plt.figure()
    plt.hist(pQCD, bins=100, label='QCD', density=True, range=(minScore, maxScore), histtype='step', fill=False)
    plt.hist(pSUEP, bins=100, label='SUEP', density=True, range=(minScore, maxScore), histtype='step', fill=False)
    plt.semilogy()
    plt.xlabel(name)
    plt.ylabel("Probability (a.u.)")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(plotdir+"compare_"+name+".pdf",dpi=1000)

    ######################
    targetSUEP = np.ones(pSUEP.shape[0])
    targetQCD  = np.zeros(pQCD.shape[0])
    trueVal    = np.concatenate((targetSUEP, targetQCD))
    predVal    = np.concatenate((pSUEP, pQCD))
    
    fpr, tpr, threshold = roc_curve(trueVal, predVal)
    auc1 = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(tpr,fpr,label='SUEP Anomaly Detection, auc = %0.1f%%'%(auc1*100.))
    plt.title(name, fontsize=15)
    plt.xlabel("sig. efficiency (TPR)")
    plt.ylabel("bkg. mistag rate (FPR)")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(plotdir+"roc_"+name+".pdf",dpi=1000)

def sumall(data_in):
    data_in = data_in.sum(-1)
    data_in = data_in.sum(-1)
    data_in = data_in.sum(-1)
    return data_in

for i in range(4):
    if (i == 0): pQCD = lossQCD;              pSUEP = lossSUEP;              roc_compare(pQCD, pSUEP, "AE_loss")
    if (i == 1): pQCD = sumall(predictedQCD); pSUEP = sumall(predictedSUEP); roc_compare(pQCD, pSUEP, "Predict_E")
    if (i == 2): pQCD = sumall(Image3D_QCD);  pSUEP = sumall(Image3D_SUEP);  roc_compare(pQCD, pSUEP, "True_E")
    if (i == 3): 
        for j in range(predictedQCD_enc.shape[1]):
            for k in range(predictedQCD_enc.shape[2]):
                for l in range(predictedQCD_enc.shape[3]):
                    pQCD = np.sum(np.sum(np.sum(predictedQCD_enc[:,j:j+1,k:k+1,l:l+1], axis=-1), axis=-1), axis=-1)#(2000, 2, 5, 2) #[:,1:2,*:*,0:1]
                    pSUEP = np.sum(np.sum(np.sum(predictedSUEP_enc[:,j:j+1,k:k+1,l:l+1], axis=-1), axis=-1), axis=-1)
                    roc_compare(pQCD, pSUEP, "Predict_E_encoder_%d_%d_%d"%(j, k, l))
    print("pQCD.shape", pQCD.shape)
    print("pSUEP.shape", pSUEP.shape)
########################################## 
