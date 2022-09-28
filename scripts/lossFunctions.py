parts = 128
nevts = 4000

# tf.config.run_functions_eagerly(True) # some losses will not work without this

########################################################################
########################################################################

def mse_tf(y_true, y_pred): # mean of sum of squares of errors
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

def mse_np(data_in, data_out): # mean of sum of squares of errors
    mse = (data_out-data_in)*(data_out-data_in)
    
    index_1 = mse.shape[1]
    index_2 = mse.shape[2]
    index_3 = mse.shape[3]

    mse = mse.sum(-1)
    mse = mse.sum(-1)
    mse = mse.sum(-1)
    mse /= (index_1*index_2*index_3)
    return mse

########################################################################
########################################################################

def mod_sse_tf(y_true, y_pred): # sum of square of errors for non-zero elements only

    sse =[]
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1] #idx_keep_in shape is (N, ) where N is # non-zero elements
        t_true_keep = tf.gather(y_true_tmp[0,:], idx_keep_in) #t_true_keep shape is (N, )
        t_pred_keep = tf.gather(y_pred_tmp[0,:], idx_keep_in) #t_pred_keep shape is (N, )

        sse.append(K.sum(K.square(t_true_keep - t_pred_keep)))
    return sse

def mod_sse_np(y_true, y_pred): # sum of square of errors for non-zero elements only

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

########################################################################
########################################################################

ALPHA = 0.8
GAMMA = 2

def FocalLoss_tf(y_true, y_pred, alpha=ALPHA, gamma=GAMMA):

    focal = []
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1]
        y_true_tmp  = tf.gather(y_true_tmp[0,:], idx_keep_in)
        y_pred_tmp  = tf.gather(y_pred_tmp[0,:], idx_keep_in)

        y_true_tmp = tf.reshape(y_true_tmp, shape=(y_true_tmp.shape[0]))
        y_pred_tmp = tf.reshape(y_pred_tmp, shape=(y_pred_tmp.shape[0]))

        BCE = K.binary_crossentropy(y_true_tmp, y_pred_tmp)
        BCE_EXP = K.exp(-BCE)
        focal.append(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal

# or we can do
# from focal_loss import BinaryFocalLoss
# model.compile(optimizer=opt, loss = BinaryFocalLoss(gamma=2))

########################################################################
########################################################################

def IoULoss_tf(targets, inputs, smooth=1e-6):
    
    BCE = K.binary_crossentropy(targets, inputs)

    inputs  = tf.reshape(inputs,  shape=((parts*288*360*3), 1))
    targets = tf.reshape(targets, shape=(1, (parts*288*360*3)))

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


########################################################################
########################################################################

def oppDiceloss_tf(y_true, y_pred, smooth=1e-6): # inverse of dice coefficient - 1

    dice = []
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1] #idx_keep_in shape is (N, ) where N is # non-zero elements
        y_true_tmp  = tf.gather(y_true_tmp[0,:], idx_keep_in) #t_true_keep shape is (N, )
        y_pred_tmp  = tf.gather(y_pred_tmp[0,:], idx_keep_in) #t_pred_keep shape is (N, )

        y_true_tmp = tf.reshape(y_true_tmp, shape=(1, y_true_tmp.shape[0]))
        y_pred_tmp = tf.reshape(y_pred_tmp, shape=(y_pred_tmp.shape[0], 1))

        intersection = K.sum(K.dot(y_true_tmp, y_pred_tmp))
        dice.append((K.sum(K.square(y_true[i])) + K.sum(K.square(y_pred[i])) + smooth) / (2 * intersection + smooth) - 1)
    return dice

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

def oppDiceloss_np(targets, inputs, smooth=1e-6): # inverse of dice coefficient 

    dice_final = np.empty([nevts, 1])
    for i in range(nevts):
        # we don't only select non-zero elements because the dot product will anyway be zero
        inputs_tmp  = inputs[i].reshape(((288*360*3), 1))
        targets_tmp = targets[i].reshape((1, (288*360*3)))

        intersection = np.sum(np.dot(targets_tmp, inputs_tmp))
        dice = (np.sum(targets_tmp * targets_tmp) + np.sum(inputs_tmp * inputs_tmp) + smooth) / (2*intersection + smooth) # here we don't do -1
        dice_final[i] = dice
    return dice_final

########################################################################
########################################################################

def Chamferloss_tf(y_true, y_pred): # loss starting from non-zero elements in input + loss starting from non-zero elements in output

    chamfer = []
    for i in range(parts):        
        inputs_R  = y_true[i,:,:,0]
        outputs_R = y_pred[i,:,:,0]
        inputs_G  = y_true[i,:,:,1]
        outputs_G = y_pred[i,:,:,1]
        inputs_B  = y_true[i,:,:,2]
        outputs_B = y_pred[i,:,:,2]

        y_true_R = tf.reshape(inputs_R , shape=(1, (288*360))) # (1, 288*360)
        y_pred_R = tf.reshape(outputs_R, shape=(1, (288*360)))
        y_true_G = tf.reshape(inputs_G , shape=(1, (288*360)))
        y_pred_G = tf.reshape(outputs_G, shape=(1, (288*360)))
        y_true_B = tf.reshape(inputs_B , shape=(1, (288*360)))
        y_pred_B = tf.reshape(outputs_B, shape=(1, (288*360)))

        ####################################
        idx_keep_in = tf.where(y_true_R[0,:]>0)[:,-1]
        t_true_keep_R = tf.gather(y_true_R[0,:], idx_keep_in) # (N, ) where N is # non-zero elements
        t_pred_keep_R = tf.gather(y_pred_R[0,:], idx_keep_in)
        t_true_keep_G = tf.gather(y_true_G[0,:], idx_keep_in)
        t_pred_keep_G = tf.gather(y_pred_G[0,:], idx_keep_in)
        t_true_keep_B = tf.gather(y_true_B[0,:], idx_keep_in)
        t_pred_keep_B = tf.gather(y_pred_B[0,:], idx_keep_in)

        t_true_keep_R = tf.reshape(t_true_keep_R, shape=(t_true_keep_R.shape[0], 1)) # (N, 1) where N is # non-zero elements
        t_pred_keep_R = tf.reshape(t_pred_keep_R, shape=(t_pred_keep_R.shape[0], 1))
        t_true_keep_G = tf.reshape(t_true_keep_G, shape=(t_true_keep_G.shape[0], 1))
        t_pred_keep_G = tf.reshape(t_pred_keep_G, shape=(t_pred_keep_G.shape[0], 1))
        t_true_keep_B = tf.reshape(t_true_keep_B, shape=(t_true_keep_B.shape[0], 1))
        t_pred_keep_B = tf.reshape(t_pred_keep_B, shape=(t_pred_keep_B.shape[0], 1))

        t_true_keep_conc = tf.concat([t_true_keep_R, t_true_keep_G, t_true_keep_B], axis=-1) # (N, 3) where N is # non-zero elements
        t_pred_keep_conc = tf.concat([t_pred_keep_R, t_pred_keep_G, t_pred_keep_B], axis=-1)

        expand_inputs  = tf.expand_dims(t_true_keep_conc, 1)
        expand_outputs = tf.expand_dims(t_pred_keep_conc, 0)

        distances = tf.math.reduce_sum(tf.math.squared_difference(expand_inputs, expand_outputs), -1)
        min_dist_to_inputs = tf.math.reduce_min(distances, 0)
        #min_dist_to_outputs = tf.math.reduce_min(distances,1)
        ####################################
        ####################################           
        idx_keep_in = tf.where(y_pred_R[0,:]>0)[:,-1] # we also tried > 5.
        t_true_keep_R = tf.gather(y_true_R[0,:], idx_keep_in) #(N, ) where N is # non-zero elements
        t_pred_keep_R = tf.gather(y_pred_R[0,:], idx_keep_in)
        t_true_keep_G = tf.gather(y_true_G[0,:], idx_keep_in)
        t_pred_keep_G = tf.gather(y_pred_G[0,:], idx_keep_in)
        t_true_keep_B = tf.gather(y_true_B[0,:], idx_keep_in)
        t_pred_keep_B = tf.gather(y_pred_B[0,:], idx_keep_in)

        t_true_keep_R = tf.reshape(t_true_keep_R, shape=(t_true_keep_R.shape[0], 1)) #(N, 1) where N is # non-zero elements
        t_pred_keep_R = tf.reshape(t_pred_keep_R, shape=(t_pred_keep_R.shape[0], 1))
        t_true_keep_G = tf.reshape(t_true_keep_G, shape=(t_true_keep_G.shape[0], 1))
        t_pred_keep_G = tf.reshape(t_pred_keep_G, shape=(t_pred_keep_G.shape[0], 1))
        t_true_keep_B = tf.reshape(t_true_keep_B, shape=(t_true_keep_B.shape[0], 1))
        t_pred_keep_B = tf.reshape(t_pred_keep_B, shape=(t_pred_keep_B.shape[0], 1))

        t_true_keep_conc = tf.concat([t_true_keep_R, t_true_keep_G, t_true_keep_B], axis=-1) #(N, 3) where N is # non-zero elements
        t_pred_keep_conc = tf.concat([t_pred_keep_R, t_pred_keep_G, t_pred_keep_B], axis=-1)

        expand_inputs  = tf.expand_dims(t_true_keep_conc, 0) # we inverted it
        expand_outputs = tf.expand_dims(t_pred_keep_conc, 1) # we inverted it

        distances = tf.math.reduce_sum(tf.math.squared_difference(expand_outputs, expand_inputs), -1)
        #min_dist_to_inputs = tf.math.reduce_min(distances,1) # we inverted it
        min_dist_to_outputs = tf.math.reduce_min(distances, 0) # we inverted it
        ####################################

        chamfer.append(tf.math.reduce_sum(min_dist_to_inputs, 0) + tf.math.reduce_sum(min_dist_to_outputs, 0))
    return chamfer

def Chamferloss_np(y_true, y_pred): # loss starting from non-zero elements in input + loss starting from non-zero elements in output

    chamfer = np.empty([nevts, 1])
    for i in range(parts):
        inputs_R  = y_true[i,:,:,0]
        outputs_R = y_pred[i,:,:,0]
        inputs_G  = y_true[i,:,:,1]
        outputs_G = y_pred[i,:,:,1]
        inputs_B  = y_true[i,:,:,2]
        outputs_B = y_pred[i,:,:,2]

        y_true_R = np.reshape(inputs_R , shape=(1, (288*360))) # (1, 288*360) 
        y_pred_R = np.reshape(outputs_R, shape=(1, (288*360)))
        y_true_G = np.reshape(inputs_G , shape=(1, (288*360)))
        y_pred_G = np.reshape(outputs_G, shape=(1, (288*360)))
        y_true_B = np.reshape(inputs_B , shape=(1, (288*360)))
        y_pred_B = np.reshape(outputs_B, shape=(1, (288*360)))

        #################################### 
        t_true_keep_R = np.take(y_true_R[i], np.nonzero(y_true_R[i])) # (N, ) where N is # non-zero elements
        t_pred_keep_R = np.take(y_true_R[i], np.nonzero(y_true_R[i]))
        t_true_keep_G = np.take(y_true_G[i], np.nonzero(y_true_R[i]))
        t_pred_keep_G = np.take(y_true_G[i], np.nonzero(y_true_R[i]))
        t_true_keep_B = np.take(y_true_B[i], np.nonzero(y_true_R[i]))
        t_pred_keep_B = np.take(y_true_B[i], np.nonzero(y_true_R[i]))

        t_true_keep_R = np.reshape(t_true_keep_R, shape=(t_true_keep_R.shape[0], 1)) # (N, 1) where N is # non-zero elements
        t_pred_keep_R = np.reshape(t_pred_keep_R, shape=(t_pred_keep_R.shape[0], 1))
        t_true_keep_G = np.reshape(t_true_keep_G, shape=(t_true_keep_G.shape[0], 1))
        t_pred_keep_G = np.reshape(t_pred_keep_G, shape=(t_pred_keep_G.shape[0], 1))
        t_true_keep_B = np.reshape(t_true_keep_B, shape=(t_true_keep_B.shape[0], 1))
        t_pred_keep_B = np.reshape(t_pred_keep_B, shape=(t_pred_keep_B.shape[0], 1))

        t_true_keep_conc = np.concatenate([t_true_keep_R, t_true_keep_G, t_true_keep_B], axis=-1) # (N, 3) where N is # non-zero elements
        t_pred_keep_conc = np.concatenate([t_pred_keep_R, t_pred_keep_G, t_pred_keep_B], axis=-1)

        expand_inputs  = np.expand_dims(t_true_keep_conc, 1)
        expand_outputs = np.expand_dims(t_pred_keep_conc, 0)

        distances = np.math.reduce_sum(np.math.squared_difference(expand_inputs, expand_outputs), -1)
        min_dist_to_inputs = np.math.reduce_min(distances,0)
        #min_dist_to_outputs = np.math.reduce_min(distances,1)
        ####################################
        ####################################           
        t_true_keep_R = np.take(y_true_R[i], np.nonzero(y_pred_R[i])) # (N, ) where N is # non-zero elements
        t_pred_keep_R = np.take(y_true_R[i], np.nonzero(y_pred_R[i]))
        t_true_keep_G = np.take(y_true_G[i], np.nonzero(y_pred_R[i]))
        t_pred_keep_G = np.take(y_true_G[i], np.nonzero(y_pred_R[i]))
        t_true_keep_B = np.take(y_true_B[i], np.nonzero(y_pred_R[i]))
        t_pred_keep_B = np.take(y_true_B[i], np.nonzero(y_pred_R[i]))

        t_true_keep_R = np.reshape(t_true_keep_R, shape=(t_true_keep_R.shape[0], 1)) # (N, 1) where N is # non-zero elements
        t_pred_keep_R = np.reshape(t_pred_keep_R, shape=(t_pred_keep_R.shape[0], 1))
        t_true_keep_G = np.reshape(t_true_keep_G, shape=(t_true_keep_G.shape[0], 1))
        t_pred_keep_G = np.reshape(t_pred_keep_G, shape=(t_pred_keep_G.shape[0], 1))
        t_true_keep_B = np.reshape(t_true_keep_B, shape=(t_true_keep_B.shape[0], 1))
        t_pred_keep_B = np.reshape(t_pred_keep_B, shape=(t_pred_keep_B.shape[0], 1))

        t_true_keep_conc = np.concatenate([t_true_keep_R, t_true_keep_G, t_true_keep_B], axis=-1) # (N, 3) where N is # non-zero elements
        t_pred_keep_conc = np.concatenate([t_pred_keep_R, t_pred_keep_G, t_pred_keep_B], axis=-1)

        expand_inputs  = np.expand_dims(t_true_keep_conc, 0) # we inverted it
        expand_outputs = np.expand_dims(t_pred_keep_conc, 1) # we inverted it

        distances = np.math.reduce_sum(np.math.squared_difference(expand_inputs, expand_outputs), -1)
        #min_dist_to_inputs = np.math.reduce_min(distances,1) # we inverted it
        min_dist_to_outputs = np.math.reduce_min(distances,0) # we inverted it
        ####################################

        chamfer[i] = np.math.reduce_sum(min_dist_to_inputs, 0) + np.math.reduce_sum(min_dist_to_outputs, 0)
    return chamfer

########################################################################
########################################################################
