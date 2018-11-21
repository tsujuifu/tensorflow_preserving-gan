# CelebA

import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.misc import imsave
import cv2
import itertools
from tqdm import tqdm

from Src.CelebA.utils import *
from Src.CelebA.neural_style import build_model
from Src.CelebA.nets_bujo1 import *
from Src.CelebA.models import *

# Hyperparameter
N_SAMPLES = 36

NPX = 64
LR = 9e-2
B1 = 0.4
B2 = 0.99
MAX_STEP = 40
MAIN_COEF = 1
MASK_POS = None
NOISE_COEF = LR*0

LAT_LAYER = 'relu4_1'
LAT_WEIGHT = 1
LAT_TYPE = 'Feature'
LAT_PROB = 0.5

MASK = 1
ALPHA = 0
MASK_RECON = 1 if MASK_POS is None else (1-MASK)
GAMMA = 0

MASK_TYPE = 'main'

def produce(img_cont, mask_pix, mask_lat):
    MASK_LAT = mask_lat
    img_cont = img_cont*mask_pix.reshape((64, 64, 1))
    img_cont = preprocess(img_cont, npx=NPX)
    
    # Init
    tf.reset_default_graph()

    with tf.variable_scope('Paraphrasing') as scope:
        pb = tf.get_variable('Paint_board', [N_SAMPLES, 64], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    
    op_init_pb = tf.variables_initializer([pb])

    init_img, var = GeneratorCNN(pb, 128, 3, 4, 'NCHW', reuse=False)
    init_img = tf.transpose(init_img, [0, 2, 3, 1])
    init_img = tf.image.resize_nearest_neighbor(init_img, [NPX, NPX])
    init_img = (init_img+1)/2*255

    init_img_tar = init_img
    init_img *= mask_pix.reshape((1, 64, 64, 1))
    init_img_vgg = vgg_img(init_img)

    net_cont = build_model(img_cont, mask=tf.constant(MASK, dtype=tf.float32))
    net_vgg = build_model(init_img_vgg, is_gen=True, mask=tf.constant(MASK, dtype=tf.float32))

    sess = tf.Session()
    saver = tf.train.Saver(var)
    saver.restore(sess, tf.train.latest_checkpoint('Model/CelebA/'))

    print('---Build Model---')
    
    # Loss
    loss_cont, loss_col = sum_content_losses(sess, net_cont, net_vgg, None, None, img_cont, content_layers=[LAT_LAYER], content_layer_weights=[LAT_WEIGHT], content_types=[LAT_TYPE], mask_types=[MASK_TYPE], latent_mask=[MASK_LAT.reshape((1, 8, 8, 1))])
    loss_total = MAIN_COEF*loss_cont
    var_alpha = tf.get_variable(name='alpha', dtype=tf.float32, initializer=ALPHA*1.0)
    
    # Optim
    init_z = pb

    with tf.variable_scope('adam_optimizer') as optim:
        optimizer = tf.train.AdamOptimizer(LR, beta1=B1, beta2=B2)
    
        gvs = optimizer.compute_gradients(loss_total, var_list=[init_z])
        gvs = [(grad+NOISE_COEF*tf.random_normal(tf.shape(grad)), var) for grad, var in gvs]
    
        op_train = optimizer.apply_gradients(gvs)
        var = tf.contrib.framework.get_variables(optim)

    op_init = tf.variables_initializer(var+[var_alpha])
    
    # Run
    sess.run(op_init_pb)
    sess.run(op_init)

    # Warmup

    # Iter
    for s in tqdm(range(MAX_STEP)):
        sess.run(op_train)

    # Result
    out = sess.run(init_img_tar)
    
    ret = []
    for i in range(N_SAMPLES):
        ret.append(np.clip(out[i], 0, 255)/255)
    
    return ret
