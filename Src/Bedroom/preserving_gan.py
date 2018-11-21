# Bedroom

import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.misc import imsave
import cv2
import itertools
from tqdm import tqdm

from Src.Bedroom.utils import *
from Src.Bedroom.neural_style import build_model
from Src.Bedroom.nets_bujo1 import *
from Src.Bedroom.models import *

# Hyperparameter
N_SAMPLES = 36

NPX = 64
LR = 1e-2
B1 = 0.4
B2 = 0.99
MAX_STEP = 400
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

    ###
    mask_pix = mask_pix*0+1
    ###

    MASK_LAT = mask_lat
    img_cont = img_cont*mask_pix.reshape((64, 64, 1))
    img_cont = preprocess(img_cont, npx=NPX)

    # Init
    tf.reset_default_graph()

    with tf.variable_scope('Paraphrasing') as scope:
        pb = tf.get_variable('Paint_board', [N_SAMPLES, 128], initializer=tf.random_normal_initializer())

    op_init_pb = tf.variables_initializer([pb])

    G, D = GeneratorAndDiscriminator()

    init_img_org = G(N_SAMPLES, noise=pb)
    init_img = tf.transpose(tf.reshape(init_img_org, [-1,3, 64, 64]), [0, 2, 3, 1])
    init_img = tf.image.resize_nearest_neighbor(init_img, [NPX, NPX])
    init_img = (init_img+1)/2*255.

    init_img_tar = init_img
    init_img *= mask_pix.reshape(1, 64, 64, 1)
    init_img_vgg = vgg_img(init_img)

    init_img_log = D(init_img_org)
    g_var = tf.contrib.framework.get_variables('Generator')
    d_var = tf.contrib.framework.get_variables('Discriminator')
    var = g_var+d_var

    net_cont = build_model(img_cont, mask=tf.constant(MASK, dtype=tf.float32))
    net_vgg = build_model(init_img_vgg, is_gen=True, mask=tf.constant(MASK, dtype=tf.float32))

    sess = tf.Session()
    saver = tf.train.Saver(var)
    saver.restore(sess, tf.train.latest_checkpoint('Model/Bedroom/'))

    print('---Build Model---')

    # Loss
    loss_cont, loss_col = sum_content_losses(sess, net_cont, net_vgg, None, None, img_cont, content_layers=[LAT_LAYER], content_layer_weights=[LAT_WEIGHT], content_types=[LAT_TYPE], mask_types=[MASK_TYPE], latent_mask=[MASK_LAT.reshape((1, 8, 8, 1))])
    loss_total = MAIN_COEF*loss_cont
    #loss_total = tf.reduce_mean((img_cont - init_img_vgg)**2)
    var_alpha = tf.get_variable(name='alpha', dtype=tf.float32, initializer=ALPHA*1.0)

    # Optim
    init_z = pb

    with tf.variable_scope('adam_optimizer') as optim:
        optimizer = tf.train.GradientDescentOptimizer(LR)

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
    losses = sess.run(loss_col[0])

    idxs = np.argsort(losses)

    ret = []
    for i in range(N_SAMPLES):
        ret.append(np.clip(out[idxs[i]], 0, 255)/255)

    return ret
