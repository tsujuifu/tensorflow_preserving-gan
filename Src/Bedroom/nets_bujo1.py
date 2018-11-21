import tensorflow as tf
import numpy as np

from tqdm import trange

def my_gram_matrix(x, area, depth):
    x_shape = x.get_shape().as_list()
    print('Latent shape: %s'%x_shape)
    x = tf.transpose(x, [0,3,1,2])

    F = tf.reshape(x, (-1, depth, area))
    #F_mean = tf.expand_dims(tf.reduce_mean(F, [1]),[1])
    F_t = tf.transpose(F, perm=[0,2,1])
    G = tf.matmul(F_t,F)
    G_shape = G.get_shape().as_list()
    print('Gram_matrix_shape: %s'%G_shape)
    return G
def content_layer_loss(p, x, loss_type='Gram', method='preservingGAN'):
    n, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    K = 1. / (2*N * M * n.value)**2

    #if args.content_loss_function   == 1:
    #  K = 1. / (2. * N**0.5 * M**0.5)
    #elif args.content_loss_function == 2:
    #  K = 1. / (2*N * M)**2
    #elif args.content_loss_function == 3:
    #  K = 1. / 2.

    if loss_type == 'Feature':
#         loss = K * tf.reduce_sum(tf.pow((x - p), 2))
        if method=='preservingGAN':
            loss =  tf.maximum(tf.reduce_mean(tf.abs((x - p)), [1,2,3]),0)
        else:
            loss = K * tf.maximum(tf.reduce_sum(tf.pow((x - p), 2), [1,2,3]),1e-9)
    elif loss_type == 'Gram':
#         _, n,_,_ = x.get_shape().as_list()
#         gram_mask = geen_mask(n, prob = 0.8).reshape(1,n,n,1)

#         k = 2
# #         p = tf.nn.max_pool(p, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
# #         x = tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
#         p = p[:,::k,::k,:]
#         x = x[:,::k,::k,:]
#         n, h, w, d = p.get_shape()
#         M = h.value * w.value
#         N = d.value
#         K = 1. / (2*N * M * n.value)**2
#         print('Processed shape: %s'%(p.get_shape().as_list(),))

#         p = tf.nn.relu(p - 0.2 * tf.reduce_max(p))
#         x = tf.nn.relu(x - 0.2 * tf.reduce_max(x))
        P = my_gram_matrix(p , M,N)
        X = my_gram_matrix(x , M,N)

        print(X.get_shape().as_list())

        gram_mask = 1
        loss = K * tf.reduce_sum(tf.abs((P-X)) * gram_mask, [1,2])
    else:
        print('Non-Implemented error')
        return None
    return loss
def sum_content_losses(sess, net, net2, net3, net4, content_img, content_layers, content_layer_weights, content_types, mask_types, mask=None, latent_mask = None, latent_sample_rate = 1.):
    sess.run(net['input'].assign(content_img))
    #sess.run(net3['input'].assign(content_img))

    content_loss = 0.
    loss_collections = []
    print(content_layers, content_layer_weights, content_types, mask_types)
    for idx, (layer, weight, content_type, mask_type) in enumerate(zip(content_layers, content_layer_weights, content_types, mask_types)):
        print('Layer: %s, %s, %d, %s'%(layer, content_type, weight, mask_type))
        if mask_type == 'main':
            p = sess.run(net[layer])
            x = net2[layer]
        elif mask_type =='comp':
            p = sess.run(net3[layer])
            x = net4[layer]
        else:
            return 0

        print("Latent shape: %s"%(p.shape,))
        if latent_mask[idx] is not None:
            p = latent_mask[idx] * p
            x = latent_mask[idx] * x

        shape = p.shape[1:3]
        sampling = np.random.rand(*shape) <= latent_sample_rate
        print(sampling.shape)
        sampling = sampling.reshape(1,shape[0],shape[1],1)
        p *= sampling
        x *= sampling
        p = tf.convert_to_tensor(p)
        tmp_loss = content_layer_loss(p, x, loss_type=content_type) * weight
        loss_collections.append(tmp_loss)
        content_loss += tf.reduce_sum(tmp_loss)
    content_loss /= float(len(content_layers))
    return content_loss, loss_collections

def igan_loss(sess, net_cont, net_vgg, img_cont, mask_pix):
    mask_pix = mask_pix.reshape((1, 64, 64, 1))

    sess.run(net_cont['input'].assign(img_cont))

    p = sess.run(net_cont['input'])
    x = net_vgg['input']

    p = mask_pix*p
    x = mask_pix*x

    shape = p.shape[1:3]
    sampling = np.random.rand(*shape) <= 1
    sampling = sampling.reshape(1, shape[0], shape[1], 1)

    p *= sampling
    x *= sampling

    p = tf.convert_to_tensor(p)

    loss = content_layer_loss(p, x, loss_type='Feature', method='iGAN')

    return loss, None


def vgg_img(img):
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    return bgr
