import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
key = ['plot_hidden_output']

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    if reuse is True:
        global key
        key = ['plot_hidden_output_other_towers']
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        tf.summary.histogram('fc0',x,collections=key)
        x = reshape(x, 8, 8, hidden_num, data_format)
        #x_skip = x
        #repeat_num+=1
        for idx in range(repeat_num):
            with tf.name_scope('conv%d'%idx):
                x = ResBlock_Dec(x, hidden_num, data_format=data_format)
                tf.summary.histogram('Res_out',x,collections=key)
            if idx < repeat_num - 1:
                with tf.name_scope('upsampling'): 
                    x = upscale(x, 2, data_format)
                    tf.summary.histogram('%d'%idx,x,collections=key)
        #x = tf.nn.elu(x)
        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)
        tf.summary.histogram('conv_gen',out,collections=key)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

first_D = True
def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    global first_D
    global key
    if first_D:
        key = ['plot_hidden_output']
        first_D = False
    else:
        key = ['plot_hidden_output_other_towers']
    with tf.variable_scope("D") as vs:
        # Encoder
        with tf.name_scope('Enc'):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            tf.summary.histogram('conv_input',x,collections=key)
            prev_channel_num = hidden_num
            #repeat_num += 1
            for idx in range(repeat_num):
                with tf.name_scope('conv%d'%idx):
                    channel_num = hidden_num * (idx + 1)
                    x = ResBlock_Enc(x, channel_num, data_format=data_format)
                    tf.summary.histogram('Res_out',x,collections=key)
                if idx < repeat_num - 1:
                    with tf.name_scope('downsampling'):
                        x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                        tf.summary.histogram('%d'%idx,x,collections=key)

            #x = tf.nn.elu(x)
            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, z_num, activation_fn=None)
            tf.summary.histogram('fc_code',x,collections=key)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        with tf.name_scope('Dec'):
            x = slim.fully_connected(x, num_output, activation_fn=None)
            tf.summary.histogram('fc_code',x,collections=key)

            x = reshape(x, 8, 8, hidden_num, data_format)
            #x_skip = x 
            for idx in range(repeat_num):
                with tf.name_scope('conv%d'%idx):
                    x = ResBlock_Dec(x, hidden_num, data_format=data_format)
                    tf.summary.histogram('Res_out',x,collections=key)
                if idx < repeat_num - 1:
                    with tf.name_scope('upsampling'):
                        x = upscale(x, 2, data_format)
                        tf.summary.histogram('%d'%idx,x,collections=key)
            #x = tf.nn.elu(x)
            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
            tf.summary.histogram('conv_reconstruct',out,collections=key)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables
def ResBlock_Enc(x, hidden_num, data_format):
    origin_x = x
    # Pre-activation
    x = tf.nn.elu(x)
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=None, data_format=data_format)
    shape_in = get_conv_shape(origin_x, data_format)
    shape_out= get_conv_shape(x, data_format)
    #print('SHAPE')
    #print(data_format)
    #print(shape_in, shape_out)
    if  shape_in[3] != shape_out[3]:
        origin_x = slim.conv2d(origin_x, shape_out[3], 1, 1, activation_fn=None, data_format=data_format)
    
    return x + origin_x
def ResBlock_Dec(x, hidden_num, data_format):
    origin_x = x
    # Pre-activation
    x = tf.nn.elu(x)
    x = slim.conv2d(x, hidden_num * 2, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    x = slim.conv2d(x, hidden_num * 1, 3, 1, activation_fn=None, data_format=data_format)
    #shape_in = get_conv_shape(origin_x, data_format)
    #shape_out= get_conv_shape(x, data_format)
    #if  shape_in[1] != shape_out[1]:
    #    origin_x = slim.conv2d(origin_x, shape_out[1], 1, 1, activation_fn=None, data_format=data_format)
    return x + origin_x

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
