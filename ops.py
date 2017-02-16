from __future__ import division
import tensorflow as tf
from tensorflow.python.framework import ops
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

import os


def batch_norm(imgs, epsilon=1e-5, momentum = 0.9, train=True, name="batch_norm"):
    imgs = imgs
    epsilon= epsilon
    momentum = momentum
    bn = tf.contrib.layers.batch_norm(imgs,
                                 decay=momentum,
                                 updates_collections=None,
                                 epsilon=epsilon,
                                 scale=True,
                                 is_training=train,
                                 scope=name)

    return bn

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, name="conv2d"):
    with tf.variable_scope(name):
        kernel_size = [k_h, k_w, input_.get_shape()[-1], output_dim]
        w = tf.get_variable('w', kernel_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        kernel_size = [k_h, k_w, output_shape[-1],input_.get_shape()[-1]]
        w = tf.get_variable('w', kernel_size, initializer=tf.random_normal_initializer(stddev=0.02))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

def lrelu(x, leak=0.3, name="lrelu"):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)

    return f1*x + f2*abs(x)


def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, matrix) + bias



def discriminator(images, batch_size, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(conv2d(images, 32, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, 64, name='d_h1_conv'), name='d_bn1'))
        h2 = lrelu(batch_norm(conv2d(h1, 128, name='d_h2_conv'), name='d_bn2'))
        h3 = lrelu(batch_norm(conv2d(h2, 256, name='d_h3_conv'), name='d_bn3'))
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
        return h4

def generator(z, output_size, batch_size, c_dim=3):
    with tf.variable_scope("generator") as scope:

        s = output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        # project `z` and reshape
        z_ = linear(z, s16*s16*512, 'g_h0_lin')

        h0 = tf.reshape(z_, [-1, s16, s16, 512])
        h0 = tf.nn.relu(batch_norm(h0, name='g_bn0'))

        h1 = deconv2d(h0,[batch_size, s8, s8, 256], name='g_h1')
        h1 = tf.nn.relu(batch_norm(h1, name='g_bn1'))

        h2 = deconv2d(h1,[batch_size, s4, s4, 64], name='g_h2')
        h2 = tf.nn.relu(batch_norm(h2, name='g_bn2'))

        h3 = deconv2d(h2, [batch_size, s2, s2, 32], name='g_h3')
        h3 = tf.nn.relu(batch_norm(h3, name='g_bn3'))

        h4 = deconv2d(h3,[batch_size, s, s, c_dim], name='g_h4')

        return tf.nn.tanh(h4)

def sampler(z, output_size, batch_size, c_dim=3):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()
        s = output_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

        # project `z` and reshape
        h0 = tf.reshape(linear(z, s16*s16*512, 'g_h0_lin'),[-1, s16, s16, 512])
        h0 = tf.nn.relu(batch_norm(h0, name='g_bn0', train=False))

        h1 = deconv2d(h0, [batch_size, s8, s8, 256], name='g_h1')
        h1 = tf.nn.relu(batch_norm(h1, name='g_bn1', train=False))

        h2 = deconv2d(h1, [batch_size, s4, s4, 64], name='g_h2')
        h2 = tf.nn.relu(batch_norm(h2, name='g_bn2', train=False))

        h3 = deconv2d(h2, [batch_size, s2, s2, 32], name='g_h3')
        h3 = tf.nn.relu(batch_norm(h3, name='g_bn3', train=False))

        h4 = deconv2d(h3, [batch_size, s, s, c_dim], name='g_h4')

        return tf.nn.tanh(h4)

def image_save(batch_img, epoch, iter, OUTPUT_PATH):
    """
    Deprocess the generator output and saves the results
    :param batch_res: list of images
    :param grid_pad: size of the blank between two images
    """
    #create an output grid tp hold the images
    grid_shape = [8, 8]
    grid_pad = 5
    img_h,img_w = batch_img.shape[1:3]
    grid_h = img_h * grid_shape[0] + grid_pad * (grid_shape[0] - 1)
    grid_w = img_w * grid_shape[1] + grid_pad * (grid_shape[1] - 1)
    img_grid = np.zeros((grid_h,grid_w,3),dtype=np.uint8)

    #loop to save generator outputs
    for i, img in enumerate(batch_img):
        if i >= grid_shape[0] * grid_shape[1]:
            break

        #deprocessing(tanh)
        img = (img + 1.0)*127.5
        img = img.astype(np.uint8)#to show

        #add the image to the image grid
        row = (i//grid_shape[0])*(img_h + grid_pad)
        col = (i%grid_shape[1])*(img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w,:] = img

    #save the output image
    fname = "{0}_{1}.jpg".format(epoch, iter)
    imsave(os.path.join(OUTPUT_PATH, fname), img_grid)

import scipy.misc
def image_read3(paths, crop_len=224):
    """
    read multiple images
    """
    imgs = []
    #force to 3-channel images
    for path in paths:
        img = imread(path)
        if img.ndim == 2:
            img = gray2rgb(img)
        elif img.shape[2]==4:
            img = img[:,:,:3]

        img = scipy.misc.imresize(img, [crop_len, crop_len])
        img = img/127.5 - 1
        imgs.append(img)

    return np.array(imgs, dtype=np.float32)

def read_name(path,file):
    f = open(os.path.join(path, file), 'r')
    list = f.readlines()
    f.close()
    name = [] #image is the name list of the image
    for item in list:
        item = item.strip()
        name.append(os.path.join(path, item))
    return name
def image_read2(paths, scale_len=150, crop_len=128):
    """
    read multiple images
    """
    imgs = []
    #force to 3-channel images
    for path in paths:
        img = imread(path)
        if img.ndim == 2:
            img = gray2rgb(img)
        elif img.shape[2]==4:
            img = img[:,:,:3]

        #compute the resize dimension
        crop_h = scale_len
        crop_w = scale_len
        h, w = img.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        img = scipy.misc.imresize(img[j:j + crop_h, i:i + crop_w], [crop_len, crop_len])
        #the size of img is [64, 64]
        img = img/127.5 - 1
        imgs.append(img)

    return np.array(imgs, dtype=np.float32)

