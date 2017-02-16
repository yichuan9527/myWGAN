import time
from ops import *
import glob
import os
import tensorflow as tf
import numpy as np


batch_size = 128
z_dim = 100
c_dim = 3
sample_size = 128
output_size = 128
learning_rate = 5e-5
image_size = 128
epoches = 600
train_size = 100000000
clamp_lower = -0.01
clamp_upper = 0.01
is_crop = True
sample_dir = 'picture1'
data_path = "/home/jinlin/tensorflow/GAN/DCGAN/data/celebA"
k = 5

images = tf.placeholder(tf.float32, [batch_size, output_size, output_size, c_dim], name='real_images')
sample_images = tf.placeholder(tf.float32, [sample_size, output_size, output_size, c_dim], name='sample_images')
z = tf.placeholder(tf.float32, [None, z_dim], name='z')

G = generator(z, image_size, batch_size)
tf.histogram_summary('g', G)
D = discriminator(images, batch_size)
tf.histogram_summary('D', D)

_sample = sampler(z, image_size, batch_size)
D_ = discriminator(G, batch_size, reuse=True)

d_loss = tf.reduce_mean(D_ - D)
tf.scalar_summary('d_loss', d_loss)
g_loss = -tf.reduce_mean(D_)
tf.scalar_summary('g_loss', g_loss)


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]


#train
d_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
clipped_var_d = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in d_vars]
with tf.control_dependencies([d_optim]):
    d_optim = tf.tuple(clipped_var_d)

saver = tf.train.Saver()
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
merge = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('logs/',sess.graph_def)

data = glob.glob(os.path.join(data_path, "*"))
sample_z = np.random.uniform(-1, 1, size=[sample_size, z_dim]).astype(np.float32)
sample_files = data[0:sample_size]
sample_images = image_read2(sample_files)


for epoch in range(epoches):
    batch_idxs = min(len(data), train_size) // batch_size - 1

    for idx in range(0, batch_idxs):
        start_time = time.time()
        batch_files = data[idx * batch_size:(idx + 1) * batch_size]
        batch_images = image_read2(batch_files)
        #imread2(image_path)
        #batch_images = np.array(batch).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                # Update D network
        for i in range(k):
            sess.run(g_optim, feed_dict={z: batch_z})

        # Update G network(twice)
        sess.run(d_optim, feed_dict={z: batch_z, images: batch_images})


        errG = sess.run(g_loss, feed_dict={z: batch_z})
        errD = sess.run(d_loss, feed_dict={z: batch_z, images: batch_images})

        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,time.time() - start_time, errD, errG))
        if idx % 100 == 0:
            save_image = sess.run(_sample, feed_dict={z: sample_z})
            image_save(save_image, epoch, idx, sample_dir)
            summary_str = sess.run(merge, feed_dict={z: batch_z, images: batch_images})
            summary_writer.add_summary(summary_str, (epoch+1)*(idx+1))

    #samples = sess.run(_sample, feed_dict={z:sample_z})

    mysamples, myd_loss, myg_loss = sess.run([_sample, d_loss, g_loss],feed_dict={z: sample_z, images: sample_images})
    image_save(mysamples, epoch, 'over', sample_dir)
    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (myd_loss, myg_loss))
    #if i % 10 == 0:
    #    save_path = saver.save(sess, 'cht_point', i)
    save_path = saver.save(sess, 'ckt_point/ckt_point', epoch)











