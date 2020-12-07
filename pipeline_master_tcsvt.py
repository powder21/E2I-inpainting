import re
import tensorflow as tf
from scipy.misc import imread,imsave,imshow
from network_test import *
from util import *
import argparse
import json
import os
import glob
from hed import FCN
import time

def coarse(sess, img_file, mask_file, reuse=False):
    mask_np = np.expand_dims(np.expand_dims(imread(mask_file), 0), -1)
    mask_np = np.cast[np.float32](mask_np // 255)

    img_np = np.cast[np.float32](imread(img_file))
    img_np = img_np*(1-mask_np[0,:,:,0:1])+mask_np[0,:,:,0:1]*255
    img_np = np.expand_dims(img_np, 0)
    img_np = np.cast[np.float32]((img_np - 127.5) / 127.5)

    edge_np = np.ones_like(mask_np)*255
    edge_np = np.cast[np.float32]((edge_np - 127.5) / 127.5)

    input_shape = list(mask_np.shape)
    if input_shape[1]%8 != 0:
        input_shape[1] = input_shape[1]//8*8
    if input_shape[2]%8 != 0:
        input_shape[2] = input_shape[2]//8*8
    img_np = img_np[:,:input_shape[1],:input_shape[2],:]
    edge_np = edge_np[:, :input_shape[1], :input_shape[2], :]
    mask_np = mask_np[:, :input_shape[1], :input_shape[2], :]

    img = tf.placeholder(tf.float32, input_shape[:-1]+[3])
    edge = tf.placeholder(tf.float32, input_shape[:-1]+[1])
    mask = tf.placeholder(tf.float32, input_shape[:-1]+[1])
    feed_dict = {img: img_np, edge: edge_np[:,:,:,0:1], mask: mask_np}

    model = network(img, edge, mask, reuse=reuse)
    model.build_model()

    if not reuse:
        all_var = tf.global_variables()
        restore_var = [v for v in all_var if 'completion_net' in v.name]
        saver = tf.train.Saver(restore_var)
        saver.restore(sess, args.checkpoints_path + 'model')

    coarse_recon = sess.run(model.reconstruct_img, feed_dict)
    coarse_recon = np.cast[np.uint8](coarse_recon * 127.5 + 127.5)
    return coarse_recon[0]


def HED(sess, input_img, reuse):
    batch = 1
    data = np.cast[np.float32](input_img[:, :, ::-1])
    data = np.expand_dims(data, 0)
    data -= np.array((104.00698793, 116.66876762, 122.67891434))
    input_shape = list(input_img.shape)
    input_data = tf.placeholder(tf.float32, [batch] + input_shape)
    with tf.variable_scope('HED', reuse=reuse):
        net = FCN({'data': input_data})
        if not reuse:
            net.load('hed.npy', sess)
        out1, out2, out3, out4, out5 = net.get_output()
        output1 = out1[:, :input_shape[0], :input_shape[1], :]
        output2 = out2[:, :input_shape[0], :input_shape[1], :]
        output3 = out3[:, :input_shape[0], :input_shape[1], :]
        output4 = out4[:, :input_shape[0], :input_shape[1], :]
        output5 = out5[:, :input_shape[0], :input_shape[1], :]
        output = tf.concat([output1, output2, output3, output4, output5], -1)
        output = tf.nn.sigmoid(tf.reduce_mean(output, -1, True))

        feed = {input_data: data}
        output_np = sess.run(output[:, :, :, 0], feed_dict=feed)
        output_np = 1 - output_np

        return output_np[0]

def inpainting(sess, img_file, mask_file, edge, reuse):
    mask_np = np.expand_dims(np.expand_dims(imread(mask_file), 0), -1)
    mask_np = np.cast[np.float32](mask_np // 255)

    img_np = np.cast[np.float32](imread(img_file))
    img_np = img_np * (1 - mask_np[0, :, :, 0:1]) + mask_np[0, :, :, 0:1] * 255
    img_np = np.expand_dims(img_np, 0)
    img_np = np.cast[np.float32]((img_np - 127.5) / 127.5)

    edge_np = np.cast[np.float32](edge)
    # edge_np = np.cast[np.float32]((edge - 127.5) / 127.5)
    edge_np = np.expand_dims(np.expand_dims(edge_np, 0),-1)

    input_shape = list(mask_np.shape)
    if input_shape[1] % 8 != 0:
        input_shape[1] = input_shape[1] // 8 * 8
    if input_shape[2] % 8 != 0:
        input_shape[2] = input_shape[2] // 8 * 8
    img_np = img_np[:, :input_shape[1], :input_shape[2], :]
    edge_np = edge_np[:, :input_shape[1], :input_shape[2], :]
    mask_np = mask_np[:, :input_shape[1], :input_shape[2], :]
    edge_np = edge_np * (1 - mask_np) + mask_np
    edge_np = (edge_np - 0.5)*2

    img = tf.placeholder(tf.float32, input_shape[:-1] + [3])
    edge = tf.placeholder(tf.float32, input_shape[:-1] + [1])
    mask = tf.placeholder(tf.float32, input_shape[:-1] + [1])
    feed_dict = {img: img_np, edge: edge_np[:, :, :, 0:1], mask: mask_np}

    model = network(img, edge, mask, reuse=reuse)
    model.build_model()

    if not reuse:
        all_var = tf.global_variables()
        restore_var = [v for v in all_var if 'completion_net' in v.name]
        saver = tf.train.Saver(restore_var)
        saver.restore(sess, args.checkpoints_path + 'model')

    reconstruct_edge, reconstruct_img = sess.run([model.reconstruct_edge, model.reconstruct_img], feed_dict)
    reconstruct_edge = np.cast[np.uint8](reconstruct_edge * 127.5 + 127.5)
    reconstruct_img = np.cast[np.uint8](reconstruct_img * 127.5 + 127.5)

    return reconstruct_edge[0,:,:,0], reconstruct_img[0]


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',  type=str, default='./tcsvt_input/', help='img_dir')
parser.add_argument('--mask_dir',  type=str, default='./tcsvt_mask/', help='mask_dir')
parser.add_argument('--output_dir',  type=str, default='./tcsvt_output/', help='output_dir')
parser.add_argument('--hed_dir',  type=str, default='./hed.npy', help='hed model')
parser.add_argument('--checkpoints_path', type=str, default='./model/places2/', help='saved model checkpoint path')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
run_config = tf.ConfigProto(allow_soft_placement=True)
run_config.gpu_options.allow_growth = True
with tf.Session(config=run_config) as sess:
    image_glob = os.path.join(args.img_dir, '*.png')
    image_files = sorted(glob.glob(image_glob))
    mask_glob = os.path.join(args.mask_dir, '*.png')
    mask_files = sorted(glob.glob(mask_glob))

    coarse_list = []
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.device('/gpu:0'):
            coarse_result = coarse(sess, img_file, mask_file, reuse)

        print('{}. '.format(i + 1) + ': Coarse Done!')

        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.device('/gpu:0'):
            hed_result = HED(sess, coarse_result, reuse)
        print('{}. '.format(i + 1) + ': HED Done!')


        reuse = True
        with tf.device('/gpu:0'):
            recon_edge, recon_img = inpainting(sess,img_file,mask_file,hed_result,reuse)
        edge_path = os.path.join(args.output_dir, os.path.basename(img_file).replace('_img','_edge'))
        img_path = os.path.join(args.output_dir, os.path.basename(img_file))
        scipy.misc.imsave(edge_path, recon_edge)
        scipy.misc.imsave(img_path, recon_img)
        print('{}. '.format(i + 1) + ': fine Done!')




